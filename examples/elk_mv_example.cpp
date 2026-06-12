/**
 * elk_mv_example — Elk movement state detection via multivariate HMM (v4 API).
 *
 * Validates the v4 IndependentComponentsDistribution API by fitting the same
 * Gamma + von Mises model used by the moveHMM R package to the Morales et al.
 * (2004) elk GPS dataset (4 animals, 725 observations).
 *
 * The conditional independence assumption (step length ⊥ turning angle | state)
 * is statistically appropriate here: within-state Pearson correlation between
 * log(step) and turning angle is ≈ −0.05 to −0.08, indistinguishable from zero.
 * See the "Model selection" note in the output for details.
 *
 * For a comparison of DiagonalGaussian vs FullCovarianceGaussian on data where
 * within-state correlation is genuinely strong (SPY + QQQ, ρ = 0.83–0.92),
 * see mv_regime_example.cpp.
 *
 * Data preparation:
 *   Rscript scripts/prepare_elk_data.R
 *   Exports elk_<id>_obs.csv to /tmp/ (or pass the directory as argument 1).
 *
 * moveHMM reference (Gamma + von Mises, 2 states, Michelot et al. 2016):
 *   State 0 (encamped):   step mean=373.8m  step sd=399.0m  angle kappa=0.592
 *   State 1 (travelling): step mean=3247.3m step sd=4393.5m angle kappa=0.208
 *   Transition:           [[0.9115, 0.0885], [0.2002, 0.7998]]
 *   Log-likelihood: -6935.6  (moveHMM includes a zero-step mass; libhmm does not)
 *   Wall time: ~2000 ms
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/distributions/gamma_distribution.h"
#include "libhmm/distributions/independent_components_distribution.h"
#include "libhmm/distributions/von_mises_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/model_selection.h"

using namespace libhmm;

// =============================================================================
// Data loading
// =============================================================================

struct RawSeq {
    std::vector<double> steps;
    std::vector<double> angles;
    std::size_t size() const { return steps.size(); }
};

static RawSeq read_csv(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    RawSeq seq;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        const std::size_t c = line.find(',');
        if (c == std::string::npos)
            continue;
        seq.steps.push_back(std::stod(line.substr(0, c)));
        seq.angles.push_back(std::stod(line.substr(c + 1)));
    }
    return seq;
}

static MultiObservationLists to_mv(const std::vector<RawSeq> &seqs) {
    MultiObservationLists out;
    for (const auto &s : seqs) {
        ObservationMatrix mat(s.size(), 2);
        for (std::size_t t = 0; t < s.size(); ++t) {
            mat(t, 0) = s.steps[t];
            mat(t, 1) = s.angles[t];
        }
        out.push_back(std::move(mat));
    }
    return out;
}

// =============================================================================
// Helpers
// =============================================================================

static double total_logp(HmmMV &hmm, const MultiObservationLists &lists) {
    double lp = 0.0;
    for (const auto &seq : lists) {
        BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, seq);
        lp += fbc.getLogProbability();
    }
    return lp;
}

static double run_bwt(HmmMV &hmm, const MultiObservationLists &lists, int max_iter = 200,
                      double tol = 1e-4) {
    BasicBaumWelchTrainer<ObservationVectorView> trainer(hmm, lists);
    double prev = total_logp(hmm, lists);
    int conv = -1;
    for (int k = 0; k < max_iter; ++k) {
        trainer.train();
        const double cur = total_logp(hmm, lists);
        const double delta = cur - prev;
        if (k > 0 && delta >= -1e-8 && delta < tol) {
            if (conv < 0)
                conv = k;
        }
        if (conv >= 0 && k >= conv + 2)
            break;
        prev = cur;
    }
    return total_logp(hmm, lists);
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char *argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "/tmp";

    std::cout << "Elk Movement -- IndependentComponents HMM vs moveHMM (v4 API)\n";
    std::cout << "=============================================================\n\n";

    // -------------------------------------------------------------------------
    // Load data
    // -------------------------------------------------------------------------
    const std::vector<std::string> elk_ids = {"elk_115", "elk_163", "elk_287", "elk_363"};
    std::vector<RawSeq> raw;
    std::size_t n_total = 0;
    std::cout << "Loading from " << data_dir << "/\n";
    for (const auto &id : elk_ids) {
        try {
            auto s = read_csv(data_dir + "/" + id + "_obs.csv");
            std::cout << "  " << id << ": " << s.size() << " obs\n";
            n_total += s.size();
            raw.push_back(std::move(s));
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << "\nRun: Rscript scripts/prepare_elk_data.R\n";
            return 1;
        }
    }
    std::cout << "Total: " << n_total << " observations\n\n";

    const auto lists = to_mv(raw);

    // -------------------------------------------------------------------------
    // Model: IndependentComponents(Gamma, VonMises) -- v4 MV API
    //
    // Fits the same conditional-independence model as moveHMM.  Each state
    // has an independent GammaDistribution for step length and a
    // VonMisesDistribution for turning angle; the joint log-probability is
    // their sum.
    // -------------------------------------------------------------------------
    std::cout << "Model: IndependentComponents(Gamma, VonMises)\n";
    std::cout << "  Observation vector: (step_length_m, turning_angle_rad)\n";
    std::cout << "  Conditional independence of step and angle given state.\n\n";

    HmmMV hmm(2);
    {
        Matrix tr(2, 2);
        tr(0, 0) = 0.9;
        tr(0, 1) = 0.1;
        tr(1, 0) = 0.1;
        tr(1, 1) = 0.9;
        hmm.setTrans(tr);
        Vector pi(2);
        pi(0) = 0.5;
        pi(1) = 0.5;
        hmm.setPi(pi);

        std::vector<std::unique_ptr<EmissionDistribution>> c0;
        c0.push_back(std::make_unique<GammaDistribution>(1.0, 100.0));
        c0.push_back(std::make_unique<VonMisesDistribution>(0.0, 0.1));
        hmm.setDistribution(0, std::make_unique<IndependentComponentsDistribution>(std::move(c0)));

        std::vector<std::unique_ptr<EmissionDistribution>> c1;
        c1.push_back(std::make_unique<GammaDistribution>(1.0, 1000.0));
        c1.push_back(std::make_unique<VonMisesDistribution>(0.0, 1.0));
        hmm.setDistribution(1, std::make_unique<IndependentComponentsDistribution>(std::move(c1)));
    }

    const auto t0 = std::chrono::steady_clock::now();
    const double ll = run_bwt(hmm, lists);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::cout << std::fixed << std::setprecision(4);
    for (int s = 0; s < 2; ++s) {
        const auto &ic =
            static_cast<const IndependentComponentsDistribution &>(hmm.getDistribution(s));
        const auto &gd = static_cast<const GammaDistribution &>(ic.getComponent(0));
        const auto &vm = static_cast<const VonMisesDistribution &>(ic.getComponent(1));
        std::cout << "State " << s << " (" << (s == 0 ? "encamped  " : "travelling") << "):\n"
                  << "  Gamma:    k=" << gd.getK() << "  theta=" << gd.getTheta()
                  << "  mean=" << gd.getK() * gd.getTheta() << "m"
                  << "  sd=" << std::sqrt(gd.getK()) * gd.getTheta() << "m\n"
                  << "  VonMises: mu=" << vm.getMu() << "  kappa=" << vm.getKappa() << "\n";
    }
    const auto &trans = hmm.getTrans();
    std::cout << "\nTransition:  [[" << trans(0, 0) << ", " << trans(0, 1) << "], [" << trans(1, 0)
              << ", " << trans(1, 1) << "]]\n"
              << "Log-likelihood: " << ll << "\n"
              << "Wall time: " << std::setprecision(1) << wall_ms << " ms\n\n";

    // -------------------------------------------------------------------------
    // Comparison table
    // -------------------------------------------------------------------------
    std::cout << "=== Comparison: libhmm v4 MV API vs moveHMM ===\n\n";
    std::cout << std::setw(28) << " " << std::setw(14) << "libhmm v4" << std::setw(14)
              << "moveHMM\n";
    std::cout << std::string(56, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);

    const auto &ic0 =
        static_cast<const IndependentComponentsDistribution &>(hmm.getDistribution(0));
    const auto &ic1 =
        static_cast<const IndependentComponentsDistribution &>(hmm.getDistribution(1));
    const auto &gd0 = static_cast<const GammaDistribution &>(ic0.getComponent(0));
    const auto &vm0 = static_cast<const VonMisesDistribution &>(ic0.getComponent(1));
    const auto &gd1 = static_cast<const GammaDistribution &>(ic1.getComponent(0));
    const auto &vm1 = static_cast<const VonMisesDistribution &>(ic1.getComponent(1));

    struct Row {
        const char *name;
        double lib;
        double ref;
    };
    const Row rows[] = {
        {"State 0 step mean (m)", gd0.getK() * gd0.getTheta(), 373.8},
        {"State 0 step sd (m)", std::sqrt(gd0.getK()) * gd0.getTheta(), 399.0},
        {"State 0 angle kappa", vm0.getKappa(), 0.592},
        {"State 1 step mean (m)", gd1.getK() * gd1.getTheta(), 3247.3},
        {"State 1 step sd (m)", std::sqrt(gd1.getK()) * gd1.getTheta(), 4393.5},
        {"State 1 angle kappa", vm1.getKappa(), 0.208},
        {"A[0->0]", trans(0, 0), 0.9115},
        {"A[1->0]", trans(1, 0), 0.2002},
        {"Log-likelihood", ll, -6935.6},
    };
    for (const auto &r : rows)
        std::cout << std::setw(28) << r.name << std::setw(14) << r.lib << std::setw(14) << r.ref
                  << "\n";
    const std::string wall_str = std::to_string(static_cast<int>(wall_ms)) + " ms";
    std::cout << std::setw(28) << "Wall time" << std::setw(14) << wall_str << std::setw(14)
              << "~2000 ms\n";

    std::cout << "\nNotes:\n"
              << "  Both models: Gamma step + von Mises angle, 4 tracks, independent\n"
              << "  step and angle given state.  moveHMM models the single zero-step\n"
              << "  observation via a zero-mass parameter; libhmm omits it, so libhmm's\n"
              << "  LL is slightly less negative.\n\n";

    // -------------------------------------------------------------------------
    // Model selection justification
    // -------------------------------------------------------------------------
    std::cout << "Model selection -- why IndependentComponents is correct here:\n"
              << "  Hard-assigning observations to states via k-means on log(step)\n"
              << "  gives within-state Pearson r(log_step, angle) = -0.05 to -0.08\n"
              << "  across both states -- indistinguishable from zero.  Adding\n"
              << "  cross-variable covariance (DiagonalGaussian or FullCovGaussian)\n"
              << "  would fit noise, not signal.\n\n"
              << "  For a dataset where within-state correlation is genuinely strong,\n"
              << "  see mv_regime_example.cpp (SPY + QQQ, rho = 0.83-0.92 per regime).\n";

    return 0;
}
