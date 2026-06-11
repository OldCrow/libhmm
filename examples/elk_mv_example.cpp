/**
 * elk_mv_example — Elk movement state detection via multivariate HMM (v4).
 *
 * Same dataset as elk_movement_example.cpp (Morales et al. 2004, 4 elk GPS
 * tracks), exercised with three models using the v4 MV API:
 *
 * Model A — IndependentComponentsDistribution (Gamma + von Mises)
 *   Observation vector: (step_length_m, turning_angle_rad)
 *   Conditional independence of step and angle given state — same statistical
 *   assumption as elk_movement_example.cpp.  Log-likelihood and parameters
 *   should match that example, validating the v4 MV API.
 *
 * Model B — DiagonalGaussianDistribution (log-step, angle)
 *   Observation vector: (log(step_length_m), turning_angle_rad)
 *   Models log-normal step lengths with independent Gaussian components.
 *   Equivalent to assuming conditional independence on the log scale.
 *
 * Model C — FullCovarianceGaussianDistribution (log-step, angle)
 *   Same observation vector as B.  Relaxes the independence assumption by
 *   fitting a full 2×2 covariance matrix per state.
 *
 * Models B and C operate on the same observation space so their BIC values
 * can be compared directly.  Within-state correlation between log_step and
 * angle is near zero for this dataset (see analysis in AGENTS.md), so we
 * expect BIC to favour the simpler Model B — validating the library's ability
 * to select the correct complexity level.
 *
 * Model A is a validation baseline only; its LL is on a different observation
 * space and cannot be compared to B or C via BIC.
 *
 * Data preparation (same script as elk_movement_example):
 *   Rscript scripts/prepare_elk_data.R
 *   Exports elk_<id>_obs.csv to /tmp/ (or pass dir as argument 1).
 *
 * moveHMM reference (Gamma + von Mises, 2 states):
 *   State 0 (encamped):   step mean=373.8m; angle kappa=0.592
 *   State 1 (travelling): step mean=3247.3m; angle kappa=0.208
 *   log-likelihood: -6935.6
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"
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

static RawSeq read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    RawSeq seq;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const std::size_t c = line.find(',');
        if (c == std::string::npos) continue;
        seq.steps.push_back(std::stod(line.substr(0, c)));
        seq.angles.push_back(std::stod(line.substr(c + 1)));
    }
    return seq;
}

// =============================================================================
// Convert raw sequences to ObservationMatrix lists
// =============================================================================

/// Model A: rows are (step_length, turning_angle).
MultiObservationLists to_mv_a(const std::vector<RawSeq>& seqs) {
    MultiObservationLists out;
    for (const auto& s : seqs) {
        ObservationMatrix mat(s.size(), 2);
        for (std::size_t t = 0; t < s.size(); ++t) {
            mat(t, 0) = s.steps[t];
            mat(t, 1) = s.angles[t];
        }
        out.push_back(std::move(mat));
    }
    return out;
}

/// Model B: rows are (log(step_length), turning_angle).
/// Zero or sub-metre steps are clamped to log(1.0) = 0 (< 1 m is measurement noise).
MultiObservationLists to_mv_b(const std::vector<RawSeq>& seqs) {
    MultiObservationLists out;
    for (const auto& s : seqs) {
        ObservationMatrix mat(s.size(), 2);
        for (std::size_t t = 0; t < s.size(); ++t) {
            mat(t, 0) = std::log(std::max(s.steps[t], 1.0));
            mat(t, 1) = s.angles[t];
        }
        out.push_back(std::move(mat));
    }
    return out;
}

// =============================================================================
// Helpers
// =============================================================================

/// Total log-probability of all sequences under hmm.
static double total_logp(HmmMV& hmm, const MultiObservationLists& lists) {
    double lp = 0.0;
    for (const auto& seq : lists) {
        BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, seq);
        lp += fbc.getLogProbability();
    }
    return lp;
}

/// Run Baum-Welch to convergence (or max_iter); return final log-likelihood.
static double run_bwt(HmmMV& hmm, const MultiObservationLists& lists,
                       int max_iter = 200, double tol = 1e-4) {
    BasicBaumWelchTrainer<ObservationVectorView> trainer(hmm, lists);
    double prev = total_logp(hmm, lists);
    int conv = -1;
    for (int k = 0; k < max_iter; ++k) {
        trainer.train();
        const double cur = total_logp(hmm, lists);
        const double delta = cur - prev;
        if (k > 0 && delta >= -1e-8 && delta < tol) {
            if (conv < 0) conv = k;
        }
        if (conv >= 0 && k >= conv + 2) break;
        prev = cur;
    }
    return total_logp(hmm, lists);
}

/// Build a uniform-transition 2-state HmmMV.
static HmmMV make_base_hmm() {
    HmmMV hmm(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.9; trans(0, 1) = 0.1;
    trans(1, 0) = 0.1; trans(1, 1) = 0.9;
    hmm.setTrans(trans);
    Vector pi(2); pi(0) = 0.5; pi(1) = 0.5;
    hmm.setPi(pi);
    return hmm;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "/tmp";

    std::cout << "Elk Movement — Multivariate HMM comparison (v4 API)\n";
    std::cout << "====================================================\n\n";

    // -------------------------------------------------------------------------
    // Load data
    // -------------------------------------------------------------------------
    const std::vector<std::string> elk_ids = {"elk_115", "elk_163", "elk_287", "elk_363"};
    std::vector<RawSeq> raw;
    std::size_t n_total = 0;
    std::cout << "Loading from " << data_dir << "/\n";
    for (const auto& id : elk_ids) {
        try {
            auto s = read_csv(data_dir + "/" + id + "_obs.csv");
            std::cout << "  " << id << ": " << s.size() << " obs\n";
            n_total += s.size();
            raw.push_back(std::move(s));
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what()
                      << "\nRun: Rscript scripts/prepare_elk_data.R\n";
            return 1;
        }
    }
    std::cout << "Total: " << n_total << " observations\n\n";

    const auto lists_a = to_mv_a(raw);
    const auto lists_b = to_mv_b(raw);

    // =========================================================================
    // Model A — IndependentComponents (Gamma + von Mises) via v4 MV API
    // =========================================================================
    std::cout << "--- Model A: IndependentComponents (Gamma + von Mises) ---\n\n";
    std::cout << "Observation vector: (step_length_m, turning_angle_rad)\n";
    std::cout << "Same conditional-independence assumption as elk_movement_example.cpp.\n\n";

    HmmMV hmm_a = make_base_hmm();
    {
        // State 0 (encamped): short steps, high kappa (concentrated around 0? or directional)
        std::vector<std::unique_ptr<EmissionDistribution>> c0;
        c0.push_back(std::make_unique<GammaDistribution>(1.0, 100.0));   // step
        c0.push_back(std::make_unique<VonMisesDistribution>(0.0, 0.1));  // angle
        hmm_a.setDistribution(0, std::make_unique<IndependentComponentsDistribution>(std::move(c0)));

        // State 1 (travelling): long steps, low kappa (diffuse angles)
        std::vector<std::unique_ptr<EmissionDistribution>> c1;
        c1.push_back(std::make_unique<GammaDistribution>(1.0, 1000.0));  // step
        c1.push_back(std::make_unique<VonMisesDistribution>(0.0, 1.0));  // angle
        hmm_a.setDistribution(1, std::make_unique<IndependentComponentsDistribution>(std::move(c1)));
    }

    const auto t_a0 = std::chrono::steady_clock::now();
    const double ll_a = run_bwt(hmm_a, lists_a);
    const double wall_a = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_a0).count();

    // Print Model A results
    std::cout << std::fixed << std::setprecision(4);
    for (int s = 0; s < 2; ++s) {
        const auto& ic = static_cast<const IndependentComponentsDistribution&>(
            hmm_a.getDistribution(s));
        const auto& gd = static_cast<const GammaDistribution&>(ic.getComponent(0));
        const auto& vm = static_cast<const VonMisesDistribution&>(ic.getComponent(1));
        std::cout << "State " << s << " (" << (s==0?"encamped  ":"travelling") << "):\n";
        std::cout << "  Gamma:    k=" << gd.getK() << "  theta=" << gd.getTheta()
                  << "  mean=" << gd.getK()*gd.getTheta() << "m\n";
        std::cout << "  VonMises: mu=" << vm.getMu() << "  kappa=" << vm.getKappa() << "\n";
    }
    std::cout << "Log-likelihood: " << ll_a << "\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_a << " ms\n\n";

    // =========================================================================
    // Model B — DiagonalGaussian on (log-step, angle)
    // =========================================================================
    std::cout << "--- Model B: DiagonalGaussian (log-step, angle) ---\n\n";
    std::cout << "Observation vector: (log(step_length_m), turning_angle_rad)\n";
    std::cout << "Assumes conditional independence on the log scale.\n\n";

    HmmMV hmm_b = make_base_hmm();
    {
        auto d0 = std::make_unique<DiagonalGaussianDistribution>(2);
        // Encamped: log(380)≈5.94, angle near 0, moderate variances
        d0->setParameters({std::log(380.0), 0.0}, {1.5, 1.0});
        hmm_b.setDistribution(0, std::move(d0));

        auto d1 = std::make_unique<DiagonalGaussianDistribution>(2);
        // Travelling: log(3200)≈8.07, angle near 0, larger step variance
        d1->setParameters({std::log(3200.0), 0.0}, {2.5, 1.0});
        hmm_b.setDistribution(1, std::move(d1));
    }

    const auto t_b0 = std::chrono::steady_clock::now();
    const double ll_b = run_bwt(hmm_b, lists_b);
    const double wall_b = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_b0).count();

    std::cout << std::fixed << std::setprecision(4);
    for (int s = 0; s < 2; ++s) {
        const auto& dg = static_cast<const DiagonalGaussianDistribution&>(
            hmm_b.getDistribution(s));
        std::cout << "State " << s << " (" << (s==0?"encamped  ":"travelling") << "):\n";
        std::cout << "  mean(log-step)=" << dg.getMean()[0]
                  << "  step mean~" << std::exp(dg.getMean()[0]) << "m"
                  << "  var(log-step)=" << dg.getVariance()[0] << "\n";
        std::cout << "  mean(angle)=" << dg.getMean()[1]
                  << "  var(angle)=" << dg.getVariance()[1] << "\n";
    }
    std::cout << "Log-likelihood: " << ll_b << "\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_b << " ms\n\n";

    // =========================================================================
    // Model C — FullCovarianceGaussian on (log-step, angle)
    // =========================================================================
    std::cout << "--- Model C: FullCovarianceGaussian (log-step, angle) ---\n\n";
    std::cout << "Observation vector: (log(step_length_m), turning_angle_rad)\n";
    std::cout << "Relaxes independence: fits a full 2x2 covariance per state.\n\n";

    HmmMV hmm_c = make_base_hmm();
    {
        // Use identity covariance as starting point; Baum-Welch will fit Σ.
        auto c0 = std::make_unique<FullCovarianceGaussianDistribution>(2);
        c0->setMean({std::log(380.0), 0.0});
        hmm_c.setDistribution(0, std::move(c0));

        auto c1 = std::make_unique<FullCovarianceGaussianDistribution>(2);
        c1->setMean({std::log(3200.0), 0.0});
        hmm_c.setDistribution(1, std::move(c1));
    }

    const auto t_c0 = std::chrono::steady_clock::now();
    const double ll_c = run_bwt(hmm_c, lists_b);
    const double wall_c = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_c0).count();

    std::cout << std::fixed << std::setprecision(4);
    for (int s = 0; s < 2; ++s) {
        const auto& fc = static_cast<const FullCovarianceGaussianDistribution&>(
            hmm_c.getDistribution(s));
        std::cout << "State " << s << " (" << (s==0?"encamped  ":"travelling") << "):\n";
        std::cout << "  mean(log-step)=" << fc.getMean()[0]
                  << "  step mean~" << std::exp(fc.getMean()[0]) << "m\n";
        const auto& C = fc.getCovariance();
        std::cout << "  Σ=[[" << C(0,0) << ", " << C(0,1) << "], ["
                  << C(1,0) << ", " << C(1,1) << "]]\n";
        const double rho = C(0,1) / std::sqrt(C(0,0) * C(1,1));
        std::cout << "  corr(log-step, angle)=" << rho << "\n";
    }
    std::cout << "Log-likelihood: " << ll_c << "\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_c << " ms\n\n";

    // =========================================================================
    // Model comparison
    // =========================================================================
    const std::size_t k_a = count_free_parameters(hmm_a);
    const std::size_t k_b = count_free_parameters(hmm_b);
    const std::size_t k_c = count_free_parameters(hmm_c);
    const double bic_b = compute_bic(ll_b, k_b, n_total);
    const double bic_c = compute_bic(ll_c, k_c, n_total);

    std::cout << "=== Model comparison ===\n\n";
    std::cout << std::setw(30) << " "
              << std::setw(12) << "Model A"
              << std::setw(12) << "Model B"
              << std::setw(12) << "Model C"
              << std::setw(14) << "moveHMM ref\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(30) << "Log-likelihood"
              << std::setw(12) << ll_a
              << std::setw(12) << ll_b
              << std::setw(12) << ll_c
              << std::setw(14) << -6935.6 << "\n";
    std::cout << std::setw(30) << "Free parameters k"
              << std::setw(12) << k_a
              << std::setw(12) << k_b
              << std::setw(12) << k_c
              << std::setw(14) << "11" << "\n";
    std::cout << std::setw(30) << "BIC (B vs C only)"
              << std::setw(12) << "n/a"
              << std::setw(12) << bic_b
              << std::setw(12) << bic_c
              << std::setw(14) << "n/a" << "\n";
    std::cout << std::setw(30) << "Wall time (ms)"
              << std::setw(12) << static_cast<int>(wall_a)
              << std::setw(12) << static_cast<int>(wall_b)
              << std::setw(12) << static_cast<int>(wall_c)
              << std::setw(14) << "~2000" << "\n";

    std::cout << "\nNotes:\n";
    std::cout << "  Model A: v4 IndependentComponents — same model as elk_movement_example.cpp.\n";
    std::cout << "  Log-likelihoods for A vs B/C are NOT directly comparable (different\n";
    std::cout << "  observation spaces: raw step vs log-step).\n";
    std::cout << "  BIC comparison is valid only between B and C (same (log-step, angle) space).\n";
    if (bic_b < bic_c)
        std::cout << "  -> Model B wins: within-state log-step/angle correlation is negligible;\n"
                     "     the independence assumption is appropriate for this dataset.\n";
    else
        std::cout << "  -> Model C wins: within-state log-step/angle correlation is informative.\n";

    return 0;
}
