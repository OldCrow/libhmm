/**
 * mv_gaussian_example — multivariate HMM demonstration (v4).
 *
 * Fits a 2-state, 2-dimensional DiagonalGaussianDistribution HMM to
 * synthetic data drawn from two well-separated Gaussian clusters.
 * No external data required — runs standalone.
 *
 * Demonstrates the full v4 MV workflow:
 *   1. Generate synthetic 2D data from a known 2-state Gaussian mixture.
 *   2. Construct HmmMV with DiagonalGaussianDistribution emissions.
 *   3. Initialise emission parameters via kmeans_init (k-means++).
 *   4. Train with BasicBaumWelchTrainer<ObservationVectorView>.
 *   5. Score sequences with BasicForwardBackwardCalculator<ObservationVectorView>.
 *   6. Decode the most likely state sequence with BasicViterbiCalculator<OVV>.
 *   7. Save and reload the model with save_json_mv / load_json_mv.
 *
 * True model (used to generate data):
 *   State 0 — "low"  cluster: μ=[0, 0],   σ²=[1, 1]
 *   State 1 — "high" cluster: μ=[5, 5],   σ²=[1, 1]
 *   Transition: self-loop 0.85, switch 0.15
 *   pi: [0.5, 0.5]
 */

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/calculators/basic_viterbi_calculator.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/io/hmm_json.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/kmeans_init.h"

using namespace libhmm;

// =============================================================================
// Helpers
// =============================================================================

namespace {

/// Build the ground-truth HmmMV.
HmmMV make_true_hmm() {
    HmmMV hmm(2);
    // State 0: low-activity cluster near origin
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2, 0.0, 1.0));
    // State 1: high-activity cluster
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(2, 5.0, 1.0));

    Matrix trans(2, 2);
    trans(0, 0) = 0.85; trans(0, 1) = 0.15;
    trans(1, 0) = 0.15; trans(1, 1) = 0.85;
    hmm.setTrans(trans);

    Vector pi(2); pi(0) = 0.5; pi(1) = 0.5;
    hmm.setPi(pi);
    return hmm;
}

/// Sample one observation sequence of length T from @p true_hmm.
ObservationMatrix sample_sequence(const HmmMV& true_hmm, std::size_t T, std::mt19937_64& rng) {
    const std::size_t N = static_cast<std::size_t>(true_hmm.getNumStates());
    const auto& pi    = true_hmm.getPi();
    const auto& trans = true_hmm.getTrans();

    // Sample initial state from pi.
    std::discrete_distribution<std::size_t> pi_dist(pi.data(), pi.data() + N);
    std::size_t state = pi_dist(rng);

    ObservationMatrix mat(T, 2);
    for (std::size_t t = 0; t < T; ++t) {
        // Emit from the current state's diagonal Gaussian.
        const auto& d = static_cast<const DiagonalGaussianDistribution&>(
            true_hmm.getDistribution(state));
        const auto& mu  = d.getMean();
        const auto& var = d.getVariance();
        for (std::size_t dim = 0; dim < 2; ++dim) {
            std::normal_distribution<double> nd(mu[dim], std::sqrt(var[dim]));
            mat(t, dim) = nd(rng);
        }
        // Transition.
        const double* row = trans.data() + state * N;
        std::discrete_distribution<std::size_t> trans_dist(row, row + N);
        state = trans_dist(rng);
    }
    return mat;
}

/// Compute total log-probability of @p lists under @p hmm.
double total_log_prob(HmmMV& hmm, const MultiObservationLists& lists) {
    double lp = 0.0;
    for (const auto& seq : lists) {
        BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, seq);
        lp += fbc.getLogProbability();
    }
    return lp;
}

/// Print the mean and variance for each state.
void print_emissions(const HmmMV& hmm) {
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());
    for (std::size_t i = 0; i < N; ++i) {
        const auto& d = static_cast<const DiagonalGaussianDistribution&>(hmm.getDistribution(i));
        std::cout << "  State " << i << ": μ=[" << std::fixed << std::setprecision(3)
                  << d.getMean()[0] << ", " << d.getMean()[1] << "]  σ²=["
                  << d.getVariance()[0] << ", " << d.getVariance()[1] << "]\n";
    }
}

} // namespace

// =============================================================================
// main
// =============================================================================

int main() {
    std::cout << "=== libhmm v4 — Multivariate HMM (DiagonalGaussian, 2D) ===\n\n";

    // -------------------------------------------------------------------------
    // 1. Generate synthetic training data
    // -------------------------------------------------------------------------
    std::mt19937_64 rng(42);
    constexpr std::size_t N_SEQS = 30;
    constexpr std::size_t SEQ_LEN = 20;

    HmmMV true_hmm = make_true_hmm();
    MultiObservationLists training_data;
    training_data.reserve(N_SEQS);
    for (std::size_t s = 0; s < N_SEQS; ++s)
        training_data.push_back(sample_sequence(true_hmm, SEQ_LEN, rng));

    std::cout << "Generated " << N_SEQS << " sequences of length " << SEQ_LEN
              << " from the true model.\n";
    std::cout << "True emission parameters:\n";
    print_emissions(true_hmm);
    std::cout << '\n';

    // -------------------------------------------------------------------------
    // 2. Construct a fresh HmmMV to fit
    // -------------------------------------------------------------------------
    HmmMV fitted_hmm(2);
    fitted_hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2));
    fitted_hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(2));

    Matrix trans(2, 2);
    trans(0, 0) = 0.7; trans(0, 1) = 0.3;
    trans(1, 0) = 0.3; trans(1, 1) = 0.7;
    fitted_hmm.setTrans(trans);
    Vector pi(2); pi(0) = 0.5; pi(1) = 0.5;
    fitted_hmm.setPi(pi);

    // -------------------------------------------------------------------------
    // 3. k-means++ initialisation
    // -------------------------------------------------------------------------
    const double lp_before_init = total_log_prob(fitted_hmm, training_data);
    kmeans_init(fitted_hmm, training_data, rng);
    const double lp_after_init = total_log_prob(fitted_hmm, training_data);

    std::cout << "After kmeans_init:\n";
    print_emissions(fitted_hmm);
    std::cout << "  total logP: " << std::fixed << std::setprecision(2)
              << lp_before_init << "  →  " << lp_after_init
              << "  (improvement: " << (lp_after_init - lp_before_init) << ")\n\n";

    // -------------------------------------------------------------------------
    // 4. Baum-Welch training
    // -------------------------------------------------------------------------
    constexpr int MAX_ITERS = 50;
    constexpr double TOLERANCE = 1e-4;

    BasicBaumWelchTrainer<ObservationVectorView> trainer(fitted_hmm, training_data);
    double prev_lp = lp_after_init;

    std::cout << "Baum-Welch training (up to " << MAX_ITERS << " iterations):\n";
    for (int k = 0; k < MAX_ITERS; ++k) {
        trainer.train();
        const double cur_lp = total_log_prob(fitted_hmm, training_data);
        if (k < 5 || (k + 1) % 10 == 0)
            std::cout << "  iter " << std::setw(3) << (k + 1)
                      << "  logP = " << std::setprecision(2) << cur_lp << '\n';
        if (std::abs(cur_lp - prev_lp) < TOLERANCE) {
            std::cout << "  converged at iteration " << (k + 1) << '\n';
            break;
        }
        prev_lp = cur_lp;
    }
    std::cout << "\nFitted emission parameters:\n";
    print_emissions(fitted_hmm);
    std::cout << '\n';

    // -------------------------------------------------------------------------
    // 5. Score a new test sequence
    // -------------------------------------------------------------------------
    const ObservationMatrix test_seq = sample_sequence(true_hmm, 15, rng);

    BasicForwardBackwardCalculator<ObservationVectorView> fbc(fitted_hmm, test_seq);
    std::cout << "Test sequence (T=15):\n";
    std::cout << "  log P(O | fitted model) = " << std::setprecision(4)
              << fbc.getLogProbability() << '\n';
    std::cout << "  P(O | fitted model)     = " << std::setprecision(6)
              << fbc.probability() << '\n';

    // -------------------------------------------------------------------------
    // 6. Viterbi decoding
    // -------------------------------------------------------------------------
    BasicViterbiCalculator<ObservationVectorView> vc(fitted_hmm, test_seq);
    std::cout << "\nViterbi decoded state sequence:\n  [";
    const StateSequence& seq = vc.getStateSequence();
    for (std::size_t t = 0; t < seq.size(); ++t) {
        if (t) std::cout << ", ";
        std::cout << seq(t);
    }
    std::cout << "]\n  (log P(O, q*) = " << vc.getLogProbability() << ")\n";

    // -------------------------------------------------------------------------
    // 7. JSON save / load round-trip
    // -------------------------------------------------------------------------
    const auto model_path = std::filesystem::temp_directory_path() / "mv_gaussian_example.json";
    save_json_mv(fitted_hmm, model_path);
    HmmMV reloaded = load_json_mv(model_path);
    std::cout << "\nJSON round-trip: saved to " << model_path << '\n';

    BasicForwardBackwardCalculator<ObservationVectorView> fbc2(reloaded, test_seq);
    const double logp_diff = std::abs(fbc.getLogProbability() - fbc2.getLogProbability());
    std::cout << "  log-probability difference after reload: " << std::scientific
              << logp_diff << " (should be ~0)\n";

    std::cout << "\nDone.\n";
    return 0;
}
