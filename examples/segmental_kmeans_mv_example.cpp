/**
 * segmental_kmeans_mv_example — Segmental k-means for multivariate HMMs (v4).
 *
 * Demonstrates SegmentalKMeansTrainerMV (= BasicSegmentalKMeansTrainer<ObservationVectorView>)
 * on a 2-state, 2-dimensional DiagonalGaussian HMM fitted to synthetic data from
 * two well-separated Gaussian clusters.
 *
 * The recommended warm-start workflow for MV HMMs:
 *   1. kmeans_init   — data-driven centroid seeding (avoids random restarts)
 *   2. SegmentalKMeansTrainerMV — fast hard-assignment convergence
 *   3. BasicBaumWelchTrainer<MV> — soft-EM refinement for the final model
 *
 * This workflow reliably converges to a good solution on well-separated data
 * and avoids the local-optima sensitivity of pure Baum-Welch from a random start.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/basic_segmental_kmeans_trainer.h"
#include "libhmm/training/kmeans_init.h"

using namespace libhmm;

// =============================================================================
// Helpers
// =============================================================================

namespace {

/// Build a fresh N-state D-dimensional DiagonalGaussian HmmMV with uniform π/A.
HmmMV make_mv_hmm(std::size_t N, std::size_t D) {
    HmmMV hmm(N);
    for (std::size_t i = 0; i < N; ++i)
        hmm.setDistribution(i, std::make_unique<DiagonalGaussianDistribution>(D));
    Matrix trans(N, N);
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i) {
        pi(i) = 1.0 / static_cast<double>(N);
        for (std::size_t j = 0; j < N; ++j)
            trans(i, j) = 1.0 / static_cast<double>(N);
    }
    hmm.setTrans(trans);
    hmm.setPi(pi);
    return hmm;
}

/// Generate n_seqs sequences of length T from two D-dim Gaussian clusters.
MultiObservationLists make_two_cluster_data(std::size_t n_seqs, std::size_t T, std::size_t D,
                                            double c0, double c1, std::mt19937_64 &rng) {
    std::normal_distribution<double> noise(0.0, 0.4);
    MultiObservationLists lists;
    lists.reserve(n_seqs);
    for (std::size_t s = 0; s < n_seqs; ++s) {
        const double centre = (s % 2 == 0) ? c0 : c1;
        ObservationMatrix mat(T, D);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t d = 0; d < D; ++d)
                mat(t, d) = centre + noise(rng);
        lists.push_back(std::move(mat));
    }
    return lists;
}

/// Total forward-backward log-probability across all sequences.
double total_log_prob(HmmMV &hmm, const MultiObservationLists &lists) {
    double lp = 0.0;
    for (const auto &seq : lists) {
        BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, seq);
        lp += fbc.getLogProbability();
    }
    return lp;
}

/// Print DiagonalGaussian mean and variance for each state.
void print_emissions(const HmmMV &hmm) {
    for (std::size_t i = 0; i < hmm.getNumStatesModern(); ++i) {
        const auto &d = static_cast<const DiagonalGaussianDistribution &>(hmm.getDistribution(i));
        std::cout << "    State " << i << ": μ=[" << std::fixed << std::setprecision(3)
                  << d.getMean()[0] << ", " << d.getMean()[1] << "]  σ²=[" << d.getVariance()[0]
                  << ", " << d.getVariance()[1] << "]\n";
    }
}

} // namespace

// =============================================================================
// main
// =============================================================================

int main() {
    std::cout << "Segmental K-Means MV Example\n";
    std::cout << "============================\n\n";

    // -------------------------------------------------------------------------
    // Synthetic training data: two clusters at [0,0] and [5,5].
    // -------------------------------------------------------------------------
    std::mt19937_64 rng(42);
    constexpr std::size_t N = 2, D = 2;
    constexpr std::size_t N_SEQS = 20, SEQ_LEN = 10;
    auto data = make_two_cluster_data(N_SEQS, SEQ_LEN, D, /*c0=*/0.0, /*c1=*/5.0, rng);

    std::cout << "Data: " << N_SEQS << " sequences, length " << SEQ_LEN << ", D=" << D
              << "  (two clusters at [0,0] and [5,5])\n\n";

    // -------------------------------------------------------------------------
    // Path A: SegmentalKMeansTrainerMV alone.
    //   Starts from a uniform HMM; the index-partition initial assignment
    //   is crude but usually sufficient for well-separated data.
    // -------------------------------------------------------------------------
    std::cout << "Path A: SegmentalKMeansTrainerMV alone\n";
    std::cout << "--------------------------------------\n";
    HmmMV hmm_a = make_mv_hmm(N, D);
    std::cout << "  Initial logP: " << std::fixed << std::setprecision(2)
              << total_log_prob(hmm_a, data) << "\n";

    SegmentalKMeansTrainerMV skm_a(hmm_a, data);
    skm_a.train();

    const double lp_a = total_log_prob(hmm_a, data);
    std::cout << "  Converged:    " << (skm_a.isTerminated() ? "yes" : "no") << "\n";
    std::cout << "  Final logP:   " << lp_a << "\n";
    std::cout << "  Emissions:\n";
    print_emissions(hmm_a);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Path B: Recommended warm-start workflow.
    //   kmeans_init  → SegmentalKMeansTrainerMV → BaumWelchTrainer<MV>
    //
    //   kmeans_init provides a data-driven starting point that is more robust
    //   than the index-partition used by SegmentalKMeansTrainerMV alone.
    //   SegmentalKMeansTrainerMV refines it quickly via hard EM.
    //   Baum-Welch then smooths the assignments and maximises P(O|λ) directly.
    // -------------------------------------------------------------------------
    std::cout << "Path B: kmeans_init → SegmentalKMeansTrainerMV → Baum-Welch\n";
    std::cout << "--------------------------------------------------------------\n";
    HmmMV hmm_b = make_mv_hmm(N, D);

    // Step 1: kmeans_init
    std::mt19937_64 rng2(7);
    const double lp_raw = total_log_prob(hmm_b, data);
    kmeans_init(hmm_b, data, rng2);
    const double lp_init = total_log_prob(hmm_b, data);
    std::cout << "  After kmeans_init:        logP = " << lp_init << "  (Δ " << std::showpos
              << (lp_init - lp_raw) << std::noshowpos << ")\n";

    // Step 2: SegmentalKMeansTrainerMV
    SegmentalKMeansTrainerMV skm_b(hmm_b, data);
    skm_b.train();
    const double lp_skm = total_log_prob(hmm_b, data);
    std::cout << "  After SegmentalKMeans MV: logP = " << lp_skm << "  (Δ " << std::showpos
              << (lp_skm - lp_init) << std::noshowpos << ")"
              << (skm_b.isTerminated() ? "  [converged]\n" : "  [max iters]\n");

    // Step 3: Baum-Welch refinement
    BasicBaumWelchTrainer<ObservationVectorView> bw(hmm_b, data);
    double prev = lp_skm;
    for (int k = 0; k < 20; ++k) {
        bw.train();
        const double cur = total_log_prob(hmm_b, data);
        if (std::abs(cur - prev) < 1e-4) {
            std::cout << "  After Baum-Welch (" << (k + 1) << " iters):   logP = " << cur << "  (Δ "
                      << std::showpos << (cur - lp_skm) << std::noshowpos << ")  [converged]\n";
            break;
        }
        prev = cur;
    }

    std::cout << "  Final emissions (Path B):\n";
    print_emissions(hmm_b);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Summary: compare Path A vs Path B.
    // -------------------------------------------------------------------------
    const double lp_b = total_log_prob(hmm_b, data);
    std::cout << "Summary\n";
    std::cout << "-------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Path A (SegmentalKMeans only): logP = " << lp_a << "\n";
    std::cout << "  Path B (warm-start + BW):      logP = " << lp_b << "\n";
    std::cout << "  BW refinement gain:            " << std::showpos << (lp_b - lp_a)
              << std::noshowpos << "\n";

    return 0;
}
