/**
 * @file test_mv_training.cpp
 * @brief Tests for MV (Obs=ObservationVectorView) training algorithms.
 *
 * Covers:
 *   - BasicBaumWelchTrainer<ObservationVectorView>: likelihood non-decreasing.
 *   - BasicViterbiTrainer<ObservationVectorView>: runs without error.
 *   - BasicMapBaumWelchTrainer<ObservationVectorView>: runs without error.
 *   - kmeans_init: dimensions are valid after init; clusters improve
 *     log-probability compared to default initialisation.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/kmeans_init.h"
#include "libhmm/training/map_baum_welch_trainer.h"
#include "libhmm/training/viterbi_trainer.h"

using namespace libhmm;

// =============================================================================
// Shared helpers
// =============================================================================

namespace {

/// Build a symmetric 2-state D-dimensional DiagonalGaussian HmmMV.
/// Both states start with mean=0, var=1; the caller should call kmeans_init
/// or set distributions before training.
HmmMV make_hmm_mv(std::size_t N, std::size_t D) {
    HmmMV hmm(N);
    for (std::size_t i = 0; i < N; ++i)
        hmm.setDistribution(i, std::make_unique<DiagonalGaussianDistribution>(D));

    // Uniform transition matrix + uniform initial
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

/// Generate T×D observations with all entries equal to @p val.
ObservationMatrix const_seq(std::size_t T, std::size_t D, double val) {
    ObservationMatrix m(T, D);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t d = 0; d < D; ++d)
            m(t, d) = val;
    return m;
}

/// Generate @p n_seqs sequences of length @p T from two D-dim clusters:
///   half near @p c0, half near @p c1 (with small Gaussian noise σ=0.3).
MultiObservationLists make_two_cluster_data(std::size_t n_seqs, std::size_t T, std::size_t D,
                                            double c0, double c1, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.3);
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

/// Compute total log-probability of @p lists under @p hmm.
double total_log_prob(HmmMV &hmm, const MultiObservationLists &lists) {
    double lp = 0.0;
    for (const auto &seq : lists) {
        BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, seq);
        lp += fbc.getLogProbability();
    }
    return lp;
}

} // namespace

// =============================================================================
// BasicBaumWelchTrainer<ObservationVectorView>
// =============================================================================

TEST(MvBWT, LikelihoodNonDecreasingOverIterations) {
    // 2-state, D=2 DiagonalGaussian; data from two well-separated clusters.
    // Baum-Welch must not decrease total log-likelihood across iterations.
    constexpr std::size_t N = 2, D = 2;
    auto lists = make_two_cluster_data(20, 10, D, /*c0=*/0.0, /*c1=*/5.0);

    HmmMV hmm = make_hmm_mv(N, D);
    // Initialise with kmeans so the starting point is reasonably good.
    std::mt19937_64 rng(7);
    kmeans_init(hmm, lists, rng);

    const double lp0 = total_log_prob(hmm, lists);

    BasicBaumWelchTrainer<ObservationVectorView> trainer(hmm, lists);
    constexpr int ITERS = 5;
    double prev = lp0;
    for (int k = 0; k < ITERS; ++k) {
        trainer.train();
        const double cur = total_log_prob(hmm, lists);
        // Allow tiny numerical epsilon for floating-point rounding.
        EXPECT_GE(cur, prev - 1e-4) << "Likelihood decreased at iteration " << k;
        prev = cur;
    }
    // Overall improvement after 5 iterations.
    EXPECT_GT(prev, lp0);
}

TEST(MvBWT, RunsOnSingleSequence) {
    HmmMV hmm = make_hmm_mv(2, 3);
    ObservationMatrix seq = const_seq(20, 3, 1.0);
    MultiObservationLists lists = {seq};
    EXPECT_NO_THROW({
        BasicBaumWelchTrainer<ObservationVectorView> trainer(hmm, lists);
        trainer.train();
    });
}

TEST(MvBWT, EmptyListThrows) {
    HmmMV hmm = make_hmm_mv(2, 2);
    MultiObservationLists empty;
    EXPECT_THROW((BasicBaumWelchTrainer<ObservationVectorView>(hmm, empty)), std::invalid_argument);
}

// =============================================================================
// BasicViterbiTrainer<ObservationVectorView>
// =============================================================================

TEST(MvViterbiTrainer, RunsWithoutError) {
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    EXPECT_NO_THROW({
        BasicViterbiTrainer<ObservationVectorView> trainer(hmm, lists);
        trainer.train();
    });
}

TEST(MvViterbiTrainer, ConvergesOrHitsMaxIter) {
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    std::mt19937_64 rng(3);
    kmeans_init(hmm, lists, rng);

    TrainingConfig cfg;
    cfg.maxIterations = 20;
    BasicViterbiTrainer<ObservationVectorView> trainer(hmm, lists, cfg);
    trainer.train();
    // At least one termination condition must be true.
    EXPECT_TRUE(trainer.hasConverged() || trainer.reachedMaxIterations());
    EXPECT_TRUE(std::isfinite(trainer.getLastLogProbability()));
}

// =============================================================================
// BasicMapBaumWelchTrainer<ObservationVectorView>
// =============================================================================

TEST(MvMapBWT, RunsWithoutError) {
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    EXPECT_NO_THROW({
        BasicMapBaumWelchTrainer<ObservationVectorView> trainer(hmm, lists, 0.5);
        trainer.train();
    });
}

TEST(MvMapBWT, ZeroPseudoCountEquivalentToBWT) {
    // With c=0, MAP-BWT must produce the same result as plain BWT.
    constexpr std::size_t D = 2;
    auto lists = make_two_cluster_data(10, 8, D, 0.0, 5.0, 99);

    HmmMV hmm_bwt = make_hmm_mv(2, D);
    HmmMV hmm_map = make_hmm_mv(2, D);
    // Identical initialisation
    std::mt19937_64 rng(123);
    kmeans_init(hmm_bwt, lists, rng);
    rng = std::mt19937_64(123);
    kmeans_init(hmm_map, lists, rng);

    BasicBaumWelchTrainer<ObservationVectorView> bwt(hmm_bwt, lists);
    BasicMapBaumWelchTrainer<ObservationVectorView> map(hmm_map, lists, 0.0);
    bwt.train();
    map.train();

    // Log-probabilities under the two trained models should be very close.
    const double lp_bwt = total_log_prob(hmm_bwt, lists);
    const double lp_map = total_log_prob(hmm_map, lists);
    EXPECT_NEAR(lp_bwt, lp_map, 1e-4);
}

TEST(MvMapBWT, ComputeLogPriorMvReturnsOnlyTransAndPiPart) {
    // MV distributions are continuous; discrete emission prior must be zero.
    // computeLogPrior() with c=1 should be finite and only reflect trans+pi.
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    BasicMapBaumWelchTrainer<ObservationVectorView> trainer(hmm, lists, 1.0);
    const double prior = trainer.computeLogPrior();
    EXPECT_TRUE(std::isfinite(prior));
    EXPECT_LT(prior, 0.0); // log of probabilities < 1 is negative
}

// =============================================================================
// kmeans_init
// =============================================================================

TEST(KmeansInit, RunsWithoutError) {
    HmmMV hmm = make_hmm_mv(2, 3);
    auto lists = make_two_cluster_data(20, 10, 3, 0.0, 8.0);
    std::mt19937_64 rng(42);
    EXPECT_NO_THROW(kmeans_init(hmm, lists, rng));
}

TEST(KmeansInit, DistributionDimensionsValidAfterInit) {
    constexpr std::size_t D = 4;
    HmmMV hmm = make_hmm_mv(3, D);
    auto lists = make_two_cluster_data(15, 8, D, 0.0, 10.0);
    std::mt19937_64 rng(0);
    kmeans_init(hmm, lists, rng);
    for (std::size_t i = 0; i < hmm.getNumStatesModern(); ++i)
        EXPECT_EQ(hmm.getDistribution(i).getDimension(), D);
}

TEST(KmeansInit, ImprovesLogProbOverDefaultInit) {
    // Default init (mean=0, var=1 for all states) should give worse log-prob
    // on well-separated data than kmeans_init.
    constexpr std::size_t D = 2;
    auto lists = make_two_cluster_data(20, 10, D, -5.0, 5.0, 17);

    HmmMV hmm_default = make_hmm_mv(2, D); // all means at 0
    HmmMV hmm_kmeans = make_hmm_mv(2, D);
    std::mt19937_64 rng(55);
    kmeans_init(hmm_kmeans, lists, rng);

    const double lp_default = total_log_prob(hmm_default, lists);
    const double lp_kmeans = total_log_prob(hmm_kmeans, lists);
    EXPECT_GT(lp_kmeans, lp_default);
}

TEST(KmeansInit, EmptyDataThrows) {
    HmmMV hmm = make_hmm_mv(2, 2);
    MultiObservationLists empty;
    std::mt19937_64 rng(0);
    EXPECT_THROW(kmeans_init(hmm, empty, rng), std::invalid_argument);
}

TEST(KmeansInit, FewerObsThanStatesThrows) {
    // M < K: cannot seed K distinct centroids from fewer observations.
    // Before fix: seed_kmeanspp produced duplicate centroids silently.
    HmmMV hmm = make_hmm_mv(5, 2); // K=5 states
    MultiObservationLists lists;
    // Only 3 observations total (< K=5)
    for (int i = 0; i < 3; ++i)
        lists.push_back(const_seq(1, 2, static_cast<double>(i)));
    std::mt19937_64 rng(0);
    EXPECT_THROW(kmeans_init(hmm, lists, rng), std::invalid_argument);
}

TEST(MvViterbiTrainer, ConvergenceWindowOneTwoThrows) {
    // convergenceWindow=1 causes premature convergence after one iteration.
    // After fix, construction throws invalid_argument.
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    TrainingConfig bad_cfg;
    bad_cfg.convergenceWindow = 1;
    EXPECT_THROW((BasicViterbiTrainer<ObservationVectorView>(hmm, lists, bad_cfg)),
                 std::invalid_argument);
    // window=0 is also invalid
    bad_cfg.convergenceWindow = 0;
    EXPECT_THROW((BasicViterbiTrainer<ObservationVectorView>(hmm, lists, bad_cfg)),
                 std::invalid_argument);
    // window=2 is the minimum valid value
    bad_cfg.convergenceWindow = 2;
    EXPECT_NO_THROW((BasicViterbiTrainer<ObservationVectorView>(hmm, lists, bad_cfg)));
}

TEST(KmeansInit, SingleClusterData) {
    // All data at the same point — should not crash or hang.
    HmmMV hmm = make_hmm_mv(2, 2);
    MultiObservationLists lists;
    for (int i = 0; i < 5; ++i)
        lists.push_back(const_seq(8, 2, 3.0));
    std::mt19937_64 rng(1);
    EXPECT_NO_THROW(kmeans_init(hmm, lists, rng));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
