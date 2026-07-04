/**
 * @file test_segmental_kmeans_mv.cpp
 * @brief Tests for BasicSegmentalKMeansTrainer<ObservationVectorView>.
 *
 * Covers:
 *   - Construction: reference, pointer, empty list, null pointer.
 *   - Training: runs without error on DiagonalGaussian and FullCovGaussian HMMs.
 *   - Convergence: terminates (converged or maxIterations) with finite log-prob.
 *   - Log-prob improvement: training on well-separated data does not degrade the model.
 *   - Scalar alias: SegmentalKMeansTrainer still constructs and trains correctly.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/basic_segmental_kmeans_trainer.h"
#include "libhmm/training/segmental_kmeans_trainer.h"
#include "libhmm/training/kmeans_init.h"

using namespace libhmm;

// =============================================================================
// Shared helpers
// =============================================================================

namespace {

/// Build an N-state D-dimensional DiagonalGaussian HmmMV with uniform π/A.
HmmMV make_hmm_mv(std::size_t N, std::size_t D) {
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

/// Generate n_seqs sequences of T timesteps from two D-dim Gaussian clusters.
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

/// Total forward-backward log-probability of all sequences under hmm.
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
// Construction
// =============================================================================

TEST(MvSegKMeans, ConstructionFromReference) {
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    EXPECT_NO_THROW((SegmentalKMeansTrainerMV(hmm, lists)));
}

TEST(MvSegKMeans, ConstructionFromPointer) {
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    EXPECT_NO_THROW((SegmentalKMeansTrainerMV(&hmm, lists)));
}

TEST(MvSegKMeans, NullHmmPointerThrows) {
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    EXPECT_THROW((SegmentalKMeansTrainerMV(static_cast<HmmMV *>(nullptr), lists)),
                 std::invalid_argument);
}

TEST(MvSegKMeans, EmptyListThrows) {
    HmmMV hmm = make_hmm_mv(2, 2);
    MultiObservationLists empty;
    EXPECT_THROW((SegmentalKMeansTrainerMV(hmm, empty)), std::invalid_argument);
}

TEST(MvSegKMeans, IsTerminatedFalseBeforeTrain) {
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    SegmentalKMeansTrainerMV trainer(hmm, lists);
    EXPECT_FALSE(trainer.isTerminated());
}

// =============================================================================
// Training — DiagonalGaussian
// =============================================================================

TEST(MvSegKMeans, RunsWithoutErrorDiagonalGaussian) {
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);
    EXPECT_NO_THROW({
        SegmentalKMeansTrainerMV trainer(hmm, lists);
        trainer.train();
    });
}

TEST(MvSegKMeans, ConvergesOrHitsMaxIter) {
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 5.0);
    std::mt19937_64 rng(42);
    kmeans_init(hmm, lists, rng);

    SegmentalKMeansTrainerMV trainer(hmm, lists, /*maxIterations=*/50);
    trainer.train();
    // Model must produce finite log-probabilities regardless of termination cause.
    EXPECT_TRUE(std::isfinite(total_log_prob(hmm, lists)));
}

TEST(MvSegKMeans, LogProbImprovesonWellSeparatedData) {
    // On clearly separated clusters, repeated train() passes should not degrade
    // the model relative to a kmeans-initialised starting point.
    constexpr std::size_t N = 2, D = 2;
    auto lists = make_two_cluster_data(20, 10, D, 0.0, 5.0, 7);

    HmmMV hmm = make_hmm_mv(N, D);
    std::mt19937_64 rng(3);
    kmeans_init(hmm, lists, rng);

    const double lp0 = total_log_prob(hmm, lists);

    SegmentalKMeansTrainerMV trainer(hmm, lists, /*maxIterations=*/30);
    trainer.train();

    const double lpFinal = total_log_prob(hmm, lists);
    EXPECT_TRUE(std::isfinite(lpFinal));
    // Hard-assignment EM is not guaranteed to be strictly monotone in
    // forward-backward log-prob, but should not dramatically degrade the model.
    EXPECT_GE(lpFinal, lp0 - 1.0);
}

TEST(MvSegKMeans, HmmReferenceUnchangedAfterTrainer) {
    // The trainer must operate on the HMM passed to it, not a copy.
    HmmMV hmm = make_hmm_mv(2, 2);
    auto lists = make_two_cluster_data(10, 8, 2, 0.0, 4.0);

    SegmentalKMeansTrainerMV trainer(hmm, lists);
    EXPECT_EQ(&trainer.getHmm(), &hmm);
}

// =============================================================================
// Training — FullCovarianceGaussian
// =============================================================================

TEST(MvSegKMeans, WorksWithFullCovarianceGaussian) {
    constexpr std::size_t N = 2, D = 2;
    HmmMV hmm(N);
    for (std::size_t i = 0; i < N; ++i)
        hmm.setDistribution(i, std::make_unique<FullCovarianceGaussianDistribution>(D));
    Matrix trans(N, N);
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i) {
        pi(i) = 0.5;
        for (std::size_t j = 0; j < N; ++j)
            trans(i, j) = 0.5;
    }
    hmm.setTrans(trans);
    hmm.setPi(pi);

    auto lists = make_two_cluster_data(10, 8, D, 0.0, 4.0, 99);
    EXPECT_NO_THROW({
        SegmentalKMeansTrainerMV trainer(hmm, lists);
        trainer.train();
    });
    EXPECT_TRUE(std::isfinite(total_log_prob(hmm, lists)));
}

// =============================================================================
// Scalar alias: SegmentalKMeansTrainer still works for discrete HMMs
// =============================================================================

TEST(ScalarSegKMeans, DiscreteHmmTrainsWithoutError) {
    Hmm hmm(2);
    auto d0 = std::make_unique<DiscreteDistribution>(4);
    auto d1 = std::make_unique<DiscreteDistribution>(4);
    for (int i = 0; i < 4; ++i) {
        d0->setProbability(i, 0.25);
        d1->setProbability(i, 0.25);
    }
    hmm.setDistribution(0, std::move(d0));
    hmm.setDistribution(1, std::move(d1));
    Matrix trans(2, 2);
    trans(0, 0) = 0.7;
    trans(0, 1) = 0.3;
    trans(1, 0) = 0.4;
    trans(1, 1) = 0.6;
    hmm.setTrans(trans);
    Vector pi(2);
    pi(0) = 0.6;
    pi(1) = 0.4;
    hmm.setPi(pi);

    ObservationSet seq(12);
    for (std::size_t i = 0; i < 12; ++i)
        seq(i) = static_cast<double>(i % 4);
    ObservationLists lists = {seq};

    EXPECT_NO_THROW({
        SegmentalKMeansTrainer trainer(hmm, lists);
        trainer.train();
    });
}

TEST(ScalarSegKMeans, GaussianHmmTrainsWithoutError) {
    // The discrete-only restriction is gone; any scalar distribution works.
    Hmm hmm(2);
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.0));
    Matrix trans(2, 2);
    trans(0, 0) = 0.7;
    trans(0, 1) = 0.3;
    trans(1, 0) = 0.3;
    trans(1, 1) = 0.7;
    hmm.setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);

    ObservationSet seq(20);
    for (std::size_t i = 0; i < 10; ++i)
        seq(i) = static_cast<double>(i) * 0.1;
    for (std::size_t i = 10; i < 20; ++i)
        seq(i) = 5.0 + static_cast<double>(i - 10) * 0.1;
    ObservationLists lists = {seq};

    EXPECT_NO_THROW({
        SegmentalKMeansTrainer trainer(hmm, lists);
        trainer.train();
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
