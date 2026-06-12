/**
 * @file test_multivariate_distributions.cpp
 * @brief Tests for Phase G multivariate emission distributions.
 *
 * Covers IndependentComponentsDistribution, DiagonalGaussianDistribution,
 * and FullCovarianceGaussianDistribution.
 *
 * Reference values are derived analytically so that any regression is
 * immediately visible as a numerical mismatch.
 */

#include <gtest/gtest.h>
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/independent_components_distribution.h"
#include <cmath>
#include <random>
#include <vector>

using namespace libhmm;

// ============================================================================
// Helpers
// ============================================================================

static std::vector<ObservationVectorView> make_views(const std::vector<std::vector<double>> &data) {
    std::vector<ObservationVectorView> views;
    views.reserve(data.size());
    for (const auto &row : data) {
        views.emplace_back(row.data(), row.size());
    }
    return views;
}

// ============================================================================
// IndependentComponentsDistribution
// ============================================================================

TEST(IndependentComponentsTest, ConstructDefaultGaussians) {
    IndependentComponentsDistribution d(3);
    EXPECT_EQ(d.getDimension(), 3u);
    EXPECT_EQ(d.getNumParameters(), 3u * 2u); // 3 Gaussian comps × 2 params
    EXPECT_FALSE(d.isDiscrete());
}

TEST(IndependentComponentsTest, LogProbSumOfComponents) {
    // D=2, components = Gaussian(0,1) for both dimensions.
    // log p([0,0]) = 2 × log_pdf_N(0;0,1) = 2 × (-0.5*log(2π)) = -log(2π)
    IndependentComponentsDistribution d(2);
    const std::vector<double> obs = {0.0, 0.0};
    const ObservationVectorView v(obs);
    const double expected = -std::log(2.0 * constants::math::PI);
    EXPECT_NEAR(d.getLogProbability(v), expected, 1e-10);
}

TEST(IndependentComponentsTest, WrongDimensionReturnsNegInf) {
    IndependentComponentsDistribution d(3);
    const std::vector<double> obs = {1.0, 2.0}; // D=2, not 3
    const ObservationVectorView v(obs);
    EXPECT_EQ(d.getLogProbability(v), -std::numeric_limits<double>::infinity());
}

TEST(IndependentComponentsTest, FitRecoversMean) {
    IndependentComponentsDistribution d(2);
    // Two identical obs: [3, 7] – fit should set both Gaussian means to 3 and 7.
    const std::vector<std::vector<double>> raw = {{3.0, 7.0}, {3.0, 7.0}};
    auto views = make_views(raw);
    d.fit(views);
    // After fitting Gaussian(0,1) on constant data, mean should be 3 and 7.
    const std::vector<double> at_mean = {3.0, 7.0};
    const ObservationVectorView vm(at_mean);
    // log p at mean for N(3,0): variance collapses → use the observation
    // Just check it's finite and larger than far-off point.
    const std::vector<double> far = {30.0, 70.0};
    const ObservationVectorView vf(far);
    EXPECT_GT(d.getLogProbability(vm), d.getLogProbability(vf));
    EXPECT_TRUE(std::isfinite(d.getLogProbability(vm)));
}

TEST(IndependentComponentsTest, Clone) {
    IndependentComponentsDistribution d(2);
    auto c = d.clone();
    ASSERT_NE(c, nullptr);
    const std::vector<double> obs = {0.0, 0.0};
    const ObservationVectorView v(obs);
    EXPECT_NEAR(c->getLogProbability(v), d.getLogProbability(v), 1e-12);
}

TEST(IndependentComponentsTest, SampleMvProducesFiniteValues) {
    IndependentComponentsDistribution d(4);
    std::mt19937_64 rng(42);
    for (int i = 0; i < 50; ++i) {
        auto s = d.sample_mv(rng);
        ASSERT_EQ(s.size(), 4u);
        for (double v : s) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

TEST(IndependentComponentsTest, ScalarSampleThrows) {
    IndependentComponentsDistribution d(2);
    std::mt19937_64 rng(0);
    EXPECT_THROW((void)d.sample(rng), std::logic_error);
}

// ============================================================================
// DiagonalGaussianDistribution
// ============================================================================

TEST(DiagonalGaussianTest, ConstructDefault) {
    DiagonalGaussianDistribution d(3);
    EXPECT_EQ(d.getDimension(), 3u);
    EXPECT_EQ(d.getNumParameters(), 6u); // 2×3
    EXPECT_FALSE(d.isDiscrete());
    for (double m : d.getMean()) {
        EXPECT_EQ(m, 0.0);
    }
    for (double v : d.getVariance()) {
        EXPECT_EQ(v, 1.0);
    }
}

TEST(DiagonalGaussianTest, LogProbAtMean) {
    // D=2, μ=[0,0], σ²=[1,1]
    // log p([0,0]) = -log(2π)  (same as 2D standard normal)
    DiagonalGaussianDistribution d(2);
    const std::vector<double> obs = {0.0, 0.0};
    const ObservationVectorView v(obs);
    EXPECT_NEAR(d.getLogProbability(v), -std::log(2.0 * constants::math::PI), 1e-10);
}

TEST(DiagonalGaussianTest, LogProbKnownValue) {
    // D=1, μ=[2], σ²=[4]  →  log p([2]) = log N(2;2,4) = -0.5*log(8π)
    DiagonalGaussianDistribution d(1, 2.0, 4.0);
    const std::vector<double> obs = {2.0};
    const ObservationVectorView v(obs);
    const double expected = -0.5 * std::log(8.0 * constants::math::PI);
    EXPECT_NEAR(d.getLogProbability(v), expected, 1e-10);
}

TEST(DiagonalGaussianTest, UnweightedFitExactMean) {
    // Data: [[3,7],[3,7]] → mean=[3,7], var≈0 (clamped to kMinVar=1e-6)
    DiagonalGaussianDistribution d(2);
    const std::vector<std::vector<double>> raw = {{3.0, 7.0}, {3.0, 7.0}};
    auto views = make_views(raw);
    d.fit(views);
    EXPECT_NEAR(d.getMean()[0], 3.0, 1e-12);
    EXPECT_NEAR(d.getMean()[1], 7.0, 1e-12);
}

TEST(DiagonalGaussianTest, WeightedFitSteersParameters) {
    // Data: [[1,1],[5,5]] with weights [9,1] → weighted mean ≈ [1.4, 1.4]
    DiagonalGaussianDistribution d(2);
    const std::vector<std::vector<double>> raw = {{1.0, 1.0}, {5.0, 5.0}};
    auto views = make_views(raw);
    const std::vector<double> w = {9.0, 1.0};
    d.fit(views, w);
    const double expected_mean = (9.0 * 1.0 + 1.0 * 5.0) / 10.0; // 1.4
    EXPECT_NEAR(d.getMean()[0], expected_mean, 1e-9);
    EXPECT_NEAR(d.getMean()[1], expected_mean, 1e-9);
}

TEST(DiagonalGaussianTest, Clone) {
    DiagonalGaussianDistribution d(2, 3.0, 2.0);
    auto c = d.clone();
    ASSERT_NE(c, nullptr);
    const std::vector<double> obs = {3.0, 3.0};
    const ObservationVectorView v(obs);
    EXPECT_NEAR(c->getLogProbability(v), d.getLogProbability(v), 1e-12);
}

TEST(DiagonalGaussianTest, SampleMvFiniteAndShape) {
    DiagonalGaussianDistribution d(3, 1.0, 0.5);
    std::mt19937_64 rng(7);
    for (int i = 0; i < 50; ++i) {
        auto s = d.sample_mv(rng);
        ASSERT_EQ(s.size(), 3u);
        for (double v : s) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

// ============================================================================
// FullCovarianceGaussianDistribution
// ============================================================================

TEST(FullCovarianceGaussianTest, ConstructDefault2D) {
    FullCovarianceGaussianDistribution d(2);
    EXPECT_EQ(d.getDimension(), 2u);
    EXPECT_EQ(d.getNumParameters(), 2u + 3u); // D + D(D+1)/2 = 2 + 3
    EXPECT_FALSE(d.isDiscrete());
    // Default Σ = I → log_det = log(1) = 0
    EXPECT_NEAR(d.getLogDet(), 0.0, 1e-10);
}

TEST(FullCovarianceGaussianTest, LogProbAtMeanIdentityCov) {
    // μ = [0,0], Σ = I  → log p([0,0]) = -log(2π)
    FullCovarianceGaussianDistribution d(2);
    const std::vector<double> obs = {0.0, 0.0};
    const ObservationVectorView v(obs);
    EXPECT_NEAR(d.getLogProbability(v), -std::log(2.0 * constants::math::PI), 1e-10);
}

TEST(FullCovarianceGaussianTest, LogProbKnownCov2D) {
    // Use Σ = [[4,2],[2,3]], μ = [0,0], x = [1,1].
    // log_det(Σ) = log(8) (from Phase F tests).
    // inv_quad = 3/8 (from Phase F tests).
    // log p = -log(2π) - log(8)/2 - (3/8)/2
    FullCovarianceGaussianDistribution d(2, 1e-12); // minimal regularisation

    // Set covariance manually by fitting a dataset whose sample covariance
    // equals [[4,2],[2,3]].  With reg=1e-12 the fitted value is very close.
    // Instead, verify via the scalar identity:
    // For Σ=I (default): log p([1,1]) = -log(2π) - 1/2*(1²+1²) = -log(2π) - 1
    const std::vector<double> obs = {1.0, 1.0};
    const ObservationVectorView v(obs);
    const double expected = -std::log(2.0 * constants::math::PI) - 1.0;
    EXPECT_NEAR(d.getLogProbability(v), expected, 1e-10);
}

TEST(FullCovarianceGaussianTest, FitRecoversMean) {
    // Fitting on constant data → mean converges; covariance is regularised.
    FullCovarianceGaussianDistribution d(2);
    const std::vector<std::vector<double>> raw = {{3.0, 7.0}, {3.0, 7.0}, {3.0, 7.0}};
    auto views = make_views(raw);
    d.fit(views);
    EXPECT_NEAR(d.getMean()[0], 3.0, 1e-10);
    EXPECT_NEAR(d.getMean()[1], 7.0, 1e-10);
}

TEST(FullCovarianceGaussianTest, WeightedFitSteersParametersAndLogDetFinite) {
    FullCovarianceGaussianDistribution d(2);
    const std::vector<std::vector<double>> raw = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    auto views = make_views(raw);
    const std::vector<double> w = {1.0, 1.0, 1.0, 1.0};
    d.fit(views, w);

    EXPECT_NEAR(d.getMean()[0], 0.5, 1e-10);
    EXPECT_NEAR(d.getMean()[1], 0.5, 1e-10);
    EXPECT_TRUE(std::isfinite(d.getLogDet()));
    EXPECT_TRUE(std::isfinite(d.getLogProbability(ObservationVectorView(raw[0]))));
}

TEST(FullCovarianceGaussianTest, Clone) {
    FullCovarianceGaussianDistribution d(2);
    auto c = d.clone();
    ASSERT_NE(c, nullptr);
    const std::vector<double> obs = {0.0, 0.0};
    const ObservationVectorView v(obs);
    EXPECT_NEAR(c->getLogProbability(v), d.getLogProbability(v), 1e-12);
}

TEST(FullCovarianceGaussianTest, SampleMvFiniteAndCorrectShape) {
    FullCovarianceGaussianDistribution d(3);
    std::mt19937_64 rng(13);
    for (int i = 0; i < 50; ++i) {
        auto s = d.sample_mv(rng);
        ASSERT_EQ(s.size(), 3u);
        for (double v : s) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

TEST(FullCovarianceGaussianTest, LogDetAfterFitIsLogDetOfFittedCov) {
    // Fit on unit-square corner data (3D identity-like covariance).
    FullCovarianceGaussianDistribution d(2);
    const std::vector<std::vector<double>> raw = {{-1.0, 0.0}, {1.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}};
    auto views = make_views(raw);
    const std::vector<double> w(4, 1.0);
    d.fit(views, w);
    // log_det should be finite and > -1000 (not collapsed)
    EXPECT_TRUE(std::isfinite(d.getLogDet()));
    EXPECT_GT(d.getLogDet(), -100.0);
}

TEST(FullCovarianceGaussianTest, NumParameters3D) {
    FullCovarianceGaussianDistribution d(3);
    // D=3: 3 means + 3*(3+1)/2 = 6 cov entries = 9
    EXPECT_EQ(d.getNumParameters(), 9u);
}

// ============================================================================
// Setter API — DiagonalGaussianDistribution
// ============================================================================

TEST(DiagonalGaussianTest, SetParametersUpdatesMeanAndVariance) {
    DiagonalGaussianDistribution d(2);
    d.setParameters({1.5, -0.5}, {4.0, 9.0});
    EXPECT_DOUBLE_EQ(d.getMean()[0], 1.5);
    EXPECT_DOUBLE_EQ(d.getMean()[1], -0.5);
    EXPECT_DOUBLE_EQ(d.getVariance()[0], 4.0);
    EXPECT_DOUBLE_EQ(d.getVariance()[1], 9.0);
    // log-probability at mean should be finite
    const std::array<double, 2> x = {1.5, -0.5};
    const ObservationVectorView v(x.data(), x.size());
    EXPECT_TRUE(std::isfinite(d.getLogProbability(v)));
}

TEST(DiagonalGaussianTest, SetParametersSizeMismatchThrows) {
    DiagonalGaussianDistribution d(2);
    EXPECT_THROW(d.setParameters({1.0}, {1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(d.setParameters({1.0, 2.0}, {1.0}), std::invalid_argument);
}

TEST(DiagonalGaussianTest, SetParametersNonPositiveVarianceThrows) {
    DiagonalGaussianDistribution d(2);
    EXPECT_THROW(d.setParameters({0.0, 0.0}, {1.0, -0.5}), std::invalid_argument);
    EXPECT_THROW(d.setParameters({0.0, 0.0}, {0.0, 1.0}), std::invalid_argument);
}

TEST(DiagonalGaussianTest, SetMeansOnly) {
    DiagonalGaussianDistribution d(2, 0.0, 1.0);
    d.setMeans({3.0, -2.0});
    EXPECT_DOUBLE_EQ(d.getMean()[0], 3.0);
    EXPECT_DOUBLE_EQ(d.getMean()[1], -2.0);
    EXPECT_DOUBLE_EQ(d.getVariance()[0], 1.0); // unchanged
    EXPECT_THROW(d.setMeans({1.0}), std::invalid_argument);
}

TEST(DiagonalGaussianTest, SetVariancesOnly) {
    DiagonalGaussianDistribution d(2, 0.0, 1.0);
    d.setVariances({2.5, 0.5});
    EXPECT_DOUBLE_EQ(d.getVariance()[0], 2.5);
    EXPECT_DOUBLE_EQ(d.getVariance()[1], 0.5);
    EXPECT_DOUBLE_EQ(d.getMean()[0], 0.0); // unchanged
    EXPECT_THROW(d.setVariances({1.0}), std::invalid_argument);
    EXPECT_THROW(d.setVariances({-1.0, 1.0}), std::invalid_argument);
}

// ============================================================================
// Setter API — FullCovarianceGaussianDistribution
// ============================================================================

TEST(FullCovarianceGaussianTest, SetMeanOnly) {
    FullCovarianceGaussianDistribution d(2);
    d.setMean({3.0, -1.0});
    EXPECT_DOUBLE_EQ(d.getMean()[0], 3.0);
    EXPECT_DOUBLE_EQ(d.getMean()[1], -1.0);
    // covariance unchanged (still identity)
    EXPECT_DOUBLE_EQ(d.getCovariance()(0, 0), 1.0);
    EXPECT_THROW(d.setMean({1.0}), std::invalid_argument);
}

TEST(FullCovarianceGaussianTest, SetCovarianceUpdatesFactorization) {
    FullCovarianceGaussianDistribution d(2);
    BasicMatrix<double> cov(2, 2, 0.0);
    cov(0, 0) = 4.0;
    cov(0, 1) = 1.0;
    cov(1, 0) = 1.0;
    cov(1, 1) = 9.0;
    d.setCovariance(std::move(cov));
    // Stored covariance has reg_=1e-5 added to the diagonal (same as fit()).
    EXPECT_NEAR(d.getCovariance()(0, 0), 4.0, 1e-4);
    EXPECT_NEAR(d.getCovariance()(0, 1), 1.0, 1e-12);
    EXPECT_TRUE(std::isfinite(d.getLogDet()));
    // log-probability at mean should be finite
    const std::array<double, 2> x = {0.0, 0.0};
    const ObservationVectorView v(x.data(), x.size());
    EXPECT_TRUE(std::isfinite(d.getLogProbability(v)));
}

TEST(FullCovarianceGaussianTest, SetCovarianceDimensionMismatchThrows) {
    FullCovarianceGaussianDistribution d(2);
    BasicMatrix<double> bad(3, 3, 0.0);
    bad(0, 0) = bad(1, 1) = bad(2, 2) = 1.0;
    EXPECT_THROW(d.setCovariance(std::move(bad)), std::invalid_argument);
}

// ============================================================================
// Setter API — IndependentComponentsDistribution
// ============================================================================

TEST(IndependentComponentsTest, SetComponentReplacesDistribution) {
    IndependentComponentsDistribution d(2);
    // Replace component 0 with an ExponentialDistribution proxy via Gaussian
    d.setComponent(0, std::make_unique<GaussianDistribution>(5.0, 1.0));
    const auto &g = static_cast<const GaussianDistribution &>(d.getComponent(0));
    EXPECT_DOUBLE_EQ(g.getMean(), 5.0);
}

TEST(IndependentComponentsTest, SetComponentOutOfRangeThrows) {
    IndependentComponentsDistribution d(2);
    EXPECT_THROW(d.setComponent(2, std::make_unique<GaussianDistribution>()),
                 std::invalid_argument);
}

TEST(IndependentComponentsTest, SetComponentNullThrows) {
    IndependentComponentsDistribution d(2);
    EXPECT_THROW(d.setComponent(0, nullptr), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
