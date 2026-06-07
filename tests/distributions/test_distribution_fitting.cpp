/**
 * @file test_distribution_fitting.cpp
 * @brief Fitting validation tests for emission distributions (issue #21).
 *
 * Tests verify that fit(data) and fit(data, weights) correctly recover
 * parameters from synthetic data with known exact sample statistics.
 *
 * Design principles:
 *   - Unweighted tests use data whose sample statistics are exactly known
 *     from simple arithmetic, so tolerances can be tight (1e-9 for exact
 *     closed-form MLE, ±0.15 for Newton–Raphson distributions).
 *   - Weighted tests with uniform weights verify that the weighted overload
 *     gives the same result as the unweighted overload.
 *   - Weighted tests with concentrated weights verify that weights genuinely
 *     steer the fitted parameters (not ignored).
 *   - No random number generation — all data is hardcoded.
 */

#include <gtest/gtest.h>
#include "libhmm/distributions/distributions.h"
#include <cmath>
#include <vector>

using namespace libhmm;

// ============================================================================
// GaussianDistribution
// ============================================================================

/**
 * Data: {1.5, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 3.5, 4.0, 4.5}
 *
 * Exact sample statistics (N=10):
 *   mean = 30.0 / 10 = 3.0
 *   SS   = 1.5²+1.0²+0.5²+0.5²+0²+0²+0.5²+0.5²+1.0²+1.5² = 7.5
 *   var  = 7.5 / 10 = 0.75   (MLE denominator N)
 *   std  = sqrt(0.75) ≈ 0.86602540378
 */
namespace {
const std::vector<double> kGaussianData = {
    1.5, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 3.5, 4.0, 4.5};
constexpr double kGaussianMean = 3.0;
constexpr double kGaussianStd  = 0.8660254037844386;  // sqrt(0.75)
}

TEST(GaussianFitTest, UnweightedRecovery) {
    GaussianDistribution d;
    d.fit(kGaussianData);
    EXPECT_NEAR(d.getMean(),              kGaussianMean, 1e-9);
    EXPECT_NEAR(d.getStandardDeviation(), kGaussianStd,  1e-9);
}

TEST(GaussianFitTest, WeightedUniformEqualsUnweighted) {
    // Uniform weights should give the same result as unweighted.
    GaussianDistribution dU, dW;
    dU.fit(kGaussianData);
    const std::vector<double> w(kGaussianData.size(), 1.0);
    dW.fit(kGaussianData, w);
    EXPECT_NEAR(dW.getMean(),              dU.getMean(),              1e-9);
    EXPECT_NEAR(dW.getStandardDeviation(), dU.getStandardDeviation(), 1e-9);
}

TEST(GaussianFitTest, WeightedConcentratedOnLowValues) {
    // Data spans [1.0, 5.0]; heavy weights on the 1.0-valued points.
    // Unweighted mean = (3*1 + 7*5) / 10 = 3.8.
    // Weighted mean must be substantially less than 3.8.
    const std::vector<double> data    = {1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
    const std::vector<double> weights = {4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    // Weighted mean = (12*1 + 7*5) / (12+7) = 47 / 19 ≈ 2.47

    GaussianDistribution dU, dW;
    dU.fit(data);
    dW.fit(data, weights);
    EXPECT_LT(dW.getMean(), dU.getMean());   // weights push mean down
    EXPECT_NEAR(dW.getMean(), 47.0 / 19.0, 1e-9);
}

// ============================================================================
// ExponentialDistribution
// ============================================================================

/**
 * Data: {0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.1}
 *
 * Exact sample statistics (N=10):
 *   sum  = 5.0
 *   mean = 0.5
 *   MLE  λ̂ = 1 / mean = 2.0
 */
namespace {
const std::vector<double> kExpData = {
    0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.1};
constexpr double kExpLambda = 2.0;
}

TEST(ExponentialFitTest, UnweightedRecovery) {
    ExponentialDistribution d;
    d.fit(kExpData);
    EXPECT_NEAR(d.getLambda(), kExpLambda, 1e-9);
}

TEST(ExponentialFitTest, WeightedUniformEqualsUnweighted) {
    ExponentialDistribution dU, dW;
    dU.fit(kExpData);
    const std::vector<double> w(kExpData.size(), 1.0);
    dW.fit(kExpData, w);
    EXPECT_NEAR(dW.getLambda(), dU.getLambda(), 1e-9);
}

TEST(ExponentialFitTest, WeightedConcentratedOnHighValues) {
    // Small data vs large data; heavy weights on large → smaller λ̂ (larger mean).
    const std::vector<double> data    = {0.1, 0.1, 2.0, 2.0};
    const std::vector<double> weights = {0.01, 0.01, 5.0, 5.0};
    // Weighted mean ≈ (0.02*0.1 + 10.0*2.0) / (0.02+10.0) = 20.002/10.02 ≈ 1.996
    // Unweighted mean = (0.2 + 4.0) / 4 = 1.05

    ExponentialDistribution dU, dW;
    dU.fit(data);
    dW.fit(data, weights);
    EXPECT_LT(dW.getLambda(), dU.getLambda());  // higher mean → smaller lambda
}

// ============================================================================
// PoissonDistribution
// ============================================================================

/**
 * Data: {1,2,2,3,3,3,3,4,4,5}
 *
 * Exact sample statistics (N=10):
 *   sum  = 30
 *   mean = 3.0
 *   MLE  λ̂ = mean = 3.0
 */
namespace {
const std::vector<double> kPoissonData = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};
constexpr double kPoissonLambda = 3.0;
}

TEST(PoissonFitTest, UnweightedRecovery) {
    PoissonDistribution d;
    d.fit(kPoissonData);
    EXPECT_NEAR(d.getLambda(), kPoissonLambda, 1e-9);
}

TEST(PoissonFitTest, WeightedUniformEqualsUnweighted) {
    PoissonDistribution dU, dW;
    dU.fit(kPoissonData);
    const std::vector<double> w(kPoissonData.size(), 2.5);  // any uniform value
    dW.fit(kPoissonData, w);
    EXPECT_NEAR(dW.getLambda(), dU.getLambda(), 1e-9);
}

TEST(PoissonFitTest, WeightedConcentratedOnLargeValues) {
    // {1, 5} with heavy weight on 5 → λ̂ substantially larger than unweighted mean.
    const std::vector<double> data    = {1.0, 5.0};
    const std::vector<double> weights = {0.1, 9.9};
    // Weighted mean = (0.1*1 + 9.9*5) / 10.0 = (0.1+49.5)/10 = 4.96
    // Unweighted mean = 3.0

    PoissonDistribution dU, dW;
    dU.fit(data);
    dW.fit(data, weights);
    EXPECT_GT(dW.getLambda(), dU.getLambda());
    EXPECT_NEAR(dW.getLambda(), 4.96, 1e-9);
}

// ============================================================================
// GammaDistribution
// ============================================================================

/**
 * Data designed so sample mean = 3.0, which equals k*theta for the true
 * parameters.  The MLE (Newton–Raphson) will not give exact integer values
 * from small data, so tolerances are loose.
 *
 * Data: {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 7.5}
 *   sum  = 30.0, N = 10, mean = 3.0
 *   Fitted k and theta should satisfy k > 0, theta > 0, k*theta ≈ mean.
 */
namespace {
const std::vector<double> kGammaData = {
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 7.5};
constexpr double kGammaMean = 3.0;
}

TEST(GammaFitTest, UnweightedConvergence) {
    GammaDistribution d;
    d.fit(kGammaData);
    EXPECT_GT(d.getK(),     0.0);
    EXPECT_GT(d.getTheta(), 0.0);
    // k * theta = fitted mean; must match sample mean within MLE error
    EXPECT_NEAR(d.getK() * d.getTheta(), kGammaMean, 0.15);
    EXPECT_TRUE(std::isfinite(d.getK()));
    EXPECT_TRUE(std::isfinite(d.getTheta()));
}

TEST(GammaFitTest, WeightedUniformEqualsUnweighted) {
    GammaDistribution dU, dW;
    dU.fit(kGammaData);
    const std::vector<double> w(kGammaData.size(), 3.0);
    dW.fit(kGammaData, w);
    EXPECT_NEAR(dW.getK(),     dU.getK(),     1e-6);
    EXPECT_NEAR(dW.getTheta(), dU.getTheta(), 1e-6);
}

TEST(GammaFitTest, WeightedConcentratedOnSmallValues) {
    // Heavy weights on small values → smaller fitted mean = k*theta.
    const std::vector<double> data    = {0.5, 0.5, 10.0, 10.0};
    const std::vector<double> weights = {8.0, 8.0, 0.5, 0.5};
    // Weighted mean = (16*0.5 + 1*10) / (16+1) = 18/17 ≈ 1.059
    // Unweighted mean = (1 + 20) / 4 = 5.25

    GammaDistribution dU, dW;
    dU.fit(data);
    dW.fit(data, weights);
    EXPECT_LT(dW.getK() * dW.getTheta(), dU.getK() * dU.getTheta());
}

// ============================================================================
// DiscreteDistribution
// ============================================================================

/**
 * Data: {0,1,1,1,2,2,2,2,3,3}  (K=4 symbols, N=10)
 *
 * Exact empirical probabilities:
 *   P(0) = 1/10 = 0.1
 *   P(1) = 3/10 = 0.3
 *   P(2) = 4/10 = 0.4
 *   P(3) = 2/10 = 0.2
 */
namespace {
const std::vector<double> kDiscreteData = {
    0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0};
}

TEST(DiscreteFitTest, UnweightedExactCounts) {
    DiscreteDistribution d(4);
    d.fit(kDiscreteData);
    EXPECT_NEAR(d.getSymbolProbability(0), 0.1, 1e-9);
    EXPECT_NEAR(d.getSymbolProbability(1), 0.3, 1e-9);
    EXPECT_NEAR(d.getSymbolProbability(2), 0.4, 1e-9);
    EXPECT_NEAR(d.getSymbolProbability(3), 0.2, 1e-9);
    // Probabilities must sum to 1.
    EXPECT_NEAR(d.getProbabilitySum(), 1.0, 1e-9);
}

TEST(DiscreteFitTest, WeightedUniformEqualsUnweighted) {
    DiscreteDistribution dU(4), dW(4);
    dU.fit(kDiscreteData);
    const std::vector<double> w(kDiscreteData.size(), 1.0);
    dW.fit(kDiscreteData, w);
    for (std::size_t s = 0; s < 4; ++s) {
        EXPECT_NEAR(dW.getSymbolProbability(s), dU.getSymbolProbability(s), 1e-9);
    }
}

TEST(DiscreteFitTest, WeightedConcentratedOnSymbol2) {
    // Weight symbol 2 very heavily → P(2) should dominate.
    const std::vector<double> data    = {0.0, 1.0, 2.0, 3.0};
    const std::vector<double> weights = {0.1, 0.1, 9.7, 0.1};
    // Weighted: P(2) = 9.7 / 10.0 = 0.97; others ≈ 0.01 each.

    DiscreteDistribution dU(4), dW(4);
    dU.fit(data);
    dW.fit(data, weights);
    EXPECT_GT(dW.getSymbolProbability(2), dU.getSymbolProbability(2));
    EXPECT_GT(dW.getSymbolProbability(2), 0.9);  // clearly dominant
}

// ============================================================================
// Edge / guard cases across distributions
// ============================================================================

TEST(FitEdgeCasesTest, SinglePointReset) {
    // fit() on a single-point dataset should not crash and should leave
    // the distribution in a valid (reset) state.
    GaussianDistribution    g;  g.fit(std::vector<double>{2.0});
    ExponentialDistribution e;  e.fit(std::vector<double>{0.5});
    GammaDistribution       gm; gm.fit(std::vector<double>{1.0});
    PoissonDistribution     p;  p.fit(std::vector<double>{3.0});

    EXPECT_TRUE(std::isfinite(g.getMean()));
    EXPECT_TRUE(std::isfinite(e.getLambda()));
    EXPECT_TRUE(std::isfinite(gm.getK()));
    EXPECT_TRUE(std::isfinite(p.getLambda()));
}

TEST(FitEdgeCasesTest, NearZeroWeightsDoNotCrash) {
    // Near-zero total weight should not crash; implementation keeps current params.
    GaussianDistribution g(5.0, 1.0);
    const std::vector<double> data    = {1.0, 2.0, 3.0};
    const std::vector<double> weights = {1e-35, 1e-35, 1e-35};
    EXPECT_NO_THROW(g.fit(data, weights));
    EXPECT_TRUE(std::isfinite(g.getMean()));
    EXPECT_TRUE(std::isfinite(g.getStandardDeviation()));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
