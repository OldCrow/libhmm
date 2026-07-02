#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <sstream>
#include "libhmm/distributions/chi_squared_distribution.h"
#include <gtest/gtest.h>

using libhmm::ChiSquaredDistribution;
using libhmm::Observation;

/**
 * Test basic Chi-squared distribution functionality
 */
TEST(ChiSquaredDistributionTest, BasicFunctionality) {

    // Test default constructor
    ChiSquaredDistribution chi_dist;
    EXPECT_EQ(chi_dist.getDegreesOfFreedom(), 1.0);

    // Test parameterized constructor
    ChiSquaredDistribution chi_dist2(5.0);
    EXPECT_EQ(chi_dist2.getDegreesOfFreedom(), 5.0);

    // Test parameter setter
    chi_dist.setDegreesOfFreedom(3.0);
    EXPECT_EQ(chi_dist.getDegreesOfFreedom(), 3.0);
}

/**
 * Test probability calculations
 */
TEST(ChiSquaredDistributionTest, Probabilities) {

    ChiSquaredDistribution chi_dist(2.0); // df = 2

    // Test that probability is zero for negative values
    EXPECT_EQ(chi_dist.getProbability(-0.1), 0.0);
    EXPECT_EQ(chi_dist.getProbability(-1.0), 0.0);
    EXPECT_EQ(chi_dist.getProbability(-10.0), 0.0);

    // Test that probability is positive for non-negative values
    double prob_at_zero = chi_dist.getProbability(0.0);
    double prob_at_one = chi_dist.getProbability(1.0);
    double prob_at_two = chi_dist.getProbability(2.0);
    double prob_at_five = chi_dist.getProbability(5.0);

    EXPECT_GT(prob_at_zero, 0.0);
    EXPECT_GT(prob_at_one, 0.0);
    EXPECT_GT(prob_at_two, 0.0);
    EXPECT_GT(prob_at_five, 0.0);

    // For Chi-squared(2), the distribution is exponential-like
    // Probability should decrease as x increases
    EXPECT_GT(prob_at_zero, prob_at_one);
    EXPECT_GT(prob_at_one, prob_at_two);
    EXPECT_GT(prob_at_two, prob_at_five);

    // Test with df = 1
    ChiSquaredDistribution chi_dist_1(1.0);
    double prob_1_at_zero = chi_dist_1.getProbability(0.0);
    double prob_1_at_small = chi_dist_1.getProbability(0.1);

    // For df=1, density at x=0 should be infinite
    EXPECT_TRUE(std::isinf(prob_1_at_zero));
    EXPECT_TRUE(prob_1_at_small > 0.0 && std::isfinite(prob_1_at_small));

    // Test with df > 2 (should have mode at df-2)
    ChiSquaredDistribution chi_dist_5(5.0);
    double mode = 5.0 - 2.0; // mode = df - 2 = 3
    double prob_at_mode = chi_dist_5.getProbability(mode);
    double prob_before_mode = chi_dist_5.getProbability(mode - 1.0);
    double prob_after_mode = chi_dist_5.getProbability(mode + 1.0);

    // Probability at mode should be higher than nearby points
    EXPECT_TRUE(prob_at_mode >= prob_before_mode);
    EXPECT_TRUE(prob_at_mode >= prob_after_mode);
}

/**
 * Test statistical properties
 */
TEST(ChiSquaredDistributionTest, StatisticalProperties) {

    // Test mean and variance formulas
    ChiSquaredDistribution chi_dist_3(3.0);
    EXPECT_EQ(chi_dist_3.getMean(), 3.0);     // Mean = k
    EXPECT_EQ(chi_dist_3.getVariance(), 6.0); // Variance = 2k
    EXPECT_NEAR(chi_dist_3.getStandardDeviation(), std::sqrt(6.0), 1e-10);

    ChiSquaredDistribution chi_dist_10(10.0);
    EXPECT_EQ(chi_dist_10.getMean(), 10.0);
    EXPECT_EQ(chi_dist_10.getVariance(), 20.0);
    EXPECT_NEAR(chi_dist_10.getStandardDeviation(), std::sqrt(20.0), 1e-10);

    // Test mode calculation
    ChiSquaredDistribution chi_dist_1(1.0);
    ChiSquaredDistribution chi_dist_2(2.0);
    ChiSquaredDistribution chi_dist_5(5.0);

    EXPECT_EQ(chi_dist_1.getMode(), 0.0); // max(0, 1-2) = 0
    EXPECT_EQ(chi_dist_2.getMode(), 0.0); // max(0, 2-2) = 0
    EXPECT_EQ(chi_dist_5.getMode(), 3.0); // max(0, 5-2) = 3
}

/**
 * Test parameter fitting
 */
TEST(ChiSquaredDistributionTest, Fitting) {

    ChiSquaredDistribution chi_dist;

    // Test with data that has known mean
    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0}; // Mean = 3.0
    chi_dist.fit(data);

    // For Chi-squared, MLE estimate of k is the sample mean
    EXPECT_NEAR(chi_dist.getDegreesOfFreedom(), 3.0, 1e-10);

    // Test with another dataset
    std::vector<Observation> data2 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5}; // Mean = 4.0
    chi_dist.fit(data2);
    EXPECT_NEAR(chi_dist.getDegreesOfFreedom(), 4.0, 1e-10);

    // Test with empty data - should throw exception
    std::vector<Observation> empty_data;
    EXPECT_THROW(chi_dist.fit(empty_data), std::invalid_argument);

    // Test with negative values - should throw exception
    std::vector<Observation> negative_data = {1.0, -2.0, 3.0};
    EXPECT_THROW(chi_dist.fit(negative_data), std::invalid_argument);
}

/**
 * Test parameter validation
 */
TEST(ChiSquaredDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    EXPECT_THROW(ChiSquaredDistribution(0.0), std::invalid_argument);  // zero df
    EXPECT_THROW(ChiSquaredDistribution(-1.0), std::invalid_argument); // negative df

    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(ChiSquaredDistribution chi_dist(nan_val), std::invalid_argument);

    EXPECT_THROW(ChiSquaredDistribution chi_dist(inf_val), std::invalid_argument);

    // Test setter validation
    ChiSquaredDistribution chi_dist(1.0);

    EXPECT_THROW(chi_dist.setDegreesOfFreedom(0.0), std::invalid_argument);

    EXPECT_THROW(chi_dist.setDegreesOfFreedom(-5.0), std::invalid_argument);
}

/**
 * Test string representation and serialization
 */
TEST(ChiSquaredDistributionTest, StringRepresentation) {

    ChiSquaredDistribution chi_dist(3.5);
    std::string str = chi_dist.toString();

    // Should contain key information
    EXPECT_NE(str.find("ChiSquared Distribution"), std::string::npos);
    EXPECT_NE(str.find("3.5"), std::string::npos);
    EXPECT_NE(str.find("k"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(ChiSquaredDistributionTest, CopyMoveSemantics) {

    ChiSquaredDistribution original(4.2);

    // Test copy constructor
    ChiSquaredDistribution copied(original);
    EXPECT_EQ(copied.getDegreesOfFreedom(), original.getDegreesOfFreedom());
    EXPECT_EQ(copied.getProbability(1.0), original.getProbability(1.0));

    // Test copy assignment
    ChiSquaredDistribution assigned(1.0);
    assigned = original;
    EXPECT_EQ(assigned.getDegreesOfFreedom(), original.getDegreesOfFreedom());
    EXPECT_EQ(assigned.getProbability(1.0), original.getProbability(1.0));

    // Test move constructor
    double original_df = original.getDegreesOfFreedom();
    ChiSquaredDistribution moved(std::move(original));
    EXPECT_EQ(moved.getDegreesOfFreedom(), original_df);
}

/**
 * Test equality operators
 */
TEST(ChiSquaredDistributionTest, EqualityOperators) {

    ChiSquaredDistribution chi1(3.0);
    ChiSquaredDistribution chi2(3.0);
    ChiSquaredDistribution chi3(4.0);

    EXPECT_EQ(chi1, chi2);
    EXPECT_FALSE(chi1 == chi3);
    EXPECT_NE(chi1, chi3);
    EXPECT_FALSE(chi1 != chi2);
}

/**
 * Test edge cases and numerical stability
 */
TEST(ChiSquaredDistributionTest, EdgeCases) {

    // Test very small degrees of freedom
    ChiSquaredDistribution chi_small(1e-5);
    double prob_small = chi_small.getProbability(0.01);
    EXPECT_TRUE(std::isfinite(prob_small) && prob_small > 0.0);

    // Test very large degrees of freedom
    ChiSquaredDistribution chi_large(1e5);
    double prob_large = chi_large.getProbability(1e5); // Around the mean
    EXPECT_TRUE(std::isfinite(prob_large) && prob_large > 0.0);

    // Test reset functionality
    ChiSquaredDistribution chi_test(10.0);
    chi_test.reset();
    EXPECT_EQ(chi_test.getDegreesOfFreedom(), 1.0);

    // Test boundary behavior for different df values
    ChiSquaredDistribution chi_half(0.5); // df < 1
    double prob_half_at_zero = chi_half.getProbability(0.0);
    EXPECT_TRUE(std::isinf(prob_half_at_zero)); // Should be infinite

    ChiSquaredDistribution chi_exactly_2(2.0); // df = 2
    double prob_2_at_zero = chi_exactly_2.getProbability(0.0);
    EXPECT_TRUE(std::isfinite(prob_2_at_zero) && prob_2_at_zero > 0.0); // Should be finite
}

/**
 * Test relationship to Gamma distribution
 */
TEST(ChiSquaredDistributionTest, GammaRelationship) {

    // Chi-squared(k) is equivalent to Gamma(k/2, 2)
    // We can't directly test this without the Gamma distribution,
    // but we can verify some properties

    ChiSquaredDistribution chi_4(4.0);

    // For Chi-squared(4), which is Gamma(2, 2):
    // Mean = k = 4
    // Variance = 2k = 8
    // Mode = k - 2 = 2 (for k >= 2)

    EXPECT_EQ(chi_4.getMean(), 4.0);
    EXPECT_EQ(chi_4.getVariance(), 8.0);
    EXPECT_EQ(chi_4.getMode(), 2.0);
}

/**
 * Test log probability calculations
 */
TEST(ChiSquaredDistributionTest, LogProbabilities) {

    ChiSquaredDistribution chi_dist(3.0);

    // Test that log probability is -infinity for negative values
    EXPECT_TRUE(std::isinf(chi_dist.getLogProbability(-0.1)) &&
                chi_dist.getLogProbability(-0.1) < 0);
    EXPECT_TRUE(std::isinf(chi_dist.getLogProbability(-1.0)) &&
                chi_dist.getLogProbability(-1.0) < 0);

    // Test that log probability is finite for positive values
    EXPECT_TRUE(std::isfinite(chi_dist.getLogProbability(0.5)));
    EXPECT_TRUE(std::isfinite(chi_dist.getLogProbability(1.0)));
    EXPECT_TRUE(std::isfinite(chi_dist.getLogProbability(2.0)));

    // Test consistency between log and regular probability
    double x = 1.5;
    double prob = chi_dist.getProbability(x);
    double log_prob = chi_dist.getLogProbability(x);
    EXPECT_NEAR(std::log(prob), log_prob, 1e-10);

    // Test special case for df < 2 at x = 0
    ChiSquaredDistribution chi_1(1.0);
    EXPECT_TRUE(std::isinf(chi_1.getLogProbability(0.0)) && chi_1.getLogProbability(0.0) > 0);

    // Test special case for df = 2 at x = 0
    ChiSquaredDistribution chi_2(2.0);
    EXPECT_TRUE(std::isfinite(chi_2.getLogProbability(0.0)));
}

/**
 * Test CDF calculations
 */
TEST(ChiSquaredDistributionTest, CDF) {

    ChiSquaredDistribution chi_dist(4.0);

    // Test CDF values at boundaries
    EXPECT_EQ(chi_dist.getCumulativeProbability(-1.0), 0.0); // Below support
    EXPECT_EQ(chi_dist.getCumulativeProbability(0.0), 0.0);  // At lower bound

    // Test CDF is monotonically increasing
    EXPECT_LT(chi_dist.getCumulativeProbability(1.0), chi_dist.getCumulativeProbability(2.0));
    EXPECT_LT(chi_dist.getCumulativeProbability(2.0), chi_dist.getCumulativeProbability(4.0));
    EXPECT_LT(chi_dist.getCumulativeProbability(4.0), chi_dist.getCumulativeProbability(8.0));

    // Test CDF approaches 1 for large values
    EXPECT_GT(chi_dist.getCumulativeProbability(100.0), 0.99);

    // Test with NaN
    EXPECT_TRUE(
        std::isnan(chi_dist.getCumulativeProbability(std::numeric_limits<double>::quiet_NaN())));

    // Test known values for Chi-squared(2) which is exponential-like
    ChiSquaredDistribution chi_2(2.0);
    // For Chi-squared(2): CDF(x) = 1 - exp(-x/2)
    double cdf_at_2 = chi_2.getCumulativeProbability(2.0);
    double expected_cdf = 1.0 - std::exp(-1.0); // 1 - exp(-2/2)

    std::cout << "CDF at x=2: " << cdf_at_2 << ", Expected: " << expected_cdf
              << ", Difference: " << std::abs(cdf_at_2 - expected_cdf) << std::endl;

    // Use precision constant for numerical tolerance
    // CDF calculations involving gamma functions need slightly more tolerance
    using namespace libhmm::constants;
    EXPECT_NEAR(cdf_at_2, expected_cdf, precision::LIMIT_TOLERANCE);
}

/**
 * Test equality operators and I/O
 */
TEST(ChiSquaredDistributionTest, EqualityAndIO) {

    ChiSquaredDistribution chi1(3.5);
    ChiSquaredDistribution chi2(3.5);
    ChiSquaredDistribution chi3(4.0);

    // Test equality operator
    EXPECT_EQ(chi1, chi2);
    EXPECT_FALSE(chi1 == chi3);

    // Test stream output
    std::ostringstream oss;
    oss << chi1;
    std::string output = oss.str();
    EXPECT_NE(output.find("ChiSquared Distribution"), std::string::npos);
    EXPECT_NE(output.find("3.5"), std::string::npos);

    std::cout << "Stream output: " << output << std::endl;

    // Test stream input (basic format check)
    std::istringstream iss("ChiSquared Distribution: k = 6.75");
    ChiSquaredDistribution inputDist;
    iss >> inputDist;

    if (iss.good()) {
        EXPECT_NEAR(inputDist.getDegreesOfFreedom(), 6.75, 1e-10);
    }
}

/**
 * Test caching mechanism
 */
TEST(ChiSquaredDistributionTest, Caching) {

    ChiSquaredDistribution chi_dist(3.0);

    // First call should update cache
    double prob1 = chi_dist.getProbability(2.0);
    double logProb1 = chi_dist.getLogProbability(2.0);

    // Second call should use cached values
    double prob2 = chi_dist.getProbability(2.5);
    double logProb2 = chi_dist.getLogProbability(2.5);

    // Both should be valid
    EXPECT_TRUE(std::isfinite(prob1) && prob1 > 0.0);
    EXPECT_TRUE(std::isfinite(prob2) && prob2 > 0.0);
    EXPECT_TRUE(std::isfinite(logProb1));
    EXPECT_TRUE(std::isfinite(logProb2));

    // Changing parameters should invalidate cache
    chi_dist.setDegreesOfFreedom(5.0);
    double prob3 = chi_dist.getProbability(2.0);
    EXPECT_NE(prob3, prob1); // Should be different now

    // Reset should also invalidate cache
    chi_dist.reset();
    double prob4 = chi_dist.getProbability(2.0);
    EXPECT_NE(prob4, prob3); // Should be different after reset
}

/**
 * Test performance characteristics
 */
TEST(ChiSquaredDistributionTest, Performance) {

    ChiSquaredDistribution chi_dist(4.0);

    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 10000;
    volatile double sum = 0.0; // volatile to prevent optimization

    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i) / 1000.0; // Range 0 to 10
        sum = sum + chi_dist.getProbability(x);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;

    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;

    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i) / 1000.0; // Range 0 to 10
        logSum = logSum + chi_dist.getLogProbability(x);
    }

    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;

    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i + 1) / 100.0; // Positive values
    }

    start = std::chrono::high_resolution_clock::now();
    chi_dist.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fitDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double fitTimePerPoint = static_cast<double>(fitDuration.count()) / fitData.size();

    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) << pdfTimePerCall
              << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3) << logPdfTimePerCall
              << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) << fitTimePerPoint
              << " μs/point (" << fitData.size() << " points)" << std::endl;

    // Performance requirements (should be reasonable)
    EXPECT_LT(pdfTimePerCall, 5.0);    // Less than 5 μs per PDF call
    EXPECT_LT(logPdfTimePerCall, 3.0); // Less than 3 μs per log PDF call
    EXPECT_LT(fitTimePerPoint, 10.0);  // Less than 10 μs per data point for fitting
}

/**
 * Accuracy tests for ChiSquaredDistribution::getCumulativeProbability().
 *
 * Chi-squared(df) ~ Gamma(df/2, 2), so CDF(x; df) = P(df/2, x/2).
 * This is a second independent call path into DistributionBase::gammap().
 *
 * For even df the CDF has an exact exponential form; for df=1 it reduces
 * to std::erf (independently correct), linking chi-squared to the normal.
 */
TEST(ChiSquaredDistributionTest, CDFAccuracy) {
    // gammap uses BW_TOLERANCE = 3e-7; max observed error ~3e-8, so 1e-6 confirms
    // correctness across integer and half-integer parameter cases.
    constexpr double kTol = 1e-6;

    // df=2: CDF(x) = 1 − e^(−x/2)
    {
        ChiSquaredDistribution d(2.0);
        EXPECT_NEAR(d.getCumulativeProbability(2.0), 1.0 - std::exp(-1.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(4.0), 1.0 - std::exp(-2.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(1.0), 1.0 - std::exp(-0.5), kTol);
    }

    // df=4: CDF(x) = 1 − e^(−x/2) · (1 + x/2)
    {
        ChiSquaredDistribution d(4.0);
        auto exact = [](double x) {
            return 1.0 - std::exp(-0.5 * x) * (1.0 + 0.5 * x);
        };
        EXPECT_NEAR(d.getCumulativeProbability(2.0), exact(2.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(4.0), exact(4.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(6.0), exact(6.0), kTol);
    }

    // df=1: CDF(x) = erf(√(x/2)).
    // P(|Z| ≤ z) = CDF_χ²(z²; 1), so this cross-checks against std::erf.
    // gammap(0.5, x/2) = erf(√(x/2)) is the non-trivial half-integer case.
    {
        ChiSquaredDistribution d(1.0);
        EXPECT_NEAR(d.getCumulativeProbability(1.0), std::erf(std::sqrt(0.5)), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(4.0), std::erf(std::sqrt(2.0)), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(9.0), std::erf(std::sqrt(4.5)), kTol);
    }

    // Boundary conditions
    {
        ChiSquaredDistribution d(3.0);
        EXPECT_EQ(d.getCumulativeProbability(0.0), 0.0);
        EXPECT_EQ(d.getCumulativeProbability(-1.0), 0.0);
        EXPECT_GT(d.getCumulativeProbability(1e6), 1.0 - 1e-9);
    }
}

// =============================================================================
// Tests added to close L-5 audit gap: weighted fit, batch log-prob, JSON type.
// =============================================================================

TEST(ChiSquaredDistributionTest, WeightedFit) {
    // Weighted MOM estimator: k̂ = Σ(w_i * x_i) / Σw_i
    ChiSquaredDistribution chi;

    // Equal weights: weighted mean == sample mean.
    std::vector<double> data = {2.0, 4.0, 6.0, 8.0};
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0};
    chi.fit(data, weights);
    EXPECT_NEAR(chi.getDegreesOfFreedom(), 5.0, 1e-10);

    // Unequal weights: k̂ = (0*2 + 0*4 + 1*6 + 0*8) / 1 = 6.
    std::vector<double> w2 = {0.0, 0.0, 1.0, 0.0};
    chi.fit(data, w2);
    EXPECT_NEAR(chi.getDegreesOfFreedom(), 6.0, 1e-10);

    // All-zero weights: sumW = 0 triggers the EM-collapse guard and parameters must
    // not change (same contract as all other distributions with zero weight).
    ChiSquaredDistribution chi2(3.0);
    std::vector<double> zero = {0.0, 0.0, 0.0, 0.0};
    chi2.fit(data, zero);
    EXPECT_NEAR(chi2.getDegreesOfFreedom(), 3.0, 1e-10);
}

TEST(ChiSquaredDistributionTest, BatchLogProbabilityMatchesScalar) {
    ChiSquaredDistribution chi(5.0);
    std::vector<double> obs = {0.5, 1.0, 2.0, 5.0, 10.0, 20.0};
    std::vector<double> out(obs.size());
    chi.getBatchLogProbabilities(obs, out);
    for (std::size_t i = 0; i < obs.size(); ++i)
        EXPECT_NEAR(out[i], chi.getLogProbability(obs[i]), 1e-12)
            << "Batch vs scalar mismatch at index " << i;
}

TEST(ChiSquaredDistributionTest, JsonTypeField) {
    // to_json() must open with {"type":"ChiSquared" (exact format required by from_json).
    ChiSquaredDistribution chi(7.5);
    const std::string j = chi.to_json();
    // Mirror the prefix check used in test_hmm_json.cpp's TypeFieldPrefix test.
    const std::string prefix = std::string(R"({"type":")") + "ChiSquared" + "\"";
    EXPECT_EQ(j.substr(0, prefix.size()), prefix);
    EXPECT_NE(j.find("7.5"), std::string::npos);
}

/**
 * Main test function
 */
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
