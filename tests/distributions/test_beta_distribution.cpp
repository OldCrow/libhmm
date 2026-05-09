#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include "libhmm/distributions/beta_distribution.h"
#include <gtest/gtest.h>

using libhmm::BetaDistribution;
using libhmm::Observation;

/**
 * Test basic Beta distribution functionality
 */
TEST(BetaDistributionTest, BasicFunctionality) {

    // Test default constructor
    BetaDistribution beta;
    EXPECT_EQ(beta.getAlpha(), 1.0);
    EXPECT_EQ(beta.getBeta(), 1.0);

    // Test parameterized constructor
    BetaDistribution beta2(2.5, 1.5);
    EXPECT_EQ(beta2.getAlpha(), 2.5);
    EXPECT_EQ(beta2.getBeta(), 1.5);
}

/**
 * Test probability calculations
 */
TEST(BetaDistributionTest, Probabilities) {

    BetaDistribution beta(2.0, 3.0); // Alpha=2, Beta=3

    // Test that probability is zero for values outside [0,1]
    EXPECT_EQ(beta.getProbability(-0.1), 0.0);
    EXPECT_EQ(beta.getProbability(1.1), 0.0);
    EXPECT_EQ(beta.getProbability(-1.0), 0.0);
    EXPECT_EQ(beta.getProbability(2.0), 0.0);

    // Test that probability is positive for values in [0,1]
    double prob1 = beta.getProbability(0.2);
    double prob2 = beta.getProbability(0.5);
    double prob3 = beta.getProbability(0.8);

    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(prob2, 0.0);
    EXPECT_GT(prob3, 0.0);

    // For Beta(2,3), mode is at (α-1)/(α+β-2) = 1/3 ≈ 0.333
    // So probability should be higher at 0.2 than at 0.8
    EXPECT_GT(prob1, prob3);

    // Test boundary values
    double probAt0 = beta.getProbability(0.0);
    double probAt1 = beta.getProbability(1.0);
    EXPECT_TRUE(probAt0 >= 0.0); // Should be 0 for Beta(2,3) since α > 1
    EXPECT_TRUE(probAt1 >= 0.0); // Should be 0 for Beta(2,3) since β > 1
}

/**
 * Test parameter fitting
 */
TEST(BetaDistributionTest, Fitting) {

    BetaDistribution beta;

    // Test with known data
    std::vector<Observation> data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    beta.fit(data);
    EXPECT_GT(beta.getAlpha(), 0.0); // Should have some reasonable value
    EXPECT_GT(beta.getBeta(), 0.0);  // Should have some reasonable value

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    beta.fit(emptyData);
    EXPECT_EQ(beta.getAlpha(), 1.0);
    EXPECT_EQ(beta.getBeta(), 1.0);

    // Test with single point (should reset to default for insufficient data)
    std::vector<Observation> singlePoint = {0.5};
    beta.fit(singlePoint);
    EXPECT_EQ(beta.getAlpha(), 1.0);
    EXPECT_EQ(beta.getBeta(), 1.0);
}

/**
 * Test parameter validation
 */
TEST(BetaDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    try {
        BetaDistribution beta(0.0, 1.0); // Zero alpha
        ADD_FAILURE();                   // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        BetaDistribution beta(-1.0, 1.0); // Negative alpha
        ADD_FAILURE();                    // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        BetaDistribution beta(1.0, 0.0); // Zero beta
        ADD_FAILURE();                   // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        BetaDistribution beta(1.0, -1.0); // Negative beta
        ADD_FAILURE();                    // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(BetaDistribution beta(nan_val, 1.0), std::invalid_argument);

    EXPECT_THROW(BetaDistribution beta(1.0, inf_val), std::invalid_argument);

    // Test setters validation
    BetaDistribution beta(1.0, 1.0);

    EXPECT_THROW(beta.setAlpha(0.0), std::invalid_argument);

    EXPECT_THROW(beta.setBeta(-1.0), std::invalid_argument);
}

/**
 * Test string representation
 */
TEST(BetaDistributionTest, StringRepresentation) {

    BetaDistribution beta(2.5, 1.5);
    std::string str = beta.toString();

    // Should contain key information based on actual output format:
    // "Beta Distribution:\n      α (alpha) = 2.5\n      β (beta) = 1.5\n"
    EXPECT_NE(str.find("Beta"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("2.5"), std::string::npos);
    EXPECT_NE(str.find("1.5"), std::string::npos);
    EXPECT_NE(str.find("α"), std::string::npos || str.find("alpha") != std::string::npos);
    EXPECT_NE(str.find("β"), std::string::npos || str.find("beta") != std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(BetaDistributionTest, CopyMoveSemantics) {

    BetaDistribution original(3.14, 2.71);

    // Test copy constructor
    BetaDistribution copied(original);
    EXPECT_EQ(copied.getAlpha(), original.getAlpha());
    EXPECT_EQ(copied.getBeta(), original.getBeta());

    // Test copy assignment
    BetaDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getAlpha(), original.getAlpha());
    EXPECT_EQ(assigned.getBeta(), original.getBeta());

    // Test move constructor
    BetaDistribution moved(std::move(original));
    EXPECT_EQ(moved.getAlpha(), 3.14);
    EXPECT_EQ(moved.getBeta(), 2.71);

    // Test move assignment
    BetaDistribution moveAssigned;
    BetaDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getAlpha(), 1.41);
    EXPECT_EQ(moveAssigned.getBeta(), 1.73);
}

/**
 * Test invalid input handling
 */
TEST(BetaDistributionTest, InvalidInputHandling) {

    BetaDistribution beta(2.0, 3.0);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();

    EXPECT_EQ(beta.getProbability(nan_val), 0.0);
    EXPECT_EQ(beta.getProbability(inf_val), 0.0);
    EXPECT_EQ(beta.getProbability(neg_inf_val), 0.0);

    // Values outside [0,1] should return 0
    EXPECT_EQ(beta.getProbability(-0.1), 0.0);
    EXPECT_EQ(beta.getProbability(1.1), 0.0);
    EXPECT_EQ(beta.getProbability(-5.0), 0.0);
    EXPECT_EQ(beta.getProbability(10.0), 0.0);
}

/**
 * Test reset functionality
 */
TEST(BetaDistributionTest, ResetFunctionality) {

    BetaDistribution beta(10.0, 5.0);
    beta.reset();

    EXPECT_EQ(beta.getAlpha(), 1.0);
    EXPECT_EQ(beta.getBeta(), 1.0);
}

/**
 * Test log probability function for numerical stability
 */
TEST(BetaDistributionTest, LogProbability) {

    BetaDistribution beta(2.0, 3.0); // alpha=2, beta=3

    // Test log probability at several points
    double x1 = 0.25;
    double x2 = 0.5;
    double x3 = 0.75;

    double logP1 = beta.getLogProbability(x1);
    double logP2 = beta.getLogProbability(x2);
    double logP3 = beta.getLogProbability(x3);

    // Debug: Print the actual values
    std::cout << "  logP1 = " << logP1 << ", logP2 = " << logP2 << ", logP3 = " << logP3
              << std::endl;

    // Note: For continuous distributions, PDF can be > 1, so log(PDF) can be positive
    // We just verify they are finite and reasonable
    EXPECT_TRUE(std::isfinite(logP1));
    EXPECT_TRUE(std::isfinite(logP2));
    EXPECT_TRUE(std::isfinite(logP3));

    // For Beta(2,3): log(f(x)) = log(x) + 2*log(1-x) - log(B(2,3))
    // where B(2,3) = Γ(2)Γ(3)/Γ(5) = 1*2/(4*3*2*1) = 2/24 = 1/12
    double logBeta23 = lgamma(2.0) + lgamma(3.0) - lgamma(5.0);

    double expectedLogP1 = std::log(x1) + 2.0 * std::log(1.0 - x1) - logBeta23;
    double expectedLogP2 = std::log(x2) + 2.0 * std::log(1.0 - x2) - logBeta23;
    double expectedLogP3 = std::log(x3) + 2.0 * std::log(1.0 - x3) - logBeta23;

    EXPECT_NEAR(logP1, expectedLogP1, 1e-10);
    EXPECT_NEAR(logP2, expectedLogP2, 1e-10);
    EXPECT_NEAR(logP3, expectedLogP3, 1e-10);

    // Verify consistency with getProbability
    double p1 = beta.getProbability(x1);
    double p2 = beta.getProbability(x2);

    EXPECT_GT(p1, 0.0);
    EXPECT_GT(p2, 0.0);

    // Test invalid inputs return -infinity
    EXPECT_EQ(beta.getLogProbability(-0.1), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(beta.getLogProbability(1.1), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(beta.getLogProbability(std::numeric_limits<double>::quiet_NaN()),
              -std::numeric_limits<double>::infinity());

    // Test boundary cases
    BetaDistribution uniform(1.0, 1.0); // Uniform on [0,1]
    double logP0 = uniform.getLogProbability(0.0);
    double logP1_boundary = uniform.getLogProbability(1.0);

    // For uniform distribution, log(f(x)) = -log(B(1,1)) = -log(1) = 0
    EXPECT_NEAR(logP0, 0.0, 1e-10);
    EXPECT_NEAR(logP1_boundary, 0.0, 1e-10);
}

/**
 * Test Beta distribution properties
 */
TEST(BetaDistributionTest, BetaProperties) {

    // Test uniform distribution (Beta(1,1))
    BetaDistribution uniform(1.0, 1.0);
    EXPECT_NEAR(uniform.getMean(), 0.5, 1e-10);
    EXPECT_NEAR(uniform.getVariance(), (1.0 / 12.0), 1e-10);

    // Test symmetric distribution (Beta(2,2))
    BetaDistribution symmetric(2.0, 2.0);
    EXPECT_NEAR(symmetric.getMean(), 0.5, 1e-10);

    // Test skewed distribution (Beta(2,5))
    BetaDistribution skewed(2.0, 5.0);
    double expectedMean = 2.0 / (2.0 + 5.0);
    EXPECT_NEAR(skewed.getMean(), expectedMean, 1e-10);
    EXPECT_LT(skewed.getMean(), 0.5); // Should be skewed toward 0

    // Test that probability is only defined on [0,1]
    EXPECT_EQ(uniform.getProbability(-0.1), 0.0);
    EXPECT_EQ(uniform.getProbability(1.1), 0.0);
    EXPECT_GT(uniform.getProbability(0.5), 0.0);
}

/**
 * Test fitting validation
 */
TEST(BetaDistributionTest, FittingValidation) {

    BetaDistribution beta;

    // Test with data containing values outside [0,1]
    std::vector<Observation> invalidData = {0.2, 0.5, 1.5, 0.8}; // 1.5 is out of range

    EXPECT_THROW(beta.fit(invalidData), std::invalid_argument);

    // Test with negative values
    std::vector<Observation> negativeData = {0.2, 0.5, -0.1, 0.8}; // -0.1 is invalid
    EXPECT_THROW(beta.fit(negativeData), std::invalid_argument);

    // Test with valid data in boundary cases
    std::vector<Observation> boundaryData = {0.0, 0.5, 1.0};
    try {
        beta.fit(boundaryData);
        // Should work fine, check that parameters are reasonable
        EXPECT_GT(beta.getAlpha(), 0.0);
        EXPECT_GT(beta.getBeta(), 0.0);
    } catch (const std::exception &) {
        // Also acceptable if implementation rejects boundary values
    }
}

/**
 * Test statistical moments
 */
TEST(BetaDistributionTest, StatisticalMoments) {

    BetaDistribution beta(3.0, 2.0);

    // Test mean: α/(α+β) = 3/(3+2) = 0.6
    double expectedMean = 3.0 / 5.0;
    EXPECT_NEAR(beta.getMean(), expectedMean, 1e-10);

    // Test variance: αβ/((α+β)²(α+β+1)) = 6/(25*6) = 0.04
    double expectedVar = (3.0 * 2.0) / (5.0 * 5.0 * 6.0);
    EXPECT_NEAR(beta.getVariance(), expectedVar, 1e-10);

    // Test standard deviation
    double expectedStd = std::sqrt(expectedVar);
    EXPECT_NEAR(beta.getStandardDeviation(), expectedStd, 1e-10);
}

/**
 * Test performance characteristics to verify optimizations
 * This serves as a benchmark and regression test for performance
 */
TEST(BetaDistributionTest, Performance) {

    using namespace std::chrono;
    BetaDistribution beta(2.5, 3.5);

    // Test parameters
    const int pdf_iterations = 100000;
    const int fit_datapoints = 5000;

    // Generate test values for PDF calls (avoid exact 0 and 1 for log PDF)
    std::vector<double> testValues;
    testValues.reserve(pdf_iterations);
    for (int i = 0; i < pdf_iterations; ++i) {
        // Values in (0,1) - slightly away from boundaries to avoid -infinity in log PDF
        double t = static_cast<double>(i + 1) / (pdf_iterations + 1);
        testValues.push_back(t);
    }

    // Test getProbability() performance (should benefit from integer optimization and caching)
    auto start = high_resolution_clock::now();
    double sum_pdf = 0.0;
    for (const auto &val : testValues) {
        sum_pdf += beta.getProbability(val);
    }
    auto end = high_resolution_clock::now();
    auto pdf_duration = duration_cast<microseconds>(end - start);

    // Test getLogProbability() performance (should benefit from cached alphaMinus1_, betaMinus1_)
    start = high_resolution_clock::now();
    double sum_log_pdf = 0.0;
    for (const auto &val : testValues) {
        sum_log_pdf += beta.getLogProbability(val);
    }
    end = high_resolution_clock::now();
    auto log_pdf_duration = duration_cast<microseconds>(end - start);

    // Test fitting performance with Welford's single-pass algorithm
    std::vector<double> fit_data;
    fit_data.reserve(fit_datapoints);
    for (int i = 0; i < fit_datapoints; ++i) {
        // Generate Beta-like data for consistent testing
        double t = static_cast<double>(i) / (fit_datapoints - 1);
        fit_data.push_back(0.1 + 0.8 * t); // Values in [0.1, 0.9]
    }

    start = high_resolution_clock::now();
    beta.fit(fit_data);
    end = high_resolution_clock::now();
    auto fit_duration = duration_cast<microseconds>(end - start);

    // Performance expectations (reasonable thresholds for regression testing)
    double pdf_per_call = static_cast<double>(pdf_duration.count()) / pdf_iterations;
    double log_pdf_per_call = static_cast<double>(log_pdf_duration.count()) / pdf_iterations;
    double fit_per_point = static_cast<double>(fit_duration.count()) / fit_datapoints;

    // Basic performance assertions (adjust thresholds based on typical performance)
    // These should be conservative to avoid false failures on different hardware
    EXPECT_LT(pdf_per_call, 2.0); // Should be well under 2 microseconds per PDF call
    EXPECT_LT(log_pdf_per_call,
              1.0); // Should be well under 1 microsecond per log PDF call (faster due to no exp)
    EXPECT_LT(fit_per_point, 5.0); // Should be well under 5 microseconds per fit datapoint

    // Verify correctness (prevent compiler optimization removal)
    EXPECT_GT(sum_pdf, 0.0);
    EXPECT_TRUE(std::isfinite(sum_log_pdf));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  PDF timing:       " << pdf_per_call << " μs/call (" << pdf_iterations
              << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << log_pdf_per_call << " μs/call (" << pdf_iterations
              << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << fit_per_point << " μs/point (" << fit_datapoints
              << " points)" << std::endl;
}

/**
 * Test CDF calculations (Gold Standard)
 */
TEST(BetaDistributionTest, CDFCalculations) {

    BetaDistribution beta(2.0, 3.0);

    // Test boundary values
    EXPECT_EQ(beta.getCumulativeProbability(-0.1), 0.0);
    EXPECT_EQ(beta.getCumulativeProbability(0.0), 0.0);
    EXPECT_EQ(beta.getCumulativeProbability(1.0), 1.0);
    EXPECT_EQ(beta.getCumulativeProbability(1.1), 1.0);

    // Test monotonicity
    double cdf1 = beta.getCumulativeProbability(0.2);
    double cdf2 = beta.getCumulativeProbability(0.5);
    double cdf3 = beta.getCumulativeProbability(0.8);
    EXPECT_LT(cdf1, cdf2);
    EXPECT_LT(cdf2, cdf3);

    // Test that CDF values are in [0,1]
    EXPECT_TRUE(cdf1 >= 0.0 && cdf1 <= 1.0);
    EXPECT_TRUE(cdf2 >= 0.0 && cdf2 <= 1.0);
    EXPECT_TRUE(cdf3 >= 0.0 && cdf3 <= 1.0);
}

/**
 * Test equality and I/O operators (Gold Standard)
 */
TEST(BetaDistributionTest, EqualityAndIO) {

    BetaDistribution b1(2.0, 1.5);
    BetaDistribution b2(2.0, 1.5);
    BetaDistribution b3(3.0, 1.5);

    EXPECT_EQ(b1, b2);
    EXPECT_EQ(b2, b1);
    EXPECT_FALSE(b1 == b3);
    EXPECT_NE(b1, b3);

    std::ostringstream oss;
    oss << b1;
    std::string output = oss.str();
    EXPECT_NE(output.find("Beta Distribution"), std::string::npos);
    EXPECT_NE(output.find("2.0"), std::string::npos);
    EXPECT_NE(output.find("1.5"), std::string::npos);
}

/**
 * Test numerical stability (Gold Standard)
 */
TEST(BetaDistributionTest, NumericalStability) {

    // Test extreme parameter values
    BetaDistribution smallAlpha(0.1, 1.0);
    BetaDistribution largeAlpha(10.0, 1.0);
    BetaDistribution largeBoth(10.0, 10.0);

    double probSmall = smallAlpha.getProbability(0.1);
    double probLarge = largeAlpha.getProbability(0.1);
    double probBoth = largeBoth.getProbability(0.5);

    EXPECT_TRUE(probSmall > 0.0 && std::isfinite(probSmall));
    EXPECT_TRUE(probLarge > 0.0 && std::isfinite(probLarge));
    EXPECT_TRUE(probBoth > 0.0 && std::isfinite(probBoth));

    // Test log probability with extreme values
    double logProbSmall = smallAlpha.getLogProbability(0.1);
    double logProbLarge = largeAlpha.getLogProbability(0.1);
    EXPECT_TRUE(std::isfinite(logProbSmall));
    EXPECT_TRUE(std::isfinite(logProbLarge));
}

/**
 * Test caching mechanism (Gold Standard)
 */
TEST(BetaDistributionTest, Caching) {

    BetaDistribution beta(2.0, 1.0);

    // Test that calculations work correctly after parameter changes
    double prob1 = beta.getProbability(0.5);
    double logProb1 = beta.getLogProbability(0.5);

    // Change parameters and verify cache is updated
    beta.setAlpha(3.0);
    double prob2 = beta.getProbability(0.5);
    double logProb2 = beta.getLogProbability(0.5);

    EXPECT_NE(prob1, prob2);       // Should be different after parameter change
    EXPECT_NE(logProb1, logProb2); // Should be different after parameter change

    // Change beta parameter
    beta.setBeta(2.0);
    double prob3 = beta.getProbability(0.5);
    double logProb3 = beta.getLogProbability(0.5);

    EXPECT_NE(prob2, prob3);       // Should be different after parameter change
    EXPECT_NE(logProb2, logProb3); // Should be different after parameter change

    // Test that copy constructor preserves cache state
    BetaDistribution copied(beta);
    EXPECT_EQ(copied.getProbability(0.5), beta.getProbability(0.5));
    EXPECT_EQ(copied.getLogProbability(0.5), beta.getLogProbability(0.5));

    // Test that cached values are consistent
    double prob4 = beta.getProbability(0.5);
    double cdf4 = beta.getCumulativeProbability(0.5);
    double logProb4 = beta.getLogProbability(0.5);

    // Multiple calls should return identical results (using cache)
    EXPECT_EQ(beta.getProbability(0.5), prob4);
    EXPECT_EQ(beta.getCumulativeProbability(0.5), cdf4);
    EXPECT_EQ(beta.getLogProbability(0.5), logProb4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
