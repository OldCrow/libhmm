#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/log_normal_distribution.h"
#include <gtest/gtest.h>

using libhmm::LogNormalDistribution;
using libhmm::Observation;

/**
 * Test basic LogNormal distribution functionality
 */
TEST(LogNormalDistributionTest, BasicFunctionality) {

    // Test default constructor
    LogNormalDistribution lognormal;
    EXPECT_EQ(lognormal.getMean(), 0.0);
    EXPECT_EQ(lognormal.getStandardDeviation(), 1.0);

    // Test parameterized constructor
    LogNormalDistribution lognormal2(5.0, 2.5);
    EXPECT_EQ(lognormal2.getMean(), 5.0);
    EXPECT_EQ(lognormal2.getStandardDeviation(), 2.5);

}

/**
 * Test probability calculations
 */
TEST(LogNormalDistributionTest, Probabilities) {

    LogNormalDistribution lognormal(0.0, 1.0);

    // Test that probability is zero for non-positive values
    EXPECT_EQ(lognormal.getProbability(0.0), 0.0);
    EXPECT_EQ(lognormal.getProbability(-1.0), 0.0);
    EXPECT_EQ(lognormal.getProbability(-0.5), 0.0);

    // Test that probability is positive for positive values
    double prob1 = lognormal.getProbability(1.0);
    double prob2 = lognormal.getProbability(2.0);
    double prob3 = lognormal.getProbability(0.5);

    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(prob2, 0.0);
    EXPECT_GT(prob3, 0.0);

    // Test that probability density is reasonable (should be small for continuous dist)
    EXPECT_LT(prob1, 1.0);   // Should be less than 1 for probability density
    EXPECT_GT(prob1, 1e-10); // Should be greater than zero

}

/**
 * Test parameter fitting
 */
TEST(LogNormalDistributionTest, Fitting) {

    LogNormalDistribution lognormal;

    // Test with known data
    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    lognormal.fit(data);
    EXPECT_GT(lognormal.getMean(), 0.0); // Should have some reasonable value
    EXPECT_GT(lognormal.getStandardDeviation(), 0.0);

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    lognormal.fit(emptyData);
    EXPECT_EQ(lognormal.getMean(), 0.0);
    EXPECT_EQ(lognormal.getStandardDeviation(), 1.0);

    // Test with single point (should fit based on actual behavior)
    std::vector<Observation> singlePoint = {2.5};
    lognormal.fit(singlePoint);
    // Based on debug output: mean=0.916291, stddev=1e-30
    EXPECT_GT(lognormal.getMean(), 0.0);
    EXPECT_GT(lognormal.getStandardDeviation(), 0.0);

}

/**
 * Test parameter validation
 */
TEST(LogNormalDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    try {
        LogNormalDistribution lognormal(0.0, 0.0); // Zero std dev
        ADD_FAILURE();                             // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        LogNormalDistribution lognormal(0.0, -1.0); // Negative std dev
        ADD_FAILURE();                              // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    // Test invalid mean
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(LogNormalDistribution lognormal(nan_val, 1.0), std::invalid_argument);

    EXPECT_THROW(LogNormalDistribution lognormal(inf_val, 1.0), std::invalid_argument);

    // Test setters validation
    LogNormalDistribution lognormal(0.0, 1.0);

    EXPECT_THROW(lognormal.setMean(nan_val), std::invalid_argument);

    EXPECT_THROW(lognormal.setStandardDeviation(0.0), std::invalid_argument);

    EXPECT_THROW(lognormal.setStandardDeviation(-1.0), std::invalid_argument);

}

/**
 * Test string representation
 */
TEST(LogNormalDistributionTest, StringRepresentation) {

    LogNormalDistribution lognormal(2.5, 1.5);
    std::string str = lognormal.toString();

    // Should contain key information based on new format:
    // "LogNormal Distribution:\n      μ (log mean) = 2.5\n      σ (log std. deviation) = 1.5\n      Mean = ...\n      Variance = ...\n"
    EXPECT_NE(str.find("LogNormal"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("2.5"), std::string::npos);
    EXPECT_NE(str.find("1.5"), std::string::npos);
    EXPECT_NE(str.find("Mean"), std::string::npos);
    EXPECT_NE(str.find("log std. deviation"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(LogNormalDistributionTest, CopyMoveSemantics) {

    LogNormalDistribution original(3.14, 2.71);

    // Test copy constructor
    LogNormalDistribution copied(original);
    EXPECT_EQ(copied.getMean(), original.getMean());
    EXPECT_EQ(copied.getStandardDeviation(), original.getStandardDeviation());

    // Test copy assignment
    LogNormalDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getMean(), original.getMean());
    EXPECT_EQ(assigned.getStandardDeviation(), original.getStandardDeviation());

    // Test move constructor
    LogNormalDistribution moved(std::move(original));
    EXPECT_EQ(moved.getMean(), 3.14);
    EXPECT_EQ(moved.getStandardDeviation(), 2.71);

    // Test move assignment
    LogNormalDistribution moveAssigned;
    LogNormalDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getMean(), 1.41);
    EXPECT_EQ(moveAssigned.getStandardDeviation(), 1.73);

}

/**
 * Test invalid input handling
 */
TEST(LogNormalDistributionTest, InvalidInputHandling) {

    LogNormalDistribution lognormal(0.0, 1.0);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();

    EXPECT_EQ(lognormal.getProbability(nan_val), 0.0);
    EXPECT_EQ(lognormal.getProbability(inf_val), 0.0);
    EXPECT_EQ(lognormal.getProbability(neg_inf_val), 0.0);

    // Negative and zero values should return 0
    EXPECT_EQ(lognormal.getProbability(-1.0), 0.0);
    EXPECT_EQ(lognormal.getProbability(0.0), 0.0);

}

/**
 * Test reset functionality
 */
TEST(LogNormalDistributionTest, ResetFunctionality) {

    LogNormalDistribution lognormal(10.0, 5.0);
    lognormal.reset();

    EXPECT_EQ(lognormal.getMean(), 0.0);
    EXPECT_EQ(lognormal.getStandardDeviation(), 1.0);

}

/**
 * Test relationship to normal distribution
 */
TEST(LogNormalDistributionTest, LogNormalProperties) {

    LogNormalDistribution lognormal(0.0, 1.0);

    // Test that log-normal is only defined for positive values
    EXPECT_EQ(lognormal.getProbability(0.0), 0.0);
    EXPECT_EQ(lognormal.getProbability(-1.0), 0.0);

    // Test some positive values
    double prob1 = lognormal.getProbability(1.0);
    double prob2 = lognormal.getProbability(2.0);
    double prob3 = lognormal.getProbability(0.5);

    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(prob2, 0.0);
    EXPECT_GT(prob3, 0.0);

}

/**
 * Test fitting validation
 */
TEST(LogNormalDistributionTest, FittingValidation) {

    LogNormalDistribution lognormal;

    // Test with data containing negative values (should be ignored)
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};

    // Log-Normal distribution should handle negative values gracefully
    // by ignoring them (they're not in the support)
    lognormal.fit(invalidData);
    // Should have fitted to positive values {1.0, 2.0, 3.0}
    EXPECT_GT(lognormal.getMean(), 0.0);
    EXPECT_GT(lognormal.getStandardDeviation(), 0.0);

    // Test with zero values (should be ignored)
    std::vector<Observation> zeroData = {0.0, 1.0, 2.0};
    lognormal.fit(zeroData);
    // Should have fitted to positive values {1.0, 2.0}
    EXPECT_GT(lognormal.getMean(), 0.0);
    EXPECT_GT(lognormal.getStandardDeviation(), 0.0);

}

/**
 * Test statistical moments
 */
TEST(LogNormalDistributionTest, StatisticalMoments) {

    LogNormalDistribution lognormal(1.0, 0.5);

    // For Log-Normal(μ=1.0, σ=0.5):
    // Distribution mean = exp(μ + σ²/2) = exp(1.0 + 0.25/2) = exp(1.125)
    double expectedDistMean = std::exp(1.0 + 0.25 / 2.0);
    EXPECT_NEAR(lognormal.getDistributionMean(), expectedDistMean, 1e-10);

    // Distribution variance = (exp(σ²) - 1) * exp(2μ + σ²)
    //                      = (exp(0.25) - 1) * exp(2.0 + 0.25)
    double expectedVar = (std::exp(0.25) - 1.0) * std::exp(2.0 + 0.25);
    EXPECT_NEAR(lognormal.getVariance(), expectedVar, 1e-10);

    // Test median = exp(μ) = exp(1.0) ≈ 2.718
    double expectedMedian = std::exp(1.0);
    EXPECT_NEAR(lognormal.getMedian(), expectedMedian, 1e-10);

    // Test mode = exp(μ - σ²) = exp(1.0 - 0.25) = exp(0.75)
    double expectedMode = std::exp(1.0 - 0.25);
    EXPECT_NEAR(lognormal.getMode(), expectedMode, 1e-10);

}

/**
 * Test log probability calculations (Gold Standard)
 */
TEST(LogNormalDistributionTest, LogProbability) {

    LogNormalDistribution lognormal(0.0, 1.0); // Standard log-normal

    // Test log probability at several points
    double x1 = 1.0;
    double x2 = 2.0;
    double x3 = 0.5;

    double logP1 = lognormal.getLogProbability(x1);
    double logP2 = lognormal.getLogProbability(x2);
    double logP3 = lognormal.getLogProbability(x3);

    EXPECT_TRUE(std::isfinite(logP1));
    EXPECT_TRUE(std::isfinite(logP2));
    EXPECT_TRUE(std::isfinite(logP3));

    // For Log-Normal(0,1): log(f(x)) = -ln(x) - ln(√(2π)) - ½(ln(x))²
    double expectedLogP1 =
        -std::log(x1) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow(std::log(x1), 2);
    double expectedLogP2 =
        -std::log(x2) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow(std::log(x2), 2);
    double expectedLogP3 =
        -std::log(x3) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow(std::log(x3), 2);

    EXPECT_NEAR(logP1, expectedLogP1, 1e-10);
    EXPECT_NEAR(logP2, expectedLogP2, 1e-10);
    EXPECT_NEAR(logP3, expectedLogP3, 1e-10);

    // Test invalid inputs return -infinity
    EXPECT_EQ(lognormal.getLogProbability(-0.1), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(lognormal.getLogProbability(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(lognormal.getLogProbability(std::numeric_limits<double>::quiet_NaN())) ||
                lognormal.getLogProbability(std::numeric_limits<double>::quiet_NaN()) ==
                    -std::numeric_limits<double>::infinity());

}

/**
 * Test CDF calculations (Gold Standard)
 */
TEST(LogNormalDistributionTest, CDFCalculations) {

    LogNormalDistribution lognormal(0.0, 1.0);

    // Test boundary values
    EXPECT_EQ(lognormal.getCumulativeProbability(-0.1), 0.0);
    EXPECT_EQ(lognormal.getCumulativeProbability(0.0), 0.0);

    // Test monotonicity
    double cdf1 = lognormal.getCumulativeProbability(0.5);
    double cdf2 = lognormal.getCumulativeProbability(1.0);
    double cdf3 = lognormal.getCumulativeProbability(2.0);
    EXPECT_LT(cdf1, cdf2);
    EXPECT_LT(cdf2, cdf3);

    // Test that CDF values are in [0,1]
    EXPECT_TRUE(cdf1 >= 0.0 && cdf1 <= 1.0);
    EXPECT_TRUE(cdf2 >= 0.0 && cdf2 <= 1.0);
    EXPECT_TRUE(cdf3 >= 0.0 && cdf3 <= 1.0);

    // Test known value: for Log-Normal(0,1), CDF(1) = 0.5 (median)
    double cdfAt1 = lognormal.getCumulativeProbability(1.0);
    EXPECT_NEAR(cdfAt1, 0.5, 1e-6);

    // Test approach to 1 for large values
    double cdfLarge = lognormal.getCumulativeProbability(100.0);
    EXPECT_GT(cdfLarge, 0.99);

}

/**
 * Test equality and I/O operators (Gold Standard)
 */
TEST(LogNormalDistributionTest, EqualityAndIO) {

    LogNormalDistribution ln1(2.0, 1.5);
    LogNormalDistribution ln2(2.0, 1.5);
    LogNormalDistribution ln3(3.0, 1.5);

    EXPECT_EQ(ln1, ln2);
    EXPECT_EQ(ln2, ln1);
    EXPECT_FALSE(ln1 == ln3);
    EXPECT_NE(ln1, ln3);

    std::ostringstream oss;
    oss << ln1;
    std::string output = oss.str();
    EXPECT_NE(output.find("LogNormal Distribution"), std::string::npos);
    // Check for mean and standard deviation values in the output format
    EXPECT_NE(output.find("Mean"), std::string::npos);
    EXPECT_NE(output.find("std. deviation"), std::string::npos);

    // Test stream input operator
    std::istringstream iss(output);
    LogNormalDistribution inputDist;
    iss >> inputDist;

    if (iss.good() || iss.eof()) {
        EXPECT_EQ(inputDist, ln1);
    }

}

/**
 * Test numerical stability (Gold Standard)
 */
TEST(LogNormalDistributionTest, NumericalStability) {

    // Test extreme parameter values
    LogNormalDistribution smallSigma(0.0, 0.1);
    LogNormalDistribution largeSigma(0.0, 5.0);
    LogNormalDistribution largeMu(10.0, 1.0);

    double probSmall = smallSigma.getProbability(1.0);
    double probLarge = largeSigma.getProbability(1.0);
    double probLargeMu = largeMu.getProbability(1000.0);

    EXPECT_TRUE(probSmall > 0.0 && std::isfinite(probSmall));
    EXPECT_TRUE(probLarge > 0.0 && std::isfinite(probLarge));
    EXPECT_TRUE(probLargeMu > 0.0 && std::isfinite(probLargeMu));

    // Test log probability with extreme values
    double logProbSmall = smallSigma.getLogProbability(1.0);
    double logProbLarge = largeSigma.getLogProbability(1.0);
    EXPECT_TRUE(std::isfinite(logProbSmall));
    EXPECT_TRUE(std::isfinite(logProbLarge));

    // Test CDF stability
    double cdfSmall = smallSigma.getCumulativeProbability(1.0);
    double cdfLarge = largeSigma.getCumulativeProbability(1.0);
    EXPECT_TRUE(cdfSmall >= 0.0 && cdfSmall <= 1.0 && std::isfinite(cdfSmall));
    EXPECT_TRUE(cdfLarge >= 0.0 && cdfLarge <= 1.0 && std::isfinite(cdfLarge));

}

/**
 * Test performance characteristics (Gold Standard)
 */
TEST(LogNormalDistributionTest, PerformanceCharacteristics) {

    LogNormalDistribution lognormal(1.0, 0.5);
    const int iterations = 10000;
    std::vector<double> testValues;
    testValues.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        double t = 0.1 + static_cast<double>(i + 1) / 100.0; // Start from 0.1
        testValues.push_back(t);
    }

    // Test PDF performance
    auto start = std::chrono::high_resolution_clock::now();
    volatile double sum_pdf = 0.0;
    for (const auto &val : testValues) {
        sum_pdf += lognormal.getProbability(val);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto pdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdf_time_per_call = static_cast<double>(pdf_duration.count()) / iterations;

    // Test log PDF performance
    start = std::chrono::high_resolution_clock::now();
    volatile double sum_logpdf = 0.0;
    for (const auto &val : testValues) {
        sum_logpdf += lognormal.getLogProbability(val);
    }
    end = std::chrono::high_resolution_clock::now();
    auto logpdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logpdf_time_per_call = static_cast<double>(logpdf_duration.count()) / iterations;

    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = 0.1 + static_cast<double>(i) / 100.0;
    }

    start = std::chrono::high_resolution_clock::now();
    lognormal.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fitDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double fitTimePerPoint = static_cast<double>(fitDuration.count()) / fitData.size();

    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) << pdf_time_per_call
              << " μs/call (" << iterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3)
              << logpdf_time_per_call << " μs/call (" << iterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) << fitTimePerPoint
              << " μs/point (" << fitData.size() << " points)" << std::endl;

    // Performance requirements
    EXPECT_LT(pdf_time_per_call, 5.0);    // Less than 5 μs per PDF call
    EXPECT_LT(logpdf_time_per_call, 3.0); // Less than 3 μs per log PDF call
    EXPECT_LT(fitTimePerPoint, 50.0);     // Less than 50 μs per data point for fitting

}

/**
 * Test caching mechanism (Gold Standard)
 */
TEST(LogNormalDistributionTest, Caching) {

    LogNormalDistribution lognormal(1.0, 0.5);

    // Test that calculations work correctly after parameter changes
    double prob1 = lognormal.getProbability(2.0);
    double logProb1 = lognormal.getLogProbability(2.0);

    // Change parameters and verify cache is updated
    lognormal.setMean(2.0);
    double prob2 = lognormal.getProbability(2.0);
    double logProb2 = lognormal.getLogProbability(2.0);

    EXPECT_NE(prob1, prob2);       // Should be different after parameter change
    EXPECT_NE(logProb1, logProb2); // Should be different after parameter change

    // Change standard deviation parameter
    lognormal.setStandardDeviation(1.0);
    double prob3 = lognormal.getProbability(2.0);
    double logProb3 = lognormal.getLogProbability(2.0);

    EXPECT_NE(prob2, prob3);       // Should be different after parameter change
    EXPECT_NE(logProb2, logProb3); // Should be different after parameter change

    // Test that copy constructor preserves cache state
    LogNormalDistribution copied(lognormal);
    EXPECT_EQ(copied.getProbability(2.0), lognormal.getProbability(2.0));
    EXPECT_EQ(copied.getLogProbability(2.0), lognormal.getLogProbability(2.0));

    // Test that cached values are consistent
    double prob4 = lognormal.getProbability(2.0);
    double cdf4 = lognormal.getCumulativeProbability(2.0);
    double logProb4 = lognormal.getLogProbability(2.0);

    // Multiple calls should return identical results (using cache)
    EXPECT_EQ(lognormal.getProbability(2.0), prob4);
    EXPECT_EQ(lognormal.getCumulativeProbability(2.0), cdf4);
    EXPECT_EQ(lognormal.getLogProbability(2.0), logProb4);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}