#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <sstream>
#include "libhmm/distributions/negative_binomial_distribution.h"
#include <gtest/gtest.h>

using libhmm::NegativeBinomialDistribution;
using libhmm::Observation;

/**
 * Test basic Negative Binomial distribution functionality
 */
TEST(NegativeBinomialDistributionTest, BasicFunctionality) {

    // Test default constructor
    NegativeBinomialDistribution negbinom;
    EXPECT_EQ(negbinom.getR(), 5.0);
    EXPECT_EQ(negbinom.getP(), 0.5);

    // Test parameterized constructor
    NegativeBinomialDistribution negbinom2(3.0, 0.7);
    EXPECT_EQ(negbinom2.getR(), 3.0);
    EXPECT_EQ(negbinom2.getP(), 0.7);

}

/**
 * Test probability calculations
 */
TEST(NegativeBinomialDistributionTest, Probabilities) {

    NegativeBinomialDistribution negbinom(5.0, 0.5);

    // Test probability at some specific values
    double prob0 = negbinom.getProbability(0.0);
    double prob1 = negbinom.getProbability(1.0);
    double prob5 = negbinom.getProbability(5.0);

    EXPECT_GT(prob0, 0.0);
    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(prob5, 0.0);

    // For negative binomial, probabilities should be positive and decreasing in general
    // (but the exact pattern depends on parameters, so we just check they're positive)

    // Test out of range values
    EXPECT_EQ(negbinom.getProbability(-1.0), 0.0);

    // Test edge case p = 1
    NegativeBinomialDistribution negbinom_p1(5.0, 1.0);
    EXPECT_EQ(negbinom_p1.getProbability(0.0), 1.0);
    EXPECT_EQ(negbinom_p1.getProbability(1.0), 0.0);

}

/**
 * Test parameter fitting
 */
TEST(NegativeBinomialDistributionTest, Fitting) {

    NegativeBinomialDistribution negbinom;

    // Test with over-dispersed data (variance > mean)
    std::vector<Observation> data = {0, 1, 2, 3, 5, 8, 10, 15, 2, 4, 7, 12};
    negbinom.fit(data);

    // After fitting, parameters should be positive and valid
    EXPECT_GT(negbinom.getR(), 0.0);
    EXPECT_TRUE(negbinom.getP() > 0.0 && negbinom.getP() <= 1.0);

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    negbinom.fit(emptyData);
    EXPECT_EQ(negbinom.getR(), 5.0);
    EXPECT_EQ(negbinom.getP(), 0.5);

    // Test with single point (should reset)
    std::vector<Observation> singlePoint = {5};
    negbinom.fit(singlePoint);
    EXPECT_EQ(negbinom.getR(), 5.0);
    EXPECT_EQ(negbinom.getP(), 0.5);

}

/**
 * Test parameter validation
 */
TEST(NegativeBinomialDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    try {
        NegativeBinomialDistribution negbinom(0.0, 0.5); // Zero r
        ADD_FAILURE();                                   // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        NegativeBinomialDistribution negbinom(-1.0, 0.5); // Negative r
        ADD_FAILURE();                                    // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        NegativeBinomialDistribution negbinom(5.0, 0.0); // Zero p
        ADD_FAILURE();                                   // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        NegativeBinomialDistribution negbinom(5.0, -0.1); // Negative p
        ADD_FAILURE();                                    // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        NegativeBinomialDistribution negbinom(5.0, 1.5); // p > 1
        ADD_FAILURE();                                   // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(NegativeBinomialDistribution negbinom(nan_val, 0.5), std::invalid_argument);

    EXPECT_THROW(NegativeBinomialDistribution negbinom(5.0, inf_val), std::invalid_argument);

    // Test setters validation
    NegativeBinomialDistribution negbinom(5.0, 0.5);

    EXPECT_THROW(negbinom.setR(0.0), std::invalid_argument);

    EXPECT_THROW(negbinom.setP(0.0), std::invalid_argument);

    EXPECT_THROW(negbinom.setP(1.5), std::invalid_argument);

}

/**
 * Test string representation
 */
TEST(NegativeBinomialDistributionTest, StringRepresentation) {

    NegativeBinomialDistribution negbinom(8.0, 0.4);
    std::string str = negbinom.toString();

    // Should contain key information based on standardized format:
    // "Negative Binomial Distribution:\n      r (successes) = 8.0\n      p (success probability) = 0.4\n      Mean = 12.0\n      Variance = 30.0\n"
    EXPECT_NE(str.find("Negative Binomial"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("8.0"), std::string::npos);
    EXPECT_NE(str.find("0.4"), std::string::npos);
    EXPECT_NE(str.find("successes"), std::string::npos);
    EXPECT_NE(str.find("success probability"), std::string::npos);
    EXPECT_NE(str.find("Mean"), std::string::npos);
    EXPECT_NE(str.find("Variance"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(NegativeBinomialDistributionTest, CopyMoveSemantics) {

    NegativeBinomialDistribution original(7.5, 0.6);

    // Test copy constructor
    NegativeBinomialDistribution copied(original);
    EXPECT_EQ(copied.getR(), original.getR());
    EXPECT_EQ(copied.getP(), original.getP());

    // Test copy assignment
    NegativeBinomialDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getR(), original.getR());
    EXPECT_EQ(assigned.getP(), original.getP());

    // Test move constructor
    NegativeBinomialDistribution moved(std::move(original));
    EXPECT_EQ(moved.getR(), 7.5);
    EXPECT_EQ(moved.getP(), 0.6);

    // Test move assignment
    NegativeBinomialDistribution moveAssigned;
    NegativeBinomialDistribution temp(3.2, 0.8);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getR(), 3.2);
    EXPECT_EQ(moveAssigned.getP(), 0.8);

}

/**
 * Test invalid input handling
 */
TEST(NegativeBinomialDistributionTest, InvalidInputHandling) {

    NegativeBinomialDistribution negbinom(5.0, 0.5);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();

    EXPECT_EQ(negbinom.getProbability(nan_val), 0.0);
    EXPECT_EQ(negbinom.getProbability(inf_val), 0.0);
    EXPECT_EQ(negbinom.getProbability(neg_inf_val), 0.0);

    // Negative values should return 0
    EXPECT_EQ(negbinom.getProbability(-1.0), 0.0);
    EXPECT_EQ(negbinom.getProbability(-0.5), 0.0);

}

/**
 * Test reset functionality
 */
TEST(NegativeBinomialDistributionTest, ResetFunctionality) {

    NegativeBinomialDistribution negbinom(10.0, 0.2);
    negbinom.reset();

    EXPECT_EQ(negbinom.getR(), 5.0);
    EXPECT_EQ(negbinom.getP(), 0.5);

}

/**
 * Test negative binomial distribution properties
 */
TEST(NegativeBinomialDistributionTest, NegativeBinomialProperties) {

    NegativeBinomialDistribution negbinom(4.0, 0.3);

    // Test statistical moments
    double mean = negbinom.getMean();
    double variance = negbinom.getVariance();
    double stddev = negbinom.getStandardDeviation();

    // For NegBinom(r,p): mean = r*(1-p)/p, variance = r*(1-p)/p²
    double expected_mean = 4.0 * (1.0 - 0.3) / 0.3;
    double expected_variance = 4.0 * (1.0 - 0.3) / (0.3 * 0.3);

    EXPECT_NEAR(mean, expected_mean, 1e-10);
    EXPECT_NEAR(variance, expected_variance, 1e-10);
    EXPECT_NEAR(stddev, std::sqrt(variance), 1e-10);

    // Test that variance > mean (over-dispersion property)
    EXPECT_GT(variance, mean);

}

/**
 * Test fitting validation
 */
TEST(NegativeBinomialDistributionTest, FittingValidation) {

    NegativeBinomialDistribution negbinom;

    // Test with data containing negative values (should handle gracefully)
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};
    try {
        negbinom.fit(invalidData);
        // If it doesn't throw, the parameters should still be valid
        EXPECT_GT(negbinom.getR(), 0.0);
        EXPECT_TRUE(negbinom.getP() > 0.0 && negbinom.getP() <= 1.0);
    } catch (const std::exception &) {
        // It's also acceptable to throw for invalid data
    }

    // Test with under-dispersed data (variance <= mean) - should reset
    std::vector<Observation> underDispersedData = {1.0, 1.0, 1.0, 1.0, 1.0};
    negbinom.fit(underDispersedData);
    // Should fall back to defaults since negative binomial is not appropriate
    EXPECT_EQ(negbinom.getR(), 5.0);
    EXPECT_EQ(negbinom.getP(), 0.5);

}

/**
 * Test statistical moments
 */
TEST(NegativeBinomialDistributionTest, StatisticalMoments) {

    NegativeBinomialDistribution negbinom(6.0, 0.4);

    double mean = negbinom.getMean();
    double variance = negbinom.getVariance();
    double stddev = negbinom.getStandardDeviation();

    // Verify theoretical relationships
    double expected_mean = 6.0 * 0.6 / 0.4;             // r*(1-p)/p
    double expected_variance = 6.0 * 0.6 / (0.4 * 0.4); // r*(1-p)/p²

    EXPECT_NEAR(mean, expected_mean, 1e-10);
    EXPECT_NEAR(variance, expected_variance, 1e-10);
    EXPECT_NEAR(stddev * stddev, variance, 1e-10);

}

/**
 * Test over-dispersion property
 */
TEST(NegativeBinomialDistributionTest, OverDispersion) {

    // Negative binomial should exhibit over-dispersion (variance > mean)
    NegativeBinomialDistribution negbinom1(2.0, 0.3);
    NegativeBinomialDistribution negbinom2(10.0, 0.7);
    NegativeBinomialDistribution negbinom3(1.5, 0.1);

    EXPECT_GT(negbinom1.getVariance(), negbinom1.getMean());
    EXPECT_GT(negbinom2.getVariance(), negbinom2.getMean());
    EXPECT_GT(negbinom3.getVariance(), negbinom3.getMean());

}

/**
 * Test log probability calculations
 */
TEST(NegativeBinomialDistributionTest, LogProbability) {

    NegativeBinomialDistribution negbinom(5.0, 0.4);

    // Test log probability for valid values
    double logProb0 = negbinom.getLogProbability(0.0);
    double logProb1 = negbinom.getLogProbability(1.0);
    double logProb5 = negbinom.getLogProbability(5.0);

    // Log probabilities should be real numbers
    EXPECT_FALSE(std::isnan(logProb0));
    EXPECT_FALSE(std::isnan(logProb1));
    EXPECT_FALSE(std::isnan(logProb5));

    // Verify relationship: exp(log_prob) ≈ prob
    double prob0 = negbinom.getProbability(0.0);
    double prob1 = negbinom.getProbability(1.0);

    EXPECT_NEAR(std::exp(logProb0), prob0, 1e-10);
    EXPECT_NEAR(std::exp(logProb1), prob1, 1e-10);

    // Test invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_EQ(negbinom.getLogProbability(nan_val), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(negbinom.getLogProbability(inf_val), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(negbinom.getLogProbability(-1.0), -std::numeric_limits<double>::infinity());

    // Test edge case p = 1
    NegativeBinomialDistribution negbinom_p1(5.0, 1.0);
    EXPECT_EQ(negbinom_p1.getLogProbability(0.0), 0.0); // log(1) = 0
    EXPECT_EQ(negbinom_p1.getLogProbability(1.0), -std::numeric_limits<double>::infinity());

}

/**
 * Test CDF calculations
 */
TEST(NegativeBinomialDistributionTest, CDF) {

    NegativeBinomialDistribution negbinom(3.0, 0.6);

    // Test CDF properties
    double cdf0 = negbinom.getCumulativeProbability(0.0);
    double cdf1 = negbinom.getCumulativeProbability(1.0);
    double cdf5 = negbinom.getCumulativeProbability(5.0);
    double cdf10 = negbinom.getCumulativeProbability(10.0);

    // CDF should be non-decreasing
    EXPECT_LE(cdf0, cdf1);
    EXPECT_LE(cdf1, cdf5);
    EXPECT_LE(cdf5, cdf10);

    // CDF should be in [0,1]
    EXPECT_TRUE(cdf0 >= 0.0 && cdf0 <= 1.0);
    EXPECT_TRUE(cdf1 >= 0.0 && cdf1 <= 1.0);
    EXPECT_TRUE(cdf5 >= 0.0 && cdf5 <= 1.0);
    EXPECT_TRUE(cdf10 >= 0.0 && cdf10 <= 1.0);

    // CDF should equal probability at 0
    EXPECT_NEAR(cdf0, negbinom.getProbability(0.0), 1e-10);

    // Test boundary cases
    EXPECT_EQ(negbinom.getCumulativeProbability(-1.0), 0.0);

    // Test invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_EQ(negbinom.getCumulativeProbability(nan_val), 0.0);
    EXPECT_EQ(negbinom.getCumulativeProbability(inf_val), 0.0);

}

/**
 * Test additional statistical properties
 */
TEST(NegativeBinomialDistributionTest, AdditionalStatistics) {

    NegativeBinomialDistribution negbinom(4.0, 0.3);

    // Test mode calculation
    int mode = negbinom.getMode();
    EXPECT_TRUE(mode >= 0);

    // For r > 1, mode should be floor((r-1)*(1-p)/p)
    int expected_mode = static_cast<int>(std::floor((4.0 - 1.0) * (1.0 - 0.3) / 0.3));
    EXPECT_EQ(mode, expected_mode);

    // Test case where r <= 1
    NegativeBinomialDistribution negbinom_small_r(0.5, 0.3);
    EXPECT_EQ(negbinom_small_r.getMode(), 0);

    // Test skewness
    double skewness = negbinom.getSkewness();
    double expected_skewness = (2.0 - 0.3) / std::sqrt(4.0 * (1.0 - 0.3));
    EXPECT_NEAR(skewness, expected_skewness, 1e-10);

    // Test kurtosis
    double kurtosis = negbinom.getKurtosis();
    double expected_kurtosis = 3.0 + (6.0 / 4.0) + (0.3 * 0.3) / (4.0 * (1.0 - 0.3));
    EXPECT_NEAR(kurtosis, expected_kurtosis, 1e-10);

}

/**
 * Test equality operators
 */
TEST(NegativeBinomialDistributionTest, EqualityOperators) {

    NegativeBinomialDistribution negbinom1(5.0, 0.4);
    NegativeBinomialDistribution negbinom2(5.0, 0.4);
    NegativeBinomialDistribution negbinom3(5.0, 0.5);
    NegativeBinomialDistribution negbinom4(6.0, 0.4);

    // Test equality
    EXPECT_EQ(negbinom1, negbinom2);
    EXPECT_FALSE(negbinom1 == negbinom3);
    EXPECT_FALSE(negbinom1 == negbinom4);

    // Test inequality
    EXPECT_FALSE(negbinom1 != negbinom2);
    EXPECT_NE(negbinom1, negbinom3);
    EXPECT_NE(negbinom1, negbinom4);

}

/**
 * Test stream operators
 */
TEST(NegativeBinomialDistributionTest, StreamOperators) {

    NegativeBinomialDistribution original(7.5, 0.35);

    // Test output operator
    std::ostringstream oss;
    oss << original;
    std::string output = oss.str();

    EXPECT_NE(output.find("Negative Binomial"), std::string::npos);
    EXPECT_NE(output.find("7.5"), std::string::npos);
    EXPECT_NE(output.find("0.35"), std::string::npos);

    // Test input operator via roundtrip
    NegativeBinomialDistribution source(3.2, 0.8);
    std::ostringstream rss;
    rss << source;
    std::istringstream iss(rss.str());
    NegativeBinomialDistribution parsed;
    iss >> parsed;

    EXPECT_NEAR(parsed.getR(), 3.2, 1e-10);
    EXPECT_NEAR(parsed.getP(), 0.8, 1e-10);

    // Test input operator with a second set of parameters
    NegativeBinomialDistribution source2(4.5, 0.6);
    std::ostringstream rss2;
    rss2 << source2;
    std::istringstream iss2(rss2.str());
    NegativeBinomialDistribution parsed2;
    iss2 >> parsed2;

    EXPECT_NEAR(parsed2.getR(), 4.5, 1e-10);
    EXPECT_NEAR(parsed2.getP(), 0.6, 1e-10);

}

/**
 * Test caching performance and correctness
 */
TEST(NegativeBinomialDistributionTest, Caching) {

    NegativeBinomialDistribution negbinom(6.0, 0.4);

    // First call should populate cache
    double prob1_first = negbinom.getProbability(1.0);
    double logProb1_first = negbinom.getLogProbability(1.0);

    // Subsequent calls should use cached values and give identical results
    double prob1_second = negbinom.getProbability(1.0);
    double logProb1_second = negbinom.getLogProbability(1.0);

    EXPECT_EQ(prob1_first, prob1_second);
    EXPECT_EQ(logProb1_first, logProb1_second);

    // Changing parameters should invalidate cache
    negbinom.setP(0.5);
    double prob1_after_change = negbinom.getProbability(1.0);

    // Result should be different after parameter change
    EXPECT_NE(prob1_after_change, prob1_first);

    // Test cache invalidation with setR
    negbinom.setR(7.0);
    double prob1_after_r_change = negbinom.getProbability(1.0);
    EXPECT_NE(prob1_after_r_change, prob1_after_change);

    // Test cache invalidation with setParameters
    negbinom.setParameters(8.0, 0.3);
    double prob1_after_both_change = negbinom.getProbability(1.0);
    EXPECT_NE(prob1_after_both_change, prob1_after_r_change);

}

/**
 * Test numerical stability
 */
TEST(NegativeBinomialDistributionTest, NumericalStability) {

    // Test with large parameters
    NegativeBinomialDistribution large_negbinom(100.0, 0.99);
    double prob_large = large_negbinom.getProbability(0.0);
    EXPECT_TRUE(prob_large > 0.0 && prob_large <= 1.0);
    EXPECT_FALSE(std::isnan(prob_large) && !std::isinf(prob_large));

    // Test with small parameters
    NegativeBinomialDistribution small_negbinom(0.1, 0.01);
    double prob_small = small_negbinom.getProbability(10.0);
    EXPECT_TRUE(prob_small >= 0.0 && prob_small <= 1.0);
    EXPECT_FALSE(std::isnan(prob_small));

    // Test log probability for better numerical stability
    double logProb_large = large_negbinom.getLogProbability(0.0);
    double logProb_small = small_negbinom.getLogProbability(10.0);

    EXPECT_FALSE(std::isnan(logProb_large));
    EXPECT_FALSE(std::isnan(logProb_small));

}

/**
 * Test performance characteristics
 */
TEST(NegativeBinomialDistributionTest, Performance) {

    NegativeBinomialDistribution negbinom(8.0, 0.4);

    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 10000;
    volatile double sum = 0.0; // volatile to prevent optimization

    for (int i = 0; i < pdfIterations; ++i) {
        sum += negbinom.getProbability(i % 30); // 0 to 29
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;

    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;

    for (int i = 0; i < pdfIterations; ++i) {
        logSum += negbinom.getLogProbability(i % 30); // 0 to 29
    }

    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;

    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i % 15); // Values 0-14
    }

    start = std::chrono::high_resolution_clock::now();
    negbinom.fit(fitData);
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
    EXPECT_LT(pdfTimePerCall, 10.0);   // Less than 10 μs per PDF call
    EXPECT_LT(logPdfTimePerCall, 5.0); // Less than 5 μs per log PDF call
    EXPECT_LT(fitTimePerPoint, 20.0);  // Less than 20 μs per data point for fitting

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}