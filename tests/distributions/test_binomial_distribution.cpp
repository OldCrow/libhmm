#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/binomial_distribution.h"
#include <gtest/gtest.h>

using libhmm::BinomialDistribution;
using libhmm::Observation;

/**
 * Test basic Binomial distribution functionality
 */
TEST(BinomialDistributionTest, BasicFunctionality) {

    // Test default constructor
    BinomialDistribution binomial;
    EXPECT_EQ(binomial.getN(), 10);
    EXPECT_EQ(binomial.getP(), 0.5);

    // Test parameterized constructor
    BinomialDistribution binomial2(20, 0.3);
    EXPECT_EQ(binomial2.getN(), 20);
    EXPECT_EQ(binomial2.getP(), 0.3);
}

/**
 * Test probability calculations
 */
TEST(BinomialDistributionTest, Probabilities) {

    BinomialDistribution binomial(10, 0.5);

    // Test probability at some specific values
    double prob0 = binomial.getProbability(0.0);
    double prob5 = binomial.getProbability(5.0);
    double prob10 = binomial.getProbability(10.0);

    EXPECT_GT(prob0, 0.0);
    EXPECT_GT(prob5, 0.0);
    EXPECT_GT(prob10, 0.0);

    // For symmetric binomial (p=0.5), P(5) should be the maximum
    EXPECT_TRUE(prob5 >= prob0);
    EXPECT_TRUE(prob5 >= prob10);

    // Test out of range values
    EXPECT_EQ(binomial.getProbability(-1.0), 0.0);
    EXPECT_EQ(binomial.getProbability(11.0), 0.0);

    // Test edge cases
    BinomialDistribution binomial_p0(10, 0.0);
    EXPECT_EQ(binomial_p0.getProbability(0.0), 1.0);
    EXPECT_EQ(binomial_p0.getProbability(1.0), 0.0);

    BinomialDistribution binomial_p1(10, 1.0);
    EXPECT_EQ(binomial_p1.getProbability(10.0), 1.0);
    EXPECT_EQ(binomial_p1.getProbability(9.0), 0.0);
}

/**
 * Test parameter fitting
 */
TEST(BinomialDistributionTest, Fitting) {

    BinomialDistribution binomial;

    // Test with data that should estimate reasonable parameters
    std::vector<Observation> data = {3, 4, 5, 6, 7, 3, 4, 5, 6, 7};
    binomial.fit(data);

    // After fitting, parameters should be positive and valid
    EXPECT_GT(binomial.getN(), 0);
    EXPECT_GT(binomial.getP(), 0.0 && binomial.getP() <= 1.0);

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    binomial.fit(emptyData);
    EXPECT_EQ(binomial.getN(), 10);
    EXPECT_EQ(binomial.getP(), 0.5);

    // Test with single point
    std::vector<Observation> singlePoint = {5};
    binomial.fit(singlePoint);
    EXPECT_TRUE(binomial.getN() >= 1);
    EXPECT_TRUE(binomial.getP() >= 0.0 && binomial.getP() <= 1.0);
}

/**
 * Test parameter validation
 */
TEST(BinomialDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    try {
        BinomialDistribution binomial(0, 0.5); // Zero n
        ADD_FAILURE();                         // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        BinomialDistribution binomial(-1, 0.5); // Negative n
        ADD_FAILURE();                          // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        BinomialDistribution binomial(10, -0.1); // Negative p
        ADD_FAILURE();                           // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        BinomialDistribution binomial(10, 1.5); // p > 1
        ADD_FAILURE();                          // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(BinomialDistribution binomial(10, nan_val), std::invalid_argument);

    EXPECT_THROW(BinomialDistribution binomial(10, inf_val), std::invalid_argument);

    // Test setters validation
    BinomialDistribution binomial(10, 0.5);

    EXPECT_THROW(binomial.setN(0), std::invalid_argument);

    EXPECT_THROW(binomial.setP(-0.1), std::invalid_argument);
}

/**
 * Test string representation
 */
TEST(BinomialDistributionTest, StringRepresentation) {

    BinomialDistribution binomial(15, 0.3);
    std::string str = binomial.toString();

    // Should contain key information based on standardized format:
    // "Binomial Distribution:\n      n (trials) = 15\n      p (success probability) = 0.3\n      Mean = 4.5\n      Variance = 3.15\n"
    EXPECT_NE(str.find("Binomial"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("15"), std::string::npos);
    EXPECT_NE(str.find("0.3"), std::string::npos);
    EXPECT_NE(str.find("trials"), std::string::npos);
    EXPECT_NE(str.find("success probability"), std::string::npos);
    EXPECT_NE(str.find("Mean"), std::string::npos);
    EXPECT_NE(str.find("Variance"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(BinomialDistributionTest, CopyMoveSemantics) {

    BinomialDistribution original(12, 0.7);

    // Test copy constructor
    BinomialDistribution copied(original);
    EXPECT_EQ(copied.getN(), original.getN());
    EXPECT_EQ(copied.getP(), original.getP());

    // Test copy assignment
    BinomialDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getN(), original.getN());
    EXPECT_EQ(assigned.getP(), original.getP());

    // Test move constructor
    BinomialDistribution moved(std::move(original));
    EXPECT_EQ(moved.getN(), 12);
    EXPECT_EQ(moved.getP(), 0.7);

    // Test move assignment
    BinomialDistribution moveAssigned;
    BinomialDistribution temp(8, 0.4);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getN(), 8);
    EXPECT_EQ(moveAssigned.getP(), 0.4);
}

/**
 * Test invalid input handling
 */
TEST(BinomialDistributionTest, InvalidInputHandling) {

    BinomialDistribution binomial(10, 0.5);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();

    EXPECT_EQ(binomial.getProbability(nan_val), 0.0);
    EXPECT_EQ(binomial.getProbability(inf_val), 0.0);
    EXPECT_EQ(binomial.getProbability(neg_inf_val), 0.0);

    // Out of range values should return 0
    EXPECT_EQ(binomial.getProbability(-1.0), 0.0);
    EXPECT_EQ(binomial.getProbability(11.0), 0.0);
}

/**
 * Test reset functionality
 */
TEST(BinomialDistributionTest, ResetFunctionality) {

    BinomialDistribution binomial(25, 0.8);
    binomial.reset();

    EXPECT_EQ(binomial.getN(), 10);
    EXPECT_EQ(binomial.getP(), 0.5);
}

/**
 * Test binomial distribution properties
 */
TEST(BinomialDistributionTest, BinomialProperties) {

    BinomialDistribution binomial(20, 0.3);

    // Test statistical moments
    double mean = binomial.getMean();
    double variance = binomial.getVariance();
    double stddev = binomial.getStandardDeviation();

    // For Binomial(n,p): mean = n*p, variance = n*p*(1-p)
    EXPECT_NEAR(mean, 20 * 0.3, 1e-10);
    EXPECT_NEAR(variance, 20 * 0.3 * 0.7, 1e-10);
    EXPECT_NEAR(stddev, std::sqrt(variance), 1e-10);

    // Test that probabilities sum to 1 (approximately)
    double total_prob = 0.0;
    for (int k = 0; k <= 20; ++k) {
        total_prob += binomial.getProbability(k);
    }
    EXPECT_NEAR(total_prob, 1.0, 1e-6); // Should sum to 1
}

/**
 * Test fitting validation
 */
TEST(BinomialDistributionTest, FittingValidation) {

    BinomialDistribution binomial;

    // Test with data containing negative values (should handle gracefully)
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};
    try {
        binomial.fit(invalidData);
        // If it doesn't throw, the parameters should still be valid
        EXPECT_GT(binomial.getN(), 0);
        EXPECT_TRUE(binomial.getP() >= 0.0 && binomial.getP() <= 1.0);
    } catch (const std::exception &) {
        // It's also acceptable to throw for invalid data
    }

    // Test with all zeros
    std::vector<Observation> zeroData = {0.0, 0.0, 0.0};
    binomial.fit(zeroData);
    EXPECT_TRUE(binomial.getN() >= 1);
    EXPECT_TRUE(binomial.getP() >= 0.0 && binomial.getP() <= 1.0);
}

/**
 * Test statistical moments
 */
TEST(BinomialDistributionTest, StatisticalMoments) {

    BinomialDistribution binomial(50, 0.4);

    double mean = binomial.getMean();
    double variance = binomial.getVariance();
    double stddev = binomial.getStandardDeviation();

    // Verify theoretical relationships
    EXPECT_NEAR(mean, 50 * 0.4, 1e-10);
    EXPECT_NEAR(variance, 50 * 0.4 * 0.6, 1e-10);
    EXPECT_NEAR(stddev * stddev, variance, 1e-10);
}

/**
 * Test log probability calculations
 */
TEST(BinomialDistributionTest, LogProbability) {

    BinomialDistribution binomial(10, 0.5);

    // Test valid values
    double logProb5 = binomial.getLogProbability(5.0);
    double prob5 = binomial.getProbability(5.0);

    // log(prob) should equal logProb (within numerical precision)
    EXPECT_NEAR(std::log(prob5), logProb5, 1e-10);

    // Test out of range (should return -infinity)
    double logProbNeg = binomial.getLogProbability(-1.0);
    double logProbHigh = binomial.getLogProbability(11.0);
    EXPECT_TRUE(std::isinf(logProbNeg) && logProbNeg < 0);
    EXPECT_TRUE(std::isinf(logProbHigh) && logProbHigh < 0);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(binomial.getLogProbability(nan_val)) &&
                binomial.getLogProbability(nan_val) < 0);
    EXPECT_TRUE(std::isinf(binomial.getLogProbability(inf_val)) &&
                binomial.getLogProbability(inf_val) < 0);
}

/**
 * Test CDF calculations
 */
TEST(BinomialDistributionTest, CDF) {

    BinomialDistribution binomial(10, 0.5);

    // Test basic properties
    double cdf0 = binomial.getCumulativeProbability(0.0);
    double cdf5 = binomial.getCumulativeProbability(5.0);
    double cdf10 = binomial.getCumulativeProbability(10.0);

    EXPECT_TRUE(cdf0 >= 0.0 && cdf0 <= 1.0);
    EXPECT_TRUE(cdf5 >= 0.0 && cdf5 <= 1.0);
    EXPECT_TRUE(cdf10 >= 0.0 && cdf10 <= 1.0);

    // CDF should be monotonic
    EXPECT_LE(cdf0, cdf5);
    EXPECT_LE(cdf5, cdf10);

    // CDF at maximum should be 1.0
    EXPECT_NEAR(cdf10, 1.0, 1e-10);

    // Test boundary cases
    EXPECT_EQ(binomial.getCumulativeProbability(-1.0), 0.0);
    EXPECT_EQ(binomial.getCumulativeProbability(15.0), 1.0);

    // Test invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_EQ(binomial.getCumulativeProbability(nan_val), 0.0);
    EXPECT_EQ(binomial.getCumulativeProbability(inf_val), 0.0);
}

/**
 * Test equality and I/O operators
 */
TEST(BinomialDistributionTest, EqualityAndIO) {

    BinomialDistribution binomial1(10, 0.5);
    BinomialDistribution binomial2(10, 0.5);
    BinomialDistribution binomial3(10, 0.6);
    BinomialDistribution binomial4(15, 0.5);

    // Test equality
    EXPECT_EQ(binomial1, binomial2);
    EXPECT_FALSE(binomial1 == binomial3);
    EXPECT_FALSE(binomial1 == binomial4);

    // Test inequality
    EXPECT_FALSE(binomial1 != binomial2);
    EXPECT_NE(binomial1, binomial3);
    EXPECT_NE(binomial1, binomial4);

    // Test stream output
    std::ostringstream oss;
    oss << binomial1;
    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("Binomial"), std::string::npos);

    // Test stream input via roundtrip
    BinomialDistribution source(20, 0.7);
    std::ostringstream rss;
    rss << source;
    std::istringstream iss(rss.str());
    BinomialDistribution inputBinomial;
    iss >> inputBinomial;
    EXPECT_EQ(inputBinomial.getN(), 20);
    EXPECT_NEAR(inputBinomial.getP(), 0.7, 1e-10);
}

/**
 * Test performance characteristics
 */
TEST(BinomialDistributionTest, Performance) {

    BinomialDistribution binomial(100, 0.3);

    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 10000;
    volatile double sum = 0.0; // volatile to prevent optimization

    for (int i = 0; i < pdfIterations; ++i) {
        sum = sum + binomial.getProbability(i % 101); // 0 to 100
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;

    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;

    for (int i = 0; i < pdfIterations; ++i) {
        logSum = logSum + binomial.getLogProbability(i % 101); // 0 to 100
    }

    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;

    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i % 31); // Values 0-30
    }

    start = std::chrono::high_resolution_clock::now();
    binomial.fit(fitData);
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
    EXPECT_LT(fitTimePerPoint, 10.0);  // Less than 10 μs per data point for fitting
}

/**
 * Test caching mechanism
 */
TEST(BinomialDistributionTest, Caching) {

    BinomialDistribution binomial(10, 0.3);

    // First calculation should populate cache
    double prob1 = binomial.getProbability(3.0);

    // Second calculation should use cache (should be identical)
    double prob2 = binomial.getProbability(3.0);
    EXPECT_EQ(prob1, prob2);

    // Changing parameters should invalidate cache - use value closer to expected mean
    binomial.setP(0.7);                          // Changes mean from 3.0 to 7.0
    double prob3 = binomial.getProbability(3.0); // Now far from new mean

    // The probabilities should be different (0.3 vs 0.7 is a significant change)
    // For Binomial(10, 0.3): P(X=3) ≈ 0.267, mean = 3
    // For Binomial(10, 0.7): P(X=3) ≈ 0.009, mean = 7
    EXPECT_GT(std::abs(prob3 - prob1), 1e-6); // Should be significantly different

    // Test that log probability also works with caching
    double logProb1 = binomial.getLogProbability(3.0);
    double logProb2 = binomial.getLogProbability(3.0);
    EXPECT_EQ(logProb1, logProb2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
