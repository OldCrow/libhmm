#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/discrete_distribution.h"
#include <gtest/gtest.h>

using libhmm::DiscreteDistribution;
using libhmm::Observation;

/**
 * Test basic Discrete distribution functionality
 */
TEST(DiscreteDistributionTest, BasicFunctionality) {

    // Test default constructor
    DiscreteDistribution discrete;
    EXPECT_EQ(discrete.getNumSymbols(), 10); // Default is 10 symbols

    // Test parameterized constructor
    DiscreteDistribution discrete2(5);
    EXPECT_EQ(discrete2.getNumSymbols(), 5);

    // Test invalid constructor parameter
    try {
        DiscreteDistribution discrete3(0); // Zero symbols
        ADD_FAILURE();                     // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }
}

/**
 * Test probability calculations
 */
TEST(DiscreteDistributionTest, Probabilities) {

    DiscreteDistribution discrete(5); // 5 symbols: 0,1,2,3,4

    // After reset, all symbols should have equal probability (1/numSymbols)
    double expectedProb = 1.0 / 5.0; // 0.2
    for (int i = 0; i < 5; i++) {
        double prob = discrete.getProbability(i);
        EXPECT_NEAR(prob, expectedProb, 1e-10);
    }

    // Test invalid symbol indices
    EXPECT_EQ(discrete.getProbability(-1), 0.0);
    EXPECT_EQ(discrete.getProbability(5), 0.0);
    EXPECT_EQ(discrete.getProbability(10), 0.0);

    // Test non-integer values (should treat as floor)
    double prob2_5 = discrete.getProbability(2.5);
    double prob2 = discrete.getProbability(2);
    EXPECT_EQ(prob2_5, prob2); // Should be same as symbol 2
}

/**
 * Test parameter fitting
 */
TEST(DiscreteDistributionTest, Fitting) {

    DiscreteDistribution discrete(5);

    // Test with known data - based on debug output: {0, 1, 1, 2, 3} -> {0.2, 0.4, 0.2, 0.2, 0}
    std::vector<Observation> data = {0, 1, 1, 2, 3};
    discrete.fit(data);

    // Check fitted probabilities
    EXPECT_NEAR(discrete.getProbability(0), 0.2, 1e-10); // 1/5
    EXPECT_NEAR(discrete.getProbability(1), 0.4, 1e-10); // 2/5
    EXPECT_NEAR(discrete.getProbability(2), 0.2, 1e-10); // 1/5
    EXPECT_NEAR(discrete.getProbability(3), 0.2, 1e-10); // 1/5
    EXPECT_NEAR(discrete.getProbability(4), 0.0, 1e-10); // 0/5

    // Test with empty data (should reset to uniform)
    std::vector<Observation> emptyData;
    discrete.fit(emptyData);
    double expectedUniform = 1.0 / 5.0;
    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(discrete.getProbability(i), expectedUniform, 1e-10);
    }

    // Test with single point
    std::vector<Observation> singlePoint = {2};
    discrete.fit(singlePoint);
    EXPECT_EQ(discrete.getProbability(0), 0.0);
    EXPECT_EQ(discrete.getProbability(1), 0.0);
    EXPECT_EQ(discrete.getProbability(2), 1.0); // All probability on symbol 2
    EXPECT_EQ(discrete.getProbability(3), 0.0);
    EXPECT_EQ(discrete.getProbability(4), 0.0);
}

/**
 * Test setProbability functionality
 */
TEST(DiscreteDistributionTest, SetProbability) {

    DiscreteDistribution discrete(3);

    // Set custom probabilities
    discrete.setProbability(0, 0.5);
    discrete.setProbability(1, 0.3);
    discrete.setProbability(2, 0.2);

    EXPECT_NEAR(discrete.getProbability(0), 0.5, 1e-10);
    EXPECT_NEAR(discrete.getProbability(1), 0.3, 1e-10);
    EXPECT_NEAR(discrete.getProbability(2), 0.2, 1e-10);

    // Test invalid probability values
    try {
        discrete.setProbability(0, -0.1); // Negative probability
        ADD_FAILURE();                    // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior (if implemented)
    } catch (...) {
        // Some implementations might not validate, which is also acceptable
    }

    try {
        discrete.setProbability(0, 1.1); // Probability > 1
        ADD_FAILURE();                   // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior (if implemented)
    } catch (...) {
        // Some implementations might not validate, which is also acceptable
    }
}

/**
 * Test string representation
 */
TEST(DiscreteDistributionTest, StringRepresentation) {

    DiscreteDistribution discrete(5);
    std::string str = discrete.toString();

    // Should contain key information (modern format focuses on content, not specific formatting)
    EXPECT_NE(str.find("Discrete"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("0.2"), std::string::npos); // Should contain the probabilities
    // Modern format is more readable - focus on content rather than specific separators

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(DiscreteDistributionTest, CopyMoveSemantics) {

    DiscreteDistribution original(3);
    original.setProbability(0, 0.6);
    original.setProbability(1, 0.3);
    original.setProbability(2, 0.1);

    // Test copy constructor
    DiscreteDistribution copied(original);
    EXPECT_EQ(copied.getNumSymbols(), original.getNumSymbols());
    EXPECT_NEAR(copied.getProbability(0), 0.6, 1e-10);
    EXPECT_NEAR(copied.getProbability(1), 0.3, 1e-10);
    EXPECT_NEAR(copied.getProbability(2), 0.1, 1e-10);

    // Test copy assignment
    DiscreteDistribution assigned(5); // Different size initially
    assigned = original;
    EXPECT_EQ(assigned.getNumSymbols(), original.getNumSymbols());
    EXPECT_NEAR(assigned.getProbability(0), 0.6, 1e-10);
    EXPECT_NEAR(assigned.getProbability(1), 0.3, 1e-10);
    EXPECT_NEAR(assigned.getProbability(2), 0.1, 1e-10);

    // Test move constructor
    DiscreteDistribution moved(std::move(original));
    EXPECT_EQ(moved.getNumSymbols(), 3);
    EXPECT_NEAR(moved.getProbability(0), 0.6, 1e-10);
    EXPECT_NEAR(moved.getProbability(1), 0.3, 1e-10);
    EXPECT_NEAR(moved.getProbability(2), 0.1, 1e-10);

    // Test move assignment
    DiscreteDistribution moveAssigned(2);
    DiscreteDistribution temp(4);
    temp.setProbability(1, 0.8);
    temp.setProbability(3, 0.2);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getNumSymbols(), 4);
    EXPECT_NEAR(moveAssigned.getProbability(1), 0.8, 1e-10);
    EXPECT_NEAR(moveAssigned.getProbability(3), 0.2, 1e-10);
}

/**
 * Test invalid input handling
 */
TEST(DiscreteDistributionTest, InvalidInputHandling) {

    DiscreteDistribution discrete(5);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();

    EXPECT_EQ(discrete.getProbability(nan_val), 0.0);
    EXPECT_EQ(discrete.getProbability(inf_val), 0.0);
    EXPECT_EQ(discrete.getProbability(neg_inf_val), 0.0);

    // Test with out-of-range indices
    EXPECT_EQ(discrete.getProbability(-1), 0.0);
    EXPECT_EQ(discrete.getProbability(5), 0.0);
    EXPECT_EQ(discrete.getProbability(100), 0.0);
}

/**
 * Test reset functionality
 */
TEST(DiscreteDistributionTest, ResetFunctionality) {

    DiscreteDistribution discrete(4);

    // Set non-uniform probabilities
    discrete.setProbability(0, 0.7);
    discrete.setProbability(1, 0.2);
    discrete.setProbability(2, 0.1);
    discrete.setProbability(3, 0.0);

    // Reset should restore uniform distribution
    discrete.reset();
    double expectedUniform = 1.0 / 4.0;
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(discrete.getProbability(i), expectedUniform, 1e-10);
    }
}

/**
 * Test discrete distribution properties
 */
TEST(DiscreteDistributionTest, DiscreteProperties) {

    DiscreteDistribution discrete(3);

    // Test that probabilities sum to 1
    double totalProb = 0.0;
    for (int i = 0; i < 3; i++) {
        totalProb += discrete.getProbability(i);
    }
    EXPECT_NEAR(totalProb, 1.0, 1e-10);

    // Test with fitted data
    std::vector<Observation> data = {0, 0, 1, 2};
    discrete.fit(data);

    totalProb = 0.0;
    for (int i = 0; i < 3; i++) {
        totalProb += discrete.getProbability(i);
    }
    EXPECT_NEAR(totalProb, 1.0, 1e-10);

    // Expected probabilities: 0->2/4=0.5, 1->1/4=0.25, 2->1/4=0.25
    EXPECT_NEAR(discrete.getProbability(0), 0.5, 1e-10);
    EXPECT_NEAR(discrete.getProbability(1), 0.25, 1e-10);
    EXPECT_NEAR(discrete.getProbability(2), 0.25, 1e-10);
}

/**
 * Test fitting validation
 */
TEST(DiscreteDistributionTest, FittingValidation) {

    DiscreteDistribution discrete(5);

    // Test with data containing out-of-range values
    std::vector<Observation> invalidData = {0, 1, 5, 2}; // 5 is out of range

    // Discrete distribution should handle out-of-range values gracefully
    try {
        discrete.fit(invalidData);
        // If it doesn't throw, check that valid symbols still have reasonable probabilities
        double totalValidProb = 0.0;
        for (int i = 0; i < 5; i++) {
            totalValidProb += discrete.getProbability(i);
        }
        // Should still sum close to 1 (might ignore invalid values)
        EXPECT_GT(totalValidProb, 0.5); // At least some probability assigned
    } catch (const std::exception &) {
        // It's also acceptable to throw for invalid data
    }

    // Test with negative values
    std::vector<Observation> negativeData = {0, 1, -1, 2};
    try {
        discrete.fit(negativeData);
        // Check that non-negative symbols still get reasonable probabilities
        EXPECT_TRUE(discrete.getProbability(0) >= 0.0);
        EXPECT_TRUE(discrete.getProbability(1) >= 0.0);
        EXPECT_TRUE(discrete.getProbability(2) >= 0.0);
    } catch (const std::exception &) {
        // Acceptable to reject negative values
    }
}

/**
 * Test log probability calculations
 */
TEST(DiscreteDistributionTest, LogProbability) {

    DiscreteDistribution discrete(3);
    discrete.setProbability(0, 0.5);
    discrete.setProbability(1, 0.3);
    discrete.setProbability(2, 0.2);

    // Test valid values
    double logProb0 = discrete.getLogProbability(0.0);
    double prob0 = discrete.getProbability(0.0);

    // log(prob) should equal logProb (within numerical precision)
    EXPECT_NEAR(std::log(prob0), logProb0, 1e-10);

    // Test out of range (should return -infinity)
    double logProbNeg = discrete.getLogProbability(-1.0);
    double logProbHigh = discrete.getLogProbability(3.0);
    EXPECT_TRUE(std::isinf(logProbNeg) && logProbNeg < 0);
    EXPECT_TRUE(std::isinf(logProbHigh) && logProbHigh < 0);

    // Test with zero probability
    discrete.setProbability(2, 0.0);
    double logProbZero = discrete.getLogProbability(2.0);
    EXPECT_TRUE(std::isinf(logProbZero) && logProbZero < 0);
}

/**
 * Test CDF calculations
 */
TEST(DiscreteDistributionTest, CDF) {

    DiscreteDistribution discrete(4);
    discrete.setProbability(0, 0.1);
    discrete.setProbability(1, 0.2);
    discrete.setProbability(2, 0.3);
    discrete.setProbability(3, 0.4);

    // Test basic properties
    double cdf0 = discrete.getCumulativeProbability(0.0);
    double cdf1 = discrete.getCumulativeProbability(1.0);
    double cdf2 = discrete.getCumulativeProbability(2.0);
    double cdf3 = discrete.getCumulativeProbability(3.0);

    EXPECT_NEAR(cdf0, 0.1, 1e-10);
    EXPECT_NEAR(cdf1, 0.3, 1e-10); // 0.1 + 0.2
    EXPECT_NEAR(cdf2, 0.6, 1e-10); // 0.1 + 0.2 + 0.3
    EXPECT_NEAR(cdf3, 1.0, 1e-10); // 0.1 + 0.2 + 0.3 + 0.4

    // CDF should be monotonic
    EXPECT_LE(cdf0, cdf1);
    EXPECT_LE(cdf1, cdf2);
    EXPECT_LE(cdf2, cdf3);

    // Test boundary cases
    EXPECT_EQ(discrete.getCumulativeProbability(-1.0), 0.0);
    EXPECT_EQ(discrete.getCumulativeProbability(10.0), 1.0);
}

/**
 * Test equality and I/O operators
 */
TEST(DiscreteDistributionTest, EqualityAndIO) {

    DiscreteDistribution discrete1(3);
    discrete1.setProbability(0, 0.5);
    discrete1.setProbability(1, 0.3);
    discrete1.setProbability(2, 0.2);

    DiscreteDistribution discrete2(3);
    discrete2.setProbability(0, 0.5);
    discrete2.setProbability(1, 0.3);
    discrete2.setProbability(2, 0.2);

    DiscreteDistribution discrete3(3);
    discrete3.setProbability(0, 0.6);
    discrete3.setProbability(1, 0.2);
    discrete3.setProbability(2, 0.2);

    DiscreteDistribution discrete4(4); // Different size

    // Test equality
    EXPECT_EQ(discrete1, discrete2);
    EXPECT_FALSE(discrete1 == discrete3);
    EXPECT_FALSE(discrete1 == discrete4);

    // Test inequality
    EXPECT_FALSE(discrete1 != discrete2);
    EXPECT_NE(discrete1, discrete3);
    EXPECT_NE(discrete1, discrete4);

    // Test stream output
    std::ostringstream oss;
    oss << discrete1;
    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("Discrete"), std::string::npos);

    // Test stream input via roundtrip
    DiscreteDistribution source(2);
    source.setProbability(0.0, 0.7);
    source.setProbability(1.0, 0.3);
    std::ostringstream rss;
    rss << source;
    std::istringstream iss(rss.str());
    DiscreteDistribution inputDiscrete;
    iss >> inputDiscrete;
    EXPECT_EQ(inputDiscrete.getNumSymbols(), 2);
    EXPECT_NEAR(inputDiscrete.getProbability(0), 0.7, 1e-10);
    EXPECT_NEAR(inputDiscrete.getProbability(1), 0.3, 1e-10);
}

/**
 * Test performance characteristics
 */
TEST(DiscreteDistributionTest, Performance) {

    DiscreteDistribution discrete(100); // Larger distribution

    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 10000;
    volatile double sum = 0.0; // volatile to prevent optimization

    for (int i = 0; i < pdfIterations; ++i) {
        sum += discrete.getProbability(i % 100); // 0 to 99
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;

    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;

    for (int i = 0; i < pdfIterations; ++i) {
        logSum += discrete.getLogProbability(i % 100); // 0 to 99
    }

    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;

    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i % 20); // Values 0-19
    }

    start = std::chrono::high_resolution_clock::now();
    discrete.fit(fitData);
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
    EXPECT_LT(pdfTimePerCall, 2.0);    // Less than 2 μs per PDF call (discrete should be very fast)
    EXPECT_LT(logPdfTimePerCall, 2.0); // Less than 2 μs per log PDF call
    EXPECT_LT(fitTimePerPoint,
              2.0); // Less than 2 μs per data point for fitting (discrete fitting is simple)
}

/**
 * Test caching mechanism
 */
TEST(DiscreteDistributionTest, Caching) {

    DiscreteDistribution discrete(3);
    discrete.setProbability(0, 0.4);
    discrete.setProbability(1, 0.3);
    discrete.setProbability(2, 0.3);

    // Test entropy calculation (uses caching)
    double entropy1 = discrete.getEntropy();
    double entropy2 = discrete.getEntropy();
    EXPECT_EQ(entropy1, entropy2); // Should be identical due to caching

    // Test probability sum calculation (uses caching)
    double sum1 = discrete.getProbabilitySum();
    double sum2 = discrete.getProbabilitySum();
    EXPECT_EQ(sum1, sum2);         // Should be identical due to caching
    EXPECT_NEAR(sum1, 1.0, 1e-10); // Should sum to 1

    // Changing probabilities should invalidate cache
    discrete.setProbability(0, 0.5);
    double newEntropy = discrete.getEntropy();
    EXPECT_NE(newEntropy, entropy1); // Should be different after change
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
