#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/poisson_distribution.h"
#include <gtest/gtest.h>

using libhmm::Observation;
using libhmm::PoissonDistribution;

/**
 * Test basic Poisson distribution functionality
 */
TEST(PoissonDistributionTest, BasicFunctionality) {

    // Test default constructor
    PoissonDistribution poisson;
    EXPECT_EQ(poisson.getLambda(), 1.0);
    EXPECT_EQ(poisson.getMean(), 1.0);
    EXPECT_EQ(poisson.getVariance(), 1.0);
    EXPECT_NEAR(poisson.getStandardDeviation(), 1.0, 1e-10);

    // Test parameterized constructor
    PoissonDistribution poisson2(3.5);
    EXPECT_EQ(poisson2.getLambda(), 3.5);
    EXPECT_EQ(poisson2.getMean(), 3.5);
    EXPECT_EQ(poisson2.getVariance(), 3.5);
    EXPECT_NEAR(poisson2.getStandardDeviation(), std::sqrt(3.5), 1e-10);

}

/**
 * Test probability calculations
 */
TEST(PoissonDistributionTest, Probabilities) {

    PoissonDistribution poisson(2.0);

    // Test known values for λ = 2.0
    // P(X=0) = e^(-2) ≈ 0.1353
    double p0 = poisson.getProbability(0.0);
    EXPECT_NEAR(p0, std::exp(-2.0), 1e-10);

    // P(X=1) = 2 * e^(-2) ≈ 0.2707
    double p1 = poisson.getProbability(1.0);
    EXPECT_NEAR(p1, 2.0 * std::exp(-2.0), 1e-10);

    // P(X=2) = 4/2 * e^(-2) = 2 * e^(-2) ≈ 0.2707
    double p2 = poisson.getProbability(2.0);
    EXPECT_NEAR(p2, 2.0 * std::exp(-2.0), 1e-10);

    // Invalid inputs should return 0
    EXPECT_EQ(poisson.getProbability(-1.0), 0.0);
    EXPECT_EQ(poisson.getProbability(1.5), 0.0); // non-integer
    EXPECT_EQ(poisson.getProbability(std::numeric_limits<double>::infinity()), 0.0);

}

/**
 * Test parameter fitting
 */
TEST(PoissonDistributionTest, Fitting) {

    PoissonDistribution poisson;

    // Test with known data (should fit λ ≈ 2.5)
    std::vector<Observation> data = {1, 2, 2, 3, 3, 3, 4, 2, 1, 4};
    double expectedMean = 2.5; // Sum = 25, n = 10

    poisson.fit(data);
    EXPECT_NEAR(poisson.getLambda(), expectedMean, 1e-10);

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    poisson.fit(emptyData);
    EXPECT_EQ(poisson.getLambda(), 1.0);

    // Test with invalid data (should throw)
    std::vector<Observation> invalidData = {1, 2, -1, 3};
    EXPECT_THROW(poisson.fit(invalidData), std::invalid_argument);

}

/**
 * Test parameter validation
 */
TEST(PoissonDistributionTest, ParameterValidation) {

    // Test invalid lambda values in constructor
    EXPECT_THROW(PoissonDistribution poisson(-1.0), std::invalid_argument);

    EXPECT_THROW(PoissonDistribution poisson(0.0), std::invalid_argument);

    EXPECT_THROW(PoissonDistribution poisson(std::numeric_limits<double>::infinity()), std::invalid_argument);

    // Test setLambda validation
    PoissonDistribution poisson(1.0);
    EXPECT_THROW(poisson.setLambda(-1.0), std::invalid_argument);

}

/**
 * Test string representation
 */
TEST(PoissonDistributionTest, StringRepresentation) {

    PoissonDistribution poisson(2.5);
    std::string str = poisson.toString();

    // Should contain key information
    EXPECT_NE(str.find("Poisson"), std::string::npos);
    EXPECT_NE(str.find("2.5"), std::string::npos);
    EXPECT_NE(str.find("Mean"), std::string::npos);
    EXPECT_NE(str.find("Variance"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(PoissonDistributionTest, CopyMoveSemantics) {

    PoissonDistribution original(3.14);

    // Test copy constructor
    PoissonDistribution copied(original);
    EXPECT_EQ(copied.getLambda(), original.getLambda());

    // Test copy assignment
    PoissonDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getLambda(), original.getLambda());

    // Test move constructor
    PoissonDistribution moved(std::move(original));
    EXPECT_EQ(moved.getLambda(), 3.14);

    // Test move assignment
    PoissonDistribution moveAssigned;
    PoissonDistribution temp(2.71);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getLambda(), 2.71);

}

/**
 * Test numerical stability with large values
 */
TEST(PoissonDistributionTest, NumericalStability) {

    // Test with large lambda
    PoissonDistribution poissonLarge(500.0);
    double probLarge = poissonLarge.getProbability(500.0); // Should be around mode
    EXPECT_TRUE(probLarge > 0.0 && probLarge < 1.0);

    // Test with very small lambda
    PoissonDistribution poissonSmall(1e-6);
    double probSmall = poissonSmall.getProbability(0.0);
    EXPECT_TRUE(probSmall > 0.0 && probSmall < 1.0);

    // Test extreme cases that might cause overflow/underflow
    PoissonDistribution poissonExtreme(100.0);
    double probExtreme = poissonExtreme.getProbability(200.0); // Far from mean
    // Should return a very small but positive number, or 0 due to underflow
    EXPECT_TRUE(probExtreme >= 0.0);

}

/**
 * Test log probability calculations
 */
TEST(PoissonDistributionTest, LogProbability) {

    PoissonDistribution poisson(2.0);

    // Test valid values
    double logProb2 = poisson.getLogProbability(2.0);
    double prob2 = poisson.getProbability(2.0);

    // log(prob) should equal logProb (within numerical precision)
    EXPECT_NEAR(std::log(prob2), logProb2, 1e-10);

    // Test invalid inputs (should return -infinity)
    double logProbNeg = poisson.getLogProbability(-1.0);
    double logProbFloat = poisson.getLogProbability(2.5);
    EXPECT_TRUE(std::isinf(logProbNeg) && logProbNeg < 0);
    EXPECT_TRUE(std::isinf(logProbFloat) && logProbFloat < 0);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(poisson.getLogProbability(nan_val)) &&
           poisson.getLogProbability(nan_val) < 0);
    EXPECT_TRUE(std::isinf(poisson.getLogProbability(inf_val)) &&
           poisson.getLogProbability(inf_val) < 0);

}

/**
 * Test CDF calculations
 */
TEST(PoissonDistributionTest, CDF) {

    PoissonDistribution poisson(2.0);

    // Test basic properties
    double cdf0 = poisson.getCumulativeProbability(0.0);
    double cdf2 = poisson.getCumulativeProbability(2.0);
    double cdf10 = poisson.getCumulativeProbability(10.0);

    EXPECT_TRUE(cdf0 >= 0.0 && cdf0 <= 1.0);
    EXPECT_TRUE(cdf2 >= 0.0 && cdf2 <= 1.0);
    EXPECT_TRUE(cdf10 >= 0.0 && cdf10 <= 1.0);

    // CDF should be monotonic
    EXPECT_LE(cdf0, cdf2);
    EXPECT_LE(cdf2, cdf10);

    // CDF should approach 1 for large values
    EXPECT_GT(cdf10, 0.99);

    // Test boundary cases
    EXPECT_EQ(poisson.getCumulativeProbability(-1.0), 0.0);

    // Test large lambda normal approximation
    PoissonDistribution poissonLarge(200.0);
    double cdfLarge = poissonLarge.getCumulativeProbability(200.0);
    EXPECT_TRUE(cdfLarge >= 0.0 && cdfLarge <= 1.0);

}

/**
 * Test equality and I/O operators
 */
TEST(PoissonDistributionTest, EqualityAndIO) {

    PoissonDistribution poisson1(2.5);
    PoissonDistribution poisson2(2.5);
    PoissonDistribution poisson3(3.0);

    // Test equality
    EXPECT_EQ(poisson1, poisson2);
    EXPECT_FALSE(poisson1 == poisson3);

    // Test inequality
    EXPECT_FALSE(poisson1 != poisson2);
    EXPECT_NE(poisson1, poisson3);

    // Test stream output
    std::ostringstream oss;
    oss << poisson1;
    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("Poisson"), std::string::npos);

}

/**
 * Test performance characteristics
 */
TEST(PoissonDistributionTest, Performance) {

    PoissonDistribution poisson(10.0);

    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 10000;
    volatile double sum = 0.0; // volatile to prevent optimization

    for (int i = 0; i < pdfIterations; ++i) {
        sum += poisson.getProbability(i % 50); // 0 to 49
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;

    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;

    for (int i = 0; i < pdfIterations; ++i) {
        logSum += poisson.getLogProbability(i % 50); // 0 to 49
    }

    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;

    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i % 25); // Values 0-24
    }

    start = std::chrono::high_resolution_clock::now();
    poisson.fit(fitData);
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
    EXPECT_LT(fitTimePerPoint, 5.0); // Less than 5 μs per data point for fitting (Poisson fitting is simple)

}

/**
 * Test caching mechanism
 */
TEST(PoissonDistributionTest, Caching) {

    PoissonDistribution poisson(3.0);

    // First calculation should populate cache
    double prob1 = poisson.getProbability(3.0);

    // Second calculation should use cache (should be identical)
    double prob2 = poisson.getProbability(3.0);
    EXPECT_EQ(prob1, prob2);

    // Changing parameters should invalidate cache
    poisson.setLambda(5.0);
    double prob3 = poisson.getProbability(3.0);

    // The probabilities should be different (λ=3 vs λ=5 significantly affects P(X=3))
    EXPECT_GT(std::abs(prob3 - prob1), 1e-6);

    // Test that log probability also works with caching
    double logProb1 = poisson.getLogProbability(3.0);
    double logProb2 = poisson.getLogProbability(3.0);
    EXPECT_EQ(logProb1, logProb2);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}