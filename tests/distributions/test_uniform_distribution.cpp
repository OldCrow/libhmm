#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/uniform_distribution.h"
#include <gtest/gtest.h>

using libhmm::Observation;
using libhmm::UniformDistribution;

/**
 * Test basic Uniform distribution functionality
 */
TEST(UniformDistributionTest, BasicFunctionality) {

    // Test default constructor
    UniformDistribution uniform;
    EXPECT_EQ(uniform.getA(), 0.0);
    EXPECT_EQ(uniform.getB(), 1.0);
    EXPECT_EQ(uniform.getMin(), 0.0); // Alternative getter
    EXPECT_EQ(uniform.getMax(), 1.0); // Alternative getter

    // Test parameterized constructor
    UniformDistribution uniform2(2.0, 5.0);
    EXPECT_EQ(uniform2.getA(), 2.0);
    EXPECT_EQ(uniform2.getB(), 5.0);
}

/**
 * Test probability calculations
 */
TEST(UniformDistributionTest, Probabilities) {

    UniformDistribution uniform(1.0, 4.0); // Uniform on [1, 4]

    // Test that probability is zero outside the interval
    EXPECT_EQ(uniform.getProbability(0.5), 0.0);
    EXPECT_EQ(uniform.getProbability(4.5), 0.0);
    EXPECT_EQ(uniform.getProbability(-1.0), 0.0);

    // Test that probability is constant within the interval
    double expectedPdf = 1.0 / (4.0 - 1.0); // 1/3
    EXPECT_NEAR(uniform.getProbability(1.5), expectedPdf, 1e-10);
    EXPECT_NEAR(uniform.getProbability(2.0), expectedPdf, 1e-10);
    EXPECT_NEAR(uniform.getProbability(3.5), expectedPdf, 1e-10);

    // Test boundary values
    EXPECT_NEAR(uniform.getProbability(1.0), expectedPdf, 1e-10);
    EXPECT_NEAR(uniform.getProbability(4.0), expectedPdf, 1e-10);

    // Test that probability density is reasonable
    EXPECT_GT(uniform.getProbability(2.0), 0.0);
    EXPECT_LT(uniform.getProbability(2.0), 1.0);
}

/**
 * Test parameter fitting
 */
TEST(UniformDistributionTest, Fitting) {

    UniformDistribution uniform;

    // Test with known data
    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    uniform.fit(data);

    // After fitting, bounds should encompass all data with some padding
    EXPECT_LE(uniform.getA(), 1.0);            // Should be at or below minimum
    EXPECT_TRUE(uniform.getB() >= 5.0);        // Should be at or above maximum
    EXPECT_LT(uniform.getA(), uniform.getB()); // Valid interval

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    uniform.fit(emptyData);
    EXPECT_EQ(uniform.getA(), 0.0);
    EXPECT_EQ(uniform.getB(), 1.0);

    // Test with single point (should reset to default for insufficient data)
    std::vector<Observation> singlePoint = {2.5};
    uniform.fit(singlePoint);
    EXPECT_EQ(uniform.getA(), 0.0);
    EXPECT_EQ(uniform.getB(), 1.0);
}

/**
 * Test parameter validation
 */
TEST(UniformDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    try {
        UniformDistribution uniform(5.0, 2.0); // a > b
        ADD_FAILURE();                         // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        UniformDistribution uniform(3.0, 3.0); // a == b
        ADD_FAILURE();                         // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(UniformDistribution uniform(nan_val, 1.0), std::invalid_argument);

    EXPECT_THROW(UniformDistribution uniform(1.0, inf_val), std::invalid_argument);

    // Test setters validation
    UniformDistribution uniform(1.0, 3.0);

    try {
        uniform.setA(4.0); // Would make a > b
        ADD_FAILURE();     // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        uniform.setB(0.5); // Would make b < a
        ADD_FAILURE();     // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }
}

/**
 * Test string representation
 */
TEST(UniformDistributionTest, StringRepresentation) {

    UniformDistribution uniform(2.5, 7.5);
    std::string str = uniform.toString();

    // Should contain key information based on actual output format:
    // "Uniform Distribution:\n      a (lower bound) = 2.5\n      b (upper bound) = 7.5\n"
    EXPECT_NE(str.find("Uniform"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("2.5"), std::string::npos);
    EXPECT_NE(str.find("7.5"), std::string::npos);
    EXPECT_NE(str.find("a"), std::string::npos);
    EXPECT_NE(str.find("lower bound"), std::string::npos);
    EXPECT_NE(str.find("b"), std::string::npos);
    EXPECT_NE(str.find("upper bound"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(UniformDistributionTest, CopyMoveSemantics) {

    UniformDistribution original(3.14, 6.28);

    // Test copy constructor
    UniformDistribution copied(original);
    EXPECT_EQ(copied.getA(), original.getA());
    EXPECT_EQ(copied.getB(), original.getB());

    // Test copy assignment
    UniformDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getA(), original.getA());
    EXPECT_EQ(assigned.getB(), original.getB());

    // Test move constructor
    UniformDistribution moved(std::move(original));
    EXPECT_EQ(moved.getA(), 3.14);
    EXPECT_EQ(moved.getB(), 6.28);

    // Test move assignment
    UniformDistribution moveAssigned;
    UniformDistribution temp(1.41, 2.73);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getA(), 1.41);
    EXPECT_EQ(moveAssigned.getB(), 2.73);
}

/**
 * Test invalid input handling
 */
TEST(UniformDistributionTest, InvalidInputHandling) {

    UniformDistribution uniform(0.0, 1.0);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();

    EXPECT_EQ(uniform.getProbability(nan_val), 0.0);
    EXPECT_EQ(uniform.getProbability(inf_val), 0.0);
    EXPECT_EQ(uniform.getProbability(neg_inf_val), 0.0);
}

/**
 * Test reset functionality
 */
TEST(UniformDistributionTest, ResetFunctionality) {

    UniformDistribution uniform(10.0, 20.0);
    uniform.reset();

    EXPECT_EQ(uniform.getA(), 0.0);
    EXPECT_EQ(uniform.getB(), 1.0);
}

/**
 * Test uniform distribution properties
 */
TEST(UniformDistributionTest, UniformProperties) {

    // Test standard uniform [0, 1]
    UniformDistribution standard(0.0, 1.0);
    EXPECT_NEAR(standard.getMean(), 0.5, 1e-10);            // Mean = (0+1)/2 = 0.5
    EXPECT_NEAR(standard.getVariance(), 1.0 / 12.0, 1e-10); // Variance = (1-0)²/12 = 1/12

    // Test general case [2, 8]
    UniformDistribution general(2.0, 8.0);
    double expectedMean = (2.0 + 8.0) / 2.0;               // 5.0
    double expectedVar = (8.0 - 2.0) * (8.0 - 2.0) / 12.0; // 36/12 = 3.0
    EXPECT_NEAR(general.getMean(), expectedMean, 1e-10);
    EXPECT_NEAR(general.getVariance(), expectedVar, 1e-10);

    // Test standard deviation relationship
    EXPECT_NEAR(general.getStandardDeviation(), std::sqrt(general.getVariance()), 1e-10);

    // Test that probability is constant within interval
    double pdf1 = general.getProbability(3.0);
    double pdf2 = general.getProbability(5.0);
    double pdf3 = general.getProbability(7.0);
    EXPECT_NEAR(pdf1, pdf2, 1e-10);
    EXPECT_NEAR(pdf2, pdf3, 1e-10);
}

/**
 * Test fitting validation
 */
TEST(UniformDistributionTest, FittingValidation) {

    UniformDistribution uniform;

    // Test with NaN values
    std::vector<Observation> nanData = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_THROW(uniform.fit(nanData), std::invalid_argument);

    // Test with infinity values
    std::vector<Observation> infData = {1.0, 2.0, std::numeric_limits<double>::infinity(), 3.0};
    EXPECT_THROW(uniform.fit(infData), std::invalid_argument);

    // Test with valid data
    std::vector<Observation> validData = {1.5, 2.0, 3.5, 4.0, 2.5};
    try {
        uniform.fit(validData);
        // Should work fine, check that parameters are reasonable
        EXPECT_LT(uniform.getA(), uniform.getB());
        EXPECT_LE(uniform.getA(), 1.5);     // Should encompass minimum
        EXPECT_TRUE(uniform.getB() >= 4.0); // Should encompass maximum
    } catch (const std::exception &) {
        ADD_FAILURE(); // Should not throw
    }
}

/**
 * Test parameter setters
 */
TEST(UniformDistributionTest, ParameterSetters) {

    UniformDistribution uniform(1.0, 5.0);

    // Test individual setters
    uniform.setA(0.5);
    EXPECT_EQ(uniform.getA(), 0.5);
    EXPECT_EQ(uniform.getB(), 5.0);

    uniform.setB(6.0);
    EXPECT_EQ(uniform.getA(), 0.5);
    EXPECT_EQ(uniform.getB(), 6.0);

    // Test setting both parameters
    uniform.setParameters(2.0, 8.0);
    EXPECT_EQ(uniform.getA(), 2.0);
    EXPECT_EQ(uniform.getB(), 8.0);
}

/**
 * Test edge cases
 */
TEST(UniformDistributionTest, EdgeCases) {

    // Test with very small interval
    UniformDistribution tiny(0.0, 1e-10);
    EXPECT_GT(tiny.getProbability(5e-11), 0.0); // Should be within interval
    EXPECT_EQ(tiny.getProbability(2e-10), 0.0); // Should be outside interval

    // Test with large interval
    UniformDistribution large(-1e6, 1e6);
    double largePdf = large.getProbability(0.0);
    EXPECT_GT(largePdf, 0.0);
    EXPECT_EQ(largePdf, 1.0 / (2e6)); // 1/(b-a)

    // Test with negative interval
    UniformDistribution negative(-5.0, -2.0);
    EXPECT_GT(negative.getProbability(-3.5), 0.0);
    EXPECT_EQ(negative.getProbability(0.0), 0.0);
    EXPECT_NEAR(negative.getMean(), (-3.5), 1e-10); // Mean = (-5 + -2)/2 = -3.5

    // Test isApproximatelyEqual
    UniformDistribution u1(1.0, 3.0);
    UniformDistribution u2(1.000000001, 3.000000001);
    UniformDistribution u3(1.1, 3.1);

    EXPECT_TRUE(u1.isApproximatelyEqual(u2, 1e-8));
    EXPECT_FALSE(u1.isApproximatelyEqual(u3, 1e-8));
}

/**
 * Test fitting with identical data
 */
TEST(UniformDistributionTest, FittingIdenticalData) {

    UniformDistribution uniform;

    // Test with all identical values
    std::vector<Observation> identicalData = {5.0, 5.0, 5.0, 5.0};
    uniform.fit(identicalData);

    // Should create a small interval around the value
    EXPECT_LT(uniform.getA(), 5.0);
    EXPECT_GT(uniform.getB(), 5.0);
    EXPECT_LT(uniform.getA(), uniform.getB());

    // The value should be within the fitted interval
    EXPECT_GT(uniform.getProbability(5.0), 0.0);
}

/**
 * Test log probability calculations
 */
TEST(UniformDistributionTest, LogProbabilities) {

    UniformDistribution uniform(1.0, 4.0); // Uniform on [1, 4]

    // Test that log probability is -infinity outside the interval
    EXPECT_TRUE(std::isinf(uniform.getLogProbability(0.5)) && uniform.getLogProbability(0.5) < 0);
    EXPECT_TRUE(std::isinf(uniform.getLogProbability(4.5)) && uniform.getLogProbability(4.5) < 0);
    EXPECT_TRUE(std::isinf(uniform.getLogProbability(-1.0)) && uniform.getLogProbability(-1.0) < 0);

    // Test that log probability is constant within the interval
    double expectedLogPdf = -std::log(4.0 - 1.0); // -log(3)
    EXPECT_NEAR(uniform.getLogProbability(1.5), expectedLogPdf, 1e-10);
    EXPECT_NEAR(uniform.getLogProbability(2.0), expectedLogPdf, 1e-10);
    EXPECT_NEAR(uniform.getLogProbability(3.5), expectedLogPdf, 1e-10);

    // Test boundary values
    EXPECT_NEAR(uniform.getLogProbability(1.0), expectedLogPdf, 1e-10);
    EXPECT_NEAR(uniform.getLogProbability(4.0), expectedLogPdf, 1e-10);

    // Test consistency between log and regular probability
    double x = 2.5;
    EXPECT_NEAR(std::log(uniform.getProbability(x)), uniform.getLogProbability(x), 1e-10);
}

/**
 * Test CDF calculations
 */
TEST(UniformDistributionTest, CDF) {

    UniformDistribution uniform(2.0, 8.0); // Uniform on [2, 8]

    // Test CDF values outside the interval
    EXPECT_EQ(uniform.CDF(1.0), 0.0); // Below lower bound
    EXPECT_EQ(uniform.CDF(9.0), 1.0); // Above upper bound

    // Test CDF at boundaries
    EXPECT_EQ(uniform.CDF(2.0), 0.0); // At lower bound
    EXPECT_EQ(uniform.CDF(8.0), 1.0); // At upper bound

    // Test CDF within the interval
    EXPECT_NEAR(uniform.CDF(5.0), 0.5, 1e-10);  // Midpoint should be 0.5
    EXPECT_NEAR(uniform.CDF(3.5), 0.25, 1e-10); // (3.5-2)/(8-2) = 0.25
    EXPECT_NEAR(uniform.CDF(6.5), 0.75, 1e-10); // (6.5-2)/(8-2) = 0.75

    // Test CDF is monotonically increasing
    EXPECT_LT(uniform.CDF(3.0), uniform.CDF(4.0));
    EXPECT_LT(uniform.CDF(4.0), uniform.CDF(5.0));
    EXPECT_LT(uniform.CDF(5.0), uniform.CDF(6.0));

    // Test with NaN and infinity
    EXPECT_TRUE(std::isnan(uniform.CDF(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_EQ(uniform.CDF(std::numeric_limits<double>::infinity()), 1.0);
    EXPECT_EQ(uniform.CDF(-std::numeric_limits<double>::infinity()), 0.0);
}

/**
 * Test equality operators and I/O
 */
TEST(UniformDistributionTest, EqualityAndIO) {

    UniformDistribution u1(1.5, 4.5);
    UniformDistribution u2(1.5, 4.5);
    UniformDistribution u3(2.0, 5.0);

    // Test equality operator
    EXPECT_EQ(u1, u2);
    EXPECT_FALSE(u1 == u3);

    // Test stream output
    std::ostringstream oss;
    oss << u1;
    std::string output = oss.str();
    EXPECT_NE(output.find("Uniform Distribution"), std::string::npos);
    EXPECT_NE(output.find("1.5"), std::string::npos);
    EXPECT_NE(output.find("4.5"), std::string::npos);

    std::cout << "Stream output: " << output << std::endl;

    // Test stream input (basic format check)
    std::istringstream iss("Uniform Distribution: a = 3.14 , b = 6.28");
    UniformDistribution inputDist;
    iss >> inputDist;

    if (iss.good()) {
        EXPECT_NEAR(inputDist.getA(), 3.14, 1e-10);
        EXPECT_NEAR(inputDist.getB(), 6.28, 1e-10);
    }
}

/**
 * Test caching mechanism
 */
TEST(UniformDistributionTest, Caching) {

    UniformDistribution uniform(1.0, 5.0);

    // First call should update cache
    double pdf1 = uniform.getProbability(3.0);
    double logPdf1 = uniform.getLogProbability(3.0);

    // Subsequent calls should use cached values
    double pdf2 = uniform.getProbability(3.5);
    double logPdf2 = uniform.getLogProbability(3.5);

    // Should be identical for uniform distribution
    EXPECT_NEAR(pdf1, pdf2, 1e-15);
    EXPECT_NEAR(logPdf1, logPdf2, 1e-15);

    // Changing parameters should invalidate cache
    uniform.setA(0.5);
    double pdf3 = uniform.getProbability(3.0);
    EXPECT_GT(std::abs(pdf3 - pdf1), 1e-10); // Should be different now

    // Reset should also invalidate cache
    uniform.reset();
    double pdf4 = uniform.getProbability(0.5);
    EXPECT_EQ(pdf4, 1.0); // Should be 1.0 for [0,1] interval
}

/**
 * Test performance characteristics and optimizations
 */
TEST(UniformDistributionTest, PerformanceCharacteristics) {

    UniformDistribution uniform(0.5, 5.5);

    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 100000;
    volatile double sum = 0.0; // volatile to prevent optimization

    for (int i = 0; i < pdfIterations; ++i) {
        double x = 1.0 + static_cast<double>(i) / 20000.0; // Values in [1, 6]
        sum += uniform.getProbability(x);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;

    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;

    for (int i = 0; i < pdfIterations; ++i) {
        double x = 1.0 + static_cast<double>(i) / 20000.0; // Values in [1, 6]
        logSum += uniform.getLogProbability(x);
    }

    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;

    // Test fitting timing
    std::vector<Observation> fitData(5000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i) / 1000.0; // Values in [0, 5]
    }

    start = std::chrono::high_resolution_clock::now();
    uniform.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fitDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double fitTimePerPoint = static_cast<double>(fitDuration.count()) / fitData.size();

    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) << pdfTimePerCall
              << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3) << logPdfTimePerCall
              << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) << fitTimePerPoint
              << " μs/point (" << fitData.size() << " points)" << std::endl;

    // Performance requirements (should be fast)
    EXPECT_LT(pdfTimePerCall, 1.0);    // Less than 1 μs per PDF call
    EXPECT_LT(logPdfTimePerCall, 1.0); // Less than 1 μs per log PDF call
    EXPECT_LT(fitTimePerPoint, 0.1);   // Less than 0.1 μs per data point for fitting
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
