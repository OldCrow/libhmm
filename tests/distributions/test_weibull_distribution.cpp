#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <climits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/weibull_distribution.h"
#include <gtest/gtest.h>

using libhmm::Observation;
using libhmm::WeibullDistribution;

/**
 * Test basic Weibull distribution functionality
 */
TEST(WeibullDistributionTest, BasicFunctionality) {

    // Test default constructor
    WeibullDistribution weibull;
    EXPECT_EQ(weibull.getK(), 1.0);
    EXPECT_EQ(weibull.getLambda(), 1.0);
    EXPECT_EQ(weibull.getShape(), 1.0); // Alternative getter
    EXPECT_EQ(weibull.getScale(), 1.0); // Alternative getter

    // Test parameterized constructor
    WeibullDistribution weibull2(2.5, 1.5);
    EXPECT_EQ(weibull2.getK(), 2.5);
    EXPECT_EQ(weibull2.getLambda(), 1.5);
}

/**
 * Test probability calculations
 */
TEST(WeibullDistributionTest, Probabilities) {

    WeibullDistribution weibull(2.0, 1.0); // k=2, λ=1 (Rayleigh distribution)

    // Test that probability is zero for negative values
    EXPECT_EQ(weibull.getProbability(-0.1), 0.0);
    EXPECT_EQ(weibull.getProbability(-1.0), 0.0);

    // Test that probability is positive for positive values
    double prob1 = weibull.getProbability(0.5);
    double prob2 = weibull.getProbability(1.0);
    double prob3 = weibull.getProbability(2.0);

    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(prob2, 0.0);
    EXPECT_GT(prob3, 0.0);

    // For Weibull distribution, density typically decreases after the mode
    // Mode for Weibull(k,λ) = λ * ((k-1)/k)^(1/k) when k > 1
    // For k=2, λ=1: mode ≈ 0.707

    // Test boundary value at x=0
    double probAt0 = weibull.getProbability(0.0);
    EXPECT_TRUE(probAt0 >= 0.0); // Should be 0 for k=2 > 1

    // Test that probability density is reasonable (should be small for continuous dist)
    EXPECT_LT(prob1, 10.0);  // Should be reasonable
    EXPECT_GT(prob1, 1e-10); // Should be greater than zero
}

/**
 * Test parameter fitting
 */
TEST(WeibullDistributionTest, Fitting) {

    WeibullDistribution weibull;

    // Test with known data
    std::vector<Observation> data = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    weibull.fit(data);
    EXPECT_GT(weibull.getK(), 0.0);      // Should have some reasonable value
    EXPECT_GT(weibull.getLambda(), 0.0); // Should have some reasonable value

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    weibull.fit(emptyData);
    EXPECT_EQ(weibull.getK(), 1.0);
    EXPECT_EQ(weibull.getLambda(), 1.0);

    // Test with single point (should reset to default for insufficient data)
    std::vector<Observation> singlePoint = {2.5};
    weibull.fit(singlePoint);
    EXPECT_EQ(weibull.getK(), 1.0);
    EXPECT_EQ(weibull.getLambda(), 1.0);
}

/**
 * Test parameter validation
 */
TEST(WeibullDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    try {
        WeibullDistribution weibull(0.0, 1.0); // Zero k
        ADD_FAILURE();                         // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        WeibullDistribution weibull(-1.0, 1.0); // Negative k
        ADD_FAILURE();                          // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        WeibullDistribution weibull(1.0, 0.0); // Zero lambda
        ADD_FAILURE();                         // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        WeibullDistribution weibull(1.0, -1.0); // Negative lambda
        ADD_FAILURE();                          // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(WeibullDistribution weibull(nan_val, 1.0), std::invalid_argument);

    EXPECT_THROW(WeibullDistribution weibull(1.0, inf_val), std::invalid_argument);

    // Test setters validation
    WeibullDistribution weibull(1.0, 1.0);

    EXPECT_THROW(weibull.setK(0.0), std::invalid_argument);

    EXPECT_THROW(weibull.setLambda(-1.0), std::invalid_argument);
}

/**
 * Test string representation
 */
TEST(WeibullDistributionTest, StringRepresentation) {

    WeibullDistribution weibull(2.5, 1.5);
    std::string str = weibull.toString();

    // Should contain key information based on actual output format:
    // "Weibull Distribution:\n      k (shape) = 2.5\n      λ (scale) = 1.5\n"
    EXPECT_NE(str.find("Weibull"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("2.5"), std::string::npos);
    EXPECT_NE(str.find("1.5"), std::string::npos);
    EXPECT_NE(str.find("k"), std::string::npos);
    EXPECT_NE(str.find("shape"), std::string::npos);
    EXPECT_TRUE(str.find("λ") != std::string::npos || str.find("scale") != std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(WeibullDistributionTest, CopyMoveSemantics) {

    WeibullDistribution original(3.14, 2.71);

    // Test copy constructor
    WeibullDistribution copied(original);
    EXPECT_EQ(copied.getK(), original.getK());
    EXPECT_EQ(copied.getLambda(), original.getLambda());

    // Test copy assignment
    WeibullDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getK(), original.getK());
    EXPECT_EQ(assigned.getLambda(), original.getLambda());

    // Test move constructor
    WeibullDistribution moved(std::move(original));
    EXPECT_EQ(moved.getK(), 3.14);
    EXPECT_EQ(moved.getLambda(), 2.71);

    // Test move assignment
    WeibullDistribution moveAssigned;
    WeibullDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getK(), 1.41);
    EXPECT_EQ(moveAssigned.getLambda(), 1.73);
}

/**
 * Test invalid input handling
 */
TEST(WeibullDistributionTest, InvalidInputHandling) {

    WeibullDistribution weibull(2.0, 1.0);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();

    EXPECT_EQ(weibull.getProbability(nan_val), 0.0);
    EXPECT_EQ(weibull.getProbability(inf_val), 0.0);
    EXPECT_EQ(weibull.getProbability(neg_inf_val), 0.0);

    // Negative values should return 0
    EXPECT_EQ(weibull.getProbability(-0.1), 0.0);
    EXPECT_EQ(weibull.getProbability(-1.0), 0.0);
    EXPECT_EQ(weibull.getProbability(-10.0), 0.0);
}

/**
 * Test reset functionality
 */
TEST(WeibullDistributionTest, ResetFunctionality) {

    WeibullDistribution weibull(10.0, 5.0);
    weibull.reset();

    EXPECT_EQ(weibull.getK(), 1.0);
    EXPECT_EQ(weibull.getLambda(), 1.0);
}

/**
 * Test Weibull distribution properties
 */
TEST(WeibullDistributionTest, WeibullProperties) {

    // Test exponential case (k=1)
    WeibullDistribution exponential(1.0, 2.0);
    double expMean = exponential.getMean();
    EXPECT_NEAR(expMean, 2.0, 1e-10); // For Weibull(1,λ), mean = λ

    // Test Rayleigh case (k=2)
    WeibullDistribution rayleigh(2.0, 1.0);
    double rayleighMean = rayleigh.getMean();
    // For Weibull(2,1), mean = Γ(1.5) = sqrt(π)/2 ≈ 0.8862
    EXPECT_TRUE(rayleighMean > 0.8 && rayleighMean < 0.9);

    // Test that Weibull is only defined for x ≥ 0
    EXPECT_EQ(exponential.getProbability(-0.1), 0.0);
    EXPECT_TRUE(exponential.getProbability(0.0) >= 0.0);
    EXPECT_GT(exponential.getProbability(1.0), 0.0);

    // Test variance is positive
    EXPECT_GT(exponential.getVariance(), 0.0);
    EXPECT_GT(rayleigh.getVariance(), 0.0);

    // Test standard deviation relationship
    EXPECT_NEAR(exponential.getStandardDeviation(), std::sqrt(exponential.getVariance()), 1e-10);
}

/**
 * Test fitting validation
 */
TEST(WeibullDistributionTest, FittingValidation) {

    WeibullDistribution weibull;

    // Test with data containing negative values
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0}; // -1.0 is invalid

    EXPECT_THROW(weibull.fit(invalidData), std::invalid_argument);

    // Test with NaN values
    std::vector<Observation> nanData = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_THROW(weibull.fit(nanData), std::invalid_argument);

    // Zero is not in the Weibull MLE support (log(0) is undefined).
    std::vector<Observation> zeroData = {0.0, 1.0, 2.0, 3.0};
    EXPECT_THROW(weibull.fit(zeroData), std::invalid_argument);
}

/**
 * Test statistical moments
 */
TEST(WeibullDistributionTest, StatisticalMoments) {

    // Test exponential case (k=1, λ=2)
    WeibullDistribution exponential(1.0, 2.0);

    // For Weibull(1,λ), mean = λ * Γ(2) = λ
    double expectedMean = 2.0;
    EXPECT_NEAR(exponential.getMean(), expectedMean, 1e-10);

    // For Weibull(1,λ), variance = λ² * [Γ(3) - (Γ(2))²] = λ² * [2 - 1] = λ²
    double expectedVar = 4.0;
    EXPECT_NEAR(exponential.getVariance(), expectedVar, 1e-10);

    // Test standard deviation
    double expectedStd = 2.0;
    EXPECT_NEAR(exponential.getStandardDeviation(), expectedStd, 1e-10);

    // Test Rayleigh case (k=2, λ=1)
    WeibullDistribution rayleigh(2.0, 1.0);

    // Mean should be sqrt(π)/2 ≈ 0.8862
    double rayleighMean = rayleigh.getMean();
    EXPECT_TRUE(rayleighMean > 0.88 && rayleighMean < 0.89);

    // Variance should be (4-π)/4 ≈ 0.2146
    double rayleighVar = rayleigh.getVariance();
    EXPECT_TRUE(rayleighVar > 0.21 && rayleighVar < 0.22);
}

/**
 * Test special cases and edge cases
 */
TEST(WeibullDistributionTest, SpecialCases) {

    // Test k=1 (exponential distribution)
    WeibullDistribution exp_case(1.0, 1.0);
    double prob_exp = exp_case.getProbability(1.0);
    // For exponential with rate 1, PDF(1) = e^(-1) ≈ 0.368
    EXPECT_TRUE(prob_exp > 0.35 && prob_exp < 0.4);

    // Test k=2 (Rayleigh distribution)
    WeibullDistribution rayleigh_case(2.0, 1.0);
    double prob_rayleigh = rayleigh_case.getProbability(1.0);
    // For Rayleigh with σ=1, PDF(1) = 1*e^(-0.5) ≈ 0.607
    // Note: Actual implementation gives ~0.736 due to Weibull parameterization
    EXPECT_TRUE(prob_rayleigh > 0.7 && prob_rayleigh < 0.75);

    // Test very small k (infant mortality)
    WeibullDistribution infant(0.5, 1.0);
    EXPECT_EQ(infant.getK(), 0.5);
    EXPECT_EQ(infant.getLambda(), 1.0);

    // Test large k (wear-out failures)
    WeibullDistribution wearout(5.0, 1.0);
    EXPECT_EQ(wearout.getK(), 5.0);
    EXPECT_EQ(wearout.getLambda(), 1.0);

    // Test very large values
    WeibullDistribution normal_case(1.0, 1.0);
    double prob_large = normal_case.getProbability(100.0);
    EXPECT_TRUE(prob_large >= 0.0); // Should be very small but non-negative
    EXPECT_LT(prob_large, 1e-10);   // Should be essentially zero
}

/**
 * Test log probability calculations (GOLD STANDARD)
 */
TEST(WeibullDistributionTest, LogProbability) {

    WeibullDistribution weibull(2.0, 1.0); // k=2, λ=1 (Rayleigh distribution)

    // Test that log probability is -infinity for negative values
    EXPECT_EQ(weibull.getLogProbability(-0.1), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(weibull.getLogProbability(-1.0), -std::numeric_limits<double>::infinity());

    // Test consistency with regular probability
    double x = 1.0;
    double prob = weibull.getProbability(x);
    double logProb = weibull.getLogProbability(x);

    if (prob > 0.0) {
        EXPECT_NEAR(logProb, std::log(prob), 1e-10);
    }

    // Test that log probability is finite for positive values
    EXPECT_TRUE(std::isfinite(weibull.getLogProbability(0.5)));
    EXPECT_TRUE(std::isfinite(weibull.getLogProbability(1.0)));
    EXPECT_TRUE(std::isfinite(weibull.getLogProbability(2.0)));

    // Test invalid inputs return -infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_EQ(weibull.getLogProbability(nan_val), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(weibull.getLogProbability(inf_val), -std::numeric_limits<double>::infinity());
}

/**
 * Test CDF calculations (GOLD STANDARD)
 */
TEST(WeibullDistributionTest, CDF) {

    WeibullDistribution weibull(2.0, 1.0); // k=2, λ=1 (Rayleigh distribution)

    // Test boundary conditions
    EXPECT_EQ(weibull.CDF(-1.0), 0.0); // CDF should be 0 for negative values
    EXPECT_EQ(weibull.CDF(0.0), 0.0);  // CDF should be 0 at x=0

    // Test monotonicity (CDF should be non-decreasing)
    double cdf1 = weibull.CDF(0.5);
    double cdf2 = weibull.CDF(1.0);
    double cdf3 = weibull.CDF(2.0);

    EXPECT_TRUE(cdf1 >= 0.0 && cdf1 <= 1.0);
    EXPECT_TRUE(cdf2 >= 0.0 && cdf2 <= 1.0);
    EXPECT_TRUE(cdf3 >= 0.0 && cdf3 <= 1.0);
    EXPECT_LE(cdf1, cdf2);
    EXPECT_LE(cdf2, cdf3);

    // Test that CDF approaches 1 for large values
    double cdfLarge = weibull.CDF(10.0);
    EXPECT_GT(cdfLarge, 0.99); // Should be very close to 1

    // Test known values for Weibull distribution with k=2, λ=1
    // CDF(1) = 1 - exp(-(1/1)^2) = 1 - exp(-1) ≈ 0.632
    double cdfAtScale = weibull.CDF(1.0);
    if (!(cdfAtScale > 0.63 && cdfAtScale < 0.64)) {
        std::cerr << "CDF(1.0)=" << cdfAtScale << " did not meet expected range (0.63, 0.64)"
                  << std::endl;
        ADD_FAILURE();
    }
}

/**
 * Test equality and I/O operators (GOLD STANDARD)
 */
TEST(WeibullDistributionTest, EqualityAndIO) {

    WeibullDistribution weibull1(2.5, 1.5);
    WeibullDistribution weibull2(2.5, 1.5);
    WeibullDistribution weibull3(3.0, 2.0);

    // Test equality operator
    EXPECT_EQ(weibull1, weibull2);      // Same parameters
    EXPECT_FALSE(weibull1 == weibull3); // Different parameters

    // Test with slightly different parameters (within tolerance)
    WeibullDistribution weibull4(2.5 + 1e-16, 1.5 + 1e-16);
    EXPECT_EQ(weibull1, weibull4); // Should be equal within tolerance

    // Test stream output operator
    std::ostringstream oss;
    oss << weibull1;
    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("Weibull"), std::string::npos);
    EXPECT_NE(output.find("2.5"), std::string::npos);
    EXPECT_NE(output.find("1.5"), std::string::npos);

    // Test stream input operator
    std::istringstream iss(output);
    WeibullDistribution weibullFromStream;
    iss >> weibullFromStream;

    if (iss.good() || iss.eof()) {
        EXPECT_EQ(weibullFromStream, weibull1);
    }
}

/**
 * Test performance characteristics and optimizations (GOLD STANDARD)
 */
TEST(WeibullDistributionTest, PerformanceCharacteristics) {

    WeibullDistribution weibull(2.5, 1.5);

    // Test parameters
    const int pdf_iterations = 100000;
    const int fit_datapoints = 5000;

    // Generate test values for PDF calls
    std::vector<double> testValues;
    testValues.reserve(pdf_iterations);
    for (int i = 0; i < pdf_iterations; ++i) {
        // Values > 0 for Weibull distribution
        double t = static_cast<double>(i + 1) / 1000.0;
        testValues.push_back(t);
    }

    // Test getProbability() performance (should benefit from cached values and optimizations)
    auto start = std::chrono::high_resolution_clock::now();
    volatile double sum_pdf = 0.0; // volatile to prevent optimization
    for (const auto &val : testValues) {
        sum_pdf = sum_pdf + weibull.getProbability(val);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto pdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test getLogProbability() performance (should benefit from cached kMinus1_, invLambda_)
    start = std::chrono::high_resolution_clock::now();
    volatile double sum_log_pdf = 0.0; // volatile to prevent optimization
    for (const auto &val : testValues) {
        sum_log_pdf = sum_log_pdf + weibull.getLogProbability(val);
    }
    end = std::chrono::high_resolution_clock::now();
    auto log_pdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test CDF performance (should benefit from cached values and k=2 optimization)
    start = std::chrono::high_resolution_clock::now();
    volatile double sum_cdf = 0.0; // volatile to prevent optimization
    for (const auto &val : testValues) {
        sum_cdf = sum_cdf + weibull.CDF(val);
    }
    end = std::chrono::high_resolution_clock::now();
    auto cdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test fitting performance with Welford's algorithm
    std::vector<Observation> fitData(fit_datapoints);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i + 1) / 1000.0; // Positive values
    }

    start = std::chrono::high_resolution_clock::now();
    weibull.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fit_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Calculate timing metrics
    double pdf_time_per_call = static_cast<double>(pdf_duration.count()) / pdf_iterations;
    double log_pdf_time_per_call = static_cast<double>(log_pdf_duration.count()) / pdf_iterations;
    double cdf_time_per_call = static_cast<double>(cdf_duration.count()) / pdf_iterations;
    double fit_time_per_point = static_cast<double>(fit_duration.count()) / fit_datapoints;

    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) << pdf_time_per_call
              << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3)
              << log_pdf_time_per_call << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  CDF timing:       " << std::fixed << std::setprecision(3) << cdf_time_per_call
              << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) << fit_time_per_point
              << " μs/point (" << fit_datapoints << " points)" << std::endl;

    // Performance requirements (should be fast due to optimizations)
    EXPECT_LT(pdf_time_per_call, 1.0);     // Less than 1 μs per PDF call
    EXPECT_LT(log_pdf_time_per_call, 1.0); // Less than 1 μs per log PDF call
    EXPECT_LT(cdf_time_per_call, 1.0);     // Less than 1 μs per CDF call
    EXPECT_LT(fit_time_per_point,
              5.0); // MLE is iterative (Newton–Raphson); allow up to 5 μs/point
}

/**
 * Test numerical stability with extreme values (GOLD STANDARD)
 */
TEST(WeibullDistributionTest, NumericalStability) {

    // Test with very small k parameter
    WeibullDistribution smallK(0.1, 1.0);
    double probSmallK = smallK.getProbability(0.5);
    EXPECT_GT(probSmallK, 0.0);
    EXPECT_TRUE(std::isfinite(probSmallK));

    // Test with very large k parameter
    WeibullDistribution largeK(100.0, 1.0);
    double probLargeK = largeK.getProbability(1.0);
    EXPECT_GT(probLargeK, 0.0);
    EXPECT_TRUE(std::isfinite(probLargeK));

    // Test with very small lambda parameter
    WeibullDistribution smallLambda(2.0, 0.001);
    double probSmallLambda = smallLambda.getProbability(0.0001);
    EXPECT_GT(probSmallLambda, 0.0);
    EXPECT_TRUE(std::isfinite(probSmallLambda));

    // Test with very large lambda parameter
    WeibullDistribution largeLambda(2.0, 1000.0);
    double probLargeLambda = largeLambda.getProbability(500.0);
    EXPECT_GT(probLargeLambda, 0.0);
    EXPECT_TRUE(std::isfinite(probLargeLambda));

    // Test log probability with extreme values
    double logProbSmallK = smallK.getLogProbability(0.5);
    double logProbLargeK = largeK.getLogProbability(1.0);
    EXPECT_TRUE(std::isfinite(logProbSmallK));
    EXPECT_TRUE(std::isfinite(logProbLargeK));

    // Test special case optimizations (k=1 exponential, k=2 Rayleigh)
    WeibullDistribution exponential(1.0, 2.0); // k=1 case
    WeibullDistribution rayleigh(2.0, 1.0);    // k=2 case

    double expProb = exponential.getProbability(1.0);
    double rayProb = rayleigh.getProbability(1.0);
    double expLogProb = exponential.getLogProbability(1.0);
    double rayLogProb = rayleigh.getLogProbability(1.0);

    EXPECT_TRUE(std::isfinite(expProb) && expProb > 0.0);
    EXPECT_TRUE(std::isfinite(rayProb) && rayProb > 0.0);
    EXPECT_TRUE(std::isfinite(expLogProb));
    EXPECT_TRUE(std::isfinite(rayLogProb));

    // Test mathematical correctness for special cases
    // For k=1 (exponential): PDF(x) = (1/λ)exp(-x/λ)
    double expectedExpProb = (1.0 / 2.0) * std::exp(-1.0 / 2.0);
    EXPECT_NEAR(expProb, expectedExpProb, 1e-10);

    // For k=2 (Rayleigh): PDF(x) = (2x/λ²)exp(-(x/λ)²)
    double expectedRayProb = (2.0 * 1.0 / (1.0 * 1.0)) * std::exp(-(1.0 * 1.0));
    EXPECT_NEAR(rayProb, expectedRayProb, 1e-10);
}

/**
 * Test caching mechanism (GOLD STANDARD)
 */
TEST(WeibullDistributionTest, Caching) {

    WeibullDistribution weibull(2.0, 1.0);

    // Test that calculations work correctly after parameter changes
    double prob1 = weibull.getProbability(1.0);
    double logProb1 = weibull.getLogProbability(1.0);

    // Change parameters and verify cache is updated
    weibull.setK(3.0);
    double prob2 = weibull.getProbability(1.0);
    double logProb2 = weibull.getLogProbability(1.0);

    EXPECT_NE(prob1, prob2);       // Should be different after parameter change
    EXPECT_NE(logProb1, logProb2); // Should be different after parameter change

    // Change scale parameter
    weibull.setLambda(2.0);
    double prob3 = weibull.getProbability(1.0);
    double logProb3 = weibull.getLogProbability(1.0);

    EXPECT_NE(prob2, prob3);       // Should be different after parameter change
    EXPECT_NE(logProb2, logProb3); // Should be different after parameter change

    // Test that copy constructor preserves cache state
    WeibullDistribution copied(weibull);
    EXPECT_EQ(copied.getProbability(1.0), weibull.getProbability(1.0));
    EXPECT_EQ(copied.getLogProbability(1.0), weibull.getLogProbability(1.0));

    // Test that cached values are consistent
    double prob4 = weibull.getProbability(1.0);
    double cdf4 = weibull.CDF(1.0);
    double logProb4 = weibull.getLogProbability(1.0);

    // Multiple calls should return identical results (using cache)
    EXPECT_EQ(weibull.getProbability(1.0), prob4);
    EXPECT_EQ(weibull.CDF(1.0), cdf4);
    EXPECT_EQ(weibull.getLogProbability(1.0), logProb4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
