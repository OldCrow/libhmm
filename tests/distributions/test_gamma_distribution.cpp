#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <climits>
#include <chrono>
#include <iomanip>
#include "libhmm/distributions/gamma_distribution.h"
#include <gtest/gtest.h>

using libhmm::GammaDistribution;
using libhmm::Observation;

/**
 * Test basic Gamma distribution functionality
 */
TEST(GammaDistributionTest, BasicFunctionality) {

    // Test default constructor
    GammaDistribution gamma;
    EXPECT_EQ(gamma.getK(), 1.0);
    EXPECT_EQ(gamma.getTheta(), 1.0);

    // Test parameterized constructor
    GammaDistribution gamma2(2.5, 1.5);
    EXPECT_EQ(gamma2.getK(), 2.5);
    EXPECT_EQ(gamma2.getTheta(), 1.5);
}

/**
 * Test probability calculations
 */
TEST(GammaDistributionTest, Probabilities) {

    GammaDistribution gamma(2.0, 1.0); // k=2, theta=1

    // Gamma distribution should be zero at x=0
    EXPECT_EQ(gamma.getProbability(0.0), 0.0);

    // Should be positive for positive values
    double prob1 = gamma.getProbability(1.0);
    double prob2 = gamma.getProbability(2.0);
    double prob3 = gamma.getProbability(3.0);

    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(prob2, 0.0);
    EXPECT_GT(prob3, 0.0);

    // Should be zero for negative values
    EXPECT_EQ(gamma.getProbability(-1.0), 0.0);
    EXPECT_EQ(gamma.getProbability(-0.5), 0.0);

    // For Gamma(2,1), the mode is at k-1 = 1, so prob at 1 should be relatively high
    EXPECT_GT(prob1, prob3); // Probability should decrease away from mode
}

/**
 * Test parameter fitting
 */
TEST(GammaDistributionTest, Fitting) {

    GammaDistribution gamma;

    // Test with known data
    std::vector<Observation> data = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    gamma.fit(data);
    // After fitting, parameters should be positive
    EXPECT_GT(gamma.getK(), 0.0);
    EXPECT_GT(gamma.getTheta(), 0.0);

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    gamma.fit(emptyData);
    EXPECT_EQ(gamma.getK(), 1.0); // New implementation resets to default
    EXPECT_EQ(gamma.getTheta(), 1.0);

    // Test with single positive point (implementation resets to default for insufficient data)
    std::vector<Observation> singlePoint = {2.5};
    gamma.fit(singlePoint);
    EXPECT_EQ(gamma.getK(), 1.0); // Implementation resets to default
    EXPECT_EQ(gamma.getTheta(), 1.0);
}

/**
 * Test parameter validation
 */
TEST(GammaDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    try {
        GammaDistribution gamma(0.0, 1.0); // Zero k
        ADD_FAILURE();                     // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        GammaDistribution gamma(-1.0, 1.0); // Negative k
        ADD_FAILURE();                      // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        GammaDistribution gamma(1.0, 0.0); // Zero theta
        ADD_FAILURE();                     // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        GammaDistribution gamma(1.0, -1.0); // Negative theta
        ADD_FAILURE();                      // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(GammaDistribution gamma(nan_val, 1.0), std::invalid_argument);

    EXPECT_THROW(GammaDistribution gamma(1.0, inf_val), std::invalid_argument);
}

/**
 * Test string representation
 */
TEST(GammaDistributionTest, StringRepresentation) {

    GammaDistribution gamma(2.5, 1.5);
    std::string str = gamma.toString();

    // Should contain key information based on new format:
    // "Gamma Distribution:\n      k (shape parameter) = 2.5\n      θ (scale parameter) = 1.5\n      Mean = 3.75\n      Variance = 5.625\n"
    EXPECT_NE(str.find("Gamma"), std::string::npos);
    EXPECT_NE(str.find("2.5"), std::string::npos);
    EXPECT_NE(str.find("1.5"), std::string::npos);
    EXPECT_NE(str.find("shape parameter"), std::string::npos);
    EXPECT_NE(str.find("scale parameter"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(GammaDistributionTest, CopyMoveSemantics) {

    GammaDistribution original(3.14, 2.71);

    // Test copy constructor
    GammaDistribution copied(original);
    EXPECT_EQ(copied.getK(), original.getK());
    EXPECT_EQ(copied.getTheta(), original.getTheta());

    // Test copy assignment
    GammaDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getK(), original.getK());
    EXPECT_EQ(assigned.getTheta(), original.getTheta());

    // Test move constructor
    GammaDistribution moved(std::move(original));
    EXPECT_EQ(moved.getK(), 3.14);
    EXPECT_EQ(moved.getTheta(), 2.71);

    // Test move assignment
    GammaDistribution moveAssigned;
    GammaDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getK(), 1.41);
    EXPECT_EQ(moveAssigned.getTheta(), 1.73);
}

/**
 * Test invalid input handling
 */
TEST(GammaDistributionTest, InvalidInputHandling) {

    GammaDistribution gamma(2.0, 1.0);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();

    EXPECT_EQ(gamma.getProbability(nan_val), 0.0);
    EXPECT_EQ(gamma.getProbability(inf_val), 0.0);
    EXPECT_EQ(gamma.getProbability(neg_inf_val), 0.0);

    // Negative values should return 0
    EXPECT_EQ(gamma.getProbability(-1.0), 0.0);
    EXPECT_EQ(gamma.getProbability(-0.1), 0.0);
}

/**
 * Test reset functionality
 */
TEST(GammaDistributionTest, ResetFunctionality) {

    GammaDistribution gamma(10.0, 5.0);
    gamma.reset();

    EXPECT_EQ(gamma.getK(), 1.0);
    EXPECT_EQ(gamma.getTheta(), 1.0);
}

/**
 * Test log probability function
 */
TEST(GammaDistributionTest, LogProbability) {

    GammaDistribution gamma(2.0, 1.0); // k=2, theta=1

    // Test log PDF at several points
    double x1 = 0.5;
    double x2 = 1.0;
    double x3 = 2.0;

    double logP1 = gamma.getLogProbability(x1);
    double logP2 = gamma.getLogProbability(x2);
    // Note: logP3 is computed for completeness but not used in current tests
    [[maybe_unused]] double logP3 = gamma.getLogProbability(x3);

    // Verify consistency between PDF and log PDF
    double p1 = gamma.getProbability(x1);
    double p2 = gamma.getProbability(x2);

    EXPECT_NEAR(p1, std::exp(logP1), 1e-10);
    EXPECT_NEAR(p2, std::exp(logP2), 1e-10);

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(gamma.getLogProbability(nan_val)));
    EXPECT_TRUE(std::isinf(gamma.getLogProbability(inf_val)));
    EXPECT_TRUE(std::isinf(gamma.getLogProbability(-1.0)));
}

/**
 * Test additional getters and setters
 */
TEST(GammaDistributionTest, AdditionalGettersSetters) {

    GammaDistribution gamma(2.0, 3.0);

    // Test statistical moments
    double mean = gamma.getMean();
    double variance = gamma.getVariance();
    double stdDev = gamma.getStandardDeviation();
    double mode = gamma.getMode();
    double rate = gamma.getRate();

    // For Gamma(k=2, theta=3): mean=6, variance=18, stddev=sqrt(18), mode=3, rate=1/3
    EXPECT_NEAR(mean, 6.0, 1e-10);
    EXPECT_NEAR(variance, 18.0, 1e-10);
    EXPECT_NEAR(stdDev, std::sqrt(18.0), 1e-10);
    EXPECT_NEAR(mode, 3.0, 1e-10); // (k-1)*theta = (2-1)*3 = 3
    EXPECT_NEAR(rate, (1.0 / 3.0), 1e-10);

    // Test setters
    gamma.setK(3.0);
    EXPECT_NEAR(gamma.getK(), 3.0, 1e-10);

    gamma.setTheta(2.0);
    EXPECT_NEAR(gamma.getTheta(), 2.0, 1e-10);

    // Test setParameters function
    gamma.setParameters(1.5, 0.5);
    EXPECT_NEAR(gamma.getK(), 1.5, 1e-10);
    EXPECT_NEAR(gamma.getTheta(), 0.5, 1e-10);

    // Test setParameters validation
    try {
        gamma.setParameters(-1.0, 1.0); // Invalid k
        ADD_FAILURE();                  // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior - setParameters should validate inputs
    }
}

/**
 * Test mathematical correctness with known values
 */
TEST(GammaDistributionTest, MathematicalCorrectness) {

    // Test with Gamma(1,1) which should be equivalent to Exponential(1)
    GammaDistribution gamma1(1.0, 1.0);

    // At x=1: PDF = 1 * exp(-1) ≈ 0.36787944
    double pdf1 = gamma1.getProbability(1.0);
    double expected1 = std::exp(-1.0);
    EXPECT_NEAR(pdf1, expected1, 1e-6);

    // Test with Gamma(2,1)
    GammaDistribution gamma2(2.0, 1.0);

    // At x=1: PDF = 1 * exp(-1) ≈ 0.36787944 (mode is at x=1)
    double pdf2 = gamma2.getProbability(1.0);
    double expected2 = std::exp(-1.0);
    EXPECT_NEAR(pdf2, expected2, 1e-6);
}

/**
 * Test fitting validation
 */
TEST(GammaDistributionTest, FittingValidation) {

    GammaDistribution gamma;

    // Test with data containing negative values (should filter negatives and fit to positives)
    std::vector<Observation> mixedData = {1.0, 2.0, -1.0, 3.0};
    gamma.fit(mixedData);
    // Should fit to the positive values {1.0, 2.0, 3.0} and not reset
    EXPECT_GT(gamma.getK(), 0.0); // Should have fitted to positive values
    EXPECT_GT(gamma.getTheta(), 0.0);

    // Test with data containing all negative values (should reset to default)
    std::vector<Observation> allNegativeData = {-1.0, -2.0, -3.0};
    gamma.fit(allNegativeData);
    EXPECT_EQ(gamma.getK(), 1.0); // Should reset to default
    EXPECT_EQ(gamma.getTheta(), 1.0);

    // Test with zero values mixed with positives (should filter zeros and fit to positives)
    std::vector<Observation> zeroMixedData = {0.0, 1.0, 2.0};
    gamma.fit(zeroMixedData);
    // Should fit to the positive values {1.0, 2.0} and not reset
    EXPECT_GT(gamma.getK(), 0.0); // Should have fitted to positive values
    EXPECT_GT(gamma.getTheta(), 0.0);

    // Test with mostly zero values (should reset to default)
    std::vector<Observation> mostlyZeroData = {0.0, 0.0, 1.0};
    gamma.fit(mostlyZeroData);
    EXPECT_EQ(gamma.getK(), 1.0); // Should reset to default (only 1 positive value)
    EXPECT_EQ(gamma.getTheta(), 1.0);

    // Test with all zero values
    std::vector<Observation> allZeros = {0.0, 0.0, 0.0};
    gamma.fit(allZeros);
    EXPECT_EQ(gamma.getK(), 1.0);
    EXPECT_EQ(gamma.getTheta(), 1.0);
}

/**
 * Test CDF functionality
 */
TEST(GammaDistributionTest, CDF) {

    GammaDistribution gamma(2.0, 1.0); // k=2, theta=1

    // Test CDF properties
    EXPECT_EQ(gamma.getCumulativeProbability(-1.0), 0.0); // CDF should be 0 for negative values
    EXPECT_EQ(gamma.getCumulativeProbability(0.0), 0.0);  // CDF should be 0 at x=0

    // Test CDF values at specific points
    double cdf1 = gamma.getCumulativeProbability(1.0);
    double cdf2 = gamma.getCumulativeProbability(2.0);
    double cdf3 = gamma.getCumulativeProbability(3.0);

    // CDF should be monotonically increasing
    EXPECT_LT(cdf1, cdf2);
    EXPECT_LT(cdf2, cdf3);

    // CDF should be between 0 and 1
    EXPECT_TRUE(cdf1 > 0.0 && cdf1 < 1.0);
    EXPECT_TRUE(cdf2 > 0.0 && cdf2 < 1.0);
    EXPECT_TRUE(cdf3 > 0.0 && cdf3 < 1.0);

    // Test that CDF approaches 1 for large values
    double cdf_large = gamma.getCumulativeProbability(10.0);
    EXPECT_GT(cdf_large, 0.95); // Should be close to 1

    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(gamma.getCumulativeProbability(nan_val) == 0.0 ||
                std::isnan(gamma.getCumulativeProbability(nan_val)));
}

/**
 * Test equality operator
 */
TEST(GammaDistributionTest, EqualityOperator) {

    GammaDistribution gamma1(2.5, 1.5);
    GammaDistribution gamma2(2.5, 1.5);
    GammaDistribution gamma3(2.5, 1.6); // Different theta
    GammaDistribution gamma4(2.6, 1.5); // Different k

    // Test equality
    EXPECT_EQ(gamma1, gamma2);
    EXPECT_EQ(gamma2, gamma1); // Symmetric

    // Test inequality
    EXPECT_FALSE(gamma1 == gamma3);
    EXPECT_FALSE(gamma1 == gamma4);
    EXPECT_NE(gamma1, gamma3);
    EXPECT_NE(gamma1, gamma4);

    // Test self-equality
    EXPECT_EQ(gamma1, gamma1);

    // Test with very small differences (within tolerance)
    GammaDistribution gamma5(2.5, 1.5 + 1e-15); // Very small difference
    EXPECT_EQ(gamma1, gamma5);                  // Should be equal within tolerance

    // Test with differences larger than tolerance
    GammaDistribution gamma6(2.5, 1.5 + 1e-5); // Larger difference (10x the tolerance)
    EXPECT_FALSE(gamma1 == gamma6);            // Should not be equal
}

/**
 * Test I/O operators
 */
TEST(GammaDistributionTest, IOOperators) {

    GammaDistribution original(3.14, 2.71);

    // Test output operator
    std::ostringstream oss;
    oss << original;
    std::string output = oss.str();

    // Check that output contains expected information
    EXPECT_NE(output.find("Gamma Distribution"), std::string::npos);
    EXPECT_NE(output.find("3.14"), std::string::npos);
    EXPECT_NE(output.find("2.71"), std::string::npos);
    EXPECT_NE(output.find("shape"), std::string::npos);
    EXPECT_NE(output.find("scale"), std::string::npos);

    // Test input operator via roundtrip
    GammaDistribution source(1.41, 1.73);
    std::ostringstream rss;
    rss << source;
    std::istringstream iss(rss.str());
    GammaDistribution reconstructed;
    iss >> reconstructed;

    // Check that parameters were correctly parsed
    EXPECT_NEAR(reconstructed.getK(), 1.41, 1e-10);
    EXPECT_NEAR(reconstructed.getTheta(), 1.73, 1e-10);

    // Test input operator with invalid data
    std::istringstream invalid_iss("invalid data format");
    GammaDistribution invalid_test;
    invalid_iss >> invalid_test;
    EXPECT_TRUE(invalid_iss.fail()); // Stream should be in failed state
}

/**
 * Test caching behavior
 */
TEST(GammaDistributionTest, Caching) {

    GammaDistribution gamma(2.0, 1.0);

    // Get some probability values (this should populate cache)
    double prob1 = gamma.getProbability(1.0);
    double logProb1 = gamma.getLogProbability(1.0);

    // Verify consistency
    EXPECT_NEAR(prob1, std::exp(logProb1), 1e-10);

    // Change parameters (this should invalidate cache)
    gamma.setK(3.0);

    // Get probability again (should use updated parameters)
    double prob2 = gamma.getProbability(1.0);
    double logProb2 = gamma.getLogProbability(1.0);

    // Values should be different due to parameter change
    EXPECT_GT(std::abs(prob1 - prob2), 1e-6);
    EXPECT_GT(std::abs(logProb1 - logProb2), 1e-6);

    // Verify consistency with new parameters
    EXPECT_NEAR(prob2, std::exp(logProb2), 1e-10);

    // Test cache invalidation with setTheta
    double prob3 = gamma.getProbability(1.0);
    gamma.setTheta(2.0);
    double prob4 = gamma.getProbability(1.0);
    EXPECT_GT(std::abs(prob3 - prob4), 1e-6);

    // Test cache invalidation with setParameters
    double prob5 = gamma.getProbability(1.0);
    gamma.setParameters(1.5, 0.8);
    double prob6 = gamma.getProbability(1.0);
    EXPECT_GT(std::abs(prob5 - prob6), 1e-6);
}

/**
 * Test numerical stability
 */
TEST(GammaDistributionTest, NumericalStability) {

    // Test with very small shape parameter
    GammaDistribution gamma_small(0.1, 1.0);
    double prob_small = gamma_small.getProbability(0.01);
    double logProb_small = gamma_small.getLogProbability(0.01);
    EXPECT_TRUE(std::isfinite(prob_small));
    EXPECT_TRUE(std::isfinite(logProb_small));

    // Test with very large shape parameter
    GammaDistribution gamma_large(100.0, 1.0);
    double prob_large = gamma_large.getProbability(100.0);
    double logProb_large = gamma_large.getLogProbability(100.0);
    EXPECT_TRUE(std::isfinite(prob_large));
    EXPECT_TRUE(std::isfinite(logProb_large));

    // Test with very small scale parameter
    GammaDistribution gamma_small_scale(2.0, 0.001);
    double prob_small_scale = gamma_small_scale.getProbability(0.001);
    EXPECT_TRUE(std::isfinite(prob_small_scale));

    // Test with very large scale parameter
    GammaDistribution gamma_large_scale(2.0, 1000.0);
    double prob_large_scale = gamma_large_scale.getProbability(1000.0);
    EXPECT_TRUE(std::isfinite(prob_large_scale));

    // Test log probability doesn't overflow/underflow
    GammaDistribution gamma_extreme(0.01, 0.01);
    double logProb_extreme = gamma_extreme.getLogProbability(1e-10);
    EXPECT_TRUE(std::isfinite(logProb_extreme) ||
                logProb_extreme == -std::numeric_limits<double>::infinity());

    // Test CDF stability
    double cdf_small = gamma_small.getCumulativeProbability(1e-6);
    double cdf_large = gamma_large.getCumulativeProbability(1000.0);
    EXPECT_TRUE(cdf_small >= 0.0 && cdf_small <= 1.0);
    EXPECT_TRUE(cdf_large >= 0.0 && cdf_large <= 1.0);

    // Test with values very close to zero
    GammaDistribution gamma_test(1.5, 1.0);
    double prob_near_zero = gamma_test.getProbability(1e-100);
    EXPECT_TRUE(prob_near_zero >= 0.0);
}

/**
 * Test performance characteristics
 */
TEST(GammaDistributionTest, Performance) {

    using namespace std::chrono;
    GammaDistribution gamma(2.0, 1.0);

    // Test parameters
    const int pdf_iterations = 50000;
    const int fit_datapoints = 1000;

    // Generate test values for PDF calls
    std::vector<double> testValues;
    testValues.reserve(pdf_iterations);
    for (int i = 0; i < pdf_iterations; ++i) {
        testValues.push_back(0.1 + (5.0 * i) / pdf_iterations); // Positive values only
    }

    // Test getProbability() performance
    auto start = high_resolution_clock::now();
    double sum_pdf = 0.0;
    for (const auto &val : testValues) {
        sum_pdf += gamma.getProbability(val);
    }
    auto end = high_resolution_clock::now();
    auto pdf_duration = duration_cast<microseconds>(end - start);

    // Test getLogProbability() performance
    start = high_resolution_clock::now();
    double sum_log_pdf = 0.0;
    for (const auto &val : testValues) {
        sum_log_pdf += gamma.getLogProbability(val);
    }
    end = high_resolution_clock::now();
    auto log_pdf_duration = duration_cast<microseconds>(end - start);

    // Test fitting performance
    std::vector<double> fit_data;
    fit_data.reserve(fit_datapoints);
    for (int i = 0; i < fit_datapoints; ++i) {
        fit_data.push_back(0.1 + i * 0.01); // Positive values
    }

    start = high_resolution_clock::now();
    gamma.fit(fit_data);
    end = high_resolution_clock::now();
    auto fit_duration = duration_cast<microseconds>(end - start);

    // Performance calculations
    double pdf_per_call = static_cast<double>(pdf_duration.count()) / pdf_iterations;
    double log_pdf_per_call = static_cast<double>(log_pdf_duration.count()) / pdf_iterations;
    double fit_per_point = static_cast<double>(fit_duration.count()) / fit_datapoints;

    // Basic performance assertions
    EXPECT_LT(pdf_per_call, 2.0);     // Should be well under 2 microseconds per PDF call
    EXPECT_LT(log_pdf_per_call, 1.0); // Should be well under 1 microsecond per log PDF call
    EXPECT_LT(fit_per_point, 10.0);   // Should be well under 10 microseconds per fit datapoint

    // Verify correctness
    EXPECT_GT(sum_pdf, 0.0);
    EXPECT_LT(sum_log_pdf, 0.0); // Log probabilities should be negative

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  PDF timing:       " << pdf_per_call << " μs/call (" << pdf_iterations
              << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << log_pdf_per_call << " μs/call (" << pdf_iterations
              << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << fit_per_point << " μs/point (" << fit_datapoints
              << " points)" << std::endl;
}

/**
 * Accuracy tests for GammaDistribution::getCumulativeProbability().
 *
 * Exercises DistributionBase::gammap() (and its internal series and
 * continued-fraction helpers) against exact closed-form CDFs.  For integer
 * shape parameter k the
 * regularised incomplete gamma has a Poisson-series exact form:
 *
 *   P(k, x) = 1 − e^(−x) · ∑_{j=0}^{k-1} x^j / j!
 *
 * These reference values are computed from std::exp only — no special
 * functions are needed — so any discrepancy isolates a bug in gammap.
 */
TEST(GammaDistributionTest, CDFAccuracy) {
    // gammap uses BW_TOLERANCE = 3e-7 as convergence criterion; max observed
    // error is ~3e-8, so 1e-6 is a safe bound that confirms correctness without
    // demanding sub-tolerance precision.
    constexpr double kTol = 1e-6;

    // k=1 (Exponential distribution): CDF(x) = 1 − e^(−x)
    {
        GammaDistribution d(1.0, 1.0);
        EXPECT_NEAR(d.getCumulativeProbability(0.5), 1.0 - std::exp(-0.5), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(1.0), 1.0 - std::exp(-1.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(2.0), 1.0 - std::exp(-2.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(5.0), 1.0 - std::exp(-5.0), kTol);
    }

    // k=2: CDF(x) = 1 − e^(−x) · (1 + x)
    {
        GammaDistribution d(2.0, 1.0);
        auto exact = [](double x) {
            return 1.0 - std::exp(-x) * (1.0 + x);
        };
        EXPECT_NEAR(d.getCumulativeProbability(1.0), exact(1.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(2.0), exact(2.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(3.0), exact(3.0), kTol);
    }

    // k=3: CDF(x) = 1 − e^(−x) · (1 + x + x²/2)
    {
        GammaDistribution d(3.0, 1.0);
        auto exact = [](double x) {
            return 1.0 - std::exp(-x) * (1.0 + x + 0.5 * x * x);
        };
        EXPECT_NEAR(d.getCumulativeProbability(1.0), exact(1.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(2.0), exact(2.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(4.0), exact(4.0), kTol);
    }

    // Scale parameter: Gamma(1, θ=2) = Exp(λ=0.5), CDF(x) = 1 − e^(−x/2)
    {
        GammaDistribution d(1.0, 2.0);
        EXPECT_NEAR(d.getCumulativeProbability(2.0), 1.0 - std::exp(-1.0), kTol);
        EXPECT_NEAR(d.getCumulativeProbability(4.0), 1.0 - std::exp(-2.0), kTol);
    }

    // Boundary conditions
    {
        GammaDistribution d(2.0, 1.0);
        EXPECT_EQ(d.getCumulativeProbability(0.0), 0.0);
        EXPECT_EQ(d.getCumulativeProbability(-1.0), 0.0);
        EXPECT_GT(d.getCumulativeProbability(1e6), 1.0 - 1e-9);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
