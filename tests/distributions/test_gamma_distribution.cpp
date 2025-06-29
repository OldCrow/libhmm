#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <climits>
#include <chrono>
#include <iomanip>
#include "libhmm/distributions/gamma_distribution.h"

using libhmm::GammaDistribution;
using libhmm::Observation;

/**
 * Test basic Gamma distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Gamma distribution functionality..." << std::endl;
    
    // Test default constructor
    GammaDistribution gamma;
    assert(gamma.getK() == 1.0);
    assert(gamma.getTheta() == 1.0);
    
    // Test parameterized constructor
    GammaDistribution gamma2(2.5, 1.5);
    assert(gamma2.getK() == 2.5);
    assert(gamma2.getTheta() == 1.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    GammaDistribution gamma(2.0, 1.0);  // k=2, theta=1
    
    // Gamma distribution should be zero at x=0
    assert(gamma.getProbability(0.0) == 0.0);
    
    // Should be positive for positive values
    double prob1 = gamma.getProbability(1.0);
    double prob2 = gamma.getProbability(2.0);
    double prob3 = gamma.getProbability(3.0);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    // Should be zero for negative values
    assert(gamma.getProbability(-1.0) == 0.0);
    assert(gamma.getProbability(-0.5) == 0.0);
    
    // For Gamma(2,1), the mode is at k-1 = 1, so prob at 1 should be relatively high
    assert(prob1 > prob3);  // Probability should decrease away from mode
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    GammaDistribution gamma;
    
    // Test with known data
    std::vector<Observation> data = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    
    gamma.fit(data);
    // After fitting, parameters should be positive
    assert(gamma.getK() > 0.0);
    assert(gamma.getTheta() > 0.0);
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    gamma.fit(emptyData);
    assert(gamma.getK() == 1.0);  // New implementation resets to default
    assert(gamma.getTheta() == 1.0);
    
    // Test with single positive point (implementation resets to default for insufficient data)
    std::vector<Observation> singlePoint = {2.5};
    gamma.fit(singlePoint);
    assert(gamma.getK() == 1.0);  // Implementation resets to default
    assert(gamma.getTheta() == 1.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        GammaDistribution gamma(0.0, 1.0);  // Zero k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GammaDistribution gamma(-1.0, 1.0);  // Negative k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GammaDistribution gamma(1.0, 0.0);  // Zero theta
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GammaDistribution gamma(1.0, -1.0);  // Negative theta
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        GammaDistribution gamma(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GammaDistribution gamma(1.0, inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "✓ Parameter validation tests passed" << std::endl;
}

/**
 * Test string representation
 */
void testStringRepresentation() {
    std::cout << "Testing string representation..." << std::endl;
    
    GammaDistribution gamma(2.5, 1.5);
    std::string str = gamma.toString();
    
    // Should contain key information based on new format:
    // "Gamma Distribution:\n      k (shape parameter) = 2.5\n      θ (scale parameter) = 1.5\n      Mean = 3.75\n      Variance = 5.625\n"
    assert(str.find("Gamma") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("1.5") != std::string::npos);
    assert(str.find("shape parameter") != std::string::npos);
    assert(str.find("scale parameter") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    GammaDistribution original(3.14, 2.71);
    
    // Test copy constructor
    GammaDistribution copied(original);
    assert(copied.getK() == original.getK());
    assert(copied.getTheta() == original.getTheta());
    
    // Test copy assignment
    GammaDistribution assigned;
    assigned = original;
    assert(assigned.getK() == original.getK());
    assert(assigned.getTheta() == original.getTheta());
    
    // Test move constructor
    GammaDistribution moved(std::move(original));
    assert(moved.getK() == 3.14);
    assert(moved.getTheta() == 2.71);
    
    // Test move assignment
    GammaDistribution moveAssigned;
    GammaDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getK() == 1.41);
    assert(moveAssigned.getTheta() == 1.73);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    GammaDistribution gamma(2.0, 1.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(gamma.getProbability(nan_val) == 0.0);
    assert(gamma.getProbability(inf_val) == 0.0);
    assert(gamma.getProbability(neg_inf_val) == 0.0);
    
    // Negative values should return 0
    assert(gamma.getProbability(-1.0) == 0.0);
    assert(gamma.getProbability(-0.1) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    GammaDistribution gamma(10.0, 5.0);
    gamma.reset();
    
    assert(gamma.getK() == 1.0);
    assert(gamma.getTheta() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test log probability function
 */
void testLogProbability() {
    std::cout << "Testing log probability function..." << std::endl;
    
    GammaDistribution gamma(2.0, 1.0);  // k=2, theta=1
    
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
    
    assert(std::abs(p1 - std::exp(logP1)) < 1e-10);
    assert(std::abs(p2 - std::exp(logP2)) < 1e-10);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    assert(std::isinf(gamma.getLogProbability(nan_val)));
    assert(std::isinf(gamma.getLogProbability(inf_val)));
    assert(std::isinf(gamma.getLogProbability(-1.0)));
    
    std::cout << "✓ Log probability tests passed" << std::endl;
}

/**
 * Test additional getters and setters
 */
void testAdditionalGettersSetters() {
    std::cout << "Testing additional getters and setters..." << std::endl;
    
    GammaDistribution gamma(2.0, 3.0);
    
    // Test statistical moments
    double mean = gamma.getMean();
    double variance = gamma.getVariance();
    double stdDev = gamma.getStandardDeviation();
    double mode = gamma.getMode();
    double rate = gamma.getRate();
    
    // For Gamma(k=2, theta=3): mean=6, variance=18, stddev=sqrt(18), mode=3, rate=1/3
    assert(std::abs(mean - 6.0) < 1e-10);
    assert(std::abs(variance - 18.0) < 1e-10);
    assert(std::abs(stdDev - std::sqrt(18.0)) < 1e-10);
    assert(std::abs(mode - 3.0) < 1e-10);  // (k-1)*theta = (2-1)*3 = 3
    assert(std::abs(rate - (1.0/3.0)) < 1e-10);
    
    // Test setters
    gamma.setK(3.0);
    assert(std::abs(gamma.getK() - 3.0) < 1e-10);
    
    gamma.setTheta(2.0);
    assert(std::abs(gamma.getTheta() - 2.0) < 1e-10);
    
    // Test setParameters function
    gamma.setParameters(1.5, 0.5);
    assert(std::abs(gamma.getK() - 1.5) < 1e-10);
    assert(std::abs(gamma.getTheta() - 0.5) < 1e-10);
    
    // Test setParameters validation
    try {
        gamma.setParameters(-1.0, 1.0);  // Invalid k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - setParameters should validate inputs
        assert(std::string(e.what()).find("Shape parameter") != std::string::npos);
    }
    
    std::cout << "✓ Additional getters/setters tests passed" << std::endl;
}

/**
 * Test mathematical correctness with known values
 */
void testMathematicalCorrectness() {
    std::cout << "Testing mathematical correctness..." << std::endl;
    
    // Test with Gamma(1,1) which should be equivalent to Exponential(1)
    GammaDistribution gamma1(1.0, 1.0);
    
    // At x=1: PDF = 1 * exp(-1) ≈ 0.36787944
    double pdf1 = gamma1.getProbability(1.0);
    double expected1 = std::exp(-1.0);
    assert(std::abs(pdf1 - expected1) < 1e-6);
    
    // Test with Gamma(2,1) 
    GammaDistribution gamma2(2.0, 1.0);
    
    // At x=1: PDF = 1 * exp(-1) ≈ 0.36787944 (mode is at x=1)
    double pdf2 = gamma2.getProbability(1.0);
    double expected2 = std::exp(-1.0);
    assert(std::abs(pdf2 - expected2) < 1e-6);
    
    std::cout << "✓ Mathematical correctness tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    GammaDistribution gamma;
    
    // Test with data containing negative values (should filter negatives and fit to positives)
    std::vector<Observation> mixedData = {1.0, 2.0, -1.0, 3.0};
    gamma.fit(mixedData);
    // Should fit to the positive values {1.0, 2.0, 3.0} and not reset
    assert(gamma.getK() > 0.0);  // Should have fitted to positive values
    assert(gamma.getTheta() > 0.0);
    
    // Test with data containing all negative values (should reset to default)
    std::vector<Observation> allNegativeData = {-1.0, -2.0, -3.0};
    gamma.fit(allNegativeData);
    assert(gamma.getK() == 1.0);  // Should reset to default
    assert(gamma.getTheta() == 1.0);
    
    // Test with zero values mixed with positives (should filter zeros and fit to positives)
    std::vector<Observation> zeroMixedData = {0.0, 1.0, 2.0};
    gamma.fit(zeroMixedData);
    // Should fit to the positive values {1.0, 2.0} and not reset
    assert(gamma.getK() > 0.0);  // Should have fitted to positive values
    assert(gamma.getTheta() > 0.0);
    
    // Test with mostly zero values (should reset to default)
    std::vector<Observation> mostlyZeroData = {0.0, 0.0, 1.0};
    gamma.fit(mostlyZeroData);
    assert(gamma.getK() == 1.0);  // Should reset to default (only 1 positive value)
    assert(gamma.getTheta() == 1.0);
    
    // Test with all zero values
    std::vector<Observation> allZeros = {0.0, 0.0, 0.0};
    gamma.fit(allZeros);
    assert(gamma.getK() == 1.0);
    assert(gamma.getTheta() == 1.0);
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test CDF functionality
 */
void testCDF() {
    std::cout << "Testing CDF functionality..." << std::endl;
    
    GammaDistribution gamma(2.0, 1.0);  // k=2, theta=1
    
    // Test CDF properties
    assert(gamma.getCumulativeProbability(-1.0) == 0.0);  // CDF should be 0 for negative values
    assert(gamma.getCumulativeProbability(0.0) == 0.0);   // CDF should be 0 at x=0
    
    // Test CDF values at specific points
    double cdf1 = gamma.getCumulativeProbability(1.0);
    double cdf2 = gamma.getCumulativeProbability(2.0);
    double cdf3 = gamma.getCumulativeProbability(3.0);
    
    // CDF should be monotonically increasing
    assert(cdf1 < cdf2);
    assert(cdf2 < cdf3);
    
    // CDF should be between 0 and 1
    assert(cdf1 > 0.0 && cdf1 < 1.0);
    assert(cdf2 > 0.0 && cdf2 < 1.0);
    assert(cdf3 > 0.0 && cdf3 < 1.0);
    
    // Test that CDF approaches 1 for large values
    double cdf_large = gamma.getCumulativeProbability(10.0);
    assert(cdf_large > 0.95);  // Should be close to 1
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    assert(gamma.getCumulativeProbability(nan_val) == 0.0 || std::isnan(gamma.getCumulativeProbability(nan_val)));
    
    std::cout << "✓ CDF tests passed" << std::endl;
}

/**
 * Test equality operator
 */
void testEqualityOperator() {
    std::cout << "Testing equality operator..." << std::endl;
    
    GammaDistribution gamma1(2.5, 1.5);
    GammaDistribution gamma2(2.5, 1.5);
    GammaDistribution gamma3(2.5, 1.6);  // Different theta
    GammaDistribution gamma4(2.6, 1.5);  // Different k
    
    // Test equality
    assert(gamma1 == gamma2);
    assert(gamma2 == gamma1);  // Symmetric
    
    // Test inequality
    assert(!(gamma1 == gamma3));
    assert(!(gamma1 == gamma4));
    assert(gamma1 != gamma3);
    assert(gamma1 != gamma4);
    
    // Test self-equality
    assert(gamma1 == gamma1);
    
    // Test with very small differences (within tolerance)
    GammaDistribution gamma5(2.5, 1.5 + 1e-15);  // Very small difference
    assert(gamma1 == gamma5);  // Should be equal within tolerance
    
    // Test with differences larger than tolerance
    GammaDistribution gamma6(2.5, 1.5 + 1e-5);  // Larger difference (10x the tolerance)
    assert(!(gamma1 == gamma6));  // Should not be equal
    
    std::cout << "✓ Equality operator tests passed" << std::endl;
}

/**
 * Test I/O operators
 */
void testIOOperators() {
    std::cout << "Testing I/O operators..." << std::endl;
    
    GammaDistribution original(3.14, 2.71);
    
    // Test output operator
    std::ostringstream oss;
    oss << original;
    std::string output = oss.str();
    
    // Check that output contains expected information
    assert(output.find("Gamma Distribution") != std::string::npos);
    assert(output.find("3.14") != std::string::npos);
    assert(output.find("2.71") != std::string::npos);
    assert(output.find("shape") != std::string::npos);
    assert(output.find("scale") != std::string::npos);
    
    // Test input operator
    std::istringstream iss("k (shape) = 1.41 theta (scale) = 1.73");
    GammaDistribution reconstructed;
    iss >> reconstructed;
    
    // Check that parameters were correctly parsed
    assert(std::abs(reconstructed.getK() - 1.41) < 1e-10);
    assert(std::abs(reconstructed.getTheta() - 1.73) < 1e-10);
    
    // Test input operator with invalid data
    std::istringstream invalid_iss("invalid data format");
    GammaDistribution invalid_test;
    invalid_iss >> invalid_test;
    assert(invalid_iss.fail());  // Stream should be in failed state
    
    std::cout << "✓ I/O operator tests passed" << std::endl;
}

/**
 * Test caching behavior
 */
void testCaching() {
    std::cout << "Testing caching behavior..." << std::endl;
    
    GammaDistribution gamma(2.0, 1.0);
    
    // Get some probability values (this should populate cache)
    double prob1 = gamma.getProbability(1.0);
    double logProb1 = gamma.getLogProbability(1.0);
    
    // Verify consistency
    assert(std::abs(prob1 - std::exp(logProb1)) < 1e-10);
    
    // Change parameters (this should invalidate cache)
    gamma.setK(3.0);
    
    // Get probability again (should use updated parameters)
    double prob2 = gamma.getProbability(1.0);
    double logProb2 = gamma.getLogProbability(1.0);
    
    // Values should be different due to parameter change
    assert(std::abs(prob1 - prob2) > 1e-6);
    assert(std::abs(logProb1 - logProb2) > 1e-6);
    
    // Verify consistency with new parameters
    assert(std::abs(prob2 - std::exp(logProb2)) < 1e-10);
    
    // Test cache invalidation with setTheta
    double prob3 = gamma.getProbability(1.0);
    gamma.setTheta(2.0);
    double prob4 = gamma.getProbability(1.0);
    assert(std::abs(prob3 - prob4) > 1e-6);
    
    // Test cache invalidation with setParameters
    double prob5 = gamma.getProbability(1.0);
    gamma.setParameters(1.5, 0.8);
    double prob6 = gamma.getProbability(1.0);
    assert(std::abs(prob5 - prob6) > 1e-6);
    
    std::cout << "✓ Caching tests passed" << std::endl;
}

/**
 * Test numerical stability
 */
void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;
    
    // Test with very small shape parameter
    GammaDistribution gamma_small(0.1, 1.0);
    double prob_small = gamma_small.getProbability(0.01);
    double logProb_small = gamma_small.getLogProbability(0.01);
    assert(std::isfinite(prob_small));
    assert(std::isfinite(logProb_small));
    
    // Test with very large shape parameter
    GammaDistribution gamma_large(100.0, 1.0);
    double prob_large = gamma_large.getProbability(100.0);
    double logProb_large = gamma_large.getLogProbability(100.0);
    assert(std::isfinite(prob_large));
    assert(std::isfinite(logProb_large));
    
    // Test with very small scale parameter
    GammaDistribution gamma_small_scale(2.0, 0.001);
    double prob_small_scale = gamma_small_scale.getProbability(0.001);
    assert(std::isfinite(prob_small_scale));
    
    // Test with very large scale parameter
    GammaDistribution gamma_large_scale(2.0, 1000.0);
    double prob_large_scale = gamma_large_scale.getProbability(1000.0);
    assert(std::isfinite(prob_large_scale));
    
    // Test log probability doesn't overflow/underflow
    GammaDistribution gamma_extreme(0.01, 0.01);
    double logProb_extreme = gamma_extreme.getLogProbability(1e-10);
    assert(std::isfinite(logProb_extreme) || logProb_extreme == -std::numeric_limits<double>::infinity());
    
    // Test CDF stability
    double cdf_small = gamma_small.getCumulativeProbability(1e-6);
    double cdf_large = gamma_large.getCumulativeProbability(1000.0);
    assert(cdf_small >= 0.0 && cdf_small <= 1.0);
    assert(cdf_large >= 0.0 && cdf_large <= 1.0);
    
    // Test with values very close to zero
    GammaDistribution gamma_test(1.5, 1.0);
    double prob_near_zero = gamma_test.getProbability(1e-100);
    assert(prob_near_zero >= 0.0);
    
    std::cout << "✓ Numerical stability tests passed" << std::endl;
}

/**
 * Test performance characteristics
 */
void testPerformance() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    using namespace std::chrono;
    GammaDistribution gamma(2.0, 1.0);
    
    // Test parameters
    const int pdf_iterations = 50000;
    const int fit_datapoints = 1000;
    
    // Generate test values for PDF calls
    std::vector<double> testValues;
    testValues.reserve(pdf_iterations);
    for (int i = 0; i < pdf_iterations; ++i) {
        testValues.push_back(0.1 + (5.0 * i) / pdf_iterations);  // Positive values only
    }
    
    // Test getProbability() performance
    auto start = high_resolution_clock::now();
    double sum_pdf = 0.0;
    for (const auto& val : testValues) {
        sum_pdf += gamma.getProbability(val);
    }
    auto end = high_resolution_clock::now();
    auto pdf_duration = duration_cast<microseconds>(end - start);
    
    // Test getLogProbability() performance
    start = high_resolution_clock::now();
    double sum_log_pdf = 0.0;
    for (const auto& val : testValues) {
        sum_log_pdf += gamma.getLogProbability(val);
    }
    end = high_resolution_clock::now();
    auto log_pdf_duration = duration_cast<microseconds>(end - start);
    
    // Test fitting performance
    std::vector<double> fit_data;
    fit_data.reserve(fit_datapoints);
    for (int i = 0; i < fit_datapoints; ++i) {
        fit_data.push_back(0.1 + i * 0.01);  // Positive values
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
    assert(pdf_per_call < 2.0);      // Should be well under 2 microseconds per PDF call
    assert(log_pdf_per_call < 1.0);  // Should be well under 1 microsecond per log PDF call
    assert(fit_per_point < 10.0);    // Should be well under 10 microseconds per fit datapoint
    
    // Verify correctness
    assert(sum_pdf > 0.0);
    assert(sum_log_pdf < 0.0);  // Log probabilities should be negative
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  PDF timing:       " << pdf_per_call << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << log_pdf_per_call << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << fit_per_point << " μs/point (" << fit_datapoints << " points)" << std::endl;
    std::cout << "✓ Performance tests passed" << std::endl;
}

int main() {
    std::cout << "Running Gamma distribution tests..." << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testLogProbability();
        testAdditionalGettersSetters();
        testMathematicalCorrectness();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        testFittingValidation();
        testCDF();
        testEqualityOperator();
        testIOOperators();
        testCaching();
        testNumericalStability();
        testPerformance();
        
        std::cout << "===================================" << std::endl;
        std::cout << "✅ All Gamma distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
