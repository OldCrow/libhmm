#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/gaussian_distribution.h"

using libhmm::GaussianDistribution;
using libhmm::Observation;
using namespace libhmm::constants;

/**
 * Test basic Gaussian distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Gaussian distribution functionality..." << std::endl;
    
    // Test default constructor
    GaussianDistribution gaussian;
    assert(gaussian.getMean() == 0.0);
    assert(gaussian.getStandardDeviation() == 1.0);
    
    // Test parameterized constructor
    GaussianDistribution gaussian2(5.0, 2.5);
    assert(gaussian2.getMean() == 5.0);
    assert(gaussian2.getStandardDeviation() == 2.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    GaussianDistribution gaussian(0.0, 1.0);  // Standard normal
    
    // Test at mean (should be highest probability)
    double probAtMean = gaussian.getProbability(0.0);
    assert(probAtMean > 0.0);
    
    // Test precise PDF value for standard normal at mean (should be 1/sqrt(2π))
    double expectedPDF = 1.0 / math::SQRT_2PI;
    assert(std::abs(probAtMean - expectedPDF) < 1e-10);
    
    // Test symmetry around mean
    double probAt1 = gaussian.getProbability(1.0);
    double probAtNeg1 = gaussian.getProbability(-1.0);
    assert(std::abs(probAt1 - probAtNeg1) < 1e-10);
    
    // Probability at mean should be higher than at ±1
    assert(probAtMean > probAt1);
    
    // Test that probability density at mean is reasonable (should be small for continuous dist)
    assert(probAtMean < 1.0);  // Should be less than 1 for probability density
    assert(probAtMean > 1e-10);  // Should be greater than zero
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    GaussianDistribution gaussian;
    
    // Test with known data (should fit mean=3.0, approximate std dev)
    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double expectedMean = 3.0;
    
    gaussian.fit(data);
    assert(std::abs(gaussian.getMean() - expectedMean) < 1e-10);
    assert(gaussian.getStandardDeviation() > 0.0);
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    gaussian.fit(emptyData);
    assert(gaussian.getMean() == 0.0);
    assert(gaussian.getStandardDeviation() == 1.0);
    
    // Test with single point (implementation resets to default for insufficient data)
    std::vector<Observation> singlePoint = {7.5};
    gaussian.fit(singlePoint);
    assert(gaussian.getMean() == 0.0);  // Implementation resets to default
    assert(gaussian.getStandardDeviation() == 1.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        GaussianDistribution gaussian(0.0, 0.0);  // Zero std dev
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - validation should catch zero std dev
        assert(std::string(e.what()).find("Standard deviation") != std::string::npos);
    }
    
    try {
        GaussianDistribution gaussian(0.0, -1.0);  // Negative std dev
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - validation should catch negative std dev
        assert(std::string(e.what()).find("Standard deviation") != std::string::npos);
    }
    
    // Test invalid mean
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        GaussianDistribution gaussian(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - validation should catch NaN mean
        assert(std::string(e.what()).find("Mean") != std::string::npos);
    }
    
    try {
        GaussianDistribution gaussian(inf_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - validation should catch infinite mean
        assert(std::string(e.what()).find("Mean") != std::string::npos);
    }
    
    // Test setters validation
    GaussianDistribution gaussian(0.0, 1.0);
    
    try {
        gaussian.setMean(nan_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - setMean should validate input
        assert(std::string(e.what()).find("Mean") != std::string::npos);
    }
    
    try {
        gaussian.setStandardDeviation(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - setStandardDeviation should validate zero
        assert(std::string(e.what()).find("Standard deviation") != std::string::npos);
    }
    
    try {
        gaussian.setStandardDeviation(-1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - setStandardDeviation should validate negative values
        assert(std::string(e.what()).find("Standard deviation") != std::string::npos);
    }
    
    std::cout << "✓ Parameter validation tests passed" << std::endl;
}

/**
 * Test string representation
 */
void testStringRepresentation() {
    std::cout << "Testing string representation..." << std::endl;
    
    GaussianDistribution gaussian(2.5, 1.5);
    std::string str = gaussian.toString();
    
    // Should contain key information based on actual output format:
    // "Gaussian Distribution:\n      μ (mean) = 2.5\n      σ (std. deviation) = 1.5\n      Mean = 2.5\n      Variance = 2.25\n"
    assert(str.find("Gaussian") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("1.5") != std::string::npos);
    assert(str.find("Mean") != std::string::npos);
    assert(str.find("std. deviation") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    GaussianDistribution original(3.14, 2.71);
    
    // Test copy constructor
    GaussianDistribution copied(original);
    assert(copied.getMean() == original.getMean());
    assert(copied.getStandardDeviation() == original.getStandardDeviation());
    
    // Test copy assignment
    GaussianDistribution assigned;
    assigned = original;
    assert(assigned.getMean() == original.getMean());
    assert(assigned.getStandardDeviation() == original.getStandardDeviation());
    
    // Test move constructor
    GaussianDistribution moved(std::move(original));
    assert(moved.getMean() == 3.14);
    assert(moved.getStandardDeviation() == 2.71);
    
    // Test move assignment
    GaussianDistribution moveAssigned;
    GaussianDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getMean() == 1.41);
    assert(moveAssigned.getStandardDeviation() == 1.73);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    GaussianDistribution gaussian(0.0, 1.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(gaussian.getProbability(nan_val) == 0.0);
    assert(gaussian.getProbability(inf_val) == 0.0);
    assert(gaussian.getProbability(neg_inf_val) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test log probability function
 */
void testLogProbability() {
    std::cout << "Testing log probability function..." << std::endl;
    
    GaussianDistribution gaussian(0.0, 1.0);  // Standard normal
    
    // Test log PDF at mean
    double logProbAtMean = gaussian.getLogProbability(0.0);
    double expectedLogPDF = std::log(1.0 / math::SQRT_2PI);
    assert(std::abs(logProbAtMean - expectedLogPDF) < 1e-10);
    
    // Test consistency between PDF and log PDF
    double prob = gaussian.getProbability(0.0);
    double logProb = gaussian.getLogProbability(0.0);
    assert(std::abs(prob - std::exp(logProb)) < 1e-10);
    
    // Test symmetry in log space
    double logProbAt1 = gaussian.getLogProbability(1.0);
    double logProbAtNeg1 = gaussian.getLogProbability(-1.0);
    assert(std::abs(logProbAt1 - logProbAtNeg1) < 1e-10);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    assert(std::isinf(gaussian.getLogProbability(nan_val)));
    assert(std::isinf(gaussian.getLogProbability(inf_val)));
    
    std::cout << "✓ Log probability tests passed" << std::endl;
}

/**
 * Test additional getters and setters
 */
void testAdditionalGettersSetters() {
    std::cout << "Testing additional getters and setters..." << std::endl;
    
    GaussianDistribution gaussian(3.0, 2.0);
    
    // Test variance getter
    double variance = gaussian.getVariance();
    assert(std::abs(variance - 4.0) < 1e-10);  // 2.0^2 = 4.0
    
    // Test setParameters function
    gaussian.setParameters(1.5, 0.5);
    assert(std::abs(gaussian.getMean() - 1.5) < 1e-10);
    assert(std::abs(gaussian.getStandardDeviation() - 0.5) < 1e-10);
    assert(std::abs(gaussian.getVariance() - 0.25) < 1e-10);
    
    // Test setParameters validation
    try {
        gaussian.setParameters(1.0, -1.0);  // Invalid std dev
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        // Expected behavior - setParameters should validate inputs
        assert(std::string(e.what()).find("Standard deviation") != std::string::npos);
    }
    
    std::cout << "✓ Additional getters/setters tests passed" << std::endl;
}

/**
 * Test mathematical correctness with known values
 */
void testMathematicalCorrectness() {
    std::cout << "Testing mathematical correctness..." << std::endl;
    
    // Test with different distributions
    GaussianDistribution gaussian1(0.0, 1.0);   // Standard normal
    GaussianDistribution gaussian2(5.0, 2.0);   // Non-standard
    
    // Test known PDF values for standard normal
    // At x=0: PDF = 1/sqrt(2π) ≈ 0.3989422804
    double pdf0 = gaussian1.getProbability(0.0);
    assert(std::abs(pdf0 - 0.3989422804) < 1e-8);
    
    // At x=1: PDF ≈ 0.2419707245
    double pdf1 = gaussian1.getProbability(1.0);
    assert(std::abs(pdf1 - 0.2419707245) < 1e-8);
    
    // Test different distribution
    // At mean (x=5): PDF = 1/(2*sqrt(2π)) ≈ 0.1994711402
    double pdf_mean = gaussian2.getProbability(5.0);
    double expected = 1.0 / (2.0 * math::SQRT_2PI);
    assert(std::abs(pdf_mean - expected) < 1e-8);
    
    std::cout << "✓ Mathematical correctness tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    GaussianDistribution gaussian(10.0, 5.0);
    gaussian.reset();
    
    assert(gaussian.getMean() == 0.0);
    assert(gaussian.getStandardDeviation() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test CDF calculations
 */
void testCDF() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    GaussianDistribution gaussian(0.0, 1.0);  // Standard normal
    
    // Test CDF at mean (should be 0.5)
    double cdfAtMean = gaussian.CDF(0.0);
    assert(std::abs(cdfAtMean - 0.5) < 1e-6);
    
    // Test CDF symmetry
    double cdfAt1 = gaussian.CDF(1.0);
    double cdfAtNeg1 = gaussian.CDF(-1.0);
    assert(std::abs(cdfAt1 + cdfAtNeg1 - 1.0) < 1e-6);  // Should sum to 1
    
    // Test CDF is monotonically increasing
    assert(gaussian.CDF(-2.0) < gaussian.CDF(-1.0));
    assert(gaussian.CDF(-1.0) < gaussian.CDF(0.0));
    assert(gaussian.CDF(0.0) < gaussian.CDF(1.0));
    assert(gaussian.CDF(1.0) < gaussian.CDF(2.0));
    
    // Test CDF bounds
    assert(gaussian.CDF(-5.0) >= 0.0 && gaussian.CDF(-5.0) <= 1.0);
    assert(gaussian.CDF(5.0) >= 0.0 && gaussian.CDF(5.0) <= 1.0);
    
    // Test that CDF approaches bounds
    assert(gaussian.CDF(-10.0) < 0.001);  // Should be very small
    assert(gaussian.CDF(10.0) > 0.999);   // Should be very close to 1
    
    // Test with different parameters
    GaussianDistribution gaussian2(2.0, 0.5);
    double cdfAtMean2 = gaussian2.CDF(2.0);
    assert(std::abs(cdfAtMean2 - 0.5) < 1e-6);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    assert(gaussian.CDF(nan_val) == 0.0);  // CDF returns 0.0 for NaN input
    
    std::cout << "✓ CDF calculation tests passed" << std::endl;
}

/**
 * Test equality operators and I/O
 */
void testEqualityAndIO() {
    std::cout << "Testing equality operators and I/O..." << std::endl;
    
    GaussianDistribution g1(2.5, 1.5);
    GaussianDistribution g2(2.5, 1.5);
    GaussianDistribution g3(2.5, 1.6);  // Different std dev
    GaussianDistribution g4(2.6, 1.5);  // Different mean
    
    // Test equality operator
    assert(g1 == g2);
    assert(g2 == g1);  // Symmetric
    
    // Test inequality
    assert(!(g1 == g3));
    assert(!(g1 == g4));
    assert(g1 != g3);
    assert(g1 != g4);
    
    // Test self-equality
    assert(g1 == g1);
    
    // Test with very small differences (within tolerance)
    GaussianDistribution g5(2.5, 1.5 + 1e-15);  // Very small difference
    assert(g1 == g5);  // Should be equal within tolerance
    
    // Test with differences larger than tolerance
    GaussianDistribution g6(2.5, 1.5 + 1e-5);  // Larger difference
    assert(!(g1 == g6));  // Should not be equal
    
    // Test stream output
    std::ostringstream oss;
    oss << g1;
    std::string output = oss.str();
    assert(output.find("Normal Distribution") != std::string::npos);
    assert(output.find("2.5") != std::string::npos);
    assert(output.find("1.5") != std::string::npos);
    
    std::cout << "Stream output: " << output << std::endl;
    
    // Test stream input
    std::istringstream iss("Mean = 3.14 Standard deviation = 2.71");
    GaussianDistribution inputDist;
    iss >> inputDist;
    
    if (iss.good()) {
        assert(std::abs(inputDist.getMean() - 3.14) < 1e-10);
        assert(std::abs(inputDist.getStandardDeviation() - 2.71) < 1e-10);
    }
    
    // Test input operator with invalid data
    std::istringstream invalid_iss("invalid data format");
    GaussianDistribution invalid_test;
    invalid_iss >> invalid_test;
    assert(invalid_iss.fail());  // Stream should be in failed state
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

/**
 * Test caching mechanism
 */
void testCaching() {
    std::cout << "Testing caching mechanism..." << std::endl;
    
    GaussianDistribution gaussian(1.0, 2.0);
    
    // Get some probability values (this should populate cache)
    // Use the mean value for maximum sensitivity
    double prob1 = gaussian.getProbability(1.0);
    double logProb1 = gaussian.getLogProbability(1.0);
    
    // Verify consistency between PDF and log PDF
    assert(std::abs(prob1 - std::exp(logProb1)) < 1e-10);
    
    // Change parameters (this should invalidate cache)
    gaussian.setMean(3.0);  // Bigger change for clearer difference
    
    // Get probability again (should use updated parameters)
    double prob2 = gaussian.getProbability(1.0);  // Same x value
    double logProb2 = gaussian.getLogProbability(1.0);
    
    // Values should be different due to parameter change
    assert(std::abs(prob1 - prob2) > 1e-6);
    assert(std::abs(logProb1 - logProb2) > 1e-6);
    
    // Verify consistency with new parameters
    assert(std::abs(prob2 - std::exp(logProb2)) < 1e-10);
    
    // Test cache invalidation with setStandardDeviation
    double prob3 = gaussian.getProbability(3.0);  // Use current mean
    gaussian.setStandardDeviation(1.0);  // Change std dev significantly
    double prob4 = gaussian.getProbability(3.0);
    assert(std::abs(prob3 - prob4) > 1e-6);
    
    // Test cache invalidation with setParameters
    double prob5 = gaussian.getProbability(3.0);
    gaussian.setParameters(0.0, 1.0);  // Change to standard normal
    double prob6 = gaussian.getProbability(3.0);
    assert(std::abs(prob5 - prob6) > 1e-6);
    
    // Test cache invalidation with reset
    // Change the parameters first then reset to see cache invalidation
    gaussian.setMean(5.0);
    double prob_before_reset = gaussian.getProbability(0.0);
    gaussian.reset();
    double prob_after_reset = gaussian.getProbability(0.0);
    assert(std::abs(prob_before_reset - prob_after_reset) > 1e-6);
    
    std::cout << "✓ Caching mechanism tests passed" << std::endl;
}

/**
 * Test performance characteristics to verify optimizations
 * This serves as a benchmark and regression test for performance
 */
void testPerformance() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    using namespace std::chrono;
    GaussianDistribution gaussian(0.0, 1.0);
    
    // Test parameters
    const int pdf_iterations = 100000;
    const int fit_datapoints = 5000;
    
    // Generate test values for PDF calls
    std::vector<double> testValues;
    testValues.reserve(pdf_iterations);
    for (int i = 0; i < pdf_iterations; ++i) {
        testValues.push_back(-3.0 + (6.0 * i) / pdf_iterations);
    }
    
    // Test getProbability() performance (should benefit from caching)
    auto start = high_resolution_clock::now();
    double sum_pdf = 0.0;
    for (const auto& val : testValues) {
        sum_pdf += gaussian.getProbability(val);
    }
    auto end = high_resolution_clock::now();
    auto pdf_duration = duration_cast<microseconds>(end - start);
    
    // Test getLogProbability() performance (should benefit from cached 1/σ and log(σ))
    start = high_resolution_clock::now();
    double sum_log_pdf = 0.0;
    for (const auto& val : testValues) {
        sum_log_pdf += gaussian.getLogProbability(val);
    }
    end = high_resolution_clock::now();
    auto log_pdf_duration = duration_cast<microseconds>(end - start);
    
    // Test fitting performance with Welford's algorithm
    std::vector<double> fit_data;
    fit_data.reserve(fit_datapoints);
    for (int i = 0; i < fit_datapoints; ++i) {
        fit_data.push_back(i * 0.001);  // Linear data for consistent testing
    }
    
    start = high_resolution_clock::now();
    gaussian.fit(fit_data);
    end = high_resolution_clock::now();
    auto fit_duration = duration_cast<microseconds>(end - start);
    
    // Performance expectations (reasonable thresholds for regression testing)
    double pdf_per_call = static_cast<double>(pdf_duration.count()) / pdf_iterations;
    double log_pdf_per_call = static_cast<double>(log_pdf_duration.count()) / pdf_iterations;
    double fit_per_point = static_cast<double>(fit_duration.count()) / fit_datapoints;
    
    // Basic performance assertions (adjust thresholds based on typical performance)
    // These should be conservative to avoid false failures on different hardware
    assert(pdf_per_call < 1.0);      // Should be well under 1 microsecond per PDF call
    assert(log_pdf_per_call < 1.0);  // Should be well under 1 microsecond per log PDF call
    assert(fit_per_point < 10.0);    // Should be well under 10 microseconds per fit datapoint
    
    // Verify correctness (prevent compiler optimization removal)
    assert(sum_pdf > 0.0);
    assert(sum_log_pdf < 0.0);  // Log probabilities should be negative
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  PDF timing:       " << pdf_per_call << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << log_pdf_per_call << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << fit_per_point << " μs/point (" << fit_datapoints << " points)" << std::endl;
    std::cout << "✓ Performance tests passed" << std::endl;
}

int main() {
    std::cout << "Running Gaussian distribution tests..." << std::endl;
    std::cout << "======================================" << std::endl;
    
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
        testCDF();
        testEqualityAndIO();
        testCaching();
        testPerformance();
        
        std::cout << "======================================" << std::endl;
        std::cout << "✅ All Gaussian distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
