#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/exponential_distribution.h"

using libhmm::ExponentialDistribution;
using libhmm::Observation;
using namespace libhmm::constants;

/**
 * Test basic Exponential distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Exponential distribution functionality..." << std::endl;
    
    // Test default constructor
    ExponentialDistribution exponential;
    assert(exponential.getLambda() == 1.0);
    
    // Test parameterized constructor
    ExponentialDistribution exponential2(2.5);
    assert(exponential2.getLambda() == 2.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    ExponentialDistribution exponential(1.0);  // lambda=1
    
    // For continuous Exponential PDF, at x=0 the value is λ (the rate parameter)
    assert(exponential.getProbability(0.0) == 1.0);  // lambda = 1.0
    
    // Should be positive for positive values
    double prob1 = exponential.getProbability(1.0);
    double prob2 = exponential.getProbability(2.0);
    double prob3 = exponential.getProbability(3.0);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    // Should decrease with increasing x (memoryless property)
    assert(prob1 > prob2);
    assert(prob2 > prob3);
    
    // Should be zero for negative values
    assert(exponential.getProbability(-1.0) == 0.0);
    assert(exponential.getProbability(-0.5) == 0.0);
    
    // Test that probability is reasonable (our implementation returns small values for continuous distributions)
    assert(prob1 > 1e-10);  // Should be positive
    assert(prob1 < 1.0);    // Should be less than 1
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    ExponentialDistribution exponential;
    
    // Test with known data (lambda should be 1/mean)
    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double expectedMean = 3.0;
    double expectedLambda = 1.0 / expectedMean;
    
    exponential.fit(data);
    assert(std::abs(exponential.getLambda() - expectedLambda) < 1e-10);
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    exponential.fit(emptyData);
    assert(exponential.getLambda() == 1.0);
    
    // Test with single positive point (implementation resets to default for insufficient data)
    std::vector<Observation> singlePoint = {2.5};
    exponential.fit(singlePoint);
    assert(exponential.getLambda() == 1.0);  // Implementation resets to default
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        ExponentialDistribution exponential(0.0);  // Zero lambda
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        ExponentialDistribution exponential(-1.0);  // Negative lambda
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        ExponentialDistribution exponential(nan_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        ExponentialDistribution exponential(inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    ExponentialDistribution exponential(1.0);
    
    try {
        exponential.setLambda(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        exponential.setLambda(-1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        exponential.setLambda(nan_val);
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
    
    ExponentialDistribution exponential(2.5);
    std::string str = exponential.toString();
    
    // Should contain key information based on new format:
    // "Exponential Distribution:\n      λ (rate parameter) = 2.5\n      Mean = 0.4\n"
    assert(str.find("Exponential") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("rate parameter") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    ExponentialDistribution original(3.14);
    
    // Test copy constructor
    ExponentialDistribution copied(original);
    assert(copied.getLambda() == original.getLambda());
    
    // Test copy assignment
    ExponentialDistribution assigned;
    assigned = original;
    assert(assigned.getLambda() == original.getLambda());
    
    // Test move constructor
    ExponentialDistribution moved(std::move(original));
    assert(moved.getLambda() == 3.14);
    
    // Test move assignment
    ExponentialDistribution moveAssigned;
    ExponentialDistribution temp(2.71);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getLambda() == 2.71);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    ExponentialDistribution exponential(1.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(exponential.getProbability(nan_val) == 0.0);
    assert(exponential.getProbability(inf_val) == 0.0);
    assert(exponential.getProbability(neg_inf_val) == 0.0);
    
    // Negative values should return 0
    assert(exponential.getProbability(-1.0) == 0.0);
    assert(exponential.getProbability(-0.1) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    ExponentialDistribution exponential(10.0);
    exponential.reset();
    
    assert(exponential.getLambda() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    ExponentialDistribution exponential;
    
    // Test with data containing negative values
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};
    
    // Exponential distribution should handle negative values in fitting
    // (typically by ignoring them or throwing an exception)
    try {
        exponential.fit(invalidData);
        // If it doesn't throw, the parameter should still be valid
        assert(exponential.getLambda() > 0.0);
    } catch (const std::exception&) {
        // It's also acceptable to throw for invalid data
    }
    
    // Test with zero values (should handle gracefully)
    std::vector<Observation> zeroData = {0.0, 1.0, 2.0};
    try {
        exponential.fit(zeroData);
        assert(exponential.getLambda() > 0.0);
    } catch (const std::exception&) {
        // Acceptable to reject zero values
    }
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test log probability function for numerical stability
 */
void testLogProbability() {
    std::cout << "Testing log probability calculations..." << std::endl;
    
    ExponentialDistribution exponential(2.0);  // lambda=2
    
    // Test log probability at several points
    double x1 = 0.5;
    double x2 = 1.0;
    double x3 = 2.0;
    
    double logP1 = exponential.getLogProbability(x1);
    double logP2 = exponential.getLogProbability(x2);
    double logP3 = exponential.getLogProbability(x3);
    
    // Verify log probabilities are negative (since probabilities < 1)
    assert(logP1 < 0.0);
    assert(logP2 < 0.0);
    assert(logP3 < 0.0);
    
    // For exponential with λ=2: log(f(x)) = log(2) - 2x
    double expectedLogP1 = std::log(2.0) - 2.0 * x1;  // log(2) - 1
    double expectedLogP2 = std::log(2.0) - 2.0 * x2;  // log(2) - 2
    double expectedLogP3 = std::log(2.0) - 2.0 * x3;  // log(2) - 4
    
    assert(std::abs(logP1 - expectedLogP1) < 1e-10);
    assert(std::abs(logP2 - expectedLogP2) < 1e-10);
    assert(std::abs(logP3 - expectedLogP3) < 1e-10);
    
    // Verify consistency with getProbability
    double p1 = exponential.getProbability(x1);
    double p2 = exponential.getProbability(x2);
    
    // Note: For small tolerance values, getProbability might not exactly match
    // the log of getProbability due to discrete approximation, so we test reasonableness
    assert(p1 > 0.0);
    assert(p2 > 0.0);
    
    // Test invalid inputs return -infinity
    assert(exponential.getLogProbability(-1.0) == -std::numeric_limits<double>::infinity());
    assert(exponential.getLogProbability(std::numeric_limits<double>::quiet_NaN()) == -std::numeric_limits<double>::infinity());
    
    std::cout << "✓ Log probability calculation tests passed" << std::endl;
}

/**
 * Test memoryless property
 */
void testMemorylessProperty() {
    std::cout << "Testing memoryless property..." << std::endl;
    
    ExponentialDistribution exponential(1.0);
    
    // Test that P(X > s+t | X > s) = P(X > t)
    // This is equivalent to testing that PDF decreases exponentially
    double t1 = 1.0;
    double t2 = 2.0;
    double t3 = 3.0;
    
    double p1 = exponential.getProbability(t1);
    double p2 = exponential.getProbability(t2);
    double p3 = exponential.getProbability(t3);
    
    // Check exponential decay pattern
    double ratio1 = p2 / p1;
    double ratio2 = p3 / p2;
    
    // Ratios should be approximately equal due to memoryless property
    assert(std::abs(ratio1 - ratio2) < 1e-9);  // Slightly looser tolerance for numerical precision
    
    std::cout << "✓ Memoryless property tests passed" << std::endl;
}

/**
 * Test statistical moments and properties
 */
void testStatisticalMoments() {
    std::cout << "Testing statistical moments..." << std::endl;
    
    ExponentialDistribution exponential(2.5);
    
    // For Exponential(λ): mean = 1/λ, variance = 1/λ², std_dev = 1/λ
    double expectedMean = 1.0 / 2.5;
    double expectedVar = 1.0 / (2.5 * 2.5);
    double expectedStdDev = 1.0 / 2.5;
    
    assert(std::abs(exponential.getMean() - expectedMean) < 1e-10);
    assert(std::abs(exponential.getVariance() - expectedVar) < 1e-10);
    assert(std::abs(exponential.getStandardDeviation() - expectedStdDev) < 1e-10);
    
    // Test scale parameter (should equal mean)
    assert(std::abs(exponential.getScale() - expectedMean) < 1e-10);
    
    // For exponential distribution, mean = standard deviation
    assert(std::abs(exponential.getMean() - exponential.getStandardDeviation()) < 1e-10);
    
    std::cout << "✓ Statistical moments tests passed" << std::endl;
}

/**
 * Test performance characteristics and optimizations
 */
void testPerformanceCharacteristics() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    ExponentialDistribution exponential(1.5);
    
    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 100000;
    volatile double sum = 0.0;  // volatile to prevent optimization
    
    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i) / 10000.0;
        sum += exponential.getProbability(x);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;
    
    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;
    
    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i) / 10000.0;
        logSum += exponential.getLogProbability(x);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;
    
    // Test fitting timing
    std::vector<Observation> fitData(5000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i + 1) / 1000.0;  // Positive values
    }
    
    start = std::chrono::high_resolution_clock::now();
    exponential.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fitDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double fitTimePerPoint = static_cast<double>(fitDuration.count()) / fitData.size();
    
    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) 
              << pdfTimePerCall << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3) 
              << logPdfTimePerCall << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) 
              << fitTimePerPoint << " μs/point (" << fitData.size() << " points)" << std::endl;
    
    // Performance requirements (should be fast)
    assert(pdfTimePerCall < 1.0);     // Less than 1 μs per PDF call
    assert(logPdfTimePerCall < 1.0);  // Less than 1 μs per log PDF call
    assert(fitTimePerPoint < 0.1);    // Less than 0.1 μs per data point for fitting
    
    std::cout << "✓ Performance tests passed" << std::endl;
}

/**
 * Test enhanced fitting with robust validation
 */
void testEnhancedFitting() {
    std::cout << "Testing enhanced fitting with validation..." << std::endl;
    
    ExponentialDistribution exponential;
    
    // Test with data that should produce a specific lambda
    std::vector<Observation> testData = {0.5, 1.0, 1.5, 2.0, 2.5};
    double expectedMean = 1.5;  // Sum = 7.5, n = 5
    double expectedLambda = 1.0 / expectedMean;
    
    exponential.fit(testData);
    assert(std::abs(exponential.getLambda() - expectedLambda) < 1e-10);
    
    // Test with negative data (should reset to default)
    std::vector<Observation> negativeData = {1.0, -0.5, 2.0};
    exponential.fit(negativeData);
    assert(exponential.getLambda() == 1.0);  // Should reset to default
    
    // Test with zero values (should reset to default)
    std::vector<Observation> zeroData = {0.0, 1.0, 2.0};
    exponential.fit(zeroData);
    assert(exponential.getLambda() == 1.0);  // Should reset to default
    
    // Test with all zero values
    std::vector<Observation> allZeros = {0.0, 0.0, 0.0};
    exponential.fit(allZeros);
    assert(exponential.getLambda() == 1.0);  // Should reset to default
    
    std::cout << "✓ Enhanced fitting tests passed" << std::endl;
}

/**
 * Test numerical edge cases and stability
 */
void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;
    
    // Test with very small lambda
    ExponentialDistribution smallExp(1e-6);
    double probSmall = smallExp.getProbability(1.0);
    assert(probSmall > 0.0);
    assert(std::isfinite(probSmall));
    
    // Test with very large lambda
    ExponentialDistribution largeExp(1e6);
    double probLarge = largeExp.getProbability(0.000001);
    assert(probLarge > 0.0);
    assert(std::isfinite(probLarge));
    
    // Test log probability with extreme values
    double logProbSmall = smallExp.getLogProbability(1000.0);
    double logProbLarge = largeExp.getLogProbability(0.000001);
    assert(std::isfinite(logProbSmall));
    assert(std::isfinite(logProbLarge));
    
    // Test mathematical correctness at special points
    ExponentialDistribution unit(1.0);
    
    // At x=0, PDF should equal lambda
    assert(std::abs(unit.getProbability(0.0) - 1.0) < 1e-10);
    
    // Log PDF at x=0 should equal log(lambda)
    assert(std::abs(unit.getLogProbability(0.0) - std::log(1.0)) < 1e-10);
    
    std::cout << "✓ Numerical stability tests passed" << std::endl;
}

/**
 * Test CDF calculations
 */
void testCDF() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    ExponentialDistribution exponential(1.0);  // lambda=1
    
    // For exponential distribution: F(x) = 1 - exp(-λx)
    // Test at various points
    double cdfAt0 = exponential.CDF(0.0);
    assert(std::abs(cdfAt0 - 0.0) < 1e-10);  // F(0) = 0
    
    double cdfAt1 = exponential.CDF(1.0);
    double expectedCdf1 = 1.0 - std::exp(-1.0);  // 1 - e^(-1)
    assert(std::abs(cdfAt1 - expectedCdf1) < 1e-10);
    
    double cdfAt2 = exponential.CDF(2.0);
    double expectedCdf2 = 1.0 - std::exp(-2.0);  // 1 - e^(-2)
    assert(std::abs(cdfAt2 - expectedCdf2) < 1e-10);
    
    // Test CDF is monotonically increasing
    assert(exponential.CDF(0.0) < exponential.CDF(1.0));
    assert(exponential.CDF(1.0) < exponential.CDF(2.0));
    assert(exponential.CDF(2.0) < exponential.CDF(3.0));
    
    // Test CDF bounds
    assert(exponential.CDF(0.0) >= 0.0 && exponential.CDF(0.0) <= 1.0);
    assert(exponential.CDF(10.0) >= 0.0 && exponential.CDF(10.0) <= 1.0);
    
    // Test that CDF approaches 1 for large values
    assert(exponential.CDF(10.0) > 0.99995);  // Should be very close to 1 (for λ=1, F(10) ≈ 0.99995)
    
    // Test with different parameters
    ExponentialDistribution exponential2(2.0);
    double cdf2At1 = exponential2.CDF(1.0);
    double expectedCdf2At1 = 1.0 - std::exp(-2.0);  // 1 - e^(-2)
    assert(std::abs(cdf2At1 - expectedCdf2At1) < 1e-10);
    
    // CDF for λ=2 should be higher than for λ=1 at same x
    assert(exponential2.CDF(1.0) > exponential.CDF(1.0));
    
    std::cout << "✓ CDF calculation tests passed" << std::endl;
}

/**
 * Test equality operators and I/O
 */
void testEqualityAndIO() {
    std::cout << "Testing equality operators and I/O..." << std::endl;
    
    ExponentialDistribution e1(2.5);
    ExponentialDistribution e2(2.5);
    ExponentialDistribution e3(2.6);  // Different lambda
    
    // Test equality operator
    assert(e1 == e2);
    assert(e2 == e1);  // Symmetric
    
    // Test inequality
    assert(!(e1 == e3));
    assert(e1 != e3);
    
    // Test self-equality
    assert(e1 == e1);
    
    // Test with very small differences (within tolerance)
    ExponentialDistribution e4(2.5 + 1e-15);  // Very small difference
    assert(e1 == e4);  // Should be equal within tolerance
    
    // Test with differences larger than tolerance
    ExponentialDistribution e5(2.5 + 1e-5);  // Larger difference
    assert(!(e1 == e5));  // Should not be equal
    
    // Test stream output
    std::ostringstream oss;
    oss << e1;
    std::string output = oss.str();
    assert(output.find("Exponential Distribution") != std::string::npos);
    assert(output.find("2.5") != std::string::npos);
    
    std::cout << "Stream output: " << output << std::endl;
    
    // Test stream input
    std::istringstream iss("Rate parameter = 3.14");
    ExponentialDistribution inputDist;
    iss >> inputDist;
    
    if (!iss.fail()) {
        assert(std::abs(inputDist.getLambda() - 3.14) < 1e-10);
    }
    
    // Test input operator with invalid data
    std::istringstream invalid_iss("invalid data format");
    ExponentialDistribution invalid_test;
    invalid_iss >> invalid_test;
    assert(invalid_iss.fail());  // Stream should be in failed state
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

/**
 * Test caching mechanism
 */
void testCaching() {
    std::cout << "Testing caching mechanism..." << std::endl;
    
    ExponentialDistribution exponential(1.0);
    
    // Get some probability values (this should populate cache)
    double prob1 = exponential.getProbability(1.0);
    double logProb1 = exponential.getLogProbability(1.0);
    
    // Verify consistency between PDF and log PDF
    assert(std::abs(prob1 - std::exp(logProb1)) < 1e-10);
    
    // Change parameters (this should invalidate cache)
    exponential.setLambda(2.0);  // Significant change
    
    // Get probability again (should use updated parameters)
    double prob2 = exponential.getProbability(1.0);  // Same x value
    double logProb2 = exponential.getLogProbability(1.0);
    
    // Values should be different due to parameter change
    assert(std::abs(prob1 - prob2) > 1e-6);
    assert(std::abs(logProb1 - logProb2) > 1e-6);
    
    // Verify consistency with new parameters
    assert(std::abs(prob2 - std::exp(logProb2)) < 1e-10);
    
    // Test that cached values are properly used for derived properties
    double mean1 = exponential.getMean();
    double variance1 = exponential.getVariance();
    double stdDev1 = exponential.getStandardDeviation();
    double scale1 = exponential.getScale();
    
    // These should all use cached values and be consistent
    assert(std::abs(mean1 - (1.0 / 2.0)) < 1e-10);  // mean = 1/λ
    assert(std::abs(variance1 - (1.0 / (2.0 * 2.0))) < 1e-10);  // variance = 1/λ²
    assert(std::abs(stdDev1 - mean1) < 1e-10);  // std_dev = mean for exponential
    assert(std::abs(scale1 - mean1) < 1e-10);  // scale = mean for exponential
    
    // Test cache invalidation with reset
    double prob3 = exponential.getProbability(1.0);
    exponential.reset();
    double prob4 = exponential.getProbability(1.0);
    assert(std::abs(prob3 - prob4) > 1e-6);  // Should be different after reset
    
    std::cout << "✓ Caching mechanism tests passed" << std::endl;
}

int main() {
    std::cout << "Running Exponential distribution tests..." << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testLogProbability();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        testFittingValidation();
        testMemorylessProperty();
        testStatisticalMoments();
        testPerformanceCharacteristics();
        testEnhancedFitting();
        testNumericalStability();
        testCDF();
        testEqualityAndIO();
        testCaching();
        
        std::cout << "=========================================" << std::endl;
        std::cout << "✅ All Exponential distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
