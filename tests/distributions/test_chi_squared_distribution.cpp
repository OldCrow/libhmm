#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <sstream>
#include "libhmm/distributions/chi_squared_distribution.h"

using libhmm::ChiSquaredDistribution;
using libhmm::Observation;

/**
 * Test basic Chi-squared distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Chi-squared distribution functionality..." << std::endl;
    
    // Test default constructor
    ChiSquaredDistribution chi_dist;
    assert(chi_dist.getDegreesOfFreedom() == 1.0);
    
    // Test parameterized constructor
    ChiSquaredDistribution chi_dist2(5.0);
    assert(chi_dist2.getDegreesOfFreedom() == 5.0);
    
    // Test parameter setter
    chi_dist.setDegreesOfFreedom(3.0);
    assert(chi_dist.getDegreesOfFreedom() == 3.0);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    ChiSquaredDistribution chi_dist(2.0);  // df = 2
    
    // Test that probability is zero for negative values
    assert(chi_dist.getProbability(-0.1) == 0.0);
    assert(chi_dist.getProbability(-1.0) == 0.0);
    assert(chi_dist.getProbability(-10.0) == 0.0);
    
    // Test that probability is positive for non-negative values
    double prob_at_zero = chi_dist.getProbability(0.0);
    double prob_at_one = chi_dist.getProbability(1.0);
    double prob_at_two = chi_dist.getProbability(2.0);
    double prob_at_five = chi_dist.getProbability(5.0);
    
    assert(prob_at_zero > 0.0);
    assert(prob_at_one > 0.0);
    assert(prob_at_two > 0.0);
    assert(prob_at_five > 0.0);
    
    // For Chi-squared(2), the distribution is exponential-like
    // Probability should decrease as x increases
    assert(prob_at_zero > prob_at_one);
    assert(prob_at_one > prob_at_two);
    assert(prob_at_two > prob_at_five);
    
    // Test with df = 1
    ChiSquaredDistribution chi_dist_1(1.0);
    double prob_1_at_zero = chi_dist_1.getProbability(0.0);
    double prob_1_at_small = chi_dist_1.getProbability(0.1);
    
    // For df=1, density at x=0 should be infinite
    assert(std::isinf(prob_1_at_zero));
    assert(prob_1_at_small > 0.0 && std::isfinite(prob_1_at_small));
    
    // Test with df > 2 (should have mode at df-2)
    ChiSquaredDistribution chi_dist_5(5.0);
    double mode = 5.0 - 2.0;  // mode = df - 2 = 3
    double prob_at_mode = chi_dist_5.getProbability(mode);
    double prob_before_mode = chi_dist_5.getProbability(mode - 1.0);
    double prob_after_mode = chi_dist_5.getProbability(mode + 1.0);
    
    // Probability at mode should be higher than nearby points
    assert(prob_at_mode >= prob_before_mode);
    assert(prob_at_mode >= prob_after_mode);
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test statistical properties
 */
void testStatisticalProperties() {
    std::cout << "Testing statistical properties..." << std::endl;
    
    // Test mean and variance formulas
    ChiSquaredDistribution chi_dist_3(3.0);
    assert(chi_dist_3.getMean() == 3.0);  // Mean = k
    assert(chi_dist_3.getVariance() == 6.0);  // Variance = 2k
    assert(std::abs(chi_dist_3.getStandardDeviation() - std::sqrt(6.0)) < 1e-10);
    
    ChiSquaredDistribution chi_dist_10(10.0);
    assert(chi_dist_10.getMean() == 10.0);
    assert(chi_dist_10.getVariance() == 20.0);
    assert(std::abs(chi_dist_10.getStandardDeviation() - std::sqrt(20.0)) < 1e-10);
    
    // Test mode calculation
    ChiSquaredDistribution chi_dist_1(1.0);
    ChiSquaredDistribution chi_dist_2(2.0);
    ChiSquaredDistribution chi_dist_5(5.0);
    
    assert(chi_dist_1.getMode() == 0.0);  // max(0, 1-2) = 0
    assert(chi_dist_2.getMode() == 0.0);  // max(0, 2-2) = 0
    assert(chi_dist_5.getMode() == 3.0);  // max(0, 5-2) = 3
    
    std::cout << "✓ Statistical properties tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    ChiSquaredDistribution chi_dist;
    
    // Test with data that has known mean
    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0};  // Mean = 3.0
    chi_dist.fit(data);
    
    // For Chi-squared, MLE estimate of k is the sample mean
    assert(std::abs(chi_dist.getDegreesOfFreedom() - 3.0) < 1e-10);
    
    // Test with another dataset
    std::vector<Observation> data2 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};  // Mean = 4.0
    chi_dist.fit(data2);
    assert(std::abs(chi_dist.getDegreesOfFreedom() - 4.0) < 1e-10);
    
    // Test with empty data - should throw exception
    std::vector<Observation> empty_data;
    try {
        chi_dist.fit(empty_data);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with negative values - should throw exception
    std::vector<Observation> negative_data = {1.0, -2.0, 3.0};
    try {
        chi_dist.fit(negative_data);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        ChiSquaredDistribution chi_dist(0.0);  // Zero degrees of freedom
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        ChiSquaredDistribution chi_dist(-1.0);  // Negative degrees of freedom
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        ChiSquaredDistribution chi_dist(nan_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        ChiSquaredDistribution chi_dist(inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setter validation
    ChiSquaredDistribution chi_dist(1.0);
    
    try {
        chi_dist.setDegreesOfFreedom(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        chi_dist.setDegreesOfFreedom(-5.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "✓ Parameter validation tests passed" << std::endl;
}

/**
 * Test string representation and serialization
 */
void testStringRepresentation() {
    std::cout << "Testing string representation..." << std::endl;
    
    ChiSquaredDistribution chi_dist(3.5);
    std::string str = chi_dist.toString();
    
    // Should contain key information
    assert(str.find("ChiSquared Distribution") != std::string::npos);
    assert(str.find("3.5") != std::string::npos);
    assert(str.find("k") != std::string::npos);
    
    // Test fromString functionality
    ChiSquaredDistribution chi_dist_from_str = ChiSquaredDistribution::fromString("ChiSquared(k=7.25)");
    assert(std::abs(chi_dist_from_str.getDegreesOfFreedom() - 7.25) < 1e-10);
    
    ChiSquaredDistribution chi_dist_from_str2 = ChiSquaredDistribution::fromString("ChiSquared(df=12.0)");
    assert(std::abs(chi_dist_from_str2.getDegreesOfFreedom() - 12.0) < 1e-10);
    
    // Test invalid string formats
    try {
        ChiSquaredDistribution::fromString("InvalidFormat");
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    ChiSquaredDistribution original(4.2);
    
    // Test copy constructor
    ChiSquaredDistribution copied(original);
    assert(copied.getDegreesOfFreedom() == original.getDegreesOfFreedom());
    assert(copied.getProbability(1.0) == original.getProbability(1.0));
    
    // Test copy assignment
    ChiSquaredDistribution assigned(1.0);
    assigned = original;
    assert(assigned.getDegreesOfFreedom() == original.getDegreesOfFreedom());
    assert(assigned.getProbability(1.0) == original.getProbability(1.0));
    
    // Test move constructor
    double original_df = original.getDegreesOfFreedom();
    ChiSquaredDistribution moved(std::move(original));
    assert(moved.getDegreesOfFreedom() == original_df);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test equality operators
 */
void testEqualityOperators() {
    std::cout << "Testing equality operators..." << std::endl;
    
    ChiSquaredDistribution chi1(3.0);
    ChiSquaredDistribution chi2(3.0);
    ChiSquaredDistribution chi3(4.0);
    
    assert(chi1 == chi2);
    assert(!(chi1 == chi3));
    assert(chi1 != chi3);
    assert(!(chi1 != chi2));
    
    std::cout << "✓ Equality operator tests passed" << std::endl;
}

/**
 * Test edge cases and numerical stability
 */
void testEdgeCases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test very small degrees of freedom
    ChiSquaredDistribution chi_small(1e-5);
    double prob_small = chi_small.getProbability(0.01);
    assert(std::isfinite(prob_small) && prob_small > 0.0);
    
    // Test very large degrees of freedom
    ChiSquaredDistribution chi_large(1e5);
    double prob_large = chi_large.getProbability(1e5);  // Around the mean
    assert(std::isfinite(prob_large) && prob_large > 0.0);
    
    // Test reset functionality
    ChiSquaredDistribution chi_test(10.0);
    chi_test.reset();
    assert(chi_test.getDegreesOfFreedom() == 1.0);
    
    // Test boundary behavior for different df values
    ChiSquaredDistribution chi_half(0.5);  // df < 1
    double prob_half_at_zero = chi_half.getProbability(0.0);
    assert(std::isinf(prob_half_at_zero));  // Should be infinite
    
    ChiSquaredDistribution chi_exactly_2(2.0);  // df = 2
    double prob_2_at_zero = chi_exactly_2.getProbability(0.0);
    assert(std::isfinite(prob_2_at_zero) && prob_2_at_zero > 0.0);  // Should be finite
    
    std::cout << "✓ Edge cases tests passed" << std::endl;
}

/**
 * Test relationship to Gamma distribution
 */
void testGammaRelationship() {
    std::cout << "Testing relationship to Gamma distribution..." << std::endl;
    
    // Chi-squared(k) is equivalent to Gamma(k/2, 2)
    // We can't directly test this without the Gamma distribution,
    // but we can verify some properties
    
    ChiSquaredDistribution chi_4(4.0);
    
    // For Chi-squared(4), which is Gamma(2, 2):
    // Mean = k = 4
    // Variance = 2k = 8
    // Mode = k - 2 = 2 (for k >= 2)
    
    assert(chi_4.getMean() == 4.0);
    assert(chi_4.getVariance() == 8.0);
    assert(chi_4.getMode() == 2.0);
    
    std::cout << "✓ Gamma relationship tests passed" << std::endl;
}

/**
 * Test log probability calculations
 */
void testLogProbabilities() {
    std::cout << "Testing log probability calculations..." << std::endl;
    
    ChiSquaredDistribution chi_dist(3.0);
    
    // Test that log probability is -infinity for negative values
    assert(std::isinf(chi_dist.getLogProbability(-0.1)) && chi_dist.getLogProbability(-0.1) < 0);
    assert(std::isinf(chi_dist.getLogProbability(-1.0)) && chi_dist.getLogProbability(-1.0) < 0);
    
    // Test that log probability is finite for positive values
    assert(std::isfinite(chi_dist.getLogProbability(0.5)));
    assert(std::isfinite(chi_dist.getLogProbability(1.0)));
    assert(std::isfinite(chi_dist.getLogProbability(2.0)));
    
    // Test consistency between log and regular probability
    double x = 1.5;
    double prob = chi_dist.getProbability(x);
    double log_prob = chi_dist.getLogProbability(x);
    assert(std::abs(std::log(prob) - log_prob) < 1e-10);
    
    // Test special case for df < 2 at x = 0
    ChiSquaredDistribution chi_1(1.0);
    assert(std::isinf(chi_1.getLogProbability(0.0)) && chi_1.getLogProbability(0.0) > 0);
    
    // Test special case for df = 2 at x = 0
    ChiSquaredDistribution chi_2(2.0);
    assert(std::isfinite(chi_2.getLogProbability(0.0)));
    
    std::cout << "✓ Log probability calculation tests passed" << std::endl;
}

/**
 * Test CDF calculations
 */
void testCDF() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    ChiSquaredDistribution chi_dist(4.0);
    
    // Test CDF values at boundaries
    assert(chi_dist.getCumulativeProbability(-1.0) == 0.0);  // Below support
    assert(chi_dist.getCumulativeProbability(0.0) == 0.0);   // At lower bound
    
    // Test CDF is monotonically increasing
    assert(chi_dist.getCumulativeProbability(1.0) < chi_dist.getCumulativeProbability(2.0));
    assert(chi_dist.getCumulativeProbability(2.0) < chi_dist.getCumulativeProbability(4.0));
    assert(chi_dist.getCumulativeProbability(4.0) < chi_dist.getCumulativeProbability(8.0));
    
    // Test CDF approaches 1 for large values
    assert(chi_dist.getCumulativeProbability(100.0) > 0.99);
    
    // Test with NaN
    assert(std::isnan(chi_dist.getCumulativeProbability(std::numeric_limits<double>::quiet_NaN())));
    
    // Test known values for Chi-squared(2) which is exponential-like
    ChiSquaredDistribution chi_2(2.0);
    // For Chi-squared(2): CDF(x) = 1 - exp(-x/2)
    double cdf_at_2 = chi_2.getCumulativeProbability(2.0);
    double expected_cdf = 1.0 - std::exp(-1.0);  // 1 - exp(-2/2)
    
    std::cout << "CDF at x=2: " << cdf_at_2 << ", Expected: " << expected_cdf 
              << ", Difference: " << std::abs(cdf_at_2 - expected_cdf) << std::endl;
    
    // Use precision constant for numerical tolerance
    // CDF calculations involving gamma functions need slightly more tolerance
    using namespace libhmm::constants;
    assert(std::abs(cdf_at_2 - expected_cdf) < precision::LIMIT_TOLERANCE);
    
    std::cout << "✓ CDF calculation tests passed" << std::endl;
}

/**
 * Test equality operators and I/O
 */
void testEqualityAndIO() {
    std::cout << "Testing equality operators and I/O..." << std::endl;
    
    ChiSquaredDistribution chi1(3.5);
    ChiSquaredDistribution chi2(3.5);
    ChiSquaredDistribution chi3(4.0);
    
    // Test equality operator
    assert(chi1 == chi2);
    assert(!(chi1 == chi3));
    
    // Test stream output
    std::ostringstream oss;
    oss << chi1;
    std::string output = oss.str();
    assert(output.find("ChiSquared Distribution") != std::string::npos);
    assert(output.find("3.5") != std::string::npos);
    
    std::cout << "Stream output: " << output << std::endl;
    
    // Test stream input (basic format check)
    std::istringstream iss("ChiSquared Distribution: k = 6.75");
    ChiSquaredDistribution inputDist;
    iss >> inputDist;
    
    if (iss.good()) {
        assert(std::abs(inputDist.getDegreesOfFreedom() - 6.75) < 1e-10);
    }
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

/**
 * Test caching mechanism
 */
void testCaching() {
    std::cout << "Testing caching mechanism..." << std::endl;
    
    ChiSquaredDistribution chi_dist(3.0);
    
    // First call should update cache
    double prob1 = chi_dist.getProbability(2.0);
    double logProb1 = chi_dist.getLogProbability(2.0);
    
    // Second call should use cached values
    double prob2 = chi_dist.getProbability(2.5);
    double logProb2 = chi_dist.getLogProbability(2.5);
    
    // Both should be valid
    assert(std::isfinite(prob1) && prob1 > 0.0);
    assert(std::isfinite(prob2) && prob2 > 0.0);
    assert(std::isfinite(logProb1));
    assert(std::isfinite(logProb2));
    
    // Changing parameters should invalidate cache
    chi_dist.setDegreesOfFreedom(5.0);
    double prob3 = chi_dist.getProbability(2.0);
    assert(prob3 != prob1);  // Should be different now
    
    // Reset should also invalidate cache
    chi_dist.reset();
    double prob4 = chi_dist.getProbability(2.0);
    assert(prob4 != prob3);  // Should be different after reset
    
    std::cout << "✓ Caching mechanism tests passed" << std::endl;
}

/**
 * Test performance characteristics
 */
void testPerformance() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    ChiSquaredDistribution chi_dist(4.0);
    
    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 10000;
    volatile double sum = 0.0;  // volatile to prevent optimization
    
    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i) / 1000.0;  // Range 0 to 10
        sum += chi_dist.getProbability(x);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;
    
    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;
    
    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i) / 1000.0;  // Range 0 to 10
        logSum += chi_dist.getLogProbability(x);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;
    
    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i + 1) / 100.0;  // Positive values
    }
    
    start = std::chrono::high_resolution_clock::now();
    chi_dist.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fitDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double fitTimePerPoint = static_cast<double>(fitDuration.count()) / fitData.size();
    
    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) 
              << pdfTimePerCall << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3) 
              << logPdfTimePerCall << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) 
              << fitTimePerPoint << " μs/point (" << fitData.size() << " points)" << std::endl;
    
    // Performance requirements (should be reasonable)
    assert(pdfTimePerCall < 5.0);     // Less than 5 μs per PDF call
    assert(logPdfTimePerCall < 3.0);  // Less than 3 μs per log PDF call
    assert(fitTimePerPoint < 10.0);   // Less than 10 μs per data point for fitting
    
    std::cout << "✓ Performance tests passed" << std::endl;
}

/**
 * Main test function
 */
int main() {
    std::cout << "=== Chi-squared Distribution Unit Tests ===" << std::endl;
    std::cout << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testStatisticalProperties();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testEqualityOperators();
        testEdgeCases();
        testGammaRelationship();
        testLogProbabilities();
        testCDF();
        testEqualityAndIO();
        testCaching();
        testPerformance();
        
        std::cout << std::endl;
        std::cout << "🎉 All Chi-squared distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
