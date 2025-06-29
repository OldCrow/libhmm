#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/log_normal_distribution.h"

using libhmm::LogNormalDistribution;
using libhmm::Observation;

/**
 * Test basic LogNormal distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic LogNormal distribution functionality..." << std::endl;
    
    // Test default constructor
    LogNormalDistribution lognormal;
    assert(lognormal.getMean() == 0.0);
    assert(lognormal.getStandardDeviation() == 1.0);
    
    // Test parameterized constructor
    LogNormalDistribution lognormal2(5.0, 2.5);
    assert(lognormal2.getMean() == 5.0);
    assert(lognormal2.getStandardDeviation() == 2.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    LogNormalDistribution lognormal(0.0, 1.0);
    
    // Test that probability is zero for non-positive values
    assert(lognormal.getProbability(0.0) == 0.0);
    assert(lognormal.getProbability(-1.0) == 0.0);
    assert(lognormal.getProbability(-0.5) == 0.0);
    
    // Test that probability is positive for positive values
    double prob1 = lognormal.getProbability(1.0);
    double prob2 = lognormal.getProbability(2.0);
    double prob3 = lognormal.getProbability(0.5);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    // Test that probability density is reasonable (should be small for continuous dist)
    assert(prob1 < 1.0);  // Should be less than 1 for probability density
    assert(prob1 > 1e-10);  // Should be greater than zero
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    LogNormalDistribution lognormal;
    
    // Test with known data
    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    lognormal.fit(data);
    assert(lognormal.getMean() > 0.0);  // Should have some reasonable value
    assert(lognormal.getStandardDeviation() > 0.0);
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    lognormal.fit(emptyData);
    assert(lognormal.getMean() == 0.0);
    assert(lognormal.getStandardDeviation() == 1.0);
    
    // Test with single point (should fit based on actual behavior)
    std::vector<Observation> singlePoint = {2.5};
    lognormal.fit(singlePoint);
    // Based on debug output: mean=0.916291, stddev=1e-30
    assert(lognormal.getMean() > 0.0);
    assert(lognormal.getStandardDeviation() > 0.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        LogNormalDistribution lognormal(0.0, 0.0);  // Zero std dev
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        LogNormalDistribution lognormal(0.0, -1.0);  // Negative std dev
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test invalid mean
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        LogNormalDistribution lognormal(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        LogNormalDistribution lognormal(inf_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    LogNormalDistribution lognormal(0.0, 1.0);
    
    try {
        lognormal.setMean(nan_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        lognormal.setStandardDeviation(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        lognormal.setStandardDeviation(-1.0);
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
    
    LogNormalDistribution lognormal(2.5, 1.5);
    std::string str = lognormal.toString();
    
    // Should contain key information based on new format:
    // "LogNormal Distribution:\n      μ (log mean) = 2.5\n      σ (log std. deviation) = 1.5\n      Mean = ...\n      Variance = ...\n"
    assert(str.find("LogNormal") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("1.5") != std::string::npos);
    assert(str.find("Mean") != std::string::npos);
    assert(str.find("log std. deviation") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    LogNormalDistribution original(3.14, 2.71);
    
    // Test copy constructor
    LogNormalDistribution copied(original);
    assert(copied.getMean() == original.getMean());
    assert(copied.getStandardDeviation() == original.getStandardDeviation());
    
    // Test copy assignment
    LogNormalDistribution assigned;
    assigned = original;
    assert(assigned.getMean() == original.getMean());
    assert(assigned.getStandardDeviation() == original.getStandardDeviation());
    
    // Test move constructor
    LogNormalDistribution moved(std::move(original));
    assert(moved.getMean() == 3.14);
    assert(moved.getStandardDeviation() == 2.71);
    
    // Test move assignment
    LogNormalDistribution moveAssigned;
    LogNormalDistribution temp(1.41, 1.73);
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
    
    LogNormalDistribution lognormal(0.0, 1.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(lognormal.getProbability(nan_val) == 0.0);
    assert(lognormal.getProbability(inf_val) == 0.0);
    assert(lognormal.getProbability(neg_inf_val) == 0.0);
    
    // Negative and zero values should return 0
    assert(lognormal.getProbability(-1.0) == 0.0);
    assert(lognormal.getProbability(0.0) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    LogNormalDistribution lognormal(10.0, 5.0);
    lognormal.reset();
    
    assert(lognormal.getMean() == 0.0);
    assert(lognormal.getStandardDeviation() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test relationship to normal distribution
 */
void testLogNormalProperties() {
    std::cout << "Testing LogNormal distribution properties..." << std::endl;
    
    LogNormalDistribution lognormal(0.0, 1.0);
    
    // Test that log-normal is only defined for positive values
    assert(lognormal.getProbability(0.0) == 0.0);
    assert(lognormal.getProbability(-1.0) == 0.0);
    
    // Test some positive values
    double prob1 = lognormal.getProbability(1.0);
    double prob2 = lognormal.getProbability(2.0);
    double prob3 = lognormal.getProbability(0.5);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    std::cout << "✓ LogNormal property tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    LogNormalDistribution lognormal;
    
    // Test with data containing negative values (should be ignored)
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};
    
    // Log-Normal distribution should handle negative values gracefully
    // by ignoring them (they're not in the support)
    lognormal.fit(invalidData);
    // Should have fitted to positive values {1.0, 2.0, 3.0}
    assert(lognormal.getMean() > 0.0);
    assert(lognormal.getStandardDeviation() > 0.0);
    
    // Test with zero values (should be ignored)
    std::vector<Observation> zeroData = {0.0, 1.0, 2.0};
    lognormal.fit(zeroData);
    // Should have fitted to positive values {1.0, 2.0}
    assert(lognormal.getMean() > 0.0);
    assert(lognormal.getStandardDeviation() > 0.0);
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test statistical moments
 */
void testStatisticalMoments() {
    std::cout << "Testing statistical moments..." << std::endl;
    
    LogNormalDistribution lognormal(1.0, 0.5);
    
    // For Log-Normal(μ=1.0, σ=0.5):
    // Distribution mean = exp(μ + σ²/2) = exp(1.0 + 0.25/2) = exp(1.125)
    double expectedDistMean = std::exp(1.0 + 0.25/2.0);
    assert(std::abs(lognormal.getDistributionMean() - expectedDistMean) < 1e-10);
    
    // Distribution variance = (exp(σ²) - 1) * exp(2μ + σ²)
    //                      = (exp(0.25) - 1) * exp(2.0 + 0.25)
    double expectedVar = (std::exp(0.25) - 1.0) * std::exp(2.0 + 0.25);
    assert(std::abs(lognormal.getVariance() - expectedVar) < 1e-10);
    
    // Test median = exp(μ) = exp(1.0) ≈ 2.718
    double expectedMedian = std::exp(1.0);
    assert(std::abs(lognormal.getMedian() - expectedMedian) < 1e-10);
    
    // Test mode = exp(μ - σ²) = exp(1.0 - 0.25) = exp(0.75)
    double expectedMode = std::exp(1.0 - 0.25);
    assert(std::abs(lognormal.getMode() - expectedMode) < 1e-10);
    
    std::cout << "✓ Statistical moments tests passed" << std::endl;
}

/**
 * Test log probability calculations (Gold Standard)
 */
void testLogProbability() {
    std::cout << "Testing log probability calculations..." << std::endl;
    
    LogNormalDistribution lognormal(0.0, 1.0);  // Standard log-normal
    
    // Test log probability at several points
    double x1 = 1.0;
    double x2 = 2.0;
    double x3 = 0.5;
    
    double logP1 = lognormal.getLogProbability(x1);
    double logP2 = lognormal.getLogProbability(x2);
    double logP3 = lognormal.getLogProbability(x3);
    
    assert(std::isfinite(logP1));
    assert(std::isfinite(logP2));
    assert(std::isfinite(logP3));
    
    // For Log-Normal(0,1): log(f(x)) = -ln(x) - ln(√(2π)) - ½(ln(x))²
    double expectedLogP1 = -std::log(x1) - 0.5*std::log(2.0*M_PI) - 0.5*std::pow(std::log(x1), 2);
    double expectedLogP2 = -std::log(x2) - 0.5*std::log(2.0*M_PI) - 0.5*std::pow(std::log(x2), 2);
    double expectedLogP3 = -std::log(x3) - 0.5*std::log(2.0*M_PI) - 0.5*std::pow(std::log(x3), 2);
    
    assert(std::abs(logP1 - expectedLogP1) < 1e-10);
    assert(std::abs(logP2 - expectedLogP2) < 1e-10);
    assert(std::abs(logP3 - expectedLogP3) < 1e-10);
    
    // Test invalid inputs return -infinity
    assert(lognormal.getLogProbability(-0.1) == -std::numeric_limits<double>::infinity());
    assert(lognormal.getLogProbability(0.0) == -std::numeric_limits<double>::infinity());
    assert(std::isnan(lognormal.getLogProbability(std::numeric_limits<double>::quiet_NaN())) || 
           lognormal.getLogProbability(std::numeric_limits<double>::quiet_NaN()) == -std::numeric_limits<double>::infinity());
    
    std::cout << "✓ Log probability tests passed" << std::endl;
}

/**
 * Test CDF calculations (Gold Standard)
 */
void testCDFCalculations() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    LogNormalDistribution lognormal(0.0, 1.0);
    
    // Test boundary values
    assert(lognormal.getCumulativeProbability(-0.1) == 0.0);
    assert(lognormal.getCumulativeProbability(0.0) == 0.0);
    
    // Test monotonicity
    double cdf1 = lognormal.getCumulativeProbability(0.5);
    double cdf2 = lognormal.getCumulativeProbability(1.0);
    double cdf3 = lognormal.getCumulativeProbability(2.0);
    assert(cdf1 < cdf2);
    assert(cdf2 < cdf3);
    
    // Test that CDF values are in [0,1]
    assert(cdf1 >= 0.0 && cdf1 <= 1.0);
    assert(cdf2 >= 0.0 && cdf2 <= 1.0);
    assert(cdf3 >= 0.0 && cdf3 <= 1.0);
    
    // Test known value: for Log-Normal(0,1), CDF(1) = 0.5 (median)
    double cdfAt1 = lognormal.getCumulativeProbability(1.0);
    assert(std::abs(cdfAt1 - 0.5) < 1e-6);
    
    // Test approach to 1 for large values
    double cdfLarge = lognormal.getCumulativeProbability(100.0);
    assert(cdfLarge > 0.99);
    
    std::cout << "✓ CDF calculation tests passed" << std::endl;
}

/**
 * Test equality and I/O operators (Gold Standard)
 */
void testEqualityAndIO() {
    std::cout << "Testing equality and I/O operators..." << std::endl;
    
    LogNormalDistribution ln1(2.0, 1.5);
    LogNormalDistribution ln2(2.0, 1.5);
    LogNormalDistribution ln3(3.0, 1.5);
    
    assert(ln1 == ln2);
    assert(ln2 == ln1);
    assert(!(ln1 == ln3));
    assert(ln1 != ln3);
    
    std::ostringstream oss;
    oss << ln1;
    std::string output = oss.str();
    assert(output.find("LogNormal Distribution") != std::string::npos);
    // Check for mean and standard deviation values in the output format
    assert(output.find("Mean") != std::string::npos);
    assert(output.find("Standard Deviation") != std::string::npos);
    
    // Test stream input operator
    std::istringstream iss(output);
    LogNormalDistribution inputDist;
    iss >> inputDist;
    
    if (iss.good() || iss.eof()) {
        assert(inputDist == ln1);
    }
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

/**
 * Test numerical stability (Gold Standard)
 */
void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;
    
    // Test extreme parameter values
    LogNormalDistribution smallSigma(0.0, 0.1);
    LogNormalDistribution largeSigma(0.0, 5.0);
    LogNormalDistribution largeMu(10.0, 1.0);
    
    double probSmall = smallSigma.getProbability(1.0);
    double probLarge = largeSigma.getProbability(1.0);
    double probLargeMu = largeMu.getProbability(1000.0);
    
    assert(probSmall > 0.0 && std::isfinite(probSmall));
    assert(probLarge > 0.0 && std::isfinite(probLarge));
    assert(probLargeMu > 0.0 && std::isfinite(probLargeMu));
    
    // Test log probability with extreme values
    double logProbSmall = smallSigma.getLogProbability(1.0);
    double logProbLarge = largeSigma.getLogProbability(1.0);
    assert(std::isfinite(logProbSmall));
    assert(std::isfinite(logProbLarge));
    
    // Test CDF stability
    double cdfSmall = smallSigma.getCumulativeProbability(1.0);
    double cdfLarge = largeSigma.getCumulativeProbability(1.0);
    assert(cdfSmall >= 0.0 && cdfSmall <= 1.0 && std::isfinite(cdfSmall));
    assert(cdfLarge >= 0.0 && cdfLarge <= 1.0 && std::isfinite(cdfLarge));
    
    std::cout << "✓ Numerical stability tests passed" << std::endl;
}

/**
 * Test performance characteristics (Gold Standard)
 */
void testPerformanceCharacteristics() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    LogNormalDistribution lognormal(1.0, 0.5);
    const int iterations = 10000;
    std::vector<double> testValues;
    testValues.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        double t = 0.1 + static_cast<double>(i + 1) / 100.0;  // Start from 0.1
        testValues.push_back(t);
    }
    
    // Test PDF performance
    auto start = std::chrono::high_resolution_clock::now();
    volatile double sum_pdf = 0.0;
    for (const auto& val : testValues) {
        sum_pdf += lognormal.getProbability(val);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto pdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdf_time_per_call = static_cast<double>(pdf_duration.count()) / iterations;
    
    // Test log PDF performance
    start = std::chrono::high_resolution_clock::now();
    volatile double sum_logpdf = 0.0;
    for (const auto& val : testValues) {
        sum_logpdf += lognormal.getLogProbability(val);
    }
    end = std::chrono::high_resolution_clock::now();
    auto logpdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logpdf_time_per_call = static_cast<double>(logpdf_duration.count()) / iterations;
    
    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = 0.1 + static_cast<double>(i) / 100.0;
    }
    
    start = std::chrono::high_resolution_clock::now();
    lognormal.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fitDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double fitTimePerPoint = static_cast<double>(fitDuration.count()) / fitData.size();
    
    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) 
              << pdf_time_per_call << " μs/call (" << iterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3) 
              << logpdf_time_per_call << " μs/call (" << iterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) 
              << fitTimePerPoint << " μs/point (" << fitData.size() << " points)" << std::endl;
    
    // Performance requirements
    assert(pdf_time_per_call < 5.0);      // Less than 5 μs per PDF call
    assert(logpdf_time_per_call < 3.0);   // Less than 3 μs per log PDF call
    assert(fitTimePerPoint < 50.0);       // Less than 50 μs per data point for fitting
    
    std::cout << "✓ Performance tests passed" << std::endl;
}

/**
 * Test caching mechanism (Gold Standard)
 */
void testCaching() {
    std::cout << "Testing caching mechanism..." << std::endl;
    
    LogNormalDistribution lognormal(1.0, 0.5);
    
    // Test that calculations work correctly after parameter changes
    double prob1 = lognormal.getProbability(2.0);
    double logProb1 = lognormal.getLogProbability(2.0);
    
    // Change parameters and verify cache is updated
    lognormal.setMean(2.0);
    double prob2 = lognormal.getProbability(2.0);
    double logProb2 = lognormal.getLogProbability(2.0);
    
    assert(prob1 != prob2);  // Should be different after parameter change
    assert(logProb1 != logProb2);  // Should be different after parameter change
    
    // Change standard deviation parameter
    lognormal.setStandardDeviation(1.0);
    double prob3 = lognormal.getProbability(2.0);
    double logProb3 = lognormal.getLogProbability(2.0);
    
    assert(prob2 != prob3);  // Should be different after parameter change
    assert(logProb2 != logProb3);  // Should be different after parameter change
    
    // Test that copy constructor preserves cache state
    LogNormalDistribution copied(lognormal);
    assert(copied.getProbability(2.0) == lognormal.getProbability(2.0));
    assert(copied.getLogProbability(2.0) == lognormal.getLogProbability(2.0));
    
    // Test that cached values are consistent
    double prob4 = lognormal.getProbability(2.0);
    double cdf4 = lognormal.getCumulativeProbability(2.0);
    double logProb4 = lognormal.getLogProbability(2.0);
    
    // Multiple calls should return identical results (using cache)
    assert(lognormal.getProbability(2.0) == prob4);
    assert(lognormal.getCumulativeProbability(2.0) == cdf4);
    assert(lognormal.getLogProbability(2.0) == logProb4);
    
    std::cout << "✓ Caching tests passed" << std::endl;
}

int main() {
    std::cout << "Running LogNormal distribution tests..." << std::endl;
    std::cout << "=======================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        testLogNormalProperties();
        testFittingValidation();
        testStatisticalMoments();
        
        // Gold Standard Tests
        testLogProbability();
        testCDFCalculations();
        testEqualityAndIO();
        testNumericalStability();
        testPerformanceCharacteristics();
        testCaching();
        
        std::cout << "=======================================" << std::endl;
        std::cout << "✅ All LogNormal distribution tests passed (including Gold Standard)!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
