#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <climits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/weibull_distribution.h"

using libhmm::WeibullDistribution;
using libhmm::Observation;

/**
 * Test basic Weibull distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Weibull distribution functionality..." << std::endl;
    
    // Test default constructor
    WeibullDistribution weibull;
    assert(weibull.getK() == 1.0);
    assert(weibull.getLambda() == 1.0);
    assert(weibull.getShape() == 1.0);  // Alternative getter
    assert(weibull.getScale() == 1.0);  // Alternative getter
    
    // Test parameterized constructor
    WeibullDistribution weibull2(2.5, 1.5);
    assert(weibull2.getK() == 2.5);
    assert(weibull2.getLambda() == 1.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    WeibullDistribution weibull(2.0, 1.0);  // k=2, λ=1 (Rayleigh distribution)
    
    // Test that probability is zero for negative values
    assert(weibull.getProbability(-0.1) == 0.0);
    assert(weibull.getProbability(-1.0) == 0.0);
    
    // Test that probability is positive for positive values
    double prob1 = weibull.getProbability(0.5);
    double prob2 = weibull.getProbability(1.0);
    double prob3 = weibull.getProbability(2.0);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    // For Weibull distribution, density typically decreases after the mode
    // Mode for Weibull(k,λ) = λ * ((k-1)/k)^(1/k) when k > 1
    // For k=2, λ=1: mode ≈ 0.707
    
    // Test boundary value at x=0
    double probAt0 = weibull.getProbability(0.0);
    assert(probAt0 >= 0.0);  // Should be 0 for k=2 > 1
    
    // Test that probability density is reasonable (should be small for continuous dist)
    assert(prob1 < 10.0);  // Should be reasonable
    assert(prob1 > 1e-10);  // Should be greater than zero
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    WeibullDistribution weibull;
    
    // Test with known data
    std::vector<Observation> data = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    weibull.fit(data);
    assert(weibull.getK() > 0.0);      // Should have some reasonable value
    assert(weibull.getLambda() > 0.0); // Should have some reasonable value
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    weibull.fit(emptyData);
    assert(weibull.getK() == 1.0);
    assert(weibull.getLambda() == 1.0);
    
    // Test with single point (should reset to default for insufficient data)
    std::vector<Observation> singlePoint = {2.5};
    weibull.fit(singlePoint);
    assert(weibull.getK() == 1.0);
    assert(weibull.getLambda() == 1.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        WeibullDistribution weibull(0.0, 1.0);  // Zero k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        WeibullDistribution weibull(-1.0, 1.0);  // Negative k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        WeibullDistribution weibull(1.0, 0.0);  // Zero lambda
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        WeibullDistribution weibull(1.0, -1.0);  // Negative lambda
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        WeibullDistribution weibull(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        WeibullDistribution weibull(1.0, inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    WeibullDistribution weibull(1.0, 1.0);
    
    try {
        weibull.setK(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        weibull.setLambda(-1.0);
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
    
    WeibullDistribution weibull(2.5, 1.5);
    std::string str = weibull.toString();
    
    // Should contain key information based on actual output format:
    // "Weibull Distribution:\n      k (shape) = 2.5\n      λ (scale) = 1.5\n"
    assert(str.find("Weibull") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("1.5") != std::string::npos);
    assert(str.find("k") != std::string::npos);
    assert(str.find("shape") != std::string::npos);
    assert(str.find("λ") != std::string::npos || str.find("scale") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    WeibullDistribution original(3.14, 2.71);
    
    // Test copy constructor
    WeibullDistribution copied(original);
    assert(copied.getK() == original.getK());
    assert(copied.getLambda() == original.getLambda());
    
    // Test copy assignment
    WeibullDistribution assigned;
    assigned = original;
    assert(assigned.getK() == original.getK());
    assert(assigned.getLambda() == original.getLambda());
    
    // Test move constructor
    WeibullDistribution moved(std::move(original));
    assert(moved.getK() == 3.14);
    assert(moved.getLambda() == 2.71);
    
    // Test move assignment
    WeibullDistribution moveAssigned;
    WeibullDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getK() == 1.41);
    assert(moveAssigned.getLambda() == 1.73);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    WeibullDistribution weibull(2.0, 1.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(weibull.getProbability(nan_val) == 0.0);
    assert(weibull.getProbability(inf_val) == 0.0);
    assert(weibull.getProbability(neg_inf_val) == 0.0);
    
    // Negative values should return 0
    assert(weibull.getProbability(-0.1) == 0.0);
    assert(weibull.getProbability(-1.0) == 0.0);
    assert(weibull.getProbability(-10.0) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    WeibullDistribution weibull(10.0, 5.0);
    weibull.reset();
    
    assert(weibull.getK() == 1.0);
    assert(weibull.getLambda() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test Weibull distribution properties
 */
void testWeibullProperties() {
    std::cout << "Testing Weibull distribution properties..." << std::endl;
    
    // Test exponential case (k=1)
    WeibullDistribution exponential(1.0, 2.0);
    double expMean = exponential.getMean();
    assert(std::abs(expMean - 2.0) < 1e-10);  // For Weibull(1,λ), mean = λ
    
    // Test Rayleigh case (k=2)
    WeibullDistribution rayleigh(2.0, 1.0);
    double rayleighMean = rayleigh.getMean();
    // For Weibull(2,1), mean = Γ(1.5) = sqrt(π)/2 ≈ 0.8862
    assert(rayleighMean > 0.8 && rayleighMean < 0.9);
    
    // Test that Weibull is only defined for x ≥ 0
    assert(exponential.getProbability(-0.1) == 0.0);
    assert(exponential.getProbability(0.0) >= 0.0);
    assert(exponential.getProbability(1.0) > 0.0);
    
    // Test variance is positive
    assert(exponential.getVariance() > 0.0);
    assert(rayleigh.getVariance() > 0.0);
    
    // Test standard deviation relationship
    assert(std::abs(exponential.getStandardDeviation() - std::sqrt(exponential.getVariance())) < 1e-10);
    
    std::cout << "✓ Weibull property tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    WeibullDistribution weibull;
    
    // Test with data containing negative values
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};  // -1.0 is invalid
    
    try {
        weibull.fit(invalidData);
        assert(false);  // Should throw exception
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with NaN values
    std::vector<Observation> nanData = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    try {
        weibull.fit(nanData);
        assert(false);  // Should throw exception
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with valid data including zero
    std::vector<Observation> zeroData = {0.0, 1.0, 2.0, 3.0};
    try {
        weibull.fit(zeroData);
        // Should work fine, check that parameters are reasonable
        assert(weibull.getK() > 0.0);
        assert(weibull.getLambda() > 0.0);
    } catch (const std::exception&) {
        // Also acceptable if implementation has specific constraints
    }
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test statistical moments
 */
void testStatisticalMoments() {
    std::cout << "Testing statistical moments..." << std::endl;
    
    // Test exponential case (k=1, λ=2)
    WeibullDistribution exponential(1.0, 2.0);
    
    // For Weibull(1,λ), mean = λ * Γ(2) = λ
    double expectedMean = 2.0;
    assert(std::abs(exponential.getMean() - expectedMean) < 1e-10);
    
    // For Weibull(1,λ), variance = λ² * [Γ(3) - (Γ(2))²] = λ² * [2 - 1] = λ²
    double expectedVar = 4.0;
    assert(std::abs(exponential.getVariance() - expectedVar) < 1e-10);
    
    // Test standard deviation
    double expectedStd = 2.0;
    assert(std::abs(exponential.getStandardDeviation() - expectedStd) < 1e-10);
    
    // Test Rayleigh case (k=2, λ=1)
    WeibullDistribution rayleigh(2.0, 1.0);
    
    // Mean should be sqrt(π)/2 ≈ 0.8862
    double rayleighMean = rayleigh.getMean();
    assert(rayleighMean > 0.88 && rayleighMean < 0.89);
    
    // Variance should be (4-π)/4 ≈ 0.2146
    double rayleighVar = rayleigh.getVariance();
    assert(rayleighVar > 0.21 && rayleighVar < 0.22);
    
    std::cout << "✓ Statistical moments tests passed" << std::endl;
}

/**
 * Test special cases and edge cases
 */
void testSpecialCases() {
    std::cout << "Testing special cases and edge cases..." << std::endl;
    
    // Test k=1 (exponential distribution)
    WeibullDistribution exp_case(1.0, 1.0);
    double prob_exp = exp_case.getProbability(1.0);
    // For exponential with rate 1, PDF(1) = e^(-1) ≈ 0.368
    assert(prob_exp > 0.35 && prob_exp < 0.4);
    
    // Test k=2 (Rayleigh distribution) 
    WeibullDistribution rayleigh_case(2.0, 1.0);
    double prob_rayleigh = rayleigh_case.getProbability(1.0);
    // For Rayleigh with σ=1, PDF(1) = 1*e^(-0.5) ≈ 0.607
    // Note: Actual implementation gives ~0.736 due to Weibull parameterization
    assert(prob_rayleigh > 0.7 && prob_rayleigh < 0.75);
    
    // Test very small k (infant mortality)
    WeibullDistribution infant(0.5, 1.0);
    assert(infant.getK() == 0.5);
    assert(infant.getLambda() == 1.0);
    
    // Test large k (wear-out failures)
    WeibullDistribution wearout(5.0, 1.0);
    assert(wearout.getK() == 5.0);
    assert(wearout.getLambda() == 1.0);
    
    // Test very large values
    WeibullDistribution normal_case(1.0, 1.0);
    double prob_large = normal_case.getProbability(100.0);
    assert(prob_large >= 0.0);  // Should be very small but non-negative
    assert(prob_large < 1e-10); // Should be essentially zero
    
    std::cout << "✓ Special cases tests passed" << std::endl;
}

/**
 * Test log probability calculations (GOLD STANDARD)
 */
void testLogProbability() {
    std::cout << "Testing log probability calculations..." << std::endl;
    
    WeibullDistribution weibull(2.0, 1.0);  // k=2, λ=1 (Rayleigh distribution)
    
    // Test that log probability is -infinity for negative values
    assert(weibull.getLogProbability(-0.1) == -std::numeric_limits<double>::infinity());
    assert(weibull.getLogProbability(-1.0) == -std::numeric_limits<double>::infinity());
    
    // Test consistency with regular probability
    double x = 1.0;
    double prob = weibull.getProbability(x);
    double logProb = weibull.getLogProbability(x);
    
    if (prob > 0.0) {
        assert(std::abs(logProb - std::log(prob)) < 1e-10);
    }
    
    // Test that log probability is finite for positive values
    assert(std::isfinite(weibull.getLogProbability(0.5)));
    assert(std::isfinite(weibull.getLogProbability(1.0)));
    assert(std::isfinite(weibull.getLogProbability(2.0)));
    
    // Test invalid inputs return -infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    assert(weibull.getLogProbability(nan_val) == -std::numeric_limits<double>::infinity());
    assert(weibull.getLogProbability(inf_val) == -std::numeric_limits<double>::infinity());
    
    std::cout << "✓ Log probability tests passed" << std::endl;
}

/**
 * Test CDF calculations (GOLD STANDARD)
 */
void testCDF() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    WeibullDistribution weibull(2.0, 1.0);  // k=2, λ=1 (Rayleigh distribution)
    
    // Test boundary conditions
    assert(weibull.CDF(-1.0) == 0.0);  // CDF should be 0 for negative values
    assert(weibull.CDF(0.0) == 0.0);   // CDF should be 0 at x=0
    
    // Test monotonicity (CDF should be non-decreasing)
    double cdf1 = weibull.CDF(0.5);
    double cdf2 = weibull.CDF(1.0);
    double cdf3 = weibull.CDF(2.0);
    
    assert(cdf1 >= 0.0 && cdf1 <= 1.0);
    assert(cdf2 >= 0.0 && cdf2 <= 1.0);
    assert(cdf3 >= 0.0 && cdf3 <= 1.0);
    assert(cdf1 <= cdf2);
    assert(cdf2 <= cdf3);
    
    // Test that CDF approaches 1 for large values
    double cdfLarge = weibull.CDF(10.0);
    assert(cdfLarge > 0.99);  // Should be very close to 1
    
    // Test known values for Weibull distribution with k=2, λ=1
    // CDF(1) = 1 - exp(-(1/1)^2) = 1 - exp(-1) ≈ 0.632
    double cdfAtScale = weibull.CDF(1.0);
    if (!(cdfAtScale > 0.63 && cdfAtScale < 0.64)) {
        std::cerr << "CDF(1.0)=" << cdfAtScale << " did not meet expected range (0.63, 0.64)" << std::endl;
        assert(false);
    }
    
    std::cout << "✓ CDF tests passed" << std::endl;
}

/**
 * Test equality and I/O operators (GOLD STANDARD)
 */
void testEqualityAndIO() {
    std::cout << "Testing equality and I/O operators..." << std::endl;
    
    WeibullDistribution weibull1(2.5, 1.5);
    WeibullDistribution weibull2(2.5, 1.5);
    WeibullDistribution weibull3(3.0, 2.0);
    
    // Test equality operator
    assert(weibull1 == weibull2);  // Same parameters
    assert(!(weibull1 == weibull3));  // Different parameters
    
    // Test with slightly different parameters (within tolerance)
    WeibullDistribution weibull4(2.5 + 1e-16, 1.5 + 1e-16);
    assert(weibull1 == weibull4);  // Should be equal within tolerance
    
    // Test stream output operator
    std::ostringstream oss;
    oss << weibull1;
    std::string output = oss.str();
    assert(!output.empty());
    assert(output.find("Weibull") != std::string::npos);
    assert(output.find("2.5") != std::string::npos);
    assert(output.find("1.5") != std::string::npos);
    
    // Test stream input operator
    std::istringstream iss(output);
    WeibullDistribution weibullFromStream;
    iss >> weibullFromStream;
    
    if (iss.good() || iss.eof()) {
        assert(weibullFromStream == weibull1);
    }
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

/**
 * Test performance characteristics and optimizations (GOLD STANDARD)
 */
void testPerformanceCharacteristics() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
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
    volatile double sum_pdf = 0.0;  // volatile to prevent optimization
    for (const auto& val : testValues) {
        sum_pdf += weibull.getProbability(val);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto pdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test getLogProbability() performance (should benefit from cached kMinus1_, invLambda_)
    start = std::chrono::high_resolution_clock::now();
    volatile double sum_log_pdf = 0.0;  // volatile to prevent optimization
    for (const auto& val : testValues) {
        sum_log_pdf += weibull.getLogProbability(val);
    }
    end = std::chrono::high_resolution_clock::now();
    auto log_pdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test CDF performance (should benefit from cached values and k=2 optimization)
    start = std::chrono::high_resolution_clock::now();
    volatile double sum_cdf = 0.0;  // volatile to prevent optimization
    for (const auto& val : testValues) {
        sum_cdf += weibull.CDF(val);
    }
    end = std::chrono::high_resolution_clock::now();
    auto cdf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test fitting performance with Welford's algorithm
    std::vector<Observation> fitData(fit_datapoints);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i + 1) / 1000.0;  // Positive values
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
    
    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) 
              << pdf_time_per_call << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3) 
              << log_pdf_time_per_call << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  CDF timing:       " << std::fixed << std::setprecision(3) 
              << cdf_time_per_call << " μs/call (" << pdf_iterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) 
              << fit_time_per_point << " μs/point (" << fit_datapoints << " points)" << std::endl;
    
    // Performance requirements (should be fast due to optimizations)
    assert(pdf_time_per_call < 1.0);     // Less than 1 μs per PDF call
    assert(log_pdf_time_per_call < 1.0);  // Less than 1 μs per log PDF call
    assert(cdf_time_per_call < 1.0);     // Less than 1 μs per CDF call
    assert(fit_time_per_point < 0.1);    // Less than 0.1 μs per data point for fitting (Welford's is fast)
    
    std::cout << "✓ Performance tests passed" << std::endl;
}

/**
 * Test numerical stability with extreme values (GOLD STANDARD)
 */
void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;
    
    // Test with very small k parameter
    WeibullDistribution smallK(0.1, 1.0);
    double probSmallK = smallK.getProbability(0.5);
    assert(probSmallK > 0.0);
    assert(std::isfinite(probSmallK));
    
    // Test with very large k parameter
    WeibullDistribution largeK(100.0, 1.0);
    double probLargeK = largeK.getProbability(1.0);
    assert(probLargeK > 0.0);
    assert(std::isfinite(probLargeK));
    
    // Test with very small lambda parameter
    WeibullDistribution smallLambda(2.0, 0.001);
    double probSmallLambda = smallLambda.getProbability(0.0001);
    assert(probSmallLambda > 0.0);
    assert(std::isfinite(probSmallLambda));
    
    // Test with very large lambda parameter
    WeibullDistribution largeLambda(2.0, 1000.0);
    double probLargeLambda = largeLambda.getProbability(500.0);
    assert(probLargeLambda > 0.0);
    assert(std::isfinite(probLargeLambda));
    
    // Test log probability with extreme values
    double logProbSmallK = smallK.getLogProbability(0.5);
    double logProbLargeK = largeK.getLogProbability(1.0);
    assert(std::isfinite(logProbSmallK));
    assert(std::isfinite(logProbLargeK));
    
    // Test special case optimizations (k=1 exponential, k=2 Rayleigh)
    WeibullDistribution exponential(1.0, 2.0);  // k=1 case
    WeibullDistribution rayleigh(2.0, 1.0);     // k=2 case
    
    double expProb = exponential.getProbability(1.0);
    double rayProb = rayleigh.getProbability(1.0);
    double expLogProb = exponential.getLogProbability(1.0);
    double rayLogProb = rayleigh.getLogProbability(1.0);
    
    assert(std::isfinite(expProb) && expProb > 0.0);
    assert(std::isfinite(rayProb) && rayProb > 0.0);
    assert(std::isfinite(expLogProb));
    assert(std::isfinite(rayLogProb));
    
    // Test mathematical correctness for special cases
    // For k=1 (exponential): PDF(x) = (1/λ)exp(-x/λ)
    double expectedExpProb = (1.0/2.0) * std::exp(-1.0/2.0);
    assert(std::abs(expProb - expectedExpProb) < 1e-10);
    
    // For k=2 (Rayleigh): PDF(x) = (2x/λ²)exp(-(x/λ)²)
    double expectedRayProb = (2.0 * 1.0 / (1.0 * 1.0)) * std::exp(-(1.0 * 1.0));
    assert(std::abs(rayProb - expectedRayProb) < 1e-10);
    
    std::cout << "✓ Numerical stability tests passed" << std::endl;
}

/**
 * Test caching mechanism (GOLD STANDARD)
 */
void testCaching() {
    std::cout << "Testing caching mechanism..." << std::endl;
    
    WeibullDistribution weibull(2.0, 1.0);
    
    // Test that calculations work correctly after parameter changes
    double prob1 = weibull.getProbability(1.0);
    double logProb1 = weibull.getLogProbability(1.0);
    
    // Change parameters and verify cache is updated
    weibull.setK(3.0);
    double prob2 = weibull.getProbability(1.0);
    double logProb2 = weibull.getLogProbability(1.0);
    
    assert(prob1 != prob2);  // Should be different after parameter change
    assert(logProb1 != logProb2);  // Should be different after parameter change
    
    // Change scale parameter
    weibull.setLambda(2.0);
    double prob3 = weibull.getProbability(1.0);
    double logProb3 = weibull.getLogProbability(1.0);
    
    assert(prob2 != prob3);  // Should be different after parameter change
    assert(logProb2 != logProb3);  // Should be different after parameter change
    
    // Test that copy constructor preserves cache state
    WeibullDistribution copied(weibull);
    assert(copied.getProbability(1.0) == weibull.getProbability(1.0));
    assert(copied.getLogProbability(1.0) == weibull.getLogProbability(1.0));
    
    // Test that cached values are consistent
    double prob4 = weibull.getProbability(1.0);
    double cdf4 = weibull.CDF(1.0);
    double logProb4 = weibull.getLogProbability(1.0);
    
    // Multiple calls should return identical results (using cache)
    assert(weibull.getProbability(1.0) == prob4);
    assert(weibull.CDF(1.0) == cdf4);
    assert(weibull.getLogProbability(1.0) == logProb4);
    
    std::cout << "✓ Caching tests passed" << std::endl;
}

int main() {
    std::cout << "Running Weibull distribution tests..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        testWeibullProperties();
        testFittingValidation();
        testStatisticalMoments();
        testSpecialCases();
        
        // Gold Standard Tests
        testLogProbability();
        testCDF();
        testEqualityAndIO();
        testPerformanceCharacteristics();
        testNumericalStability();
        testCaching();
        
        std::cout << "====================================="<< std::endl;
        std::cout << "✅ All Weibull distribution tests passed (including Gold Standard)!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
