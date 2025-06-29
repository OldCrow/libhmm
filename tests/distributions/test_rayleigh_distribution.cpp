#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <climits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/rayleigh_distribution.h"
#include "libhmm/common/common.h"

using libhmm::RayleighDistribution;
using libhmm::Observation;

void testBasicFunctionality() {
    std::cout << "Testing basic Rayleigh distribution functionality..." << std::endl;
    
    // Test default constructor
    RayleighDistribution rayleigh;
    assert(rayleigh.getSigma() == 1.0);
    
    // Test parameterized constructor
    RayleighDistribution rayleigh2(2.5);
    assert(rayleigh2.getSigma() == 2.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    RayleighDistribution rayleigh(1.0);  // σ=1
    
    // Test that probability is zero for negative and zero values
    assert(rayleigh.getProbability(-0.1) == 0.0);
    assert(rayleigh.getProbability(0.0) == 0.0);
    
    // Test that probability is positive for positive values
    double prob1 = rayleigh.getProbability(0.5);
    double prob2 = rayleigh.getProbability(1.0);
    double prob3 = rayleigh.getProbability(2.0);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    // Test boundary value at x=0
    double probAt0 = rayleigh.getProbability(0.0);
    assert(probAt0 == 0.0);
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    RayleighDistribution rayleigh;
    
    // Test with known data
    std::vector<Observation> data = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    rayleigh.fit(data);
    assert(rayleigh.getSigma() > 0.0);
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    rayleigh.fit(emptyData);
    assert(rayleigh.getSigma() == 1.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        RayleighDistribution rayleigh(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        RayleighDistribution rayleigh(-1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        RayleighDistribution rayleigh(nan_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        RayleighDistribution rayleigh(inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "✓ Parameter validation tests passed" << std::endl;
}

void testStatistics() {
    std::cout << "Testing Rayleigh distribution statistics..." << std::endl;
    
    RayleighDistribution rayleigh(1.0);

    // Statistical properties
    double mean = rayleigh.getMean();
    double variance = rayleigh.getVariance();
    double stddev = rayleigh.getStandardDeviation();

    assert(mean > 1.25 && mean < 1.26);  // σ * √(π/2) ≈ 1.253
    assert(variance > 0.42 && variance < 0.43);  // σ² * (4-π)/2 ≈ 0.429
    assert(std::abs(stddev - std::sqrt(variance)) < 1e-10);

    std::cout << "✓ Statistics tests passed" << std::endl;
}

void testStatisticalMoments() {
    std::cout << "Testing statistical moments and properties..." << std::endl;
    
    RayleighDistribution rayleigh(1.0);
    
    double expectedMean = libhmm::constants::math::SQRT_PI_OVER_TWO;
    double expectedVar = libhmm::constants::math::FOUR_MINUS_PI_OVER_TWO;
    
    assert(std::abs(rayleigh.getMean() - expectedMean) < 1e-10);
    assert(std::abs(rayleigh.getVariance() - expectedVar) < 1e-10);
    assert(std::abs(rayleigh.getStandardDeviation() - std::sqrt(expectedVar)) < 1e-10);
    
    std::cout << "✓ Statistical moments tests passed" << std::endl;
}

void testStringRepresentation() {
    std::cout << "Testing string representation..." << std::endl;
    
    RayleighDistribution rayleigh(1.0);
    std::string str = rayleigh.toString();
    
    assert(str.find("Rayleigh") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("1.0") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    RayleighDistribution original(2.0);
    
    RayleighDistribution copied(original);
    assert(copied.getSigma() == original.getSigma());
    
    RayleighDistribution assigned;
    assigned = original;
    assert(assigned.getSigma() == original.getSigma());
    
    RayleighDistribution moved(std::move(original));
    assert(moved.getSigma() == 2.0);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

void testEqualityAndIO() {
    std::cout << "Testing equality and I/O operators..." << std::endl;
    
    RayleighDistribution r1(1.0);
    RayleighDistribution r2(1.0);
    RayleighDistribution r3(2.0);
    
    assert(r1 == r2);
    assert(r2 == r1);
    assert(!(r1 == r3));
    assert(r1 != r3);
    
    std::ostringstream oss;
    oss << r1.toString();
    std::string output = oss.str();
    assert(output.find("Rayleigh") != std::string::npos);
    assert(output.find("1.0") != std::string::npos);
    
    std::cout << "Stream output: " << output << std::endl;
    
    std::istringstream iss("Scale parameter = 3.14");
    RayleighDistribution inputDist;
    iss >> inputDist;
    if (!iss.fail()) {
        assert(std::abs(inputDist.getSigma() - 3.14) < 1e-10);
    }
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

void testCaching() {
    std::cout << "Testing caching mechanism..." << std::endl;
    
    RayleighDistribution rayleigh(1.0);
    
    double prob1 = rayleigh.getProbability(1.0);
    rayleigh.setSigma(2.0);
    double prob2 = rayleigh.getProbability(1.0);
    
    assert(prob1 != prob2);
    
    rayleigh.reset();  // Reset back to sigma=1.0
    double prob3 = rayleigh.getProbability(1.0);
    assert(std::abs(prob1 - prob3) < 1e-10);  // Should be the same as original
    
    // Test that cache invalidation works correctly
    rayleigh.setSigma(3.0);
    double prob4 = rayleigh.getProbability(1.0);
    assert(prob1 != prob4);
    assert(prob2 != prob4);
    
    std::cout << "✓ Caching mechanism tests passed" << std::endl;
}

void testCDFCalculations() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    RayleighDistribution rayleigh(1.0);
    
    // Test boundary values
    assert(rayleigh.getCumulativeProbability(-1.0) == 0.0);
    assert(rayleigh.getCumulativeProbability(0.0) == 0.0);
    
    // Test known values
    double cdf_at_sigma = rayleigh.getCumulativeProbability(1.0);  // CDF at x = σ
    double expected_cdf_at_sigma = 1.0 - std::exp(-0.5);  // 1 - exp(-1²/(2*1²))
    assert(std::abs(cdf_at_sigma - expected_cdf_at_sigma) < 1e-10);
    
    // Test monotonicity
    double cdf1 = rayleigh.getCumulativeProbability(0.5);
    double cdf2 = rayleigh.getCumulativeProbability(1.0);
    double cdf3 = rayleigh.getCumulativeProbability(2.0);
    assert(cdf1 < cdf2);
    assert(cdf2 < cdf3);
    
    // Test that CDF approaches 1 for large values
    double cdf_large = rayleigh.getCumulativeProbability(10.0);
    assert(cdf_large > 0.99);
    
    std::cout << "✓ CDF calculation tests passed" << std::endl;
}

void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;
    
    RayleighDistribution smallSigma(0.001);
    RayleighDistribution largeSigma(1000.0);
    
    double probSmall = smallSigma.getProbability(0.0001);
    double probLarge = largeSigma.getProbability(500.0);
    
    assert(probSmall > 0.0 && std::isfinite(probSmall));
    assert(probLarge > 0.0 && std::isfinite(probLarge));
    
    // Test CDF stability
    double cdfSmall = smallSigma.getCumulativeProbability(0.0001);
    double cdfLarge = largeSigma.getCumulativeProbability(500.0);
    
    assert(cdfSmall >= 0.0 && cdfSmall <= 1.0 && std::isfinite(cdfSmall));
    assert(cdfLarge >= 0.0 && cdfLarge <= 1.0 && std::isfinite(cdfLarge));
    
    std::cout << "✓ Numerical stability tests passed" << std::endl;
}

void testPerformanceCharacteristics() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    RayleighDistribution rayleigh(1.0);
    
    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 100000;
    volatile double sum = 0.0;  // volatile to prevent optimization
    
    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i + 1) / 10000.0;  // Range 0.0001 to 10
        sum += rayleigh.getProbability(x);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;
    
    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;
    
    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i + 1) / 10000.0;  // Range 0.0001 to 10
        logSum += rayleigh.getLogProbability(x);
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
    rayleigh.fit(fitData);
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

int main() {
    std::cout << "Running Rayleigh distribution tests..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testParameterValidation();
        testStatistics();
        testStatisticalMoments();
        testStringRepresentation();
        testCopyMoveSemantics();
        testEqualityAndIO();
        testCaching();
        testCDFCalculations();
        testNumericalStability();
        testPerformanceCharacteristics();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "✅ All Rayleigh distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
