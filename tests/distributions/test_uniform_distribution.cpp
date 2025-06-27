#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/uniform_distribution.h"

using libhmm::UniformDistribution;
using libhmm::Observation;

/**
 * Test basic Uniform distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Uniform distribution functionality..." << std::endl;
    
    // Test default constructor
    UniformDistribution uniform;
    assert(uniform.getA() == 0.0);
    assert(uniform.getB() == 1.0);
    assert(uniform.getMin() == 0.0);  // Alternative getter
    assert(uniform.getMax() == 1.0);  // Alternative getter
    
    // Test parameterized constructor
    UniformDistribution uniform2(2.0, 5.0);
    assert(uniform2.getA() == 2.0);
    assert(uniform2.getB() == 5.0);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    UniformDistribution uniform(1.0, 4.0);  // Uniform on [1, 4]
    
    // Test that probability is zero outside the interval
    assert(uniform.getProbability(0.5) == 0.0);
    assert(uniform.getProbability(4.5) == 0.0);
    assert(uniform.getProbability(-1.0) == 0.0);
    
    // Test that probability is constant within the interval
    double expectedPdf = 1.0 / (4.0 - 1.0); // 1/3
    assert(std::abs(uniform.getProbability(1.5) - expectedPdf) < 1e-10);
    assert(std::abs(uniform.getProbability(2.0) - expectedPdf) < 1e-10);
    assert(std::abs(uniform.getProbability(3.5) - expectedPdf) < 1e-10);
    
    // Test boundary values
    assert(std::abs(uniform.getProbability(1.0) - expectedPdf) < 1e-10);
    assert(std::abs(uniform.getProbability(4.0) - expectedPdf) < 1e-10);
    
    // Test that probability density is reasonable
    assert(uniform.getProbability(2.0) > 0.0);
    assert(uniform.getProbability(2.0) < 1.0);
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    UniformDistribution uniform;
    
    // Test with known data
    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    uniform.fit(data);
    
    // After fitting, bounds should encompass all data with some padding
    assert(uniform.getA() <= 1.0);  // Should be at or below minimum
    assert(uniform.getB() >= 5.0);  // Should be at or above maximum
    assert(uniform.getA() < uniform.getB());  // Valid interval
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    uniform.fit(emptyData);
    assert(uniform.getA() == 0.0);
    assert(uniform.getB() == 1.0);
    
    // Test with single point (should reset to default for insufficient data)
    std::vector<Observation> singlePoint = {2.5};
    uniform.fit(singlePoint);
    assert(uniform.getA() == 0.0);
    assert(uniform.getB() == 1.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        UniformDistribution uniform(5.0, 2.0);  // a > b
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        UniformDistribution uniform(3.0, 3.0);  // a == b
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        UniformDistribution uniform(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        UniformDistribution uniform(1.0, inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    UniformDistribution uniform(1.0, 3.0);
    
    try {
        uniform.setA(4.0);  // Would make a > b
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        uniform.setB(0.5);  // Would make b < a
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
    
    UniformDistribution uniform(2.5, 7.5);
    std::string str = uniform.toString();
    
    // Should contain key information based on actual output format:
    // "Uniform Distribution:\n      a (lower bound) = 2.5\n      b (upper bound) = 7.5\n"
    assert(str.find("Uniform") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("7.5") != std::string::npos);
    assert(str.find("a") != std::string::npos);
    assert(str.find("lower bound") != std::string::npos);
    assert(str.find("b") != std::string::npos);
    assert(str.find("upper bound") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    UniformDistribution original(3.14, 6.28);
    
    // Test copy constructor
    UniformDistribution copied(original);
    assert(copied.getA() == original.getA());
    assert(copied.getB() == original.getB());
    
    // Test copy assignment
    UniformDistribution assigned;
    assigned = original;
    assert(assigned.getA() == original.getA());
    assert(assigned.getB() == original.getB());
    
    // Test move constructor
    UniformDistribution moved(std::move(original));
    assert(moved.getA() == 3.14);
    assert(moved.getB() == 6.28);
    
    // Test move assignment
    UniformDistribution moveAssigned;
    UniformDistribution temp(1.41, 2.73);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getA() == 1.41);
    assert(moveAssigned.getB() == 2.73);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    UniformDistribution uniform(0.0, 1.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(uniform.getProbability(nan_val) == 0.0);
    assert(uniform.getProbability(inf_val) == 0.0);
    assert(uniform.getProbability(neg_inf_val) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    UniformDistribution uniform(10.0, 20.0);
    uniform.reset();
    
    assert(uniform.getA() == 0.0);
    assert(uniform.getB() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test uniform distribution properties
 */
void testUniformProperties() {
    std::cout << "Testing uniform distribution properties..." << std::endl;
    
    // Test standard uniform [0, 1]
    UniformDistribution standard(0.0, 1.0);
    assert(std::abs(standard.getMean() - 0.5) < 1e-10);  // Mean = (0+1)/2 = 0.5
    assert(std::abs(standard.getVariance() - 1.0/12.0) < 1e-10);  // Variance = (1-0)²/12 = 1/12
    
    // Test general case [2, 8]
    UniformDistribution general(2.0, 8.0);
    double expectedMean = (2.0 + 8.0) / 2.0;  // 5.0
    double expectedVar = (8.0 - 2.0) * (8.0 - 2.0) / 12.0;  // 36/12 = 3.0
    assert(std::abs(general.getMean() - expectedMean) < 1e-10);
    assert(std::abs(general.getVariance() - expectedVar) < 1e-10);
    
    // Test standard deviation relationship
    assert(std::abs(general.getStandardDeviation() - std::sqrt(general.getVariance())) < 1e-10);
    
    // Test that probability is constant within interval
    double pdf1 = general.getProbability(3.0);
    double pdf2 = general.getProbability(5.0);
    double pdf3 = general.getProbability(7.0);
    assert(std::abs(pdf1 - pdf2) < 1e-10);
    assert(std::abs(pdf2 - pdf3) < 1e-10);
    
    std::cout << "✓ Uniform property tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    UniformDistribution uniform;
    
    // Test with NaN values
    std::vector<Observation> nanData = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    try {
        uniform.fit(nanData);
        assert(false);  // Should throw exception
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with infinity values
    std::vector<Observation> infData = {1.0, 2.0, std::numeric_limits<double>::infinity(), 3.0};
    try {
        uniform.fit(infData);
        assert(false);  // Should throw exception
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with valid data
    std::vector<Observation> validData = {1.5, 2.0, 3.5, 4.0, 2.5};
    try {
        uniform.fit(validData);
        // Should work fine, check that parameters are reasonable
        assert(uniform.getA() < uniform.getB());
        assert(uniform.getA() <= 1.5);  // Should encompass minimum
        assert(uniform.getB() >= 4.0);  // Should encompass maximum
    } catch (const std::exception&) {
        assert(false);  // Should not throw
    }
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test parameter setters
 */
void testParameterSetters() {
    std::cout << "Testing parameter setters..." << std::endl;
    
    UniformDistribution uniform(1.0, 5.0);
    
    // Test individual setters
    uniform.setA(0.5);
    assert(uniform.getA() == 0.5);
    assert(uniform.getB() == 5.0);
    
    uniform.setB(6.0);
    assert(uniform.getA() == 0.5);
    assert(uniform.getB() == 6.0);
    
    // Test setting both parameters
    uniform.setParameters(2.0, 8.0);
    assert(uniform.getA() == 2.0);
    assert(uniform.getB() == 8.0);
    
    std::cout << "✓ Parameter setter tests passed" << std::endl;
}

/**
 * Test edge cases
 */
void testEdgeCases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test with very small interval
    UniformDistribution tiny(0.0, 1e-10);
    assert(tiny.getProbability(5e-11) > 0.0);  // Should be within interval
    assert(tiny.getProbability(2e-10) == 0.0); // Should be outside interval
    
    // Test with large interval
    UniformDistribution large(-1e6, 1e6);
    double largePdf = large.getProbability(0.0);
    assert(largePdf > 0.0);
    assert(largePdf == 1.0 / (2e6));  // 1/(b-a)
    
    // Test with negative interval
    UniformDistribution negative(-5.0, -2.0);
    assert(negative.getProbability(-3.5) > 0.0);
    assert(negative.getProbability(0.0) == 0.0);
    assert(std::abs(negative.getMean() - (-3.5)) < 1e-10);  // Mean = (-5 + -2)/2 = -3.5
    
    // Test isApproximatelyEqual
    UniformDistribution u1(1.0, 3.0);
    UniformDistribution u2(1.000000001, 3.000000001);
    UniformDistribution u3(1.1, 3.1);
    
    assert(u1.isApproximatelyEqual(u2, 1e-8));
    assert(!u1.isApproximatelyEqual(u3, 1e-8));
    
    std::cout << "✓ Edge cases tests passed" << std::endl;
}

/**
 * Test fitting with identical data
 */
void testFittingIdenticalData() {
    std::cout << "Testing fitting with identical data..." << std::endl;
    
    UniformDistribution uniform;
    
    // Test with all identical values
    std::vector<Observation> identicalData = {5.0, 5.0, 5.0, 5.0};
    uniform.fit(identicalData);
    
    // Should create a small interval around the value
    assert(uniform.getA() < 5.0);
    assert(uniform.getB() > 5.0);
    assert(uniform.getA() < uniform.getB());
    
    // The value should be within the fitted interval
    assert(uniform.getProbability(5.0) > 0.0);
    
    std::cout << "✓ Fitting with identical data tests passed" << std::endl;
}

/**
 * Test log probability calculations
 */
void testLogProbabilities() {
    std::cout << "Testing log probability calculations..." << std::endl;
    
    UniformDistribution uniform(1.0, 4.0);  // Uniform on [1, 4]
    
    // Test that log probability is -infinity outside the interval
    assert(std::isinf(uniform.getLogProbability(0.5)) && uniform.getLogProbability(0.5) < 0);
    assert(std::isinf(uniform.getLogProbability(4.5)) && uniform.getLogProbability(4.5) < 0);
    assert(std::isinf(uniform.getLogProbability(-1.0)) && uniform.getLogProbability(-1.0) < 0);
    
    // Test that log probability is constant within the interval
    double expectedLogPdf = -std::log(4.0 - 1.0); // -log(3)
    assert(std::abs(uniform.getLogProbability(1.5) - expectedLogPdf) < 1e-10);
    assert(std::abs(uniform.getLogProbability(2.0) - expectedLogPdf) < 1e-10);
    assert(std::abs(uniform.getLogProbability(3.5) - expectedLogPdf) < 1e-10);
    
    // Test boundary values
    assert(std::abs(uniform.getLogProbability(1.0) - expectedLogPdf) < 1e-10);
    assert(std::abs(uniform.getLogProbability(4.0) - expectedLogPdf) < 1e-10);
    
    // Test consistency between log and regular probability
    double x = 2.5;
    assert(std::abs(std::log(uniform.getProbability(x)) - uniform.getLogProbability(x)) < 1e-10);
    
    std::cout << "✓ Log probability calculation tests passed" << std::endl;
}

/**
 * Test CDF calculations
 */
void testCDF() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    UniformDistribution uniform(2.0, 8.0);  // Uniform on [2, 8]
    
    // Test CDF values outside the interval
    assert(uniform.CDF(1.0) == 0.0);  // Below lower bound
    assert(uniform.CDF(9.0) == 1.0);  // Above upper bound
    
    // Test CDF at boundaries
    assert(uniform.CDF(2.0) == 0.0);  // At lower bound
    assert(uniform.CDF(8.0) == 1.0);  // At upper bound
    
    // Test CDF within the interval
    assert(std::abs(uniform.CDF(5.0) - 0.5) < 1e-10);  // Midpoint should be 0.5
    assert(std::abs(uniform.CDF(3.5) - 0.25) < 1e-10); // (3.5-2)/(8-2) = 0.25
    assert(std::abs(uniform.CDF(6.5) - 0.75) < 1e-10); // (6.5-2)/(8-2) = 0.75
    
    // Test CDF is monotonically increasing
    assert(uniform.CDF(3.0) < uniform.CDF(4.0));
    assert(uniform.CDF(4.0) < uniform.CDF(5.0));
    assert(uniform.CDF(5.0) < uniform.CDF(6.0));
    
    // Test with NaN and infinity
    assert(std::isnan(uniform.CDF(std::numeric_limits<double>::quiet_NaN())));
    assert(uniform.CDF(std::numeric_limits<double>::infinity()) == 1.0);
    assert(uniform.CDF(-std::numeric_limits<double>::infinity()) == 0.0);
    
    std::cout << "✓ CDF calculation tests passed" << std::endl;
}

/**
 * Test equality operators and I/O
 */
void testEqualityAndIO() {
    std::cout << "Testing equality operators and I/O..." << std::endl;
    
    UniformDistribution u1(1.5, 4.5);
    UniformDistribution u2(1.5, 4.5);
    UniformDistribution u3(2.0, 5.0);
    
    // Test equality operator
    assert(u1 == u2);
    assert(!(u1 == u3));
    
    // Test stream output
    std::ostringstream oss;
    oss << u1;
    std::string output = oss.str();
    assert(output.find("Uniform Distribution") != std::string::npos);
    assert(output.find("1.5") != std::string::npos);
    assert(output.find("4.5") != std::string::npos);
    
    std::cout << "Stream output: " << output << std::endl;
    
    // Test stream input (basic format check)
    std::istringstream iss("Uniform Distribution: a = 3.14 , b = 6.28");
    UniformDistribution inputDist;
    iss >> inputDist;
    
    if (iss.good()) {
        assert(std::abs(inputDist.getA() - 3.14) < 1e-10);
        assert(std::abs(inputDist.getB() - 6.28) < 1e-10);
    }
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

/**
 * Test caching mechanism
 */
void testCaching() {
    std::cout << "Testing caching mechanism..." << std::endl;
    
    UniformDistribution uniform(1.0, 5.0);
    
    // First call should update cache
    double pdf1 = uniform.getProbability(3.0);
    double logPdf1 = uniform.getLogProbability(3.0);
    
    // Subsequent calls should use cached values
    double pdf2 = uniform.getProbability(3.5);
    double logPdf2 = uniform.getLogProbability(3.5);
    
    // Should be identical for uniform distribution
    assert(std::abs(pdf1 - pdf2) < 1e-15);
    assert(std::abs(logPdf1 - logPdf2) < 1e-15);
    
    // Changing parameters should invalidate cache
    uniform.setA(0.5);
    double pdf3 = uniform.getProbability(3.0);
    assert(std::abs(pdf3 - pdf1) > 1e-10);  // Should be different now
    
    // Reset should also invalidate cache
    uniform.reset();
    double pdf4 = uniform.getProbability(0.5);
    assert(pdf4 == 1.0);  // Should be 1.0 for [0,1] interval
    
    std::cout << "✓ Caching mechanism tests passed" << std::endl;
}

/**
 * Test performance characteristics and optimizations
 */
void testPerformanceCharacteristics() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    UniformDistribution uniform(0.5, 5.5);
    
    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 100000;
    volatile double sum = 0.0;  // volatile to prevent optimization
    
    for (int i = 0; i < pdfIterations; ++i) {
        double x = 1.0 + static_cast<double>(i) / 20000.0;  // Values in [1, 6]
        sum += uniform.getProbability(x);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;
    
    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;
    
    for (int i = 0; i < pdfIterations; ++i) {
        double x = 1.0 + static_cast<double>(i) / 20000.0;  // Values in [1, 6]
        logSum += uniform.getLogProbability(x);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;
    
    // Test fitting timing
    std::vector<Observation> fitData(5000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i) / 1000.0;  // Values in [0, 5]
    }
    
    start = std::chrono::high_resolution_clock::now();
    uniform.fit(fitData);
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
    std::cout << "Running Uniform distribution tests..." << std::endl;
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
        testUniformProperties();
        testFittingValidation();
        testParameterSetters();
        testEdgeCases();
        testFittingIdenticalData();
        testLogProbabilities();
        testCDF();
        testEqualityAndIO();
        testCaching();
        testPerformanceCharacteristics();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "✅ All Uniform distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
