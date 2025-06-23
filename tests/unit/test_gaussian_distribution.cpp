#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/gaussian_distribution.h"

using libhmm::GaussianDistribution;
using libhmm::Observation;

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
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GaussianDistribution gaussian(0.0, -1.0);  // Negative std dev
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test invalid mean
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        GaussianDistribution gaussian(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GaussianDistribution gaussian(inf_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    GaussianDistribution gaussian(0.0, 1.0);
    
    try {
        gaussian.setMean(nan_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        gaussian.setStandardDeviation(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        gaussian.setStandardDeviation(-1.0);
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

int main() {
    std::cout << "Running Gaussian distribution tests..." << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        
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
