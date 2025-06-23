#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
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
        
        std::cout << "=======================================" << std::endl;
        std::cout << "✅ All LogNormal distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
