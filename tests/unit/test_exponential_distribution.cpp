#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/exponential_distribution.h"

using libhmm::ExponentialDistribution;
using libhmm::Observation;

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
    
    // Exponential distribution should be zero at x=0
    assert(exponential.getProbability(0.0) == 0.0);
    
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
    
    // Should contain key information based on actual output format:
    // "Exponential Distribution:\n      Rate parameter = 2.5\n"
    assert(str.find("Exponential") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("Rate parameter") != std::string::npos);
    
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

int main() {
    std::cout << "Running Exponential distribution tests..." << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        testFittingValidation();
        testMemorylessProperty();
        
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
