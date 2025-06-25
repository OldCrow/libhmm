#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/gamma_distribution.h"

using libhmm::GammaDistribution;
using libhmm::Observation;

/**
 * Test basic Gamma distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Gamma distribution functionality..." << std::endl;
    
    // Test default constructor
    GammaDistribution gamma;
    assert(gamma.getK() == 1.0);
    assert(gamma.getTheta() == 1.0);
    
    // Test parameterized constructor
    GammaDistribution gamma2(2.5, 1.5);
    assert(gamma2.getK() == 2.5);
    assert(gamma2.getTheta() == 1.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    GammaDistribution gamma(2.0, 1.0);  // k=2, theta=1
    
    // Gamma distribution should be zero at x=0
    assert(gamma.getProbability(0.0) == 0.0);
    
    // Should be positive for positive values
    double prob1 = gamma.getProbability(1.0);
    double prob2 = gamma.getProbability(2.0);
    double prob3 = gamma.getProbability(3.0);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    // Should be zero for negative values
    assert(gamma.getProbability(-1.0) == 0.0);
    assert(gamma.getProbability(-0.5) == 0.0);
    
    // For Gamma(2,1), the mode is at k-1 = 1, so prob at 1 should be relatively high
    assert(prob1 > prob3);  // Probability should decrease away from mode
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    GammaDistribution gamma;
    
    // Test with known data
    std::vector<Observation> data = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    
    gamma.fit(data);
    // After fitting, parameters should be positive
    assert(gamma.getK() > 0.0);
    assert(gamma.getTheta() > 0.0);
    
    // Test with empty data (implementation sets to ZERO)
    std::vector<Observation> emptyData;
    gamma.fit(emptyData);
    assert(gamma.getK() == 1e-30);  // Implementation uses ZERO constant
    assert(gamma.getTheta() == 1e-30);
    
    // Test with single positive point
    std::vector<Observation> singlePoint = {2.5};
    gamma.fit(singlePoint);
    assert(gamma.getK() > 0.0);
    assert(gamma.getTheta() > 0.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        GammaDistribution gamma(0.0, 1.0);  // Zero k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GammaDistribution gamma(-1.0, 1.0);  // Negative k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GammaDistribution gamma(1.0, 0.0);  // Zero theta
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GammaDistribution gamma(1.0, -1.0);  // Negative theta
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        GammaDistribution gamma(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        GammaDistribution gamma(1.0, inf_val);
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
    
    GammaDistribution gamma(2.5, 1.5);
    std::string str = gamma.toString();
    
    // Should contain key information based on new format:
    // "Gamma Distribution:\n      k (shape parameter) = 2.5\n      θ (scale parameter) = 1.5\n      Mean = 3.75\n      Variance = 5.625\n"
    assert(str.find("Gamma") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("1.5") != std::string::npos);
    assert(str.find("shape parameter") != std::string::npos);
    assert(str.find("scale parameter") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    GammaDistribution original(3.14, 2.71);
    
    // Test copy constructor
    GammaDistribution copied(original);
    assert(copied.getK() == original.getK());
    assert(copied.getTheta() == original.getTheta());
    
    // Test copy assignment
    GammaDistribution assigned;
    assigned = original;
    assert(assigned.getK() == original.getK());
    assert(assigned.getTheta() == original.getTheta());
    
    // Test move constructor
    GammaDistribution moved(std::move(original));
    assert(moved.getK() == 3.14);
    assert(moved.getTheta() == 2.71);
    
    // Test move assignment
    GammaDistribution moveAssigned;
    GammaDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getK() == 1.41);
    assert(moveAssigned.getTheta() == 1.73);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    GammaDistribution gamma(2.0, 1.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(gamma.getProbability(nan_val) == 0.0);
    assert(gamma.getProbability(inf_val) == 0.0);
    assert(gamma.getProbability(neg_inf_val) == 0.0);
    
    // Negative values should return 0
    assert(gamma.getProbability(-1.0) == 0.0);
    assert(gamma.getProbability(-0.1) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    GammaDistribution gamma(10.0, 5.0);
    gamma.reset();
    
    assert(gamma.getK() == 1.0);
    assert(gamma.getTheta() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    GammaDistribution gamma;
    
    // Test with data containing negative values (should handle gracefully or throw)
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};
    
    // Gamma distribution should handle negative values in fitting
    // (typically by ignoring them or throwing an exception)
    try {
        gamma.fit(invalidData);
        // If it doesn't throw, the parameters should still be valid
        assert(gamma.getK() > 0.0);
        assert(gamma.getTheta() > 0.0);
    } catch (const std::exception&) {
        // It's also acceptable to throw for invalid data
    }
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

int main() {
    std::cout << "Running Gamma distribution tests..." << std::endl;
    std::cout << "===================================" << std::endl;
    
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
        
        std::cout << "===================================" << std::endl;
        std::cout << "✅ All Gamma distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
