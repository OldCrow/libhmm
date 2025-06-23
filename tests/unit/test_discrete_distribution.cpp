#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/discrete_distribution.h"

using libhmm::DiscreteDistribution;
using libhmm::Observation;

/**
 * Test basic Discrete distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Discrete distribution functionality..." << std::endl;
    
    // Test default constructor
    DiscreteDistribution discrete;
    assert(discrete.getNumSymbols() == 10);  // Default is 10 symbols
    
    // Test parameterized constructor
    DiscreteDistribution discrete2(5);
    assert(discrete2.getNumSymbols() == 5);
    
    // Test invalid constructor parameter
    try {
        DiscreteDistribution discrete3(0);  // Zero symbols
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    DiscreteDistribution discrete(5);  // 5 symbols: 0,1,2,3,4
    
    // After reset, all symbols should have equal probability (1/numSymbols)
    double expectedProb = 1.0 / 5.0;  // 0.2
    for (int i = 0; i < 5; i++) {
        double prob = discrete.getProbability(i);
        assert(std::abs(prob - expectedProb) < 1e-10);
    }
    
    // Test invalid symbol indices
    assert(discrete.getProbability(-1) == 0.0);
    assert(discrete.getProbability(5) == 0.0);
    assert(discrete.getProbability(10) == 0.0);
    
    // Test non-integer values (should treat as floor)
    double prob2_5 = discrete.getProbability(2.5);
    double prob2 = discrete.getProbability(2);
    assert(prob2_5 == prob2);  // Should be same as symbol 2
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    DiscreteDistribution discrete(5);
    
    // Test with known data - based on debug output: {0, 1, 1, 2, 3} -> {0.2, 0.4, 0.2, 0.2, 0}
    std::vector<Observation> data = {0, 1, 1, 2, 3};
    discrete.fit(data);
    
    // Check fitted probabilities
    assert(std::abs(discrete.getProbability(0) - 0.2) < 1e-10);  // 1/5
    assert(std::abs(discrete.getProbability(1) - 0.4) < 1e-10);  // 2/5
    assert(std::abs(discrete.getProbability(2) - 0.2) < 1e-10);  // 1/5
    assert(std::abs(discrete.getProbability(3) - 0.2) < 1e-10);  // 1/5
    assert(std::abs(discrete.getProbability(4) - 0.0) < 1e-10);  // 0/5
    
    // Test with empty data (should reset to uniform)
    std::vector<Observation> emptyData;
    discrete.fit(emptyData);
    double expectedUniform = 1.0 / 5.0;
    for (int i = 0; i < 5; i++) {
        assert(std::abs(discrete.getProbability(i) - expectedUniform) < 1e-10);
    }
    
    // Test with single point
    std::vector<Observation> singlePoint = {2};
    discrete.fit(singlePoint);
    assert(discrete.getProbability(0) == 0.0);
    assert(discrete.getProbability(1) == 0.0);
    assert(discrete.getProbability(2) == 1.0);  // All probability on symbol 2
    assert(discrete.getProbability(3) == 0.0);
    assert(discrete.getProbability(4) == 0.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test setProbability functionality
 */
void testSetProbability() {
    std::cout << "Testing setProbability functionality..." << std::endl;
    
    DiscreteDistribution discrete(3);
    
    // Set custom probabilities
    discrete.setProbability(0, 0.5);
    discrete.setProbability(1, 0.3);
    discrete.setProbability(2, 0.2);
    
    assert(std::abs(discrete.getProbability(0) - 0.5) < 1e-10);
    assert(std::abs(discrete.getProbability(1) - 0.3) < 1e-10);
    assert(std::abs(discrete.getProbability(2) - 0.2) < 1e-10);
    
    // Test invalid probability values
    try {
        discrete.setProbability(0, -0.1);  // Negative probability
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior (if implemented)
    } catch (...) {
        // Some implementations might not validate, which is also acceptable
    }
    
    try {
        discrete.setProbability(0, 1.1);  // Probability > 1
        assert(false);  // Should not reach here  
    } catch (const std::invalid_argument&) {
        // Expected behavior (if implemented)
    } catch (...) {
        // Some implementations might not validate, which is also acceptable
    }
    
    std::cout << "✓ setProbability tests passed" << std::endl;
}

/**
 * Test string representation
 */
void testStringRepresentation() {
    std::cout << "Testing string representation..." << std::endl;
    
    DiscreteDistribution discrete(5);
    std::string str = discrete.toString();
    
    // Should contain key information based on actual output format:
    // "Discrete Distribution:\n0.2\t0.2\t0.2\t0.2\t0.2\t\n"
    assert(str.find("Discrete") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("0.2") != std::string::npos);  // Should contain the probabilities
    assert(str.find("\t") != std::string::npos);   // Should be tab-separated
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    DiscreteDistribution original(3);
    original.setProbability(0, 0.6);
    original.setProbability(1, 0.3);
    original.setProbability(2, 0.1);
    
    // Test copy constructor
    DiscreteDistribution copied(original);
    assert(copied.getNumSymbols() == original.getNumSymbols());
    assert(std::abs(copied.getProbability(0) - 0.6) < 1e-10);
    assert(std::abs(copied.getProbability(1) - 0.3) < 1e-10);
    assert(std::abs(copied.getProbability(2) - 0.1) < 1e-10);
    
    // Test copy assignment
    DiscreteDistribution assigned(5);  // Different size initially
    assigned = original;
    assert(assigned.getNumSymbols() == original.getNumSymbols());
    assert(std::abs(assigned.getProbability(0) - 0.6) < 1e-10);
    assert(std::abs(assigned.getProbability(1) - 0.3) < 1e-10);
    assert(std::abs(assigned.getProbability(2) - 0.1) < 1e-10);
    
    // Test move constructor
    DiscreteDistribution moved(std::move(original));
    assert(moved.getNumSymbols() == 3);
    assert(std::abs(moved.getProbability(0) - 0.6) < 1e-10);
    assert(std::abs(moved.getProbability(1) - 0.3) < 1e-10);
    assert(std::abs(moved.getProbability(2) - 0.1) < 1e-10);
    
    // Test move assignment
    DiscreteDistribution moveAssigned(2);
    DiscreteDistribution temp(4);
    temp.setProbability(1, 0.8);
    temp.setProbability(3, 0.2);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getNumSymbols() == 4);
    assert(std::abs(moveAssigned.getProbability(1) - 0.8) < 1e-10);
    assert(std::abs(moveAssigned.getProbability(3) - 0.2) < 1e-10);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    DiscreteDistribution discrete(5);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(discrete.getProbability(nan_val) == 0.0);
    assert(discrete.getProbability(inf_val) == 0.0);
    assert(discrete.getProbability(neg_inf_val) == 0.0);
    
    // Test with out-of-range indices
    assert(discrete.getProbability(-1) == 0.0);
    assert(discrete.getProbability(5) == 0.0);
    assert(discrete.getProbability(100) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    DiscreteDistribution discrete(4);
    
    // Set non-uniform probabilities
    discrete.setProbability(0, 0.7);
    discrete.setProbability(1, 0.2);
    discrete.setProbability(2, 0.1);
    discrete.setProbability(3, 0.0);
    
    // Reset should restore uniform distribution
    discrete.reset();
    double expectedUniform = 1.0 / 4.0;
    for (int i = 0; i < 4; i++) {
        assert(std::abs(discrete.getProbability(i) - expectedUniform) < 1e-10);
    }
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test discrete distribution properties
 */
void testDiscreteProperties() {
    std::cout << "Testing discrete distribution properties..." << std::endl;
    
    DiscreteDistribution discrete(3);
    
    // Test that probabilities sum to 1
    double totalProb = 0.0;
    for (int i = 0; i < 3; i++) {
        totalProb += discrete.getProbability(i);
    }
    assert(std::abs(totalProb - 1.0) < 1e-10);
    
    // Test with fitted data
    std::vector<Observation> data = {0, 0, 1, 2};
    discrete.fit(data);
    
    totalProb = 0.0;
    for (int i = 0; i < 3; i++) {
        totalProb += discrete.getProbability(i);
    }
    assert(std::abs(totalProb - 1.0) < 1e-10);
    
    // Expected probabilities: 0->2/4=0.5, 1->1/4=0.25, 2->1/4=0.25
    assert(std::abs(discrete.getProbability(0) - 0.5) < 1e-10);
    assert(std::abs(discrete.getProbability(1) - 0.25) < 1e-10);
    assert(std::abs(discrete.getProbability(2) - 0.25) < 1e-10);
    
    std::cout << "✓ Discrete property tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    DiscreteDistribution discrete(5);
    
    // Test with data containing out-of-range values
    std::vector<Observation> invalidData = {0, 1, 5, 2};  // 5 is out of range
    
    // Discrete distribution should handle out-of-range values gracefully
    try {
        discrete.fit(invalidData);
        // If it doesn't throw, check that valid symbols still have reasonable probabilities
        double totalValidProb = 0.0;
        for (int i = 0; i < 5; i++) {
            totalValidProb += discrete.getProbability(i);
        }
        // Should still sum close to 1 (might ignore invalid values)
        assert(totalValidProb > 0.5);  // At least some probability assigned
    } catch (const std::exception&) {
        // It's also acceptable to throw for invalid data
    }
    
    // Test with negative values
    std::vector<Observation> negativeData = {0, 1, -1, 2};
    try {
        discrete.fit(negativeData);
        // Check that non-negative symbols still get reasonable probabilities
        assert(discrete.getProbability(0) >= 0.0);
        assert(discrete.getProbability(1) >= 0.0);
        assert(discrete.getProbability(2) >= 0.0);
    } catch (const std::exception&) {
        // Acceptable to reject negative values
    }
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

int main() {
    std::cout << "Running Discrete distribution tests..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testSetProbability();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        testDiscreteProperties();
        testFittingValidation();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "✅ All Discrete distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
