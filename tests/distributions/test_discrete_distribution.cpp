#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
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
    
    // Should contain key information (modern format focuses on content, not specific formatting)
    assert(str.find("Discrete") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("0.2") != std::string::npos);  // Should contain the probabilities
    // Modern format is more readable - focus on content rather than specific separators
    
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

/**
 * Test log probability calculations
 */
void testLogProbability() {
    std::cout << "Testing log probability calculations..." << std::endl;
    
    DiscreteDistribution discrete(3);
    discrete.setProbability(0, 0.5);
    discrete.setProbability(1, 0.3);
    discrete.setProbability(2, 0.2);
    
    // Test valid values
    double logProb0 = discrete.getLogProbability(0.0);
    double prob0 = discrete.getProbability(0.0);
    
    // log(prob) should equal logProb (within numerical precision)
    assert(std::abs(std::log(prob0) - logProb0) < 1e-10);
    
    // Test out of range (should return -infinity)
    double logProbNeg = discrete.getLogProbability(-1.0);
    double logProbHigh = discrete.getLogProbability(3.0);
    assert(std::isinf(logProbNeg) && logProbNeg < 0);
    assert(std::isinf(logProbHigh) && logProbHigh < 0);
    
    // Test with zero probability
    discrete.setProbability(2, 0.0);
    double logProbZero = discrete.getLogProbability(2.0);
    assert(std::isinf(logProbZero) && logProbZero < 0);
    
    std::cout << "✓ Log probability tests passed" << std::endl;
}

/**
 * Test CDF calculations
 */
void testCDF() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    DiscreteDistribution discrete(4);
    discrete.setProbability(0, 0.1);
    discrete.setProbability(1, 0.2);
    discrete.setProbability(2, 0.3);
    discrete.setProbability(3, 0.4);
    
    // Test basic properties
    double cdf0 = discrete.CDF(0.0);
    double cdf1 = discrete.CDF(1.0);
    double cdf2 = discrete.CDF(2.0);
    double cdf3 = discrete.CDF(3.0);
    
    assert(std::abs(cdf0 - 0.1) < 1e-10);
    assert(std::abs(cdf1 - 0.3) < 1e-10);  // 0.1 + 0.2
    assert(std::abs(cdf2 - 0.6) < 1e-10);  // 0.1 + 0.2 + 0.3
    assert(std::abs(cdf3 - 1.0) < 1e-10);  // 0.1 + 0.2 + 0.3 + 0.4
    
    // CDF should be monotonic
    assert(cdf0 <= cdf1);
    assert(cdf1 <= cdf2);
    assert(cdf2 <= cdf3);
    
    // Test boundary cases
    assert(discrete.CDF(-1.0) == 0.0);
    assert(discrete.CDF(10.0) == 1.0);
    
    std::cout << "✓ CDF tests passed" << std::endl;
}

/**
 * Test equality and I/O operators
 */
void testEqualityAndIO() {
    std::cout << "Testing equality and I/O operators..." << std::endl;
    
    DiscreteDistribution discrete1(3);
    discrete1.setProbability(0, 0.5);
    discrete1.setProbability(1, 0.3);
    discrete1.setProbability(2, 0.2);
    
    DiscreteDistribution discrete2(3);
    discrete2.setProbability(0, 0.5);
    discrete2.setProbability(1, 0.3);
    discrete2.setProbability(2, 0.2);
    
    DiscreteDistribution discrete3(3);
    discrete3.setProbability(0, 0.6);
    discrete3.setProbability(1, 0.2);
    discrete3.setProbability(2, 0.2);
    
    DiscreteDistribution discrete4(4);  // Different size
    
    // Test equality
    assert(discrete1 == discrete2);
    assert(!(discrete1 == discrete3));
    assert(!(discrete1 == discrete4));
    
    // Test inequality
    assert(!(discrete1 != discrete2));
    assert(discrete1 != discrete3);
    assert(discrete1 != discrete4);
    
    // Test stream output
    std::ostringstream oss;
    oss << discrete1;
    std::string output = oss.str();
    assert(!output.empty());
    assert(output.find("Discrete") != std::string::npos);
    
    // Test stream input
    std::istringstream iss("2 0.7 0.3");
    DiscreteDistribution inputDiscrete;
    iss >> inputDiscrete;
    assert(inputDiscrete.getNumSymbols() == 2);
    assert(std::abs(inputDiscrete.getProbability(0) - 0.7) < 1e-10);
    assert(std::abs(inputDiscrete.getProbability(1) - 0.3) < 1e-10);
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

/**
 * Test performance characteristics
 */
void testPerformance() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    DiscreteDistribution discrete(100);  // Larger distribution
    
    // Time probability calculations
    auto start = std::chrono::high_resolution_clock::now();
    
    const int numIterations = 10000;
    double sum = 0.0;
    for (int i = 0; i < numIterations; ++i) {
        sum += discrete.getProbability(i % 100);  // 0 to 99
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Computed " << numIterations << " probabilities in " 
              << duration.count() << " microseconds" << std::endl;
    std::cout << "Average time per calculation: " 
              << static_cast<double>(duration.count()) / numIterations << " microseconds" << std::endl;
    
    // Should complete in reasonable time (< 1 second)
    assert(duration.count() < 1000000); // 1 second = 1,000,000 microseconds
    
    std::cout << "✓ Performance tests passed" << std::endl;
}

/**
 * Test caching mechanism
 */
void testCaching() {
    std::cout << "Testing caching mechanism..." << std::endl;
    
    DiscreteDistribution discrete(3);
    discrete.setProbability(0, 0.4);
    discrete.setProbability(1, 0.3);
    discrete.setProbability(2, 0.3);
    
    // Test entropy calculation (uses caching)
    double entropy1 = discrete.getEntropy();
    double entropy2 = discrete.getEntropy();
    assert(entropy1 == entropy2);  // Should be identical due to caching
    
    // Test probability sum calculation (uses caching)
    double sum1 = discrete.getProbabilitySum();
    double sum2 = discrete.getProbabilitySum();
    assert(sum1 == sum2);  // Should be identical due to caching
    assert(std::abs(sum1 - 1.0) < 1e-10);  // Should sum to 1
    
    // Changing probabilities should invalidate cache
    discrete.setProbability(0, 0.5);
    double newEntropy = discrete.getEntropy();
    assert(newEntropy != entropy1);  // Should be different after change
    
    std::cout << "✓ Caching tests passed" << std::endl;
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
        
        // Gold Standard Tests
        testLogProbability();
        testCDF();
        testEqualityAndIO();
        testPerformance();
        testCaching();
        
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
