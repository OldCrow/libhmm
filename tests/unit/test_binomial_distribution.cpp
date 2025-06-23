#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/binomial_distribution.h"

using libhmm::BinomialDistribution;
using libhmm::Observation;

/**
 * Test basic Binomial distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Binomial distribution functionality..." << std::endl;
    
    // Test default constructor
    BinomialDistribution binomial;
    assert(binomial.getN() == 10);
    assert(binomial.getP() == 0.5);
    
    // Test parameterized constructor
    BinomialDistribution binomial2(20, 0.3);
    assert(binomial2.getN() == 20);
    assert(binomial2.getP() == 0.3);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    BinomialDistribution binomial(10, 0.5);
    
    // Test probability at some specific values
    double prob0 = binomial.getProbability(0.0);
    double prob5 = binomial.getProbability(5.0);
    double prob10 = binomial.getProbability(10.0);
    
    assert(prob0 > 0.0);
    assert(prob5 > 0.0);
    assert(prob10 > 0.0);
    
    // For symmetric binomial (p=0.5), P(5) should be the maximum
    assert(prob5 >= prob0);
    assert(prob5 >= prob10);
    
    // Test out of range values
    assert(binomial.getProbability(-1.0) == 0.0);
    assert(binomial.getProbability(11.0) == 0.0);
    
    // Test edge cases
    BinomialDistribution binomial_p0(10, 0.0);
    assert(binomial_p0.getProbability(0.0) == 1.0);
    assert(binomial_p0.getProbability(1.0) == 0.0);
    
    BinomialDistribution binomial_p1(10, 1.0);
    assert(binomial_p1.getProbability(10.0) == 1.0);
    assert(binomial_p1.getProbability(9.0) == 0.0);
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    BinomialDistribution binomial;
    
    // Test with data that should estimate reasonable parameters
    std::vector<Observation> data = {3, 4, 5, 6, 7, 3, 4, 5, 6, 7};
    binomial.fit(data);
    
    // After fitting, parameters should be positive and valid
    assert(binomial.getN() > 0);
    assert(binomial.getP() > 0.0 && binomial.getP() <= 1.0);
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    binomial.fit(emptyData);
    assert(binomial.getN() == 10);
    assert(binomial.getP() == 0.5);
    
    // Test with single point
    std::vector<Observation> singlePoint = {5};
    binomial.fit(singlePoint);
    assert(binomial.getN() >= 1);
    assert(binomial.getP() >= 0.0 && binomial.getP() <= 1.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        BinomialDistribution binomial(0, 0.5);  // Zero n
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        BinomialDistribution binomial(-1, 0.5);  // Negative n
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        BinomialDistribution binomial(10, -0.1);  // Negative p
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        BinomialDistribution binomial(10, 1.5);  // p > 1
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        BinomialDistribution binomial(10, nan_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        BinomialDistribution binomial(10, inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    BinomialDistribution binomial(10, 0.5);
    
    try {
        binomial.setN(0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        binomial.setP(-0.1);
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
    
    BinomialDistribution binomial(15, 0.3);
    std::string str = binomial.toString();
    
    // Should contain key information based on standardized format:
    // "Binomial Distribution:\n      n (trials) = 15\n      p (success probability) = 0.3\n      Mean = 4.5\n      Variance = 3.15\n"
    assert(str.find("Binomial") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("15") != std::string::npos);
    assert(str.find("0.3") != std::string::npos);
    assert(str.find("trials") != std::string::npos);
    assert(str.find("success probability") != std::string::npos);
    assert(str.find("Mean") != std::string::npos);
    assert(str.find("Variance") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    BinomialDistribution original(12, 0.7);
    
    // Test copy constructor
    BinomialDistribution copied(original);
    assert(copied.getN() == original.getN());
    assert(copied.getP() == original.getP());
    
    // Test copy assignment
    BinomialDistribution assigned;
    assigned = original;
    assert(assigned.getN() == original.getN());
    assert(assigned.getP() == original.getP());
    
    // Test move constructor
    BinomialDistribution moved(std::move(original));
    assert(moved.getN() == 12);
    assert(moved.getP() == 0.7);
    
    // Test move assignment
    BinomialDistribution moveAssigned;
    BinomialDistribution temp(8, 0.4);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getN() == 8);
    assert(moveAssigned.getP() == 0.4);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    BinomialDistribution binomial(10, 0.5);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(binomial.getProbability(nan_val) == 0.0);
    assert(binomial.getProbability(inf_val) == 0.0);
    assert(binomial.getProbability(neg_inf_val) == 0.0);
    
    // Out of range values should return 0
    assert(binomial.getProbability(-1.0) == 0.0);
    assert(binomial.getProbability(11.0) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    BinomialDistribution binomial(25, 0.8);
    binomial.reset();
    
    assert(binomial.getN() == 10);
    assert(binomial.getP() == 0.5);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test binomial distribution properties
 */
void testBinomialProperties() {
    std::cout << "Testing Binomial distribution properties..." << std::endl;
    
    BinomialDistribution binomial(20, 0.3);
    
    // Test statistical moments
    double mean = binomial.getMean();
    double variance = binomial.getVariance();
    double stddev = binomial.getStandardDeviation();
    
    // For Binomial(n,p): mean = n*p, variance = n*p*(1-p)
    assert(std::abs(mean - 20 * 0.3) < 1e-10);
    assert(std::abs(variance - 20 * 0.3 * 0.7) < 1e-10);
    assert(std::abs(stddev - std::sqrt(variance)) < 1e-10);
    
    // Test that probabilities sum to 1 (approximately)
    double total_prob = 0.0;
    for (int k = 0; k <= 20; ++k) {
        total_prob += binomial.getProbability(k);
    }
    assert(std::abs(total_prob - 1.0) < 1e-6);  // Should sum to 1
    
    std::cout << "✓ Binomial property tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    BinomialDistribution binomial;
    
    // Test with data containing negative values (should handle gracefully)
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};
    try {
        binomial.fit(invalidData);
        // If it doesn't throw, the parameters should still be valid
        assert(binomial.getN() > 0);
        assert(binomial.getP() >= 0.0 && binomial.getP() <= 1.0);
    } catch (const std::exception&) {
        // It's also acceptable to throw for invalid data
    }
    
    // Test with all zeros
    std::vector<Observation> zeroData = {0.0, 0.0, 0.0};
    binomial.fit(zeroData);
    assert(binomial.getN() >= 1);
    assert(binomial.getP() >= 0.0 && binomial.getP() <= 1.0);
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test statistical moments
 */
void testStatisticalMoments() {
    std::cout << "Testing statistical moments..." << std::endl;
    
    BinomialDistribution binomial(50, 0.4);
    
    double mean = binomial.getMean();
    double variance = binomial.getVariance();
    double stddev = binomial.getStandardDeviation();
    
    // Verify theoretical relationships
    assert(std::abs(mean - 50 * 0.4) < 1e-10);
    assert(std::abs(variance - 50 * 0.4 * 0.6) < 1e-10);
    assert(std::abs(stddev * stddev - variance) < 1e-10);
    
    std::cout << "✓ Statistical moments tests passed" << std::endl;
}

int main() {
    std::cout << "Running Binomial distribution tests..." << std::endl;
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
        testBinomialProperties();
        testFittingValidation();
        testStatisticalMoments();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "✅ All Binomial distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
