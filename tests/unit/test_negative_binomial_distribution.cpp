#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/negative_binomial_distribution.h"

using libhmm::NegativeBinomialDistribution;
using libhmm::Observation;

/**
 * Test basic Negative Binomial distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Negative Binomial distribution functionality..." << std::endl;
    
    // Test default constructor
    NegativeBinomialDistribution negbinom;
    assert(negbinom.getR() == 5.0);
    assert(negbinom.getP() == 0.5);
    
    // Test parameterized constructor
    NegativeBinomialDistribution negbinom2(3.0, 0.7);
    assert(negbinom2.getR() == 3.0);
    assert(negbinom2.getP() == 0.7);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    NegativeBinomialDistribution negbinom(5.0, 0.5);
    
    // Test probability at some specific values
    double prob0 = negbinom.getProbability(0.0);
    double prob1 = negbinom.getProbability(1.0);
    double prob5 = negbinom.getProbability(5.0);
    
    assert(prob0 > 0.0);
    assert(prob1 > 0.0);
    assert(prob5 > 0.0);
    
    // For negative binomial, probabilities should be positive and decreasing in general
    // (but the exact pattern depends on parameters, so we just check they're positive)
    
    // Test out of range values
    assert(negbinom.getProbability(-1.0) == 0.0);
    
    // Test edge case p = 1
    NegativeBinomialDistribution negbinom_p1(5.0, 1.0);
    assert(negbinom_p1.getProbability(0.0) == 1.0);
    assert(negbinom_p1.getProbability(1.0) == 0.0);
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    NegativeBinomialDistribution negbinom;
    
    // Test with over-dispersed data (variance > mean)
    std::vector<Observation> data = {0, 1, 2, 3, 5, 8, 10, 15, 2, 4, 7, 12};
    negbinom.fit(data);
    
    // After fitting, parameters should be positive and valid
    assert(negbinom.getR() > 0.0);
    assert(negbinom.getP() > 0.0 && negbinom.getP() <= 1.0);
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    negbinom.fit(emptyData);
    assert(negbinom.getR() == 5.0);
    assert(negbinom.getP() == 0.5);
    
    // Test with single point (should reset)
    std::vector<Observation> singlePoint = {5};
    negbinom.fit(singlePoint);
    assert(negbinom.getR() == 5.0);
    assert(negbinom.getP() == 0.5);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        NegativeBinomialDistribution negbinom(0.0, 0.5);  // Zero r
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        NegativeBinomialDistribution negbinom(-1.0, 0.5);  // Negative r
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        NegativeBinomialDistribution negbinom(5.0, 0.0);  // Zero p
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        NegativeBinomialDistribution negbinom(5.0, -0.1);  // Negative p
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        NegativeBinomialDistribution negbinom(5.0, 1.5);  // p > 1
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        NegativeBinomialDistribution negbinom(nan_val, 0.5);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        NegativeBinomialDistribution negbinom(5.0, inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    NegativeBinomialDistribution negbinom(5.0, 0.5);
    
    try {
        negbinom.setR(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        negbinom.setP(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        negbinom.setP(1.5);
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
    
    NegativeBinomialDistribution negbinom(8.0, 0.4);
    std::string str = negbinom.toString();
    
    // Should contain key information based on standardized format:
    // "Negative Binomial Distribution:\n      r (successes) = 8.0\n      p (success probability) = 0.4\n      Mean = 12.0\n      Variance = 30.0\n"
    assert(str.find("Negative Binomial") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("8.0") != std::string::npos);
    assert(str.find("0.4") != std::string::npos);
    assert(str.find("successes") != std::string::npos);
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
    
    NegativeBinomialDistribution original(7.5, 0.6);
    
    // Test copy constructor
    NegativeBinomialDistribution copied(original);
    assert(copied.getR() == original.getR());
    assert(copied.getP() == original.getP());
    
    // Test copy assignment
    NegativeBinomialDistribution assigned;
    assigned = original;
    assert(assigned.getR() == original.getR());
    assert(assigned.getP() == original.getP());
    
    // Test move constructor
    NegativeBinomialDistribution moved(std::move(original));
    assert(moved.getR() == 7.5);
    assert(moved.getP() == 0.6);
    
    // Test move assignment
    NegativeBinomialDistribution moveAssigned;
    NegativeBinomialDistribution temp(3.2, 0.8);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getR() == 3.2);
    assert(moveAssigned.getP() == 0.8);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    NegativeBinomialDistribution negbinom(5.0, 0.5);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(negbinom.getProbability(nan_val) == 0.0);
    assert(negbinom.getProbability(inf_val) == 0.0);
    assert(negbinom.getProbability(neg_inf_val) == 0.0);
    
    // Negative values should return 0
    assert(negbinom.getProbability(-1.0) == 0.0);
    assert(negbinom.getProbability(-0.5) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    NegativeBinomialDistribution negbinom(10.0, 0.2);
    negbinom.reset();
    
    assert(negbinom.getR() == 5.0);
    assert(negbinom.getP() == 0.5);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test negative binomial distribution properties
 */
void testNegativeBinomialProperties() {
    std::cout << "Testing Negative Binomial distribution properties..." << std::endl;
    
    NegativeBinomialDistribution negbinom(4.0, 0.3);
    
    // Test statistical moments
    double mean = negbinom.getMean();
    double variance = negbinom.getVariance();
    double stddev = negbinom.getStandardDeviation();
    
    // For NegBinom(r,p): mean = r*(1-p)/p, variance = r*(1-p)/p²
    double expected_mean = 4.0 * (1.0 - 0.3) / 0.3;
    double expected_variance = 4.0 * (1.0 - 0.3) / (0.3 * 0.3);
    
    assert(std::abs(mean - expected_mean) < 1e-10);
    assert(std::abs(variance - expected_variance) < 1e-10);
    assert(std::abs(stddev - std::sqrt(variance)) < 1e-10);
    
    // Test that variance > mean (over-dispersion property)
    assert(variance > mean);
    
    std::cout << "✓ Negative Binomial property tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    NegativeBinomialDistribution negbinom;
    
    // Test with data containing negative values (should handle gracefully)
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};
    try {
        negbinom.fit(invalidData);
        // If it doesn't throw, the parameters should still be valid
        assert(negbinom.getR() > 0.0);
        assert(negbinom.getP() > 0.0 && negbinom.getP() <= 1.0);
    } catch (const std::exception&) {
        // It's also acceptable to throw for invalid data
    }
    
    // Test with under-dispersed data (variance <= mean) - should reset
    std::vector<Observation> underDispersedData = {1.0, 1.0, 1.0, 1.0, 1.0};
    negbinom.fit(underDispersedData);
    // Should fall back to defaults since negative binomial is not appropriate
    assert(negbinom.getR() == 5.0);
    assert(negbinom.getP() == 0.5);
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test statistical moments
 */
void testStatisticalMoments() {
    std::cout << "Testing statistical moments..." << std::endl;
    
    NegativeBinomialDistribution negbinom(6.0, 0.4);
    
    double mean = negbinom.getMean();
    double variance = negbinom.getVariance();
    double stddev = negbinom.getStandardDeviation();
    
    // Verify theoretical relationships
    double expected_mean = 6.0 * 0.6 / 0.4;  // r*(1-p)/p
    double expected_variance = 6.0 * 0.6 / (0.4 * 0.4);  // r*(1-p)/p²
    
    assert(std::abs(mean - expected_mean) < 1e-10);
    assert(std::abs(variance - expected_variance) < 1e-10);
    assert(std::abs(stddev * stddev - variance) < 1e-10);
    
    std::cout << "✓ Statistical moments tests passed" << std::endl;
}

/**
 * Test over-dispersion property
 */
void testOverDispersion() {
    std::cout << "Testing over-dispersion property..." << std::endl;
    
    // Negative binomial should exhibit over-dispersion (variance > mean)
    NegativeBinomialDistribution negbinom1(2.0, 0.3);
    NegativeBinomialDistribution negbinom2(10.0, 0.7);
    NegativeBinomialDistribution negbinom3(1.5, 0.1);
    
    assert(negbinom1.getVariance() > negbinom1.getMean());
    assert(negbinom2.getVariance() > negbinom2.getMean());
    assert(negbinom3.getVariance() > negbinom3.getMean());
    
    std::cout << "✓ Over-dispersion property tests passed" << std::endl;
}

int main() {
    std::cout << "Running Negative Binomial distribution tests..." << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        testNegativeBinomialProperties();
        testFittingValidation();
        testStatisticalMoments();
        testOverDispersion();
        
        std::cout << "===============================================" << std::endl;
        std::cout << "✅ All Negative Binomial distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
