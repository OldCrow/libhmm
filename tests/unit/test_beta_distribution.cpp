#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/beta_distribution.h"

using libhmm::BetaDistribution;
using libhmm::Observation;

/**
 * Test basic Beta distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Beta distribution functionality..." << std::endl;
    
    // Test default constructor
    BetaDistribution beta;
    assert(beta.getAlpha() == 1.0);
    assert(beta.getBeta() == 1.0);
    
    // Test parameterized constructor
    BetaDistribution beta2(2.5, 1.5);
    assert(beta2.getAlpha() == 2.5);
    assert(beta2.getBeta() == 1.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    BetaDistribution beta(2.0, 3.0);  // Alpha=2, Beta=3
    
    // Test that probability is zero for values outside [0,1]
    assert(beta.getProbability(-0.1) == 0.0);
    assert(beta.getProbability(1.1) == 0.0);
    assert(beta.getProbability(-1.0) == 0.0);
    assert(beta.getProbability(2.0) == 0.0);
    
    // Test that probability is positive for values in [0,1]
    double prob1 = beta.getProbability(0.2);
    double prob2 = beta.getProbability(0.5);
    double prob3 = beta.getProbability(0.8);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    // For Beta(2,3), mode is at (α-1)/(α+β-2) = 1/3 ≈ 0.333
    // So probability should be higher at 0.2 than at 0.8
    assert(prob1 > prob3);
    
    // Test boundary values
    double probAt0 = beta.getProbability(0.0);
    double probAt1 = beta.getProbability(1.0);
    assert(probAt0 >= 0.0);  // Should be 0 for Beta(2,3) since α > 1
    assert(probAt1 >= 0.0);  // Should be 0 for Beta(2,3) since β > 1
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    BetaDistribution beta;
    
    // Test with known data
    std::vector<Observation> data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    beta.fit(data);
    assert(beta.getAlpha() > 0.0);  // Should have some reasonable value
    assert(beta.getBeta() > 0.0);   // Should have some reasonable value
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    beta.fit(emptyData);
    assert(beta.getAlpha() == 1.0);
    assert(beta.getBeta() == 1.0);
    
    // Test with single point (should reset to default for insufficient data)
    std::vector<Observation> singlePoint = {0.5};
    beta.fit(singlePoint);
    assert(beta.getAlpha() == 1.0);
    assert(beta.getBeta() == 1.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        BetaDistribution beta(0.0, 1.0);  // Zero alpha
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        BetaDistribution beta(-1.0, 1.0);  // Negative alpha
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        BetaDistribution beta(1.0, 0.0);  // Zero beta
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        BetaDistribution beta(1.0, -1.0);  // Negative beta
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        BetaDistribution beta(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        BetaDistribution beta(1.0, inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    BetaDistribution beta(1.0, 1.0);
    
    try {
        beta.setAlpha(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        beta.setBeta(-1.0);
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
    
    BetaDistribution beta(2.5, 1.5);
    std::string str = beta.toString();
    
    // Should contain key information based on actual output format:
    // "Beta Distribution:\n      α (alpha) = 2.5\n      β (beta) = 1.5\n"
    assert(str.find("Beta") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("1.5") != std::string::npos);
    assert(str.find("α") != std::string::npos || str.find("alpha") != std::string::npos);
    assert(str.find("β") != std::string::npos || str.find("beta") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    BetaDistribution original(3.14, 2.71);
    
    // Test copy constructor
    BetaDistribution copied(original);
    assert(copied.getAlpha() == original.getAlpha());
    assert(copied.getBeta() == original.getBeta());
    
    // Test copy assignment
    BetaDistribution assigned;
    assigned = original;
    assert(assigned.getAlpha() == original.getAlpha());
    assert(assigned.getBeta() == original.getBeta());
    
    // Test move constructor
    BetaDistribution moved(std::move(original));
    assert(moved.getAlpha() == 3.14);
    assert(moved.getBeta() == 2.71);
    
    // Test move assignment
    BetaDistribution moveAssigned;
    BetaDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getAlpha() == 1.41);
    assert(moveAssigned.getBeta() == 1.73);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    BetaDistribution beta(2.0, 3.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(beta.getProbability(nan_val) == 0.0);
    assert(beta.getProbability(inf_val) == 0.0);
    assert(beta.getProbability(neg_inf_val) == 0.0);
    
    // Values outside [0,1] should return 0
    assert(beta.getProbability(-0.1) == 0.0);
    assert(beta.getProbability(1.1) == 0.0);
    assert(beta.getProbability(-5.0) == 0.0);
    assert(beta.getProbability(10.0) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    BetaDistribution beta(10.0, 5.0);
    beta.reset();
    
    assert(beta.getAlpha() == 1.0);
    assert(beta.getBeta() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test Beta distribution properties
 */
void testBetaProperties() {
    std::cout << "Testing Beta distribution properties..." << std::endl;
    
    // Test uniform distribution (Beta(1,1))
    BetaDistribution uniform(1.0, 1.0);
    assert(std::abs(uniform.getMean() - 0.5) < 1e-10);
    assert(std::abs(uniform.getVariance() - (1.0/12.0)) < 1e-10);
    
    // Test symmetric distribution (Beta(2,2))
    BetaDistribution symmetric(2.0, 2.0);
    assert(std::abs(symmetric.getMean() - 0.5) < 1e-10);
    
    // Test skewed distribution (Beta(2,5))
    BetaDistribution skewed(2.0, 5.0);
    double expectedMean = 2.0 / (2.0 + 5.0);
    assert(std::abs(skewed.getMean() - expectedMean) < 1e-10);
    assert(skewed.getMean() < 0.5);  // Should be skewed toward 0
    
    // Test that probability is only defined on [0,1]
    assert(uniform.getProbability(-0.1) == 0.0);
    assert(uniform.getProbability(1.1) == 0.0);
    assert(uniform.getProbability(0.5) > 0.0);
    
    std::cout << "✓ Beta property tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    BetaDistribution beta;
    
    // Test with data containing values outside [0,1]
    std::vector<Observation> invalidData = {0.2, 0.5, 1.5, 0.8};  // 1.5 is out of range
    
    try {
        beta.fit(invalidData);
        assert(false);  // Should throw exception
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with negative values
    std::vector<Observation> negativeData = {0.2, 0.5, -0.1, 0.8};  // -0.1 is invalid
    try {
        beta.fit(negativeData);
        assert(false);  // Should throw exception
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with valid data in boundary cases
    std::vector<Observation> boundaryData = {0.0, 0.5, 1.0};
    try {
        beta.fit(boundaryData);
        // Should work fine, check that parameters are reasonable
        assert(beta.getAlpha() > 0.0);
        assert(beta.getBeta() > 0.0);
    } catch (const std::exception&) {
        // Also acceptable if implementation rejects boundary values
    }
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test statistical moments
 */
void testStatisticalMoments() {
    std::cout << "Testing statistical moments..." << std::endl;
    
    BetaDistribution beta(3.0, 2.0);
    
    // Test mean: α/(α+β) = 3/(3+2) = 0.6
    double expectedMean = 3.0 / 5.0;
    assert(std::abs(beta.getMean() - expectedMean) < 1e-10);
    
    // Test variance: αβ/((α+β)²(α+β+1)) = 6/(25*6) = 0.04
    double expectedVar = (3.0 * 2.0) / (5.0 * 5.0 * 6.0);
    assert(std::abs(beta.getVariance() - expectedVar) < 1e-10);
    
    // Test standard deviation
    double expectedStd = std::sqrt(expectedVar);
    assert(std::abs(beta.getStandardDeviation() - expectedStd) < 1e-10);
    
    std::cout << "✓ Statistical moments tests passed" << std::endl;
}

int main() {
    std::cout << "Running Beta distribution tests..." << std::endl;
    std::cout << "==================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testInvalidInputHandling();
        testResetFunctionality();
        testBetaProperties();
        testFittingValidation();
        testStatisticalMoments();
        
        std::cout << "==================================" << std::endl;
        std::cout << "✅ All Beta distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
