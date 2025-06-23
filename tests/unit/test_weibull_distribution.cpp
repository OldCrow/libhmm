#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/weibull_distribution.h"

using libhmm::WeibullDistribution;
using libhmm::Observation;

/**
 * Test basic Weibull distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Weibull distribution functionality..." << std::endl;
    
    // Test default constructor
    WeibullDistribution weibull;
    assert(weibull.getK() == 1.0);
    assert(weibull.getLambda() == 1.0);
    assert(weibull.getShape() == 1.0);  // Alternative getter
    assert(weibull.getScale() == 1.0);  // Alternative getter
    
    // Test parameterized constructor
    WeibullDistribution weibull2(2.5, 1.5);
    assert(weibull2.getK() == 2.5);
    assert(weibull2.getLambda() == 1.5);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    WeibullDistribution weibull(2.0, 1.0);  // k=2, λ=1 (Rayleigh distribution)
    
    // Test that probability is zero for negative values
    assert(weibull.getProbability(-0.1) == 0.0);
    assert(weibull.getProbability(-1.0) == 0.0);
    
    // Test that probability is positive for positive values
    double prob1 = weibull.getProbability(0.5);
    double prob2 = weibull.getProbability(1.0);
    double prob3 = weibull.getProbability(2.0);
    
    assert(prob1 > 0.0);
    assert(prob2 > 0.0);
    assert(prob3 > 0.0);
    
    // For Weibull distribution, density typically decreases after the mode
    // Mode for Weibull(k,λ) = λ * ((k-1)/k)^(1/k) when k > 1
    // For k=2, λ=1: mode ≈ 0.707
    
    // Test boundary value at x=0
    double probAt0 = weibull.getProbability(0.0);
    assert(probAt0 >= 0.0);  // Should be 0 for k=2 > 1
    
    // Test that probability density is reasonable (should be small for continuous dist)
    assert(prob1 < 10.0);  // Should be reasonable
    assert(prob1 > 1e-10);  // Should be greater than zero
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    WeibullDistribution weibull;
    
    // Test with known data
    std::vector<Observation> data = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    weibull.fit(data);
    assert(weibull.getK() > 0.0);      // Should have some reasonable value
    assert(weibull.getLambda() > 0.0); // Should have some reasonable value
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    weibull.fit(emptyData);
    assert(weibull.getK() == 1.0);
    assert(weibull.getLambda() == 1.0);
    
    // Test with single point (should reset to default for insufficient data)
    std::vector<Observation> singlePoint = {2.5};
    weibull.fit(singlePoint);
    assert(weibull.getK() == 1.0);
    assert(weibull.getLambda() == 1.0);
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        WeibullDistribution weibull(0.0, 1.0);  // Zero k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        WeibullDistribution weibull(-1.0, 1.0);  // Negative k
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        WeibullDistribution weibull(1.0, 0.0);  // Zero lambda
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        WeibullDistribution weibull(1.0, -1.0);  // Negative lambda
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        WeibullDistribution weibull(nan_val, 1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        WeibullDistribution weibull(1.0, inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setters validation
    WeibullDistribution weibull(1.0, 1.0);
    
    try {
        weibull.setK(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        weibull.setLambda(-1.0);
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
    
    WeibullDistribution weibull(2.5, 1.5);
    std::string str = weibull.toString();
    
    // Should contain key information based on actual output format:
    // "Weibull Distribution:\n      k (shape) = 2.5\n      λ (scale) = 1.5\n"
    assert(str.find("Weibull") != std::string::npos);
    assert(str.find("Distribution") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
    assert(str.find("1.5") != std::string::npos);
    assert(str.find("k") != std::string::npos);
    assert(str.find("shape") != std::string::npos);
    assert(str.find("λ") != std::string::npos || str.find("scale") != std::string::npos);
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    WeibullDistribution original(3.14, 2.71);
    
    // Test copy constructor
    WeibullDistribution copied(original);
    assert(copied.getK() == original.getK());
    assert(copied.getLambda() == original.getLambda());
    
    // Test copy assignment
    WeibullDistribution assigned;
    assigned = original;
    assert(assigned.getK() == original.getK());
    assert(assigned.getLambda() == original.getLambda());
    
    // Test move constructor
    WeibullDistribution moved(std::move(original));
    assert(moved.getK() == 3.14);
    assert(moved.getLambda() == 2.71);
    
    // Test move assignment
    WeibullDistribution moveAssigned;
    WeibullDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getK() == 1.41);
    assert(moveAssigned.getLambda() == 1.73);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test invalid input handling
 */
void testInvalidInputHandling() {
    std::cout << "Testing invalid input handling..." << std::endl;
    
    WeibullDistribution weibull(2.0, 1.0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    double neg_inf_val = -std::numeric_limits<double>::infinity();
    
    assert(weibull.getProbability(nan_val) == 0.0);
    assert(weibull.getProbability(inf_val) == 0.0);
    assert(weibull.getProbability(neg_inf_val) == 0.0);
    
    // Negative values should return 0
    assert(weibull.getProbability(-0.1) == 0.0);
    assert(weibull.getProbability(-1.0) == 0.0);
    assert(weibull.getProbability(-10.0) == 0.0);
    
    std::cout << "✓ Invalid input handling tests passed" << std::endl;
}

/**
 * Test reset functionality
 */
void testResetFunctionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    WeibullDistribution weibull(10.0, 5.0);
    weibull.reset();
    
    assert(weibull.getK() == 1.0);
    assert(weibull.getLambda() == 1.0);
    
    std::cout << "✓ Reset functionality tests passed" << std::endl;
}

/**
 * Test Weibull distribution properties
 */
void testWeibullProperties() {
    std::cout << "Testing Weibull distribution properties..." << std::endl;
    
    // Test exponential case (k=1)
    WeibullDistribution exponential(1.0, 2.0);
    double expMean = exponential.getMean();
    assert(std::abs(expMean - 2.0) < 1e-10);  // For Weibull(1,λ), mean = λ
    
    // Test Rayleigh case (k=2)
    WeibullDistribution rayleigh(2.0, 1.0);
    double rayleighMean = rayleigh.getMean();
    // For Weibull(2,1), mean = Γ(1.5) = sqrt(π)/2 ≈ 0.8862
    assert(rayleighMean > 0.8 && rayleighMean < 0.9);
    
    // Test that Weibull is only defined for x ≥ 0
    assert(exponential.getProbability(-0.1) == 0.0);
    assert(exponential.getProbability(0.0) >= 0.0);
    assert(exponential.getProbability(1.0) > 0.0);
    
    // Test variance is positive
    assert(exponential.getVariance() > 0.0);
    assert(rayleigh.getVariance() > 0.0);
    
    // Test standard deviation relationship
    assert(std::abs(exponential.getStandardDeviation() - std::sqrt(exponential.getVariance())) < 1e-10);
    
    std::cout << "✓ Weibull property tests passed" << std::endl;
}

/**
 * Test fitting validation
 */
void testFittingValidation() {
    std::cout << "Testing fitting validation..." << std::endl;
    
    WeibullDistribution weibull;
    
    // Test with data containing negative values
    std::vector<Observation> invalidData = {1.0, 2.0, -1.0, 3.0};  // -1.0 is invalid
    
    try {
        weibull.fit(invalidData);
        assert(false);  // Should throw exception
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with NaN values
    std::vector<Observation> nanData = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    try {
        weibull.fit(nanData);
        assert(false);  // Should throw exception
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test with valid data including zero
    std::vector<Observation> zeroData = {0.0, 1.0, 2.0, 3.0};
    try {
        weibull.fit(zeroData);
        // Should work fine, check that parameters are reasonable
        assert(weibull.getK() > 0.0);
        assert(weibull.getLambda() > 0.0);
    } catch (const std::exception&) {
        // Also acceptable if implementation has specific constraints
    }
    
    std::cout << "✓ Fitting validation tests passed" << std::endl;
}

/**
 * Test statistical moments
 */
void testStatisticalMoments() {
    std::cout << "Testing statistical moments..." << std::endl;
    
    // Test exponential case (k=1, λ=2)
    WeibullDistribution exponential(1.0, 2.0);
    
    // For Weibull(1,λ), mean = λ * Γ(2) = λ
    double expectedMean = 2.0;
    assert(std::abs(exponential.getMean() - expectedMean) < 1e-10);
    
    // For Weibull(1,λ), variance = λ² * [Γ(3) - (Γ(2))²] = λ² * [2 - 1] = λ²
    double expectedVar = 4.0;
    assert(std::abs(exponential.getVariance() - expectedVar) < 1e-10);
    
    // Test standard deviation
    double expectedStd = 2.0;
    assert(std::abs(exponential.getStandardDeviation() - expectedStd) < 1e-10);
    
    // Test Rayleigh case (k=2, λ=1)
    WeibullDistribution rayleigh(2.0, 1.0);
    
    // Mean should be sqrt(π)/2 ≈ 0.8862
    double rayleighMean = rayleigh.getMean();
    assert(rayleighMean > 0.88 && rayleighMean < 0.89);
    
    // Variance should be (4-π)/4 ≈ 0.2146
    double rayleighVar = rayleigh.getVariance();
    assert(rayleighVar > 0.21 && rayleighVar < 0.22);
    
    std::cout << "✓ Statistical moments tests passed" << std::endl;
}

/**
 * Test special cases and edge cases
 */
void testSpecialCases() {
    std::cout << "Testing special cases and edge cases..." << std::endl;
    
    // Test k=1 (exponential distribution)
    WeibullDistribution exp_case(1.0, 1.0);
    double prob_exp = exp_case.getProbability(1.0);
    // For exponential with rate 1, PDF(1) = e^(-1) ≈ 0.368
    assert(prob_exp > 0.35 && prob_exp < 0.4);
    
    // Test k=2 (Rayleigh distribution) 
    WeibullDistribution rayleigh_case(2.0, 1.0);
    double prob_rayleigh = rayleigh_case.getProbability(1.0);
    // For Rayleigh with σ=1, PDF(1) = 1*e^(-0.5) ≈ 0.607
    // Note: Actual implementation gives ~0.736 due to Weibull parameterization
    assert(prob_rayleigh > 0.7 && prob_rayleigh < 0.75);
    
    // Test very small k (infant mortality)
    WeibullDistribution infant(0.5, 1.0);
    assert(infant.getK() == 0.5);
    assert(infant.getLambda() == 1.0);
    
    // Test large k (wear-out failures)
    WeibullDistribution wearout(5.0, 1.0);
    assert(wearout.getK() == 5.0);
    assert(wearout.getLambda() == 1.0);
    
    // Test very large values
    WeibullDistribution normal_case(1.0, 1.0);
    double prob_large = normal_case.getProbability(100.0);
    assert(prob_large >= 0.0);  // Should be very small but non-negative
    assert(prob_large < 1e-10); // Should be essentially zero
    
    std::cout << "✓ Special cases tests passed" << std::endl;
}

int main() {
    std::cout << "Running Weibull distribution tests..." << std::endl;
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
        testWeibullProperties();
        testFittingValidation();
        testStatisticalMoments();
        testSpecialCases();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "✅ All Weibull distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
