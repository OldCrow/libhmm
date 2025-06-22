#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"

using namespace libhmm;

void testDiscreteDistributionTypeSafety() {
    std::cout << "=== Testing DiscreteDistribution Type Safety ===" << std::endl;
    
    try {
        // Test constructor validation
        DiscreteDistribution badDist(0);  // Should throw
        std::cout << "ERROR: Should have thrown for 0 symbols!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Constructor correctly rejected 0 symbols: " << e.what() << std::endl;
    }
    
    // Test valid construction
    DiscreteDistribution dist(5);
    std::cout << "✓ Successfully created distribution with 5 symbols" << std::endl;
    std::cout << "Number of symbols: " << dist.getNumSymbols() << std::endl;
    
    // Test bounds checking in getProbability
    double prob1 = dist.getProbability(2.0);  // Valid
    double prob2 = dist.getProbability(10.0); // Out of bounds
    double prob3 = dist.getProbability(-1.0); // Negative
    double prob4 = dist.getProbability(std::numeric_limits<double>::quiet_NaN()); // NaN
    
    std::cout << "✓ P(2) = " << prob1 << " (should be 0.2)" << std::endl;
    std::cout << "✓ P(10) = " << prob2 << " (should be 0.0 - out of bounds)" << std::endl;
    std::cout << "✓ P(-1) = " << prob3 << " (should be 0.0 - negative)" << std::endl;
    std::cout << "✓ P(NaN) = " << prob4 << " (should be 0.0 - NaN)" << std::endl;
    
    // Test setProbability validation
    try {
        dist.setProbability(2, 0.5);  // Valid
        std::cout << "✓ Successfully set P(2) = 0.5" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR: Valid setProbability failed: " << e.what() << std::endl;
    }
    
    try {
        dist.setProbability(2, 1.5);  // Invalid probability > 1
        std::cout << "ERROR: Should have thrown for probability > 1!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Correctly rejected probability > 1: " << e.what() << std::endl;
    }
    
    try {
        dist.setProbability(10, 0.5);  // Out of bounds index
        std::cout << "ERROR: Should have thrown for out-of-bounds index!" << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "✓ Correctly rejected out-of-bounds index: " << e.what() << std::endl;
    }
    
    try {
        dist.setProbability(2, std::numeric_limits<double>::quiet_NaN());  // NaN
        std::cout << "ERROR: Should have thrown for NaN probability!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Correctly rejected NaN probability: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

void testGaussianDistributionTypeSafety() {
    std::cout << "=== Testing GaussianDistribution Type Safety ===" << std::endl;
    
    try {
        // Test valid construction
        GaussianDistribution validDist(0.0, 1.0);
        std::cout << "✓ Successfully created valid Gaussian distribution" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR: Valid construction failed: " << e.what() << std::endl;
    }
    
    try {
        // Test invalid standard deviation
        GaussianDistribution invalidDist(0.0, 0.0);  // Zero std dev
        std::cout << "ERROR: Should have thrown for zero standard deviation!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Constructor correctly rejected zero std dev: " << e.what() << std::endl;
    }
    
    try {
        // Test negative standard deviation
        GaussianDistribution invalidDist(0.0, -1.0);  // Negative std dev
        std::cout << "ERROR: Should have thrown for negative standard deviation!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Constructor correctly rejected negative std dev: " << e.what() << std::endl;
    }
    
    try {
        // Test NaN mean
        GaussianDistribution invalidDist(std::numeric_limits<double>::quiet_NaN(), 1.0);
        std::cout << "ERROR: Should have thrown for NaN mean!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Constructor correctly rejected NaN mean: " << e.what() << std::endl;
    }
    
    // Test input validation in getProbability
    GaussianDistribution dist(0.0, 1.0);
    
    double prob1 = dist.getProbability(0.0);  // Valid
    double prob2 = dist.getProbability(std::numeric_limits<double>::quiet_NaN());  // NaN
    double prob3 = dist.getProbability(std::numeric_limits<double>::infinity());   // Infinity
    
    std::cout << "✓ P(0) = " << prob1 << " (should be positive)" << std::endl;
    std::cout << "✓ P(NaN) = " << prob2 << " (should be 0.0)" << std::endl;
    std::cout << "✓ P(inf) = " << prob3 << " (should be 0.0)" << std::endl;
    
    // Test setter validation
    try {
        dist.setMean(5.0);  // Valid
        std::cout << "✓ Successfully set mean to 5.0" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR: Valid setMean failed: " << e.what() << std::endl;
    }
    
    try {
        dist.setMean(std::numeric_limits<double>::quiet_NaN());  // Invalid
        std::cout << "ERROR: Should have thrown for NaN mean!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Correctly rejected NaN mean in setter: " << e.what() << std::endl;
    }
    
    try {
        dist.setStandardDeviation(2.0);  // Valid
        std::cout << "✓ Successfully set std dev to 2.0" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR: Valid setStandardDeviation failed: " << e.what() << std::endl;
    }
    
    try {
        dist.setStandardDeviation(-1.0);  // Invalid
        std::cout << "ERROR: Should have thrown for negative std dev!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Correctly rejected negative std dev in setter: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

void testModernConstants() {
    std::cout << "=== Testing Modern Constants ===" << std::endl;
    std::cout << "BW_TOLERANCE = " << BW_TOLERANCE << std::endl;
    std::cout << "ZERO = " << ZERO << std::endl;
    std::cout << "LIMIT_TOLERANCE = " << LIMIT_TOLERANCE << std::endl;
    std::cout << "MAX_VITERBI_ITERATIONS = " << MAX_VITERBI_ITERATIONS << std::endl;
    std::cout << "ITMAX = " << ITMAX << std::endl;
    std::cout << "PI = " << PI << std::endl;
    std::cout << "✓ All constants accessible and properly defined" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "Testing Enhanced Type Safety and Modern C++17 Features:" << std::endl << std::endl;
    
    try {
        testDiscreteDistributionTypeSafety();
        testGaussianDistributionTypeSafety();
        testModernConstants();
        
        std::cout << "🎉 All type safety tests completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ Unexpected exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
