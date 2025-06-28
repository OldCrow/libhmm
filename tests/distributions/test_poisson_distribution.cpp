#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/poisson_distribution.h"

using libhmm::PoissonDistribution;
using libhmm::Observation;

/**
 * Test basic Poisson distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Poisson distribution functionality..." << std::endl;
    
    // Test default constructor
    PoissonDistribution poisson;
    assert(poisson.getLambda() == 1.0);
    assert(poisson.getMean() == 1.0);
    assert(poisson.getVariance() == 1.0);
    assert(std::abs(poisson.getStandardDeviation() - 1.0) < 1e-10);
    
    // Test parameterized constructor
    PoissonDistribution poisson2(3.5);
    assert(poisson2.getLambda() == 3.5);
    assert(poisson2.getMean() == 3.5);
    assert(poisson2.getVariance() == 3.5);
    assert(std::abs(poisson2.getStandardDeviation() - std::sqrt(3.5)) < 1e-10);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    PoissonDistribution poisson(2.0);
    
    // Test known values for λ = 2.0
    // P(X=0) = e^(-2) ≈ 0.1353
    double p0 = poisson.getProbability(0.0);
    assert(std::abs(p0 - std::exp(-2.0)) < 1e-10);
    
    // P(X=1) = 2 * e^(-2) ≈ 0.2707
    double p1 = poisson.getProbability(1.0);
    assert(std::abs(p1 - 2.0 * std::exp(-2.0)) < 1e-10);
    
    // P(X=2) = 4/2 * e^(-2) = 2 * e^(-2) ≈ 0.2707
    double p2 = poisson.getProbability(2.0);
    assert(std::abs(p2 - 2.0 * std::exp(-2.0)) < 1e-10);
    
    // Invalid inputs should return 0
    assert(poisson.getProbability(-1.0) == 0.0);
    assert(poisson.getProbability(1.5) == 0.0);  // non-integer
    assert(poisson.getProbability(std::numeric_limits<double>::infinity()) == 0.0);
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    PoissonDistribution poisson;
    
    // Test with known data (should fit λ ≈ 2.5)
    std::vector<Observation> data = {1, 2, 2, 3, 3, 3, 4, 2, 1, 4};
    double expectedMean = 2.5;  // Sum = 25, n = 10
    
    poisson.fit(data);
    assert(std::abs(poisson.getLambda() - expectedMean) < 1e-10);
    
    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    poisson.fit(emptyData);
    assert(poisson.getLambda() == 1.0);
    
    // Test with invalid data (should throw)
    std::vector<Observation> invalidData = {1, 2, -1, 3};
    try {
        poisson.fit(invalidData);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "✓ Parameter fitting tests passed" << std::endl;
}

/**
 * Test parameter validation
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid lambda values in constructor
    try {
        PoissonDistribution poisson(-1.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        PoissonDistribution poisson(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        PoissonDistribution poisson(std::numeric_limits<double>::infinity());
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setLambda validation
    PoissonDistribution poisson(1.0);
    try {
        poisson.setLambda(-1.0);
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
    
    PoissonDistribution poisson(2.5);
    std::string str = poisson.toString();
    
    // Should contain key information
    assert(str.find("Poisson") != std::string::npos);
    assert(str.find("2.5") != std::string::npos);
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
    
    PoissonDistribution original(3.14);
    
    // Test copy constructor
    PoissonDistribution copied(original);
    assert(copied.getLambda() == original.getLambda());
    
    // Test copy assignment
    PoissonDistribution assigned;
    assigned = original;
    assert(assigned.getLambda() == original.getLambda());
    
    // Test move constructor
    PoissonDistribution moved(std::move(original));
    assert(moved.getLambda() == 3.14);
    
    // Test move assignment
    PoissonDistribution moveAssigned;
    PoissonDistribution temp(2.71);
    moveAssigned = std::move(temp);
    assert(moveAssigned.getLambda() == 2.71);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test numerical stability with large values
 */
void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;
    
    // Test with large lambda
    PoissonDistribution poissonLarge(500.0);
    double probLarge = poissonLarge.getProbability(500.0);  // Should be around mode
    assert(probLarge > 0.0 && probLarge < 1.0);
    
    // Test with very small lambda
    PoissonDistribution poissonSmall(1e-6);
    double probSmall = poissonSmall.getProbability(0.0);
    assert(probSmall > 0.0 && probSmall < 1.0);
    
    // Test extreme cases that might cause overflow/underflow
    PoissonDistribution poissonExtreme(100.0);
    double probExtreme = poissonExtreme.getProbability(200.0);  // Far from mean
    // Should return a very small but positive number, or 0 due to underflow
    assert(probExtreme >= 0.0);
    
    std::cout << "✓ Numerical stability tests passed" << std::endl;
}

/**
 * Test log probability calculations
 */
void testLogProbability() {
    std::cout << "Testing log probability calculations..." << std::endl;
    
    PoissonDistribution poisson(2.0);
    
    // Test valid values
    double logProb2 = poisson.getLogProbability(2.0);
    double prob2 = poisson.getProbability(2.0);
    
    // log(prob) should equal logProb (within numerical precision)
    assert(std::abs(std::log(prob2) - logProb2) < 1e-10);
    
    // Test invalid inputs (should return -infinity)
    double logProbNeg = poisson.getLogProbability(-1.0);
    double logProbFloat = poisson.getLogProbability(2.5);
    assert(std::isinf(logProbNeg) && logProbNeg < 0);
    assert(std::isinf(logProbFloat) && logProbFloat < 0);
    
    // Test with invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    assert(std::isinf(poisson.getLogProbability(nan_val)) && poisson.getLogProbability(nan_val) < 0);
    assert(std::isinf(poisson.getLogProbability(inf_val)) && poisson.getLogProbability(inf_val) < 0);
    
    std::cout << "✓ Log probability tests passed" << std::endl;
}

/**
 * Test CDF calculations
 */
void testCDF() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    PoissonDistribution poisson(2.0);
    
    // Test basic properties
    double cdf0 = poisson.CDF(0.0);
    double cdf2 = poisson.CDF(2.0);
    double cdf10 = poisson.CDF(10.0);
    
    assert(cdf0 >= 0.0 && cdf0 <= 1.0);
    assert(cdf2 >= 0.0 && cdf2 <= 1.0);
    assert(cdf10 >= 0.0 && cdf10 <= 1.0);
    
    // CDF should be monotonic
    assert(cdf0 <= cdf2);
    assert(cdf2 <= cdf10);
    
    // CDF should approach 1 for large values
    assert(cdf10 > 0.99);
    
    // Test boundary cases
    assert(poisson.CDF(-1.0) == 0.0);
    
    // Test large lambda normal approximation
    PoissonDistribution poissonLarge(200.0);
    double cdfLarge = poissonLarge.CDF(200.0);
    assert(cdfLarge >= 0.0 && cdfLarge <= 1.0);
    
    std::cout << "✓ CDF tests passed" << std::endl;
}

/**
 * Test equality and I/O operators
 */
void testEqualityAndIO() {
    std::cout << "Testing equality and I/O operators..." << std::endl;
    
    PoissonDistribution poisson1(2.5);
    PoissonDistribution poisson2(2.5);
    PoissonDistribution poisson3(3.0);
    
    // Test equality
    assert(poisson1 == poisson2);
    assert(!(poisson1 == poisson3));
    
    // Test inequality
    assert(!(poisson1 != poisson2));
    assert(poisson1 != poisson3);
    
    // Test stream output
    std::ostringstream oss;
    oss << poisson1;
    std::string output = oss.str();
    assert(!output.empty());
    assert(output.find("Poisson") != std::string::npos);
    
    std::cout << "✓ Equality and I/O tests passed" << std::endl;
}

/**
 * Test performance characteristics
 */
void testPerformance() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    PoissonDistribution poisson(10.0);
    
    // Time probability calculations
    auto start = std::chrono::high_resolution_clock::now();
    
    const int numIterations = 10000;
    double sum = 0.0;
    for (int i = 0; i < numIterations; ++i) {
        sum += poisson.getProbability(i % 50);  // 0 to 49
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
    
    PoissonDistribution poisson(3.0);
    
    // First calculation should populate cache
    double prob1 = poisson.getProbability(3.0);
    
    // Second calculation should use cache (should be identical)
    double prob2 = poisson.getProbability(3.0);
    assert(prob1 == prob2);
    
    // Changing parameters should invalidate cache
    poisson.setLambda(5.0);
    double prob3 = poisson.getProbability(3.0);
    
    // The probabilities should be different (λ=3 vs λ=5 significantly affects P(X=3))
    assert(std::abs(prob3 - prob1) > 1e-6);
    
    // Test that log probability also works with caching
    double logProb1 = poisson.getLogProbability(3.0);
    double logProb2 = poisson.getLogProbability(3.0);
    assert(logProb1 == logProb2);
    
    std::cout << "✓ Caching tests passed" << std::endl;
}

int main() {
    std::cout << "Running Poisson distribution tests..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testNumericalStability();
        
        // Gold Standard Tests
        testLogProbability();
        testCDF();
        testEqualityAndIO();
        testPerformance();
        testCaching();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "✅ All Poisson distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
