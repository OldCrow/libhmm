#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <sstream>
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

/**
 * Test log probability calculations
 */
void testLogProbability() {
    std::cout << "Testing log probability calculations..." << std::endl;
    
    NegativeBinomialDistribution negbinom(5.0, 0.4);
    
    // Test log probability for valid values
    double logProb0 = negbinom.getLogProbability(0.0);
    double logProb1 = negbinom.getLogProbability(1.0);
    double logProb5 = negbinom.getLogProbability(5.0);
    
    // Log probabilities should be real numbers
    assert(!std::isnan(logProb0));
    assert(!std::isnan(logProb1));
    assert(!std::isnan(logProb5));
    
    // Verify relationship: exp(log_prob) ≈ prob
    double prob0 = negbinom.getProbability(0.0);
    double prob1 = negbinom.getProbability(1.0);
    
    assert(std::abs(std::exp(logProb0) - prob0) < 1e-10);
    assert(std::abs(std::exp(logProb1) - prob1) < 1e-10);
    
    // Test invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    assert(negbinom.getLogProbability(nan_val) == -std::numeric_limits<double>::infinity());
    assert(negbinom.getLogProbability(inf_val) == -std::numeric_limits<double>::infinity());
    assert(negbinom.getLogProbability(-1.0) == -std::numeric_limits<double>::infinity());
    
    // Test edge case p = 1
    NegativeBinomialDistribution negbinom_p1(5.0, 1.0);
    assert(negbinom_p1.getLogProbability(0.0) == 0.0);  // log(1) = 0
    assert(negbinom_p1.getLogProbability(1.0) == -std::numeric_limits<double>::infinity());
    
    std::cout << "✓ Log probability calculation tests passed" << std::endl;
}

/**
 * Test CDF calculations
 */
void testCDF() {
    std::cout << "Testing CDF calculations..." << std::endl;
    
    NegativeBinomialDistribution negbinom(3.0, 0.6);
    
    // Test CDF properties
    double cdf0 = negbinom.CDF(0.0);
    double cdf1 = negbinom.CDF(1.0);
    double cdf5 = negbinom.CDF(5.0);
    double cdf10 = negbinom.CDF(10.0);
    
    // CDF should be non-decreasing
    assert(cdf0 <= cdf1);
    assert(cdf1 <= cdf5);
    assert(cdf5 <= cdf10);
    
    // CDF should be in [0,1]
    assert(cdf0 >= 0.0 && cdf0 <= 1.0);
    assert(cdf1 >= 0.0 && cdf1 <= 1.0);
    assert(cdf5 >= 0.0 && cdf5 <= 1.0);
    assert(cdf10 >= 0.0 && cdf10 <= 1.0);
    
    // CDF should equal probability at 0
    assert(std::abs(cdf0 - negbinom.getProbability(0.0)) < 1e-10);
    
    // Test boundary cases
    assert(negbinom.CDF(-1.0) == 0.0);
    
    // Test invalid inputs
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    assert(negbinom.CDF(nan_val) == 0.0);
    assert(negbinom.CDF(inf_val) == 0.0);
    
    std::cout << "✓ CDF calculation tests passed" << std::endl;
}

/**
 * Test additional statistical properties
 */
void testAdditionalStatistics() {
    std::cout << "Testing additional statistical properties..." << std::endl;
    
    NegativeBinomialDistribution negbinom(4.0, 0.3);
    
    // Test mode calculation
    int mode = negbinom.getMode();
    assert(mode >= 0);
    
    // For r > 1, mode should be floor((r-1)*(1-p)/p)
    int expected_mode = static_cast<int>(std::floor((4.0 - 1.0) * (1.0 - 0.3) / 0.3));
    assert(mode == expected_mode);
    
    // Test case where r <= 1
    NegativeBinomialDistribution negbinom_small_r(0.5, 0.3);
    assert(negbinom_small_r.getMode() == 0);
    
    // Test skewness
    double skewness = negbinom.getSkewness();
    double expected_skewness = (2.0 - 0.3) / std::sqrt(4.0 * (1.0 - 0.3));
    assert(std::abs(skewness - expected_skewness) < 1e-10);
    
    // Test kurtosis
    double kurtosis = negbinom.getKurtosis();
    double expected_kurtosis = 3.0 + (6.0 / 4.0) + (0.3 * 0.3) / (4.0 * (1.0 - 0.3));
    assert(std::abs(kurtosis - expected_kurtosis) < 1e-10);
    
    std::cout << "✓ Additional statistical properties tests passed" << std::endl;
}

/**
 * Test equality operators
 */
void testEqualityOperators() {
    std::cout << "Testing equality operators..." << std::endl;
    
    NegativeBinomialDistribution negbinom1(5.0, 0.4);
    NegativeBinomialDistribution negbinom2(5.0, 0.4);
    NegativeBinomialDistribution negbinom3(5.0, 0.5);
    NegativeBinomialDistribution negbinom4(6.0, 0.4);
    
    // Test equality
    assert(negbinom1 == negbinom2);
    assert(!(negbinom1 == negbinom3));
    assert(!(negbinom1 == negbinom4));
    
    // Test inequality
    assert(!(negbinom1 != negbinom2));
    assert(negbinom1 != negbinom3);
    assert(negbinom1 != negbinom4);
    
    std::cout << "✓ Equality operator tests passed" << std::endl;
}

/**
 * Test stream operators
 */
void testStreamOperators() {
    std::cout << "Testing stream operators..." << std::endl;
    
    NegativeBinomialDistribution original(7.5, 0.35);
    
    // Test output operator
    std::ostringstream oss;
    oss << original;
    std::string output = oss.str();
    
    assert(output.find("Negative Binomial") != std::string::npos);
    assert(output.find("7.5") != std::string::npos);
    assert(output.find("0.35") != std::string::npos);
    
    // Test input operator with simple format
    std::istringstream iss("3.2 0.8");
    NegativeBinomialDistribution parsed;
    iss >> parsed;
    
    assert(std::abs(parsed.getR() - 3.2) < 1e-10);
    assert(std::abs(parsed.getP() - 0.8) < 1e-10);
    
    // Test input operator with NegativeBinomial format
    std::istringstream iss2("NegativeBinomial(4.5,0.6)");
    NegativeBinomialDistribution parsed2;
    iss2 >> parsed2;
    
    assert(std::abs(parsed2.getR() - 4.5) < 1e-10);
    assert(std::abs(parsed2.getP() - 0.6) < 1e-10);
    
    std::cout << "✓ Stream operator tests passed" << std::endl;
}

/**
 * Test caching performance and correctness
 */
void testCaching() {
    std::cout << "Testing caching performance and correctness..." << std::endl;
    
    NegativeBinomialDistribution negbinom(6.0, 0.4);
    
    // First call should populate cache
    double prob1_first = negbinom.getProbability(1.0);
    double logProb1_first = negbinom.getLogProbability(1.0);
    
    // Subsequent calls should use cached values and give identical results
    double prob1_second = negbinom.getProbability(1.0);
    double logProb1_second = negbinom.getLogProbability(1.0);
    
    assert(prob1_first == prob1_second);
    assert(logProb1_first == logProb1_second);
    
    // Changing parameters should invalidate cache
    negbinom.setP(0.5);
    double prob1_after_change = negbinom.getProbability(1.0);
    
    // Result should be different after parameter change
    assert(prob1_after_change != prob1_first);
    
    // Test cache invalidation with setR
    negbinom.setR(7.0);
    double prob1_after_r_change = negbinom.getProbability(1.0);
    assert(prob1_after_r_change != prob1_after_change);
    
    // Test cache invalidation with setParameters
    negbinom.setParameters(8.0, 0.3);
    double prob1_after_both_change = negbinom.getProbability(1.0);
    assert(prob1_after_both_change != prob1_after_r_change);
    
    std::cout << "✓ Caching tests passed" << std::endl;
}

/**
 * Test numerical stability
 */
void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;
    
    // Test with large parameters
    NegativeBinomialDistribution large_negbinom(100.0, 0.99);
    double prob_large = large_negbinom.getProbability(0.0);
    assert(prob_large > 0.0 && prob_large <= 1.0);
    assert(!std::isnan(prob_large) && !std::isinf(prob_large));
    
    // Test with small parameters
    NegativeBinomialDistribution small_negbinom(0.1, 0.01);
    double prob_small = small_negbinom.getProbability(10.0);
    assert(prob_small >= 0.0 && prob_small <= 1.0);
    assert(!std::isnan(prob_small));
    
    // Test log probability for better numerical stability
    double logProb_large = large_negbinom.getLogProbability(0.0);
    double logProb_small = small_negbinom.getLogProbability(10.0);
    
    assert(!std::isnan(logProb_large));
    assert(!std::isnan(logProb_small));
    
    std::cout << "✓ Numerical stability tests passed" << std::endl;
}

/**
 * Test performance characteristics
 */
void testPerformance() {
    std::cout << "Testing performance characteristics..." << std::endl;
    
    NegativeBinomialDistribution negbinom(8.0, 0.4);
    
    // Time probability calculations
    auto start = std::chrono::high_resolution_clock::now();
    
    const int numIterations = 10000;
    double sum = 0.0;
    for (int i = 0; i < numIterations; ++i) {
        sum += negbinom.getProbability(i % 30);  // 0 to 29
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Computed " << numIterations << " probabilities in " 
              << duration.count() << " microseconds" << std::endl;
    std::cout << "Average time per calculation: " 
              << static_cast<double>(duration.count()) / numIterations << " microseconds" << std::endl;
    
    // Should complete in reasonable time (< 1 second)
    assert(duration.count() < 1000000); // 1 second = 1,000,000 microseconds
    
    // Test log probability performance
    auto logStart = std::chrono::high_resolution_clock::now();
    double logSum = 0.0;
    for (int i = 0; i < numIterations; ++i) {
        logSum += negbinom.getLogProbability(i % 30);
    }
    auto logEnd = std::chrono::high_resolution_clock::now();
    auto logDuration = std::chrono::duration_cast<std::chrono::microseconds>(logEnd - logStart);
    
    std::cout << "Log probability performance: " << logDuration.count() << " microseconds" << std::endl;
    
    // Log probability should be reasonably fast (similar to regular probability due to caching)
    assert(logDuration.count() < 2000000); // Allow 2x regular probability time
    
    std::cout << "✓ Performance tests passed" << std::endl;
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
        testLogProbability();
        testCDF();
        testAdditionalStatistics();
        testEqualityOperators();
        testStreamOperators();
        testCaching();
        testNumericalStability();
        testPerformance();
        
        std::cout << "===============================================" << std::endl;
        std::cout << "✅ All Gold Standard Negative Binomial distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
