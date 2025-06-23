#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include "libhmm/distributions/student_t_distribution.h"

using libhmm::StudentTDistribution;
using libhmm::Observation;

/**
 * Test basic Student's t-distribution functionality
 */
void testBasicFunctionality() {
    std::cout << "Testing basic Student's t-distribution functionality..." << std::endl;
    
    // Test default constructor
    StudentTDistribution t_dist;
    assert(t_dist.getDegreesOfFreedom() == 1.0);
    
    // Test parameterized constructor
    StudentTDistribution t_dist2(5.0);
    assert(t_dist2.getDegreesOfFreedom() == 5.0);
    
    // Test parameter setter
    t_dist.setDegreesOfFreedom(3.0);
    assert(t_dist.getDegreesOfFreedom() == 3.0);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

/**
 * Test probability calculations
 */
void testProbabilities() {
    std::cout << "Testing probability calculations..." << std::endl;
    
    StudentTDistribution t_dist(2.0);  // df = 2
    
    // Test symmetric property: f(x) = f(-x)
    double prob_pos = t_dist.getProbability(1.0);
    double prob_neg = t_dist.getProbability(-1.0);
    assert(std::abs(prob_pos - prob_neg) < 1e-10);
    
    // Test that probability is highest at x = 0 (mode)
    double prob_at_zero = t_dist.getProbability(0.0);
    double prob_at_one = t_dist.getProbability(1.0);
    double prob_at_two = t_dist.getProbability(2.0);
    
    assert(prob_at_zero > prob_at_one);
    assert(prob_at_one > prob_at_two);
    assert(prob_at_zero > 0.0);
    assert(prob_at_one > 0.0);
    assert(prob_at_two > 0.0);
    
    // Test with different degrees of freedom
    StudentTDistribution t_dist_1(1.0);   // Cauchy distribution
    StudentTDistribution t_dist_30(30.0); // Close to normal
    
    double prob_1 = t_dist_1.getProbability(1.0);
    double prob_30 = t_dist_30.getProbability(1.0);
    
    // Higher df should have lower tails (lower probability at extreme values)
    // But for t(1) vs t(30), t(30) should have higher peak and lower tails
    assert(prob_1 > 0.0);
    assert(prob_30 > 0.0);
    
    std::cout << "✓ Probability calculation tests passed" << std::endl;
}

/**
 * Test statistical properties
 */
void testStatisticalProperties() {
    std::cout << "Testing statistical properties..." << std::endl;
    
    // Test mean properties
    StudentTDistribution t_dist_1(1.0);   // df = 1, mean undefined
    StudentTDistribution t_dist_2(2.0);   // df = 2, mean = 0
    StudentTDistribution t_dist_5(5.0);   // df = 5, mean = 0
    
    assert(!t_dist_1.hasFiniteMean());
    assert(t_dist_2.hasFiniteMean());
    assert(t_dist_5.hasFiniteMean());
    
    assert(std::isnan(t_dist_1.getMean()));
    assert(t_dist_2.getMean() == 0.0);
    assert(t_dist_5.getMean() == 0.0);
    
    // Test variance properties
    assert(!t_dist_1.hasFiniteVariance());
    assert(!t_dist_2.hasFiniteVariance());  // df = 2, variance infinite
    
    StudentTDistribution t_dist_3(3.0);   // df = 3, variance = 3/(3-2) = 3
    assert(t_dist_3.hasFiniteVariance());
    assert(std::abs(t_dist_3.getVariance() - 3.0) < 1e-10);
    assert(std::abs(t_dist_3.getStandardDeviation() - std::sqrt(3.0)) < 1e-10);
    
    // Test variance formula: Var = ν/(ν-2) for ν > 2
    StudentTDistribution t_dist_10(10.0);
    double expected_var = 10.0 / (10.0 - 2.0);
    assert(std::abs(t_dist_10.getVariance() - expected_var) < 1e-10);
    
    std::cout << "✓ Statistical properties tests passed" << std::endl;
}

/**
 * Test parameter fitting
 */
void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    StudentTDistribution t_dist;
    
    // Test with data that should give reasonable estimates
    std::vector<Observation> data = {-2.1, -0.8, 0.1, 0.5, 1.2, -1.5, 0.9, -0.3, 1.8, -1.1};
    t_dist.fit(data);
    
    // After fitting, degrees of freedom should be positive and reasonable
    assert(t_dist.getDegreesOfFreedom() > 0.0);
    assert(t_dist.getDegreesOfFreedom() < 1000.0);  // Reasonable upper bound
    
    // Test with insufficient data
    std::vector<Observation> single_point = {1.0};
    t_dist.fit(single_point);
    assert(t_dist.getDegreesOfFreedom() == 1.0);  // Should reset to default
    
    // Test with empty data - should throw exception
    std::vector<Observation> empty_data;
    try {
        t_dist.fit(empty_data);
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
    
    // Test invalid constructor parameters
    try {
        StudentTDistribution t_dist(0.0);  // Zero degrees of freedom
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        StudentTDistribution t_dist(-1.0);  // Negative degrees of freedom
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    try {
        StudentTDistribution t_dist(nan_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        StudentTDistribution t_dist(inf_val);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test setter validation
    StudentTDistribution t_dist(1.0);
    
    try {
        t_dist.setDegreesOfFreedom(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        t_dist.setDegreesOfFreedom(-5.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "✓ Parameter validation tests passed" << std::endl;
}

/**
 * Test string representation and serialization
 */
void testStringRepresentation() {
    std::cout << "Testing string representation..." << std::endl;
    
    StudentTDistribution t_dist(3.5);
    std::string str = t_dist.toString();
    
    // Should contain key information
    assert(str.find("StudentT Distribution") != std::string::npos);
    assert(str.find("3.5") != std::string::npos);
    assert(str.find("nu") != std::string::npos);
    
    // Test fromString functionality
    StudentTDistribution t_dist_from_str = StudentTDistribution::fromString("StudentT(ν=7.25)");
    assert(std::abs(t_dist_from_str.getDegreesOfFreedom() - 7.25) < 1e-10);
    
    StudentTDistribution t_dist_from_str2 = StudentTDistribution::fromString("StudentT(df=12.0)");
    assert(std::abs(t_dist_from_str2.getDegreesOfFreedom() - 12.0) < 1e-10);
    
    // Test invalid string formats
    try {
        StudentTDistribution::fromString("InvalidFormat");
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation tests passed" << std::endl;
}

/**
 * Test copy/move semantics
 */
void testCopyMoveSemantics() {
    std::cout << "Testing copy/move semantics..." << std::endl;
    
    StudentTDistribution original(4.2);
    
    // Test copy constructor
    StudentTDistribution copied(original);
    assert(copied.getDegreesOfFreedom() == original.getDegreesOfFreedom());
    assert(copied.getProbability(1.0) == original.getProbability(1.0));
    
    // Test copy assignment
    StudentTDistribution assigned(1.0);
    assigned = original;
    assert(assigned.getDegreesOfFreedom() == original.getDegreesOfFreedom());
    assert(assigned.getProbability(1.0) == original.getProbability(1.0));
    
    // Test move constructor
    double original_df = original.getDegreesOfFreedom();
    StudentTDistribution moved(std::move(original));
    assert(moved.getDegreesOfFreedom() == original_df);
    
    std::cout << "✓ Copy/move semantics tests passed" << std::endl;
}

/**
 * Test equality operators
 */
void testEqualityOperators() {
    std::cout << "Testing equality operators..." << std::endl;
    
    StudentTDistribution t1(3.0);
    StudentTDistribution t2(3.0);
    StudentTDistribution t3(4.0);
    
    assert(t1 == t2);
    assert(!(t1 == t3));
    assert(t1 != t3);
    assert(!(t1 != t2));
    
    std::cout << "✓ Equality operator tests passed" << std::endl;
}

/**
 * Test edge cases and numerical stability
 */
void testEdgeCases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test very small degrees of freedom
    StudentTDistribution t_small(1e-5);
    double prob_small = t_small.getProbability(0.0);
    assert(std::isfinite(prob_small) && prob_small > 0.0);
    
    // Test very large degrees of freedom (should approach normal distribution)
    StudentTDistribution t_large(1e5);
    double prob_large = t_large.getProbability(0.0);
    assert(std::isfinite(prob_large) && prob_large > 0.0);
    
    // Test reset functionality
    StudentTDistribution t_test(10.0);
    t_test.reset();
    assert(t_test.getDegreesOfFreedom() == 1.0);
    
    std::cout << "✓ Edge cases tests passed" << std::endl;
}

/**
 * Main test function
 */
int main() {
    std::cout << "=== Student's t-Distribution Unit Tests ===" << std::endl;
    std::cout << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilities();
        testStatisticalProperties();
        testFitting();
        testParameterValidation();
        testStringRepresentation();
        testCopyMoveSemantics();
        testEqualityOperators();
        testEdgeCases();
        
        std::cout << std::endl;
        std::cout << "🎉 All Student's t-distribution tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
