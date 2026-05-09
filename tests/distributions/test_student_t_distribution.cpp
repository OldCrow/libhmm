#include <gtest/gtest.h>
#include "libhmm/distributions/student_t_distribution.h"
#include <memory>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace libhmm;

/**
 * Test basic Student's t-distribution functionality
 */
TEST(StudentTDistributionTest, BasicFunctionality) {

    // Test default constructor
    StudentTDistribution t_dist;
    EXPECT_EQ(t_dist.getDegreesOfFreedom(), 1.0);

    // Test parameterized constructor
    StudentTDistribution t_dist2(5.0);
    EXPECT_EQ(t_dist2.getDegreesOfFreedom(), 5.0);

    // Test parameter setter
    t_dist.setDegreesOfFreedom(3.0);
    EXPECT_EQ(t_dist.getDegreesOfFreedom(), 3.0);
}

/**
 * Test probability calculations
 */
TEST(StudentTDistributionTest, Probabilities) {

    StudentTDistribution t_dist(2.0); // df = 2

    // Test symmetric property: f(x) = f(-x)
    double prob_pos = t_dist.getProbability(1.0);
    double prob_neg = t_dist.getProbability(-1.0);
    EXPECT_NEAR(prob_pos, prob_neg, 1e-10);

    // Test that probability is highest at x = 0 (mode)
    double prob_at_zero = t_dist.getProbability(0.0);
    double prob_at_one = t_dist.getProbability(1.0);
    double prob_at_two = t_dist.getProbability(2.0);

    EXPECT_GT(prob_at_zero, prob_at_one);
    EXPECT_GT(prob_at_one, prob_at_two);
    EXPECT_GT(prob_at_zero, 0.0);
    EXPECT_GT(prob_at_one, 0.0);
    EXPECT_GT(prob_at_two, 0.0);

    // Test with different degrees of freedom
    StudentTDistribution t_dist_1(1.0);   // Cauchy distribution
    StudentTDistribution t_dist_30(30.0); // Close to normal

    double prob_1 = t_dist_1.getProbability(1.0);
    double prob_30 = t_dist_30.getProbability(1.0);

    // Higher df should have lower tails (lower probability at extreme values)
    // But for t(1) vs t(30), t(30) should have higher peak and lower tails
    EXPECT_GT(prob_1, 0.0);
    EXPECT_GT(prob_30, 0.0);
}

/**
 * Test statistical properties
 */
TEST(StudentTDistributionTest, StatisticalProperties) {

    // Test mean properties
    StudentTDistribution t_dist_1(1.0); // df = 1, mean undefined
    StudentTDistribution t_dist_2(2.0); // df = 2, mean = 0
    StudentTDistribution t_dist_5(5.0); // df = 5, mean = 0

    EXPECT_FALSE(t_dist_1.hasFiniteMean());
    EXPECT_TRUE(t_dist_2.hasFiniteMean());
    EXPECT_TRUE(t_dist_5.hasFiniteMean());

    EXPECT_TRUE(std::isnan(t_dist_1.getMean()));
    EXPECT_EQ(t_dist_2.getMean(), 0.0);
    EXPECT_EQ(t_dist_5.getMean(), 0.0);

    // Test variance properties
    EXPECT_FALSE(t_dist_1.hasFiniteVariance());
    EXPECT_FALSE(t_dist_2.hasFiniteVariance()); // df = 2, variance infinite

    StudentTDistribution t_dist_3(3.0); // df = 3, variance = 3/(3-2) = 3
    EXPECT_TRUE(t_dist_3.hasFiniteVariance());
    EXPECT_NEAR(t_dist_3.getVariance(), 3.0, 1e-10);
    EXPECT_NEAR(t_dist_3.getStandardDeviation(), std::sqrt(3.0), 1e-10);

    // Test variance formula: Var = ν/(ν-2) for ν > 2
    StudentTDistribution t_dist_10(10.0);
    double expected_var = 10.0 / (10.0 - 2.0);
    EXPECT_NEAR(t_dist_10.getVariance(), expected_var, 1e-10);
}

/**
 * Test parameter fitting
 */
TEST(StudentTDistributionTest, Fitting) {

    StudentTDistribution t_dist;

    // Test with data that should give reasonable estimates
    std::vector<Observation> data = {-2.1, -0.8, 0.1, 0.5, 1.2, -1.5, 0.9, -0.3, 1.8, -1.1};
    t_dist.fit(data);

    // After fitting, degrees of freedom should be positive and reasonable
    EXPECT_GT(t_dist.getDegreesOfFreedom(), 0.0);
    EXPECT_LT(t_dist.getDegreesOfFreedom(), 1000.0); // Reasonable upper bound

    // Test with insufficient data
    std::vector<Observation> single_point = {1.0};
    t_dist.fit(single_point);
    EXPECT_EQ(t_dist.getDegreesOfFreedom(), 1.0); // Should reset to default

    // Test with empty data - should reset to defaults
    std::vector<Observation> empty_data;
    t_dist.fit(empty_data);
    EXPECT_EQ(t_dist.getDegreesOfFreedom(), 1.0);
    EXPECT_EQ(t_dist.getLocation(), 0.0);
    EXPECT_EQ(t_dist.getScale(), 1.0);
}

/**
 * Test parameter validation
 */
TEST(StudentTDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    try {
        StudentTDistribution t_dist(0.0); // Zero degrees of freedom
        ADD_FAILURE();                    // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    try {
        StudentTDistribution t_dist(-1.0); // Negative degrees of freedom
        ADD_FAILURE();                     // Should not reach here
    } catch (const std::invalid_argument &) {
        // Expected behavior
    }

    // Test invalid parameters with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(StudentTDistribution t_dist(nan_val), std::invalid_argument);

    EXPECT_THROW(StudentTDistribution t_dist(inf_val), std::invalid_argument);

    // Test setter validation
    StudentTDistribution t_dist(1.0);

    EXPECT_THROW(t_dist.setDegreesOfFreedom(0.0), std::invalid_argument);

    EXPECT_THROW(t_dist.setDegreesOfFreedom(-5.0), std::invalid_argument);
}

/**
 * Test string representation and serialization
 */
TEST(StudentTDistributionTest, StringRepresentation) {

    StudentTDistribution t_dist(3.5);
    std::string str = t_dist.toString();

    // Should contain key information
    EXPECT_NE(str.find("StudentT Distribution"), std::string::npos);
    EXPECT_NE(str.find("3.5"), std::string::npos);
    EXPECT_NE(str.find("nu"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

/**
 * Test copy/move semantics
 */
TEST(StudentTDistributionTest, CopyMoveSemantics) {

    StudentTDistribution original(4.2);

    // Test copy constructor
    StudentTDistribution copied(original);
    EXPECT_EQ(copied.getDegreesOfFreedom(), original.getDegreesOfFreedom());
    EXPECT_EQ(copied.getProbability(1.0), original.getProbability(1.0));

    // Test copy assignment
    StudentTDistribution assigned(1.0);
    assigned = original;
    EXPECT_EQ(assigned.getDegreesOfFreedom(), original.getDegreesOfFreedom());
    EXPECT_EQ(assigned.getProbability(1.0), original.getProbability(1.0));

    // Test move constructor
    double original_df = original.getDegreesOfFreedom();
    StudentTDistribution moved(std::move(original));
    EXPECT_EQ(moved.getDegreesOfFreedom(), original_df);
}

/**
 * Test equality operators
 */
TEST(StudentTDistributionTest, EqualityOperators) {

    StudentTDistribution t1(3.0);
    StudentTDistribution t2(3.0);
    StudentTDistribution t3(4.0);

    EXPECT_EQ(t1, t2);
    EXPECT_FALSE(t1 == t3);
    EXPECT_NE(t1, t3);
    EXPECT_FALSE(t1 != t2);
}

/**
 * Test edge cases and numerical stability
 */
TEST(StudentTDistributionTest, EdgeCases) {

    // Test very small degrees of freedom
    StudentTDistribution t_small(1e-5);
    double prob_small = t_small.getProbability(0.0);
    EXPECT_TRUE(std::isfinite(prob_small) && prob_small > 0.0);

    // Test very large degrees of freedom (should approach normal distribution)
    StudentTDistribution t_large(1e5);
    double prob_large = t_large.getProbability(0.0);
    EXPECT_TRUE(std::isfinite(prob_large) && prob_large > 0.0);

    // Test reset functionality
    StudentTDistribution t_test(10.0);
    t_test.reset();
    EXPECT_EQ(t_test.getDegreesOfFreedom(), 1.0);
}

/**
 * Test log probability calculations
 */
TEST(StudentTDistributionTest, LogProbability) {

    StudentTDistribution t_dist(2.0); // df=2

    // Test log probability at a few points
    double log_prob_0 = t_dist.getLogProbability(0.0);
    double log_prob_1 = t_dist.getLogProbability(1.0);
    double log_prob_2 = t_dist.getLogProbability(2.0);

    EXPECT_TRUE(std::isfinite(log_prob_0));
    EXPECT_TRUE(std::isfinite(log_prob_1));
    EXPECT_TRUE(std::isfinite(log_prob_2));

    // Test consistency between probability and log probability
    double prob_0 = t_dist.getProbability(0.0);
    double calculated_log_prob_0 = std::log(prob_0);
    EXPECT_NEAR(calculated_log_prob_0, log_prob_0, 1e-10);
}

/**
 * Test cumulative distribution function
 */
TEST(StudentTDistributionTest, CDF) {

    StudentTDistribution t_dist(3.0); // df=3

    // Test CDF values at a few points
    double cdf_0 = t_dist.getCumulativeProbability(0.0);
    double cdf_1 = t_dist.getCumulativeProbability(1.0);
    double cdf_2 = t_dist.getCumulativeProbability(2.0);

    EXPECT_TRUE(cdf_0 > 0.0 && cdf_0 < 1.0);
    EXPECT_GT(cdf_1, cdf_0);
    EXPECT_GT(cdf_2, cdf_1);
}

/**
 * Test invalid input handling
 */
TEST(StudentTDistributionTest, InvalidInputHandling) {

    StudentTDistribution t_dist(4.0);

    // Test probability with NaN (should return 0.0 like other distributions)
    EXPECT_EQ(t_dist.getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);

    // Test CDF with NaN (should return NaN for CDF)
    EXPECT_TRUE(
        std::isnan(t_dist.getCumulativeProbability(std::numeric_limits<double>::quiet_NaN())));
}

/**
 * Test caching behavior
 */
TEST(StudentTDistributionTest, Caching) {

    StudentTDistribution t_dist(5.0);

    // First call should update cache
    double prob1 = t_dist.getProbability(1.5);

    // Change degrees of freedom to invalidate cache
    t_dist.setDegreesOfFreedom(10.0);
    double prob2 = t_dist.getProbability(1.5);
    EXPECT_NE(prob1, prob2); // Should differ because cache was invalidated
}

/**
 * Test performance of key functions
 */
TEST(StudentTDistributionTest, Performance) {

    StudentTDistribution t_dist(6.0);

    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 10000;
    volatile double sum = 0.0; // volatile to prevent optimization

    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i) / 1000.0 - 5.0; // Range -5 to 5
        sum += t_dist.getProbability(x);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;

    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;

    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i) / 1000.0 - 5.0; // Range -5 to 5
        logSum += t_dist.getLogProbability(x);
    }

    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;

    // Test fitting timing
    std::vector<Observation> fitData(1000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i) / 100.0 - 5.0; // Range -5 to 5
    }

    start = std::chrono::high_resolution_clock::now();
    t_dist.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fitDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double fitTimePerPoint = static_cast<double>(fitDuration.count()) / fitData.size();

    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) << pdfTimePerCall
              << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3) << logPdfTimePerCall
              << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) << fitTimePerPoint
              << " μs/point (" << fitData.size() << " points)" << std::endl;

    // Performance requirements (should be reasonable)
    EXPECT_LT(pdfTimePerCall, 10.0);   // Less than 10 μs per PDF call
    EXPECT_LT(logPdfTimePerCall, 5.0); // Less than 5 μs per log PDF call
    EXPECT_LT(fitTimePerPoint, 50.0);  // Less than 50 μs per data point for fitting
}

/**
 * Main test function
 */
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
