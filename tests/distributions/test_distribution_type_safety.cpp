#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "libhmm/distributions/distributions.h"

using namespace libhmm;

class DistributionTypeSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test values
        epsilon = 1e-10;
    }
    
    double epsilon;
};

TEST_F(DistributionTypeSafetyTest, DiscreteDistributionConstructorValidation) {
    // Test invalid constructor arguments
    EXPECT_THROW(DiscreteDistribution(0), std::invalid_argument);
    
    // Test valid construction
    EXPECT_NO_THROW(DiscreteDistribution(5));
    
    DiscreteDistribution dist(5);
    EXPECT_EQ(dist.getNumSymbols(), 5);
}

TEST_F(DistributionTypeSafetyTest, DiscreteDistributionBoundsChecking) {
    DiscreteDistribution dist(5);
    
    // Test valid probability access
    double prob1 = dist.getProbability(2.0);
    EXPECT_NEAR(prob1, 0.2, epsilon);  // Uniform distribution: 1/5 = 0.2
    
    // Test out-of-bounds access (should return 0.0, not throw)
    double prob2 = dist.getProbability(10.0);
    EXPECT_DOUBLE_EQ(prob2, 0.0);
    
    // Test negative index (should return 0.0)
    double prob3 = dist.getProbability(-1.0);
    EXPECT_DOUBLE_EQ(prob3, 0.0);
    
    // Test NaN input (should return 0.0)
    double prob4 = dist.getProbability(std::numeric_limits<double>::quiet_NaN());
    EXPECT_DOUBLE_EQ(prob4, 0.0);
}

TEST_F(DistributionTypeSafetyTest, DiscreteDistributionSetProbabilityValidation) {
    DiscreteDistribution dist(5);
    
    // Test valid setProbability
    EXPECT_NO_THROW(dist.setProbability(2, 0.5));
    
    // Test invalid probability > 1
    EXPECT_THROW(dist.setProbability(2, 1.5), std::invalid_argument);
    
    // Test out-of-bounds index
    EXPECT_THROW(dist.setProbability(10, 0.5), std::out_of_range);
    
    // Test NaN probability
    double nan_prob = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(dist.setProbability(2, nan_prob), std::invalid_argument);
    
    // Test negative probability
    EXPECT_THROW(dist.setProbability(2, -0.1), std::invalid_argument);
}

TEST_F(DistributionTypeSafetyTest, GaussianDistributionConstructorValidation) {
    // Test valid construction
    EXPECT_NO_THROW(GaussianDistribution(0.0, 1.0));
    
    // Test zero standard deviation
    EXPECT_THROW(GaussianDistribution(0.0, 0.0), std::invalid_argument);
    
    // Test negative standard deviation
    EXPECT_THROW(GaussianDistribution(0.0, -1.0), std::invalid_argument);
    
    // Test NaN mean
    double nan_mean = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(GaussianDistribution(nan_mean, 1.0), std::invalid_argument);
    
    // Test NaN standard deviation  
    double nan_std = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(GaussianDistribution(0.0, nan_std), std::invalid_argument);
    
    // Test infinite values
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW(GaussianDistribution(inf_val, 1.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(0.0, inf_val), std::invalid_argument);
}

TEST_F(DistributionTypeSafetyTest, GaussianDistributionInputValidation) {
    GaussianDistribution dist(0.0, 1.0);
    
    // Test valid input
    double prob1 = dist.getProbability(0.0);
    EXPECT_GT(prob1, 0.0);  // Should be positive
    
    // Test NaN input (should return 0.0)
    double prob2 = dist.getProbability(std::numeric_limits<double>::quiet_NaN());
    EXPECT_DOUBLE_EQ(prob2, 0.0);
    
    // Test infinite input (should return 0.0)
    double prob3 = dist.getProbability(std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(prob3, 0.0);
    
    double prob4 = dist.getProbability(-std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(prob4, 0.0);
}

TEST_F(DistributionTypeSafetyTest, GaussianDistributionSetterValidation) {
    GaussianDistribution dist(0.0, 1.0);
    
    // Test valid setters
    EXPECT_NO_THROW(dist.setMean(5.0));
    EXPECT_NO_THROW(dist.setStandardDeviation(2.0));
    
    // Test invalid mean
    double nan_mean = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW(dist.setMean(nan_mean), std::invalid_argument);
    EXPECT_THROW(dist.setMean(inf_val), std::invalid_argument);
    
    // Test invalid standard deviation
    EXPECT_THROW(dist.setStandardDeviation(-1.0), std::invalid_argument);
    EXPECT_THROW(dist.setStandardDeviation(0.0), std::invalid_argument);
    double nan_std = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(dist.setStandardDeviation(nan_std), std::invalid_argument);
    EXPECT_THROW(dist.setStandardDeviation(inf_val), std::invalid_argument);
}

TEST_F(DistributionTypeSafetyTest, ExponentialDistributionValidation) {
    // Test valid construction
    EXPECT_NO_THROW(ExponentialDistribution(1.0));
    
    // Test invalid rate parameter (non-positive)
    EXPECT_THROW(ExponentialDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(ExponentialDistribution(-1.0), std::invalid_argument);
    
    // Test NaN parameter
    EXPECT_THROW((ExponentialDistribution(std::numeric_limits<double>::quiet_NaN())), std::invalid_argument);
    
    // Test input validation
    ExponentialDistribution dist(1.0);
    
    // Valid inputs - note: exponential dist has getProbability(0.0) = 0.0
    EXPECT_GE(dist.getProbability(0.0), 0.0);  // Can be 0 at x=0
    EXPECT_GT(dist.getProbability(1.0), 0.0);
    
    // Invalid inputs (should return 0.0)
    EXPECT_DOUBLE_EQ(dist.getProbability(-1.0), 0.0);  // Negative values
    EXPECT_DOUBLE_EQ(dist.getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
}

TEST_F(DistributionTypeSafetyTest, PoissonDistributionValidation) {
    // Test valid construction
    EXPECT_NO_THROW(PoissonDistribution(2.0));
    
    // Test invalid rate parameter (non-positive)
    EXPECT_THROW(PoissonDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(PoissonDistribution(-1.0), std::invalid_argument);
    
    // Test NaN parameter
    EXPECT_THROW((PoissonDistribution(std::numeric_limits<double>::quiet_NaN())), std::invalid_argument);
    
    // Test input validation
    PoissonDistribution dist(2.0);
    
    // Valid inputs (non-negative integers)
    EXPECT_GT(dist.getProbability(0.0), 0.0);
    EXPECT_GT(dist.getProbability(2.0), 0.0);
    
    // Invalid inputs (should return 0.0)
    EXPECT_DOUBLE_EQ(dist.getProbability(-1.0), 0.0);  // Negative values
    EXPECT_DOUBLE_EQ(dist.getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
}

TEST_F(DistributionTypeSafetyTest, BetaDistributionValidation) {
    // Test valid construction
    EXPECT_NO_THROW(BetaDistribution(2.0, 3.0));
    
    // Test invalid parameters (non-positive)
    EXPECT_THROW(BetaDistribution(0.0, 3.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(2.0, 0.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(-1.0, 3.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(2.0, -1.0), std::invalid_argument);
    
    // Test input validation
    BetaDistribution dist(2.0, 3.0);
    
    // Valid inputs (in [0,1])
    EXPECT_GT(dist.getProbability(0.5), 0.0);
    EXPECT_GE(dist.getProbability(0.0), 0.0);  // May be 0 at boundaries
    EXPECT_GE(dist.getProbability(1.0), 0.0);
    
    // Invalid inputs (outside [0,1] should return 0.0)
    EXPECT_DOUBLE_EQ(dist.getProbability(-0.1), 0.0);
    EXPECT_DOUBLE_EQ(dist.getProbability(1.1), 0.0);
    EXPECT_DOUBLE_EQ(dist.getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
}

// Integration test for multiple distributions
TEST_F(DistributionTypeSafetyTest, MultipleDistributionTypeSafety) {
    // Create various distributions
    std::vector<std::unique_ptr<ProbabilityDistribution>> distributions;
    
    distributions.push_back(std::make_unique<DiscreteDistribution>(4));
    distributions.push_back(std::make_unique<GaussianDistribution>(0.0, 1.0));
    distributions.push_back(std::make_unique<ExponentialDistribution>(1.0));
    distributions.push_back(std::make_unique<PoissonDistribution>(3.0));
    distributions.push_back(std::make_unique<BetaDistribution>(2.0, 2.0));
    
    // Test that all distributions handle invalid inputs gracefully
    for (auto& dist : distributions) {
        // NaN input should return 0.0 for all distributions
        double prob_nan = dist->getProbability(std::numeric_limits<double>::quiet_NaN());
        EXPECT_DOUBLE_EQ(prob_nan, 0.0) << "Distribution failed NaN input test";
        
        // Should not crash on any reasonable input
        EXPECT_NO_THROW(dist->getProbability(0.0));
        EXPECT_NO_THROW(dist->getProbability(1.0));
    }
}
