#include <gtest/gtest.h>
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gamma_distribution.h"
#include "libhmm/distributions/exponential_distribution.h"
#include "libhmm/distributions/log_normal_distribution.h"
#include "libhmm/distributions/pareto_distribution.h"
#include "libhmm/distributions/poisson_distribution.h"
#include <memory>
#include <vector>
#include <cmath>
#include <limits>

using namespace libhmm;

// Test fixture for each distribution type
template<typename DistributionType>
class DistributionTest : public ::testing::Test {
protected:
    virtual std::unique_ptr<DistributionType> createDefaultDistribution() = 0;
    virtual std::vector<double> getValidTestValues() = 0;
    virtual std::vector<double> getInvalidTestValues() = 0;
};

// Gaussian Distribution Tests
class GaussianDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<GaussianDistribution>(0.0, 1.0);
    }
    
    std::unique_ptr<GaussianDistribution> dist_;
};

TEST_F(GaussianDistributionTest, DefaultConstructor) {
    GaussianDistribution defaultDist;
    EXPECT_DOUBLE_EQ(defaultDist.getMean(), 0.0);
    EXPECT_DOUBLE_EQ(defaultDist.getStandardDeviation(), 1.0);
}

TEST_F(GaussianDistributionTest, ParameterizedConstructor) {
    GaussianDistribution dist(5.0, 2.0);
    EXPECT_DOUBLE_EQ(dist.getMean(), 5.0);
    EXPECT_DOUBLE_EQ(dist.getStandardDeviation(), 2.0);
}

TEST_F(GaussianDistributionTest, InvalidConstructorParameters) {
    // Invalid standard deviation
    EXPECT_THROW(GaussianDistribution(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(0.0, -1.0), std::invalid_argument);
    
    // Invalid mean
    EXPECT_THROW(GaussianDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(std::numeric_limits<double>::infinity(), 1.0), std::invalid_argument);
}

TEST_F(GaussianDistributionTest, ProbabilityCalculation) {
    // Test at mean (should be highest probability)
    double probAtMean = dist_->getProbability(0.0);
    EXPECT_GT(probAtMean, 0.0);
    
    // Test at other values
    double probAt1 = dist_->getProbability(1.0);
    double probAtNeg1 = dist_->getProbability(-1.0);
    
    EXPECT_GT(probAt1, 0.0);
    EXPECT_GT(probAtNeg1, 0.0);
    
    // Due to symmetry, should be equal
    EXPECT_NEAR(probAt1, probAtNeg1, 1e-10);
    
    // Probability at mean should be higher than at ±1
    EXPECT_GT(probAtMean, probAt1);
}

TEST_F(GaussianDistributionTest, InvalidInputHandling) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(GaussianDistributionTest, ParameterSettersAndGetters) {
    dist_->setMean(10.0);
    EXPECT_DOUBLE_EQ(dist_->getMean(), 10.0);
    
    dist_->setStandardDeviation(3.0);
    EXPECT_DOUBLE_EQ(dist_->getStandardDeviation(), 3.0);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setMean(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(dist_->setStandardDeviation(0.0), std::invalid_argument);
    EXPECT_THROW(dist_->setStandardDeviation(-1.0), std::invalid_argument);
}

TEST_F(GaussianDistributionTest, FittingToData) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    dist_->fit(data);
    
    // Mean should be approximately 3.0
    EXPECT_NEAR(dist_->getMean(), 3.0, 1e-10);
    
    // Standard deviation should be calculated correctly
    EXPECT_GT(dist_->getStandardDeviation(), 0.0);
}

TEST_F(GaussianDistributionTest, ResetFunctionality) {
    dist_->setMean(100.0);
    dist_->setStandardDeviation(50.0);
    
    dist_->reset();
    
    EXPECT_DOUBLE_EQ(dist_->getMean(), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getStandardDeviation(), 1.0);
}

// Discrete Distribution Tests
class DiscreteDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<DiscreteDistribution>(5); // 5 symbols
    }
    
    std::unique_ptr<DiscreteDistribution> dist_;
};

TEST_F(DiscreteDistributionTest, ConstructorValidation) {
    EXPECT_THROW(DiscreteDistribution(0), std::invalid_argument);
    EXPECT_NO_THROW(DiscreteDistribution(1));
    EXPECT_NO_THROW(DiscreteDistribution(100));
}

TEST_F(DiscreteDistributionTest, UniformInitialization) {
    // Should start with uniform distribution
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(dist_->getProbability(i), 0.2); // 1/5
    }
}

TEST_F(DiscreteDistributionTest, ProbabilitySettersAndGetters) {
    dist_->setProbability(0, 0.5);
    EXPECT_DOUBLE_EQ(dist_->getProbability(0), 0.5);
    
    // Test bounds checking
    EXPECT_THROW(dist_->setProbability(-1, 0.1), std::out_of_range);
    EXPECT_THROW(dist_->setProbability(5, 0.1), std::out_of_range);
    
    // Test probability validation
    EXPECT_THROW(dist_->setProbability(0, -0.1), std::invalid_argument);
    EXPECT_THROW(dist_->setProbability(0, 1.1), std::invalid_argument);
    EXPECT_THROW(dist_->setProbability(0, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST_F(DiscreteDistributionTest, OutOfBoundsAccess) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(10), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
}

TEST_F(DiscreteDistributionTest, FittingToData) {
    std::vector<double> data = {0, 0, 1, 1, 1, 2}; // 2 zeros, 3 ones, 1 two
    
    dist_->fit(data);
    
    EXPECT_NEAR(dist_->getProbability(0), 2.0/6.0, 1e-10);
    EXPECT_NEAR(dist_->getProbability(1), 3.0/6.0, 1e-10);
    EXPECT_NEAR(dist_->getProbability(2), 1.0/6.0, 1e-10);
    EXPECT_NEAR(dist_->getProbability(3), 0.0, 1e-10);
    EXPECT_NEAR(dist_->getProbability(4), 0.0, 1e-10);
}

// Gamma Distribution Tests
class GammaDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<GammaDistribution>(2.0, 1.0); // k=2, theta=1
    }
    
    std::unique_ptr<GammaDistribution> dist_;
};

TEST_F(GammaDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(GammaDistribution(1.0, 1.0));
    EXPECT_THROW(GammaDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(1.0, -1.0), std::invalid_argument);
}

TEST_F(GammaDistributionTest, ProbabilityCalculation) {
    // Gamma distribution is zero at x=0
    EXPECT_DOUBLE_EQ(dist_->getProbability(0.0), 0.0);
    
    // Should be positive for positive values
    EXPECT_GT(dist_->getProbability(1.0), 0.0);
    EXPECT_GT(dist_->getProbability(2.0), 0.0);
    
    // Should be zero for negative values
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
}

// Exponential Distribution Tests
class ExponentialDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<ExponentialDistribution>(1.0); // lambda=1
    }
    
    std::unique_ptr<ExponentialDistribution> dist_;
};

TEST_F(ExponentialDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(ExponentialDistribution(1.0));
    EXPECT_THROW(ExponentialDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(ExponentialDistribution(-1.0), std::invalid_argument);
}

TEST_F(ExponentialDistributionTest, ProbabilityProperties) {
    // Should be zero at x=0
    EXPECT_DOUBLE_EQ(dist_->getProbability(0.0), 0.0);
    
    // Should be positive for positive values
    EXPECT_GT(dist_->getProbability(1.0), 0.0);
    
    // Should be zero for negative values
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
    
    // Should decrease with increasing x
    double prob1 = dist_->getProbability(1.0);
    double prob2 = dist_->getProbability(2.0);
    EXPECT_GT(prob1, prob2);
}

// LogNormal Distribution Tests
class LogNormalDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<LogNormalDistribution>(0.0, 1.0);
    }
    
    std::unique_ptr<LogNormalDistribution> dist_;
};

TEST_F(LogNormalDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(LogNormalDistribution(0.0, 1.0));
    EXPECT_THROW(LogNormalDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(LogNormalDistribution(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(LogNormalDistribution(0.0, -1.0), std::invalid_argument);
}

TEST_F(LogNormalDistributionTest, ProbabilityProperties) {
    // Should be zero at x=0
    EXPECT_DOUBLE_EQ(dist_->getProbability(0.0), 0.0);
    
    // Should be positive for positive values
    EXPECT_GT(dist_->getProbability(1.0), 0.0);
    
    // Should be zero for negative values
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
}

// Pareto Distribution Tests
class ParetoDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<ParetoDistribution>(1.0, 1.0); // k=1, xm=1
    }
    
    std::unique_ptr<ParetoDistribution> dist_;
};

TEST_F(ParetoDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(ParetoDistribution(1.0, 1.0));
    EXPECT_THROW(ParetoDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(1.0, -1.0), std::invalid_argument);
}

TEST_F(ParetoDistributionTest, ProbabilityProperties) {
    // Should be zero for x <= xm
    EXPECT_DOUBLE_EQ(dist_->getProbability(0.5), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(1.0), 0.0);
    
    // Should be positive for x > xm
    EXPECT_GT(dist_->getProbability(1.5), 0.0);
    EXPECT_GT(dist_->getProbability(2.0), 0.0);
}

// Poisson Distribution Tests
class PoissonDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<PoissonDistribution>(2.0); // lambda=2.0
    }
    
    std::unique_ptr<PoissonDistribution> dist_;
};

TEST_F(PoissonDistributionTest, DefaultConstructor) {
    PoissonDistribution defaultDist;
    EXPECT_DOUBLE_EQ(defaultDist.getLambda(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist.getMean(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist.getVariance(), 1.0);
}

TEST_F(PoissonDistributionTest, ParameterizedConstructor) {
    PoissonDistribution dist(3.5);
    EXPECT_DOUBLE_EQ(dist.getLambda(), 3.5);
    EXPECT_DOUBLE_EQ(dist.getMean(), 3.5);
    EXPECT_DOUBLE_EQ(dist.getVariance(), 3.5);
    EXPECT_NEAR(dist.getStandardDeviation(), std::sqrt(3.5), 1e-10);
}

TEST_F(PoissonDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(PoissonDistribution(1.0));
    EXPECT_THROW(PoissonDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(PoissonDistribution(-1.0), std::invalid_argument);
    
    // Test with NaN and infinity - create variables outside of macros
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    // Now use variables in separate statements
    EXPECT_THROW({
        PoissonDistribution temp_nan(nan_val);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        PoissonDistribution temp_inf(inf_val);
    }, std::invalid_argument);
}

TEST_F(PoissonDistributionTest, ProbabilityCalculation) {
    // Test known values for λ=2.0
    // P(X=0) = e^(-2) ≈ 0.1353
    double p0 = dist_->getProbability(0.0);
    EXPECT_NEAR(p0, std::exp(-2.0), 1e-10);
    
    // P(X=1) = 2 * e^(-2) ≈ 0.2707
    double p1 = dist_->getProbability(1.0);
    EXPECT_NEAR(p1, 2.0 * std::exp(-2.0), 1e-10);
    
    // P(X=2) = 2 * e^(-2) ≈ 0.2707 
    double p2 = dist_->getProbability(2.0);
    EXPECT_NEAR(p2, 2.0 * std::exp(-2.0), 1e-10);
}

TEST_F(PoissonDistributionTest, InvalidInputHandling) {
    // Invalid inputs should return 0
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(1.5), 0.0);  // non-integer
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(PoissonDistributionTest, ParameterSettersAndGetters) {
    dist_->setLambda(5.0);
    EXPECT_DOUBLE_EQ(dist_->getLambda(), 5.0);
    EXPECT_DOUBLE_EQ(dist_->getMean(), 5.0);
    EXPECT_DOUBLE_EQ(dist_->getVariance(), 5.0);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setLambda(0.0), std::invalid_argument);
    EXPECT_THROW(dist_->setLambda(-1.0), std::invalid_argument);
    
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(dist_->setLambda(nan_val), std::invalid_argument);
}

TEST_F(PoissonDistributionTest, FittingToData) {
    // Test with known data (should fit λ ≈ 2.5)
    std::vector<double> data = {1, 2, 2, 3, 3, 3, 4, 2, 1, 4};
    double expectedMean = 2.5;  // Sum = 25, n = 10
    
    dist_->fit(data);
    EXPECT_NEAR(dist_->getLambda(), expectedMean, 1e-10);
    
    // Test with empty data (should reset to default)
    std::vector<double> emptyData;
    dist_->fit(emptyData);
    EXPECT_DOUBLE_EQ(dist_->getLambda(), 1.0);
}

TEST_F(PoissonDistributionTest, FittingValidation) {
    // Test with invalid data (should throw)
    std::vector<double> invalidData = {1, 2, -1, 3};
    EXPECT_THROW(dist_->fit(invalidData), std::invalid_argument);
    
    std::vector<double> nonIntegerData = {1.5, 2.0, 3.0};
    EXPECT_THROW(dist_->fit(nonIntegerData), std::invalid_argument);
}

TEST_F(PoissonDistributionTest, ResetFunctionality) {
    dist_->setLambda(10.0);
    dist_->reset();
    EXPECT_DOUBLE_EQ(dist_->getLambda(), 1.0);
}

TEST_F(PoissonDistributionTest, NumericalStability) {
    // Test with large lambda
    PoissonDistribution largeDist(500.0);
    double probLarge = largeDist.getProbability(500.0);  // Around mode
    EXPECT_GT(probLarge, 0.0);
    EXPECT_LT(probLarge, 1.0);
    
    // Test with very small lambda
    PoissonDistribution smallDist(1e-6);
    double probSmall = smallDist.getProbability(0.0);
    EXPECT_GT(probSmall, 0.0);
    EXPECT_LT(probSmall, 1.0);
    
    // Test extreme cases
    PoissonDistribution extremeDist(100.0);
    double probExtreme = extremeDist.getProbability(200.0);  // Far from mean
    EXPECT_GE(probExtreme, 0.0);  // Should be very small but non-negative
}

// Common Distribution Interface Tests
class CommonDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        distributions_.push_back(std::make_unique<GaussianDistribution>());
        distributions_.push_back(std::make_unique<DiscreteDistribution>(6));
        distributions_.push_back(std::make_unique<GammaDistribution>());
        distributions_.push_back(std::make_unique<ExponentialDistribution>());
        distributions_.push_back(std::make_unique<LogNormalDistribution>());
        distributions_.push_back(std::make_unique<ParetoDistribution>());
        distributions_.push_back(std::make_unique<PoissonDistribution>());
    }
    
    std::vector<std::unique_ptr<ProbabilityDistribution>> distributions_;
};

TEST_F(CommonDistributionTest, PolymorphicInterface) {
    // Test that all distributions implement the base interface
    for (auto& dist : distributions_) {
        EXPECT_NO_THROW(dist->toString());
        EXPECT_NO_THROW(dist->reset());
        
        // Test with some reasonable value
        double prob = dist->getProbability(1.0);
        EXPECT_GE(prob, 0.0);
        EXPECT_LE(prob, 1.0);
    }
}

TEST_F(CommonDistributionTest, ToStringFunctionality) {
    // Each distribution should produce a non-empty string
    for (auto& dist : distributions_) {
        std::string str = dist->toString();
        EXPECT_FALSE(str.empty());
        EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    }
}

TEST_F(CommonDistributionTest, ResetFunctionality) {
    // After reset, distributions should be in a valid state
    for (auto& dist : distributions_) {
        EXPECT_NO_THROW(dist->reset());
        
        // Should still be able to calculate probabilities
        EXPECT_NO_THROW(dist->getProbability(1.0));
        
        // toString should still work
        EXPECT_NO_THROW(dist->toString());
    }
}

// Edge Cases and Error Handling
class DistributionEdgeCaseTest : public ::testing::Test {};

TEST_F(DistributionEdgeCaseTest, ExtremeProbabilityValues) {
    GaussianDistribution gauss(0.0, 1.0);
    
    // Very large positive and negative values
    EXPECT_GE(gauss.getProbability(1000.0), 0.0);
    EXPECT_GE(gauss.getProbability(-1000.0), 0.0);
    
    // Should handle these gracefully (likely very small probabilities)
    EXPECT_LE(gauss.getProbability(1000.0), 1.0);
    EXPECT_LE(gauss.getProbability(-1000.0), 1.0);
}

TEST_F(DistributionEdgeCaseTest, EmptyDataFitting) {
    GaussianDistribution gauss;
    std::vector<double> emptyData;
    
    // Should handle empty data gracefully
    EXPECT_NO_THROW(gauss.fit(emptyData));
}

TEST_F(DistributionEdgeCaseTest, SingleDataPointFitting) {
    GaussianDistribution gauss;
    std::vector<double> singlePoint = {5.0};
    
    // Should handle single data point gracefully
    EXPECT_NO_THROW(gauss.fit(singlePoint));
}

TEST_F(DistributionEdgeCaseTest, NumericalStabilityTest) {
    // Test with very small standard deviation
    GaussianDistribution narrowGauss(0.0, 1e-10);
    
    // Should still produce valid probabilities
    double prob = narrowGauss.getProbability(0.0);
    EXPECT_GE(prob, 0.0);
    EXPECT_LE(prob, 1.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
