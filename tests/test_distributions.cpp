#include <gtest/gtest.h>
#include "libhmm/distributions/distributions.h"
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

// Beta Distribution Tests
class BetaDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<BetaDistribution>(2.0, 3.0); // α=2, β=3
    }
    
    std::unique_ptr<BetaDistribution> dist_;
};

TEST_F(BetaDistributionTest, DefaultConstructor) {
    BetaDistribution defaultDist;
    EXPECT_DOUBLE_EQ(defaultDist.getAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist.getBeta(), 1.0);
}

TEST_F(BetaDistributionTest, ParameterizedConstructor) {
    BetaDistribution dist(3.5, 2.5);
    EXPECT_DOUBLE_EQ(dist.getAlpha(), 3.5);
    EXPECT_DOUBLE_EQ(dist.getBeta(), 2.5);
}

TEST_F(BetaDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(BetaDistribution(1.0, 1.0));
    EXPECT_THROW(BetaDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(1.0, -1.0), std::invalid_argument);
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW(BetaDistribution(nan_val, 1.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(1.0, inf_val), std::invalid_argument);
}

TEST_F(BetaDistributionTest, ProbabilityCalculation) {
    // Beta distribution is defined on [0,1]
    EXPECT_DOUBLE_EQ(dist_->getProbability(-0.1), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(1.1), 0.0);
    
    // Should be positive within [0,1]
    EXPECT_GT(dist_->getProbability(0.5), 0.0);
    EXPECT_GT(dist_->getProbability(0.3), 0.0);
    EXPECT_GT(dist_->getProbability(0.7), 0.0);
    
    // Beta(2,3) should be 0 at endpoints
    EXPECT_DOUBLE_EQ(dist_->getProbability(0.0), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(1.0), 0.0);
}

TEST_F(BetaDistributionTest, InvalidInputHandling) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(BetaDistributionTest, ParameterSettersAndGetters) {
    dist_->setAlpha(5.0);
    EXPECT_DOUBLE_EQ(dist_->getAlpha(), 5.0);
    
    dist_->setBeta(4.0);
    EXPECT_DOUBLE_EQ(dist_->getBeta(), 4.0);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setAlpha(0.0), std::invalid_argument);
    EXPECT_THROW(dist_->setBeta(-1.0), std::invalid_argument);
    EXPECT_THROW(dist_->setAlpha(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST_F(BetaDistributionTest, FittingToData) {
    // Test with data in [0,1]
    std::vector<double> data = {0.1, 0.3, 0.5, 0.7, 0.9};
    
    EXPECT_NO_THROW(dist_->fit(data));
    EXPECT_GT(dist_->getAlpha(), 0.0);
    EXPECT_GT(dist_->getBeta(), 0.0);
    
    // Test with empty data (should reset to default)
    std::vector<double> emptyData;
    dist_->fit(emptyData);
    EXPECT_DOUBLE_EQ(dist_->getAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(dist_->getBeta(), 1.0);
}

TEST_F(BetaDistributionTest, FittingValidation) {
    // Test with data outside [0,1]
    std::vector<double> invalidData = {0.5, 1.5, 0.3};
    EXPECT_THROW(dist_->fit(invalidData), std::invalid_argument);
    
    std::vector<double> negativeData = {-0.1, 0.5, 0.8};
    EXPECT_THROW(dist_->fit(negativeData), std::invalid_argument);
}

TEST_F(BetaDistributionTest, ResetFunctionality) {
    dist_->setAlpha(10.0);
    dist_->setBeta(5.0);
    
    dist_->reset();
    
    EXPECT_DOUBLE_EQ(dist_->getAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(dist_->getBeta(), 1.0);
}

TEST_F(BetaDistributionTest, StatisticalProperties) {
    // For Beta(2,3): mean = α/(α+β) = 2/5 = 0.4
    double expectedMean = 2.0 / (2.0 + 3.0);
    EXPECT_NEAR(dist_->getMean(), expectedMean, 1e-10);
    
    // Variance = αβ/((α+β)²(α+β+1))
    double expectedVar = (2.0 * 3.0) / (std::pow(5.0, 2) * 6.0);
    EXPECT_NEAR(dist_->getVariance(), expectedVar, 1e-10);
    
    EXPECT_GT(dist_->getStandardDeviation(), 0.0);
}

// Weibull Distribution Tests
class WeibullDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<WeibullDistribution>(2.0, 1.0); // k=2, λ=1 (Rayleigh)
    }
    
    std::unique_ptr<WeibullDistribution> dist_;
};

TEST_F(WeibullDistributionTest, DefaultConstructor) {
    WeibullDistribution defaultDist;
    EXPECT_DOUBLE_EQ(defaultDist.getK(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist.getLambda(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist.getShape(), 1.0);  // Alternative getter
    EXPECT_DOUBLE_EQ(defaultDist.getScale(), 1.0);  // Alternative getter
}

TEST_F(WeibullDistributionTest, ParameterizedConstructor) {
    WeibullDistribution dist(3.5, 2.5);
    EXPECT_DOUBLE_EQ(dist.getK(), 3.5);
    EXPECT_DOUBLE_EQ(dist.getLambda(), 2.5);
}

TEST_F(WeibullDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(WeibullDistribution(1.0, 1.0));
    EXPECT_THROW(WeibullDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(1.0, -1.0), std::invalid_argument);
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW(WeibullDistribution(nan_val, 1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(1.0, inf_val), std::invalid_argument);
}

TEST_F(WeibullDistributionTest, ProbabilityCalculation) {
    // Weibull distribution is zero for negative values
    EXPECT_DOUBLE_EQ(dist_->getProbability(-0.1), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
    
    // Should be positive for positive values
    EXPECT_GT(dist_->getProbability(0.5), 0.0);
    EXPECT_GT(dist_->getProbability(1.0), 0.0);
    EXPECT_GT(dist_->getProbability(2.0), 0.0);
    
    // Should be defined at x=0 for k>=1
    EXPECT_GE(dist_->getProbability(0.0), 0.0);
}

TEST_F(WeibullDistributionTest, InvalidInputHandling) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(WeibullDistributionTest, ParameterSettersAndGetters) {
    dist_->setK(3.0);
    EXPECT_DOUBLE_EQ(dist_->getK(), 3.0);
    
    dist_->setLambda(2.0);
    EXPECT_DOUBLE_EQ(dist_->getLambda(), 2.0);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setK(0.0), std::invalid_argument);
    EXPECT_THROW(dist_->setLambda(-1.0), std::invalid_argument);
    EXPECT_THROW(dist_->setK(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST_F(WeibullDistributionTest, FittingToData) {
    // Test with positive data
    std::vector<double> data = {0.5, 1.0, 1.5, 2.0, 2.5};
    
    EXPECT_NO_THROW(dist_->fit(data));
    EXPECT_GT(dist_->getK(), 0.0);
    EXPECT_GT(dist_->getLambda(), 0.0);
    
    // Test with empty data (should reset to default)
    std::vector<double> emptyData;
    dist_->fit(emptyData);
    EXPECT_DOUBLE_EQ(dist_->getK(), 1.0);
    EXPECT_DOUBLE_EQ(dist_->getLambda(), 1.0);
}

TEST_F(WeibullDistributionTest, FittingValidation) {
    // Test with negative data
    std::vector<double> invalidData = {0.5, -1.0, 2.0};
    EXPECT_THROW(dist_->fit(invalidData), std::invalid_argument);
    
    // Test with NaN data
    std::vector<double> nanData = {1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
    EXPECT_THROW(dist_->fit(nanData), std::invalid_argument);
}

TEST_F(WeibullDistributionTest, ResetFunctionality) {
    dist_->setK(5.0);
    dist_->setLambda(3.0);
    
    dist_->reset();
    
    EXPECT_DOUBLE_EQ(dist_->getK(), 1.0);
    EXPECT_DOUBLE_EQ(dist_->getLambda(), 1.0);
}

TEST_F(WeibullDistributionTest, StatisticalProperties) {
    // Test exponential case (k=1, λ=2)
    WeibullDistribution exponential(1.0, 2.0);
    EXPECT_NEAR(exponential.getMean(), 2.0, 1e-10);  // For Weibull(1,λ), mean = λ
    EXPECT_NEAR(exponential.getVariance(), 4.0, 1e-10); // For Weibull(1,λ), variance = λ²
    
    // Test Rayleigh case (k=2, λ=1)
    WeibullDistribution rayleigh(2.0, 1.0);
    double rayleighMean = rayleigh.getMean();
    EXPECT_GT(rayleighMean, 0.8);
    EXPECT_LT(rayleighMean, 0.9);  // Should be around sqrt(π)/2 ≈ 0.8862
    
    EXPECT_GT(dist_->getStandardDeviation(), 0.0);
}

TEST_F(WeibullDistributionTest, SpecialCases) {
    // Test k=1 (exponential distribution)
    WeibullDistribution expCase(1.0, 1.0);
    double probExp = expCase.getProbability(1.0);
    EXPECT_GT(probExp, 0.35);
    EXPECT_LT(probExp, 0.4);  // Should be around e^(-1) ≈ 0.368
    
    // Test k=2 (Rayleigh distribution)
    WeibullDistribution rayleighCase(2.0, 1.0);
    double probRayleigh = rayleighCase.getProbability(1.0);
    EXPECT_GT(probRayleigh, 0.7);
    EXPECT_LT(probRayleigh, 0.75);  // Should be around 0.736
}

// Uniform Distribution Tests
class UniformDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<UniformDistribution>(2.0, 8.0); // a=2, b=8
    }
    
    std::unique_ptr<UniformDistribution> dist_;
};

TEST_F(UniformDistributionTest, DefaultConstructor) {
    UniformDistribution defaultDist;
    EXPECT_DOUBLE_EQ(defaultDist.getA(), 0.0);
    EXPECT_DOUBLE_EQ(defaultDist.getB(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist.getMin(), 0.0);  // Alternative getter
    EXPECT_DOUBLE_EQ(defaultDist.getMax(), 1.0);  // Alternative getter
}

TEST_F(UniformDistributionTest, ParameterizedConstructor) {
    UniformDistribution dist(3.5, 7.5);
    EXPECT_DOUBLE_EQ(dist.getA(), 3.5);
    EXPECT_DOUBLE_EQ(dist.getB(), 7.5);
}

TEST_F(UniformDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(UniformDistribution(1.0, 2.0));
    EXPECT_THROW(UniformDistribution(2.0, 1.0), std::invalid_argument);  // a > b
    EXPECT_THROW(UniformDistribution(3.0, 3.0), std::invalid_argument);  // a == b
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW(UniformDistribution(nan_val, 1.0), std::invalid_argument);
    EXPECT_THROW(UniformDistribution(1.0, inf_val), std::invalid_argument);
}

TEST_F(UniformDistributionTest, ProbabilityCalculation) {
    // Uniform distribution on [2, 8] has PDF = 1/(8-2) = 1/6
    double expectedPdf = 1.0 / 6.0;
    
    // Should be constant within the interval
    EXPECT_NEAR(dist_->getProbability(3.0), expectedPdf, 1e-10);
    EXPECT_NEAR(dist_->getProbability(5.0), expectedPdf, 1e-10);
    EXPECT_NEAR(dist_->getProbability(7.0), expectedPdf, 1e-10);
    
    // Should be zero outside the interval
    EXPECT_DOUBLE_EQ(dist_->getProbability(1.0), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(9.0), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
    
    // Test boundary values
    EXPECT_NEAR(dist_->getProbability(2.0), expectedPdf, 1e-10);
    EXPECT_NEAR(dist_->getProbability(8.0), expectedPdf, 1e-10);
}

TEST_F(UniformDistributionTest, InvalidInputHandling) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(UniformDistributionTest, ParameterSettersAndGetters) {
    dist_->setA(1.0);
    EXPECT_DOUBLE_EQ(dist_->getA(), 1.0);
    EXPECT_DOUBLE_EQ(dist_->getB(), 8.0);  // Should remain unchanged
    
    dist_->setB(10.0);
    EXPECT_DOUBLE_EQ(dist_->getA(), 1.0);
    EXPECT_DOUBLE_EQ(dist_->getB(), 10.0);
    
    // Test setting both parameters
    dist_->setParameters(0.5, 5.5);
    EXPECT_DOUBLE_EQ(dist_->getA(), 0.5);
    EXPECT_DOUBLE_EQ(dist_->getB(), 5.5);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setA(6.0), std::invalid_argument);  // Would make a > b
    EXPECT_THROW(dist_->setB(0.0), std::invalid_argument);  // Would make b < a
    EXPECT_THROW(dist_->setParameters(5.0, 3.0), std::invalid_argument);  // a > b
}

TEST_F(UniformDistributionTest, FittingToData) {
    // Test with data that should fit within bounds
    std::vector<double> data = {3.0, 4.5, 6.0, 7.5};
    
    dist_->fit(data);
    
    // After fitting, bounds should encompass all data with padding
    EXPECT_LE(dist_->getA(), 3.0);  // Should be at or below minimum
    EXPECT_GE(dist_->getB(), 7.5);  // Should be at or above maximum
    EXPECT_LT(dist_->getA(), dist_->getB());  // Valid interval
    
    // Test with empty data (should reset to default)
    std::vector<double> emptyData;
    dist_->fit(emptyData);
    EXPECT_DOUBLE_EQ(dist_->getA(), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getB(), 1.0);
}

TEST_F(UniformDistributionTest, FittingValidation) {
    // Test with NaN values
    std::vector<double> nanData = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_THROW(dist_->fit(nanData), std::invalid_argument);
    
    // Test with infinity values
    std::vector<double> infData = {1.0, 2.0, std::numeric_limits<double>::infinity(), 3.0};
    EXPECT_THROW(dist_->fit(infData), std::invalid_argument);
}

TEST_F(UniformDistributionTest, ResetFunctionality) {
    dist_->setParameters(10.0, 20.0);
    dist_->reset();
    
    EXPECT_DOUBLE_EQ(dist_->getA(), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getB(), 1.0);
}

TEST_F(UniformDistributionTest, StatisticalProperties) {
    // For Uniform(2, 8): mean = (2+8)/2 = 5
    double expectedMean = 5.0;
    EXPECT_NEAR(dist_->getMean(), expectedMean, 1e-10);
    
    // For Uniform(2, 8): variance = (8-2)²/12 = 36/12 = 3
    double expectedVar = 3.0;
    EXPECT_NEAR(dist_->getVariance(), expectedVar, 1e-10);
    
    // Standard deviation should be sqrt(variance)
    EXPECT_NEAR(dist_->getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
    
    // Test standard uniform [0,1]
    UniformDistribution standard(0.0, 1.0);
    EXPECT_NEAR(standard.getMean(), 0.5, 1e-10);
    EXPECT_NEAR(standard.getVariance(), 1.0/12.0, 1e-10);
}

TEST_F(UniformDistributionTest, SpecialCases) {
    // Test with very small interval
    UniformDistribution tiny(0.0, 1e-6);
    EXPECT_GT(tiny.getProbability(5e-7), 0.0);  // Within interval
    EXPECT_DOUBLE_EQ(tiny.getProbability(2e-6), 0.0);  // Outside interval
    
    // Test with negative interval
    UniformDistribution negative(-5.0, -2.0);
    EXPECT_GT(negative.getProbability(-3.5), 0.0);
    EXPECT_DOUBLE_EQ(negative.getProbability(0.0), 0.0);
    EXPECT_NEAR(negative.getMean(), -3.5, 1e-10);  // Mean = (-5 + -2)/2
    
    // Test isApproximatelyEqual
    UniformDistribution u1(1.0, 3.0);
    UniformDistribution u2(1.000000001, 3.000000001);
    UniformDistribution u3(1.1, 3.1);
    
    EXPECT_TRUE(u1.isApproximatelyEqual(u2, 1e-8));
    EXPECT_FALSE(u1.isApproximatelyEqual(u3, 1e-8));
}

// Binomial Distribution Tests
class BinomialDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<BinomialDistribution>(10, 0.3); // n=10, p=0.3
    }
    
    std::unique_ptr<BinomialDistribution> dist_;
};

TEST_F(BinomialDistributionTest, DefaultConstructor) {
    BinomialDistribution defaultDist;
    EXPECT_EQ(defaultDist.getN(), 10);
    EXPECT_DOUBLE_EQ(defaultDist.getP(), 0.5);
}

TEST_F(BinomialDistributionTest, ParameterizedConstructor) {
    BinomialDistribution dist(20, 0.7);
    EXPECT_EQ(dist.getN(), 20);
    EXPECT_DOUBLE_EQ(dist.getP(), 0.7);
}

TEST_F(BinomialDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(BinomialDistribution(1, 0.5));
    EXPECT_THROW(BinomialDistribution(0, 0.5), std::invalid_argument);  // n <= 0
    EXPECT_THROW(BinomialDistribution(-1, 0.5), std::invalid_argument); // n < 0
    EXPECT_THROW(BinomialDistribution(10, -0.1), std::invalid_argument); // p < 0
    EXPECT_THROW(BinomialDistribution(10, 1.1), std::invalid_argument);  // p > 1
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW(BinomialDistribution(10, nan_val), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, inf_val), std::invalid_argument);
}

TEST_F(BinomialDistributionTest, ProbabilityCalculation) {
    // Test valid range [0, n]
    EXPECT_GT(dist_->getProbability(0.0), 0.0);
    EXPECT_GT(dist_->getProbability(5.0), 0.0);
    EXPECT_GT(dist_->getProbability(10.0), 0.0);
    
    // Test out of range
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(11.0), 0.0);
    
    // Test edge cases
    BinomialDistribution binom_p0(10, 0.0);
    EXPECT_DOUBLE_EQ(binom_p0.getProbability(0.0), 1.0);
    EXPECT_DOUBLE_EQ(binom_p0.getProbability(1.0), 0.0);
    
    BinomialDistribution binom_p1(10, 1.0);
    EXPECT_DOUBLE_EQ(binom_p1.getProbability(10.0), 1.0);
    EXPECT_DOUBLE_EQ(binom_p1.getProbability(9.0), 0.0);
}

TEST_F(BinomialDistributionTest, InvalidInputHandling) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(BinomialDistributionTest, ParameterSettersAndGetters) {
    dist_->setN(15);
    EXPECT_EQ(dist_->getN(), 15);
    
    dist_->setP(0.8);
    EXPECT_DOUBLE_EQ(dist_->getP(), 0.8);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setN(0), std::invalid_argument);
    EXPECT_THROW(dist_->setP(-0.1), std::invalid_argument);
    EXPECT_THROW(dist_->setP(1.1), std::invalid_argument);
}

TEST_F(BinomialDistributionTest, FittingToData) {
    std::vector<double> data = {3, 4, 5, 6, 7, 3, 4, 5, 6, 7};
    
    dist_->fit(data);
    
    // After fitting, parameters should be valid
    EXPECT_GT(dist_->getN(), 0);
    EXPECT_GE(dist_->getP(), 0.0);
    EXPECT_LE(dist_->getP(), 1.0);
    
    // Test with empty data (should reset to default)
    std::vector<double> emptyData;
    dist_->fit(emptyData);
    EXPECT_EQ(dist_->getN(), 10);
    EXPECT_DOUBLE_EQ(dist_->getP(), 0.5);
}

TEST_F(BinomialDistributionTest, ResetFunctionality) {
    dist_->setN(25);
    dist_->setP(0.8);
    
    dist_->reset();
    
    EXPECT_EQ(dist_->getN(), 10);
    EXPECT_DOUBLE_EQ(dist_->getP(), 0.5);
}

TEST_F(BinomialDistributionTest, StatisticalProperties) {
    // For Binomial(10, 0.3): mean = n*p = 3, variance = n*p*(1-p) = 2.1
    double expectedMean = 10 * 0.3;
    double expectedVar = 10 * 0.3 * 0.7;
    
    EXPECT_NEAR(dist_->getMean(), expectedMean, 1e-10);
    EXPECT_NEAR(dist_->getVariance(), expectedVar, 1e-10);
    EXPECT_NEAR(dist_->getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
}

// Negative Binomial Distribution Tests
class NegativeBinomialDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<NegativeBinomialDistribution>(5.0, 0.4); // r=5, p=0.4
    }
    
    std::unique_ptr<NegativeBinomialDistribution> dist_;
};

TEST_F(NegativeBinomialDistributionTest, DefaultConstructor) {
    NegativeBinomialDistribution defaultDist;
    EXPECT_DOUBLE_EQ(defaultDist.getR(), 5.0);
    EXPECT_DOUBLE_EQ(defaultDist.getP(), 0.5);
}

TEST_F(NegativeBinomialDistributionTest, ParameterizedConstructor) {
    NegativeBinomialDistribution dist(3.5, 0.7);
    EXPECT_DOUBLE_EQ(dist.getR(), 3.5);
    EXPECT_DOUBLE_EQ(dist.getP(), 0.7);
}

TEST_F(NegativeBinomialDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(NegativeBinomialDistribution(1.0, 0.5));
    EXPECT_THROW(NegativeBinomialDistribution(0.0, 0.5), std::invalid_argument);   // r <= 0
    EXPECT_THROW(NegativeBinomialDistribution(-1.0, 0.5), std::invalid_argument);  // r < 0
    EXPECT_THROW(NegativeBinomialDistribution(5.0, 0.0), std::invalid_argument);   // p <= 0
    EXPECT_THROW(NegativeBinomialDistribution(5.0, -0.1), std::invalid_argument);  // p < 0
    EXPECT_THROW(NegativeBinomialDistribution(5.0, 1.1), std::invalid_argument);   // p > 1
    
    // Test with NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW(NegativeBinomialDistribution(nan_val, 0.5), std::invalid_argument);
    EXPECT_THROW(NegativeBinomialDistribution(5.0, inf_val), std::invalid_argument);
}

TEST_F(NegativeBinomialDistributionTest, ProbabilityCalculation) {
    // Test non-negative range [0, ∞)
    EXPECT_GT(dist_->getProbability(0.0), 0.0);
    EXPECT_GT(dist_->getProbability(5.0), 0.0);
    EXPECT_GT(dist_->getProbability(10.0), 0.0);
    
    // Test out of range
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
    
    // Test edge case p = 1
    NegativeBinomialDistribution negbinom_p1(5.0, 1.0);
    EXPECT_DOUBLE_EQ(negbinom_p1.getProbability(0.0), 1.0);
    EXPECT_DOUBLE_EQ(negbinom_p1.getProbability(1.0), 0.0);
}

TEST_F(NegativeBinomialDistributionTest, InvalidInputHandling) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(NegativeBinomialDistributionTest, ParameterSettersAndGetters) {
    dist_->setR(8.0);
    EXPECT_DOUBLE_EQ(dist_->getR(), 8.0);
    
    dist_->setP(0.6);
    EXPECT_DOUBLE_EQ(dist_->getP(), 0.6);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setR(0.0), std::invalid_argument);
    EXPECT_THROW(dist_->setP(0.0), std::invalid_argument);
    EXPECT_THROW(dist_->setP(1.1), std::invalid_argument);
}

TEST_F(NegativeBinomialDistributionTest, FittingToData) {
    // Test with over-dispersed data (variance > mean)
    std::vector<double> data = {0, 1, 2, 3, 5, 8, 10, 15, 2, 4, 7, 12};
    
    dist_->fit(data);
    
    // After fitting, parameters should be valid
    EXPECT_GT(dist_->getR(), 0.0);
    EXPECT_GT(dist_->getP(), 0.0);
    EXPECT_LE(dist_->getP(), 1.0);
    
    // Test with empty data (should reset to default)
    std::vector<double> emptyData;
    dist_->fit(emptyData);
    EXPECT_DOUBLE_EQ(dist_->getR(), 5.0);
    EXPECT_DOUBLE_EQ(dist_->getP(), 0.5);
}

TEST_F(NegativeBinomialDistributionTest, ResetFunctionality) {
    dist_->setR(10.0);
    dist_->setP(0.2);
    
    dist_->reset();
    
    EXPECT_DOUBLE_EQ(dist_->getR(), 5.0);
    EXPECT_DOUBLE_EQ(dist_->getP(), 0.5);
}

TEST_F(NegativeBinomialDistributionTest, StatisticalProperties) {
    // For NegBinom(5, 0.4): mean = r*(1-p)/p = 7.5, variance = r*(1-p)/p² = 18.75
    double expectedMean = 5.0 * (1.0 - 0.4) / 0.4;
    double expectedVar = 5.0 * (1.0 - 0.4) / (0.4 * 0.4);
    
    EXPECT_NEAR(dist_->getMean(), expectedMean, 1e-10);
    EXPECT_NEAR(dist_->getVariance(), expectedVar, 1e-10);
    EXPECT_NEAR(dist_->getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
    
    // Test over-dispersion property (variance > mean)
    EXPECT_GT(dist_->getVariance(), dist_->getMean());
}

TEST_F(NegativeBinomialDistributionTest, OverDispersionProperty) {
    // Negative binomial should always exhibit over-dispersion
    NegativeBinomialDistribution nb1(2.0, 0.3);
    NegativeBinomialDistribution nb2(10.0, 0.7);
    NegativeBinomialDistribution nb3(1.5, 0.1);
    
    EXPECT_GT(nb1.getVariance(), nb1.getMean());
    EXPECT_GT(nb2.getVariance(), nb2.getMean());
    EXPECT_GT(nb3.getVariance(), nb3.getMean());
}

// Student's t-Distribution Tests
class StudentTDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<StudentTDistribution>(5.0); // ν=5
    }
    
    std::unique_ptr<StudentTDistribution> dist_;
};

TEST_F(StudentTDistributionTest, DefaultConstructor) {
    StudentTDistribution defaultDist;
    EXPECT_DOUBLE_EQ(defaultDist.getDegreesOfFreedom(), 1.0);
}

TEST_F(StudentTDistributionTest, ParameterizedConstructor) {
    StudentTDistribution dist(3.5);
    EXPECT_DOUBLE_EQ(dist.getDegreesOfFreedom(), 3.5);
}

TEST_F(StudentTDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(StudentTDistribution(1.0));
    EXPECT_THROW(StudentTDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(StudentTDistribution(-1.0), std::invalid_argument);
    
    // Test with NaN and infinity - create separate variable declarations
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW({
        StudentTDistribution temp_nan(nan_val);
    }, std::invalid_argument);
    EXPECT_THROW({
        StudentTDistribution temp_inf(inf_val);
    }, std::invalid_argument);
}

TEST_F(StudentTDistributionTest, ProbabilityCalculation) {
    // t-distribution is symmetric around 0
    double prob0 = dist_->getProbability(0.0);
    double prob1 = dist_->getProbability(1.0);
    double probNeg1 = dist_->getProbability(-1.0);
    
    EXPECT_GT(prob0, 0.0);
    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(probNeg1, 0.0);
    
    // Symmetry property
    EXPECT_NEAR(prob1, probNeg1, 1e-10);
    
    // Maximum at x=0
    EXPECT_GT(prob0, prob1);
}

TEST_F(StudentTDistributionTest, InvalidInputHandling) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(StudentTDistributionTest, ParameterSettersAndGetters) {
    dist_->setDegreesOfFreedom(3.0);
    EXPECT_DOUBLE_EQ(dist_->getDegreesOfFreedom(), 3.0);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setDegreesOfFreedom(0.0), std::invalid_argument);
    EXPECT_THROW(dist_->setDegreesOfFreedom(-1.0), std::invalid_argument);
    EXPECT_THROW(dist_->setDegreesOfFreedom(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST_F(StudentTDistributionTest, FittingToData) {
    // Test with symmetric data
    std::vector<double> data = {-2.0, -1.0, 0.0, 1.0, 2.0};
    
    EXPECT_NO_THROW(dist_->fit(data));
    EXPECT_GT(dist_->getDegreesOfFreedom(), 0.0);
    
    // Test with empty data (should throw exception)
    std::vector<double> emptyData;
    EXPECT_THROW(dist_->fit(emptyData), std::invalid_argument);
}

TEST_F(StudentTDistributionTest, ResetFunctionality) {
    dist_->setDegreesOfFreedom(10.0);
    dist_->reset();
    EXPECT_DOUBLE_EQ(dist_->getDegreesOfFreedom(), 1.0);
}

TEST_F(StudentTDistributionTest, StatisticalProperties) {
    // For ν > 1: mean = 0
    EXPECT_NEAR(dist_->getMean(), 0.0, 1e-10);
    
    // For ν > 2: variance = ν/(ν-2)
    double expectedVar = 5.0 / (5.0 - 2.0);
    EXPECT_NEAR(dist_->getVariance(), expectedVar, 1e-10);
    EXPECT_NEAR(dist_->getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
    
    // Test edge cases
    StudentTDistribution t1(1.0);  // ν = 1, variance undefined
    EXPECT_TRUE(std::isinf(t1.getVariance()) || std::isnan(t1.getVariance()));
    
    StudentTDistribution t2(2.0);  // ν = 2, variance = ∞
    EXPECT_TRUE(std::isinf(t2.getVariance()) || std::isnan(t2.getVariance()));
}

TEST_F(StudentTDistributionTest, SpecialCases) {
    // Test Cauchy distribution (ν = 1)
    StudentTDistribution cauchy(1.0);
    double prob1 = cauchy.getProbability(1.0);
    double probNeg1 = cauchy.getProbability(-1.0);
    EXPECT_NEAR(prob1, probNeg1, 1e-10);  // Symmetry
    EXPECT_GT(cauchy.getProbability(0.0), prob1);  // Maximum at 0
    
    // Test convergence to normal as ν → ∞
    StudentTDistribution largeNu(1000.0);
    GaussianDistribution normal(0.0, 1.0);
    
    double tProb = largeNu.getProbability(1.0);
    double normProb = normal.getProbability(1.0);
    EXPECT_NEAR(tProb, normProb, 0.5);  // Should be reasonably close to normal
}

// Chi-squared Distribution Tests
class ChiSquaredDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<ChiSquaredDistribution>(4.0); // k=4
    }
    
    std::unique_ptr<ChiSquaredDistribution> dist_;
};

TEST_F(ChiSquaredDistributionTest, DefaultConstructor) {
    ChiSquaredDistribution defaultDist;
    EXPECT_DOUBLE_EQ(defaultDist.getDegreesOfFreedom(), 1.0);
}

TEST_F(ChiSquaredDistributionTest, ParameterizedConstructor) {
    ChiSquaredDistribution dist(7.5);
    EXPECT_DOUBLE_EQ(dist.getDegreesOfFreedom(), 7.5);
}

TEST_F(ChiSquaredDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(ChiSquaredDistribution(1.0));
    EXPECT_THROW(ChiSquaredDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(ChiSquaredDistribution(-1.0), std::invalid_argument);
    
    // Test with NaN and infinity - create separate variable declarations  
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    EXPECT_THROW({
        ChiSquaredDistribution temp_nan(nan_val);
    }, std::invalid_argument);
    EXPECT_THROW({
        ChiSquaredDistribution temp_inf(inf_val);
    }, std::invalid_argument);
}

TEST_F(ChiSquaredDistributionTest, ProbabilityCalculation) {
    // Chi-squared is zero for negative values
    EXPECT_DOUBLE_EQ(dist_->getProbability(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-0.1), 0.0);
    
    // Should be positive for positive values
    EXPECT_GT(dist_->getProbability(1.0), 0.0);
    EXPECT_GT(dist_->getProbability(4.0), 0.0);
    EXPECT_GT(dist_->getProbability(10.0), 0.0);
    
    // At x=0: depends on k
    ChiSquaredDistribution chi1(1.0);  // k=1, should be infinite at x=0
    ChiSquaredDistribution chi2(2.0);  // k=2, should be 0.5 at x=0
    ChiSquaredDistribution chi3(4.0);  // k>2, should be 0 at x=0
    
    EXPECT_DOUBLE_EQ(chi3.getProbability(0.0), 0.0);
}

TEST_F(ChiSquaredDistributionTest, InvalidInputHandling) {
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(dist_->getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST_F(ChiSquaredDistributionTest, ParameterSettersAndGetters) {
    dist_->setDegreesOfFreedom(6.0);
    EXPECT_DOUBLE_EQ(dist_->getDegreesOfFreedom(), 6.0);
    
    // Test invalid setters
    EXPECT_THROW(dist_->setDegreesOfFreedom(0.0), std::invalid_argument);
    EXPECT_THROW(dist_->setDegreesOfFreedom(-1.0), std::invalid_argument);
    EXPECT_THROW(dist_->setDegreesOfFreedom(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST_F(ChiSquaredDistributionTest, FittingToData) {
    // Test with positive data
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    EXPECT_NO_THROW(dist_->fit(data));
    EXPECT_GT(dist_->getDegreesOfFreedom(), 0.0);
    
    // Test with empty data (should throw exception)
    std::vector<double> emptyData;
    EXPECT_THROW(dist_->fit(emptyData), std::invalid_argument);
}

TEST_F(ChiSquaredDistributionTest, FittingValidation) {
    // Test with negative data
    std::vector<double> invalidData = {1.0, -1.0, 3.0};
    EXPECT_THROW(dist_->fit(invalidData), std::invalid_argument);
    
    // Test with NaN data
    std::vector<double> nanData = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_THROW(dist_->fit(nanData), std::invalid_argument);
}

TEST_F(ChiSquaredDistributionTest, ResetFunctionality) {
    dist_->setDegreesOfFreedom(15.0);
    dist_->reset();
    EXPECT_DOUBLE_EQ(dist_->getDegreesOfFreedom(), 1.0);
}

TEST_F(ChiSquaredDistributionTest, StatisticalProperties) {
    // For χ²(k): mean = k, variance = 2k
    double expectedMean = 4.0;
    double expectedVar = 2.0 * 4.0;
    
    EXPECT_NEAR(dist_->getMean(), expectedMean, 1e-10);
    EXPECT_NEAR(dist_->getVariance(), expectedVar, 1e-10);
    EXPECT_NEAR(dist_->getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
}

TEST_F(ChiSquaredDistributionTest, SpecialCases) {
    // Test χ²(1) - half-normal squared
    ChiSquaredDistribution chi1(1.0);
    EXPECT_GT(chi1.getProbability(0.1), 0.0);
    EXPECT_GT(chi1.getProbability(1.0), 0.0);
    
    // Test χ²(2) - exponential distribution
    ChiSquaredDistribution chi2(2.0);
    double prob0 = chi2.getProbability(0.0);
    EXPECT_NEAR(prob0, 0.5, 1e-10);  // For χ²(2), PDF(0) = 0.5
    
    // Test large k (approaches normal)
    ChiSquaredDistribution largeChi(100.0);
    double meanProb = largeChi.getProbability(100.0);  // At mean
    EXPECT_GT(meanProb, 0.0);
    EXPECT_LT(meanProb, 1.0);
}

TEST_F(ChiSquaredDistributionTest, RelationshipToGamma) {
    // χ²(k) is related to Gamma distribution
    // Note: The exact relationship depends on parameterization
    ChiSquaredDistribution chi(4.0);
    
    // Test that chi-squared produces reasonable values
    std::vector<double> testPoints = {0.5, 1.0, 2.0, 4.0, 8.0};
    for (double x : testPoints) {
        double chiProb = chi.getProbability(x);
        EXPECT_GT(chiProb, 0.0);  // Should be positive
        EXPECT_LT(chiProb, 1.0);  // Should be a valid probability density
    }
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
        distributions_.push_back(std::make_unique<BetaDistribution>());
        distributions_.push_back(std::make_unique<WeibullDistribution>());
        distributions_.push_back(std::make_unique<UniformDistribution>());
        distributions_.push_back(std::make_unique<BinomialDistribution>());
        distributions_.push_back(std::make_unique<NegativeBinomialDistribution>());
        distributions_.push_back(std::make_unique<StudentTDistribution>());
        distributions_.push_back(std::make_unique<ChiSquaredDistribution>());
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
