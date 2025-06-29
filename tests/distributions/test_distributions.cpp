
#include <gtest/gtest.h>
#include "libhmm/distributions/distributions.h"
#include <memory>
#include <vector>
#include <cmath>
#include <climits>

using namespace libhmm;

// Base test class for standardized distribution testing
template<typename DistributionType>
class StandardizedDistributionTest : public ::testing::Test {
protected:
    virtual std::unique_ptr<DistributionType> createDefaultDistribution() = 0;
    virtual std::unique_ptr<DistributionType> createParameterizedDistribution() = 0;
    virtual std::vector<double> getValidTestValues() = 0;
    virtual std::vector<double> getInvalidTestValues() = 0;
    virtual std::vector<double> getValidFittingData() = 0;
    
    // Standard test methods
    void testBasicFunctionality() {
        auto defaultDist = createDefaultDistribution();
        auto paramDist = createParameterizedDistribution();
        EXPECT_TRUE(defaultDist != nullptr);
        EXPECT_TRUE(paramDist != nullptr);
    }
    
    void testProbabilityCalculation() {
        auto dist = createDefaultDistribution();
        for (double val : getValidTestValues()) {
            double prob = dist->getProbability(val);
            EXPECT_GE(prob, 0.0) << "Probability should be non-negative for value " << val;
            EXPECT_TRUE(std::isfinite(prob)) << "Probability should be finite for value " << val;
        }
    }
    
    void testInvalidInputHandling() {
        auto dist = createDefaultDistribution();
        for (double invalidVal : getInvalidTestValues()) {
            double prob = dist->getProbability(invalidVal);
            EXPECT_GE(prob, 0.0) << "Even invalid inputs should return non-negative probability";
        }
    }
    
    void testStringRepresentation() {
        auto dist = createDefaultDistribution();
        std::string str = dist->toString();
        EXPECT_FALSE(str.empty()) << "toString() should return non-empty string";
        EXPECT_TRUE(str.find("Distribution") != std::string::npos) << "toString() should contain 'Distribution'";
    }
    
    void testResetFunctionality() {
        auto dist = createParameterizedDistribution();
        EXPECT_NO_THROW(dist->reset()) << "reset() should not throw";
        double prob = 0.0;
        EXPECT_NO_THROW(prob = dist->getProbability(1.0)) << "Should work after reset";
        (void)prob; // Suppress unused variable warning
    }
    
    void testParameterFitting() {
        auto dist = createDefaultDistribution();
        auto data = getValidFittingData();
        if (!data.empty()) {
            EXPECT_NO_THROW(dist->fit(data)) << "fit() should handle valid data";
        }
        
        // Test empty data - some distributions may throw for empty data
        std::vector<double> emptyData;
        try {
            dist->fit(emptyData);
            // If no exception, that's also acceptable
        } catch (const std::exception&) {
            // Some distributions throw for empty data, which is also acceptable
        }
    }
};

// Gaussian Distribution Tests
class GaussianDistributionTest : public StandardizedDistributionTest<GaussianDistribution> {
protected:
    std::unique_ptr<GaussianDistribution> createDefaultDistribution() override {
        return std::make_unique<GaussianDistribution>();
    }
    
    std::unique_ptr<GaussianDistribution> createParameterizedDistribution() override {
        return std::make_unique<GaussianDistribution>(5.0, 2.0);
    }
    
    std::vector<double> getValidTestValues() override {
        return {-3.0, -1.0, 0.0, 1.0, 3.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {1.0, 2.0, 3.0, 4.0, 5.0};
    }
};

// Generate test cases for GaussianDistribution
TEST_F(GaussianDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(GaussianDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(GaussianDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(GaussianDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(GaussianDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(GaussianDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for GaussianDistribution
TEST_F(GaussianDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(GaussianDistribution(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(std::numeric_limits<double>::infinity(), 1.0), std::invalid_argument);
}

TEST_F(GaussianDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // mean=5.0, stddev=2.0
    EXPECT_NEAR(dist->getMean(), 5.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 4.0, 1e-10);  // σ² = 2² = 4
    EXPECT_NEAR(dist->getStandardDeviation(), 2.0, 1e-10);
}

TEST_F(GaussianDistributionTest, ProbabilitySymmetry) {
    auto dist = createDefaultDistribution(); // mean=0, stddev=1
    // Test symmetry around mean
    EXPECT_NEAR(dist->getProbability(1.0), dist->getProbability(-1.0), 1e-10);
    EXPECT_NEAR(dist->getProbability(2.0), dist->getProbability(-2.0), 1e-10);
    
    // Maximum at mean
    EXPECT_GT(dist->getProbability(0.0), dist->getProbability(1.0));
}

TEST_F(GaussianDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

// Student's t-Distribution Tests
class StudentTDistributionTest : public StandardizedDistributionTest<StudentTDistribution> {
protected:
    std::unique_ptr<StudentTDistribution> createDefaultDistribution() override {
        return std::make_unique<StudentTDistribution>();
    }
    
    std::unique_ptr<StudentTDistribution> createParameterizedDistribution() override {
        return std::make_unique<StudentTDistribution>(3.0);
    }
    
    std::vector<double> getValidTestValues() override {
        return {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {-1.5, -0.5, 0.5, 1.0, 2.0};
    }
};

// Generate test cases for StudentTDistribution
TEST_F(StudentTDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(StudentTDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(StudentTDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(StudentTDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(StudentTDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(StudentTDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for StudentTDistribution
TEST_F(StudentTDistributionTest, ParameterValidation) {
    // Test invalid degrees of freedom
    EXPECT_THROW(StudentTDistribution(0.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return StudentTDistribution(std::numeric_limits<double>::quiet_NaN()); }(), std::invalid_argument);
    EXPECT_THROW([]() { return StudentTDistribution(std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(StudentTDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // ν=3.0
    // For Student's t with ν=3.0, mean = 0, variance = ν/(ν-2) = 3.0
    EXPECT_NEAR(dist->getMean(), 0.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 3.0, 1e-10);
}

TEST_F(StudentTDistributionTest, SymmetryProperty) {
    auto dist = createParameterizedDistribution(); // ν=3.0
    EXPECT_NEAR(dist->getProbability(1.0), dist->getProbability(-1.0), 1e-10);
    EXPECT_NEAR(dist->getProbability(2.0), dist->getProbability(-2.0), 1e-10);
}

TEST_F(StudentTDistributionTest, DegreesOfFreedomEffects) {
    // Test distributions with varying degrees of freedom
    StudentTDistribution t_low_df(1.0);  // Cauchy
    StudentTDistribution t_high_df(10.0); // Close to normal
    
    EXPECT_GT(t_high_df.getProbability(0.0), t_low_df.getProbability(0.0));
}

TEST_F(StudentTDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isinf(dist->getLogProbability(std::numeric_limits<double>::infinity())));
}

// Log Normal Distribution Tests
class LogNormalDistributionTest : public StandardizedDistributionTest<LogNormalDistribution> {
protected:
    std::unique_ptr<LogNormalDistribution> createDefaultDistribution() override {
        return std::make_unique<LogNormalDistribution>();
    }
    
    std::unique_ptr<LogNormalDistribution> createParameterizedDistribution() override {
        return std::make_unique<LogNormalDistribution>(1.0, 0.5);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0.1, 0.5, 1.0, 2.0, 5.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {0.5, 1.0, 1.5, 2.0, 3.0};
    }
};

// Generate test cases for LogNormalDistribution
TEST_F(LogNormalDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(LogNormalDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(LogNormalDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(LogNormalDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(LogNormalDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(LogNormalDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for LogNormalDistribution
TEST_F(LogNormalDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(LogNormalDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(LogNormalDistribution(std::numeric_limits<double>::infinity(), 1.0), std::invalid_argument);
    EXPECT_THROW(LogNormalDistribution(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(LogNormalDistribution(0.0, -1.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return LogNormalDistribution(0.0, std::numeric_limits<double>::quiet_NaN()); }(), std::invalid_argument);
    EXPECT_THROW([]() { return LogNormalDistribution(0.0, std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(LogNormalDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // μ=1.0, σ=0.5
    // For LogNormal(μ=1, σ=0.5): mean = exp(μ + σ²/2) = exp(1.125) ≈ 3.08
    double expectedMean = std::exp(1.0 + 0.25/2);
    EXPECT_NEAR(dist->getDistributionMean(), expectedMean, 1e-10);
    EXPECT_GT(dist->getVariance(), 0.0);
    EXPECT_GT(dist->getDistributionStandardDeviation(), 0.0);
}

TEST_F(LogNormalDistributionTest, PositiveSupport) {
    auto dist = createDefaultDistribution();
    // Log-normal is zero for negative values and zero
    EXPECT_DOUBLE_EQ(dist->getProbability(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(0.0), 0.0);
    
    // Positive for positive values
    EXPECT_GT(dist->getProbability(0.5), 0.0);
    EXPECT_GT(dist->getProbability(1.0), 0.0);
    EXPECT_GT(dist->getProbability(2.0), 0.0);
}

TEST_F(LogNormalDistributionTest, MedianProperty) {
    auto dist = createParameterizedDistribution(); // μ=1.0, σ=0.5
    // For log-normal, median = exp(μ) = exp(1) ≈ 2.718
    EXPECT_NEAR(dist->getMedian(), std::exp(1.0), 1e-10);
}

TEST_F(LogNormalDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1.0)) && dist->getLogProbability(-1.0) < 0);
    EXPECT_TRUE(std::isinf(dist->getLogProbability(0.0)) && dist->getLogProbability(0.0) < 0);
}

// Exponential Distribution Tests
class ExponentialDistributionTest : public StandardizedDistributionTest<ExponentialDistribution> {
protected:
    std::unique_ptr<ExponentialDistribution> createDefaultDistribution() override {
        return std::make_unique<ExponentialDistribution>();
    }
    
    std::unique_ptr<ExponentialDistribution> createParameterizedDistribution() override {
        return std::make_unique<ExponentialDistribution>(2.0);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0.0, 0.5, 1.0, 2.0, 5.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1.0, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {0.5, 1.0, 1.5, 2.0, 3.0};
    }
};

// Generate test cases for ExponentialDistribution
TEST_F(ExponentialDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(ExponentialDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(ExponentialDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(ExponentialDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(ExponentialDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(ExponentialDistributionTest, ParameterFitting) { testParameterFitting(); }

TEST_F(ExponentialDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(ExponentialDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(ExponentialDistribution(-1.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return ExponentialDistribution(std::numeric_limits<double>::quiet_NaN()); }(), std::invalid_argument);
    EXPECT_THROW([]() { return ExponentialDistribution(std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(ExponentialDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // lambda=2.0
    // For Exponential(λ=2): mean = 1/λ = 0.5, variance = 1/λ² = 0.25
    EXPECT_NEAR(dist->getMean(), 0.5, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 0.25, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), 0.5, 1e-10);
}

TEST_F(ExponentialDistributionTest, ProbabilityProperties) {
    auto dist = createDefaultDistribution(); // lambda=1.0
    // Exponential at x=0 should equal λ
    EXPECT_DOUBLE_EQ(dist->getProbability(0.0), 1.0);
    
    // Should decrease with increasing x
    EXPECT_GT(dist->getProbability(0.5), dist->getProbability(1.0));
    EXPECT_GT(dist->getProbability(1.0), dist->getProbability(2.0));
}

// Weibull Distribution Tests
class WeibullDistributionTest : public StandardizedDistributionTest<WeibullDistribution> {
protected:
    std::unique_ptr<WeibullDistribution> createDefaultDistribution() override {
        return std::make_unique<WeibullDistribution>(2.0, 1.0); // k=2, λ=1 (Rayleigh-like)
    }

    std::unique_ptr<WeibullDistribution> createParameterizedDistribution() override {
        return std::make_unique<WeibullDistribution>(1.5, 2.0); // k=1.5, λ=2
    }

    std::vector<double> getValidTestValues() override {
        return {0.0, 0.5, 1.0, 2.0, 3.0};
    }

    std::vector<double> getInvalidTestValues() override {
        return {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    }

    std::vector<double> getValidFittingData() override {
        return {0.5, 1.0, 1.5, 2.0, 3.0};
    }
};

// Generate test cases for WeibullDistribution
TEST_F(WeibullDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(WeibullDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(WeibullDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(WeibullDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(WeibullDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(WeibullDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for WeibullDistribution
TEST_F(WeibullDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(WeibullDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(1.0, -1.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return WeibullDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0); }(), std::invalid_argument);
    EXPECT_THROW([]() { return WeibullDistribution(1.0, std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(WeibullDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // k=1.5, λ=2
    // For Weibull(k=1.5, λ=2): mean = λ * Γ(1 + 1/k), variance = λ² * [Γ(1 + 2/k) - (Γ(1 + 1/k))²]
    EXPECT_GT(dist->getMean(), 0.0);
    EXPECT_GT(dist->getVariance(), 0.0);
    EXPECT_GT(dist->getStandardDeviation(), 0.0);
}

TEST_F(WeibullDistributionTest, NonNegativeSupport) {
    auto dist = createDefaultDistribution();
    // Weibull distribution is zero for negative values
    EXPECT_DOUBLE_EQ(dist->getProbability(-1.0), 0.0);
    
    // Positive for non-negative values
    EXPECT_GE(dist->getProbability(0.0), 0.0);
    EXPECT_GT(dist->getProbability(0.5), 0.0);
    EXPECT_GT(dist->getProbability(1.0), 0.0);
    EXPECT_GT(dist->getProbability(2.0), 0.0);
}

TEST_F(WeibullDistributionTest, ShapeParameterEffects) {
    // Test different shape parameters
    WeibullDistribution weibull_k1(1.0, 1.0);  // Exponential distribution
    WeibullDistribution weibull_k2(2.0, 1.0);  // Rayleigh-like distribution
    WeibullDistribution weibull_k3(3.0, 1.0);  // More peaked distribution
    
    // At x=0: k=1 should be highest (exponential), k>1 should be 0
    EXPECT_GT(weibull_k1.getProbability(0.0), weibull_k2.getProbability(0.0));
    EXPECT_DOUBLE_EQ(weibull_k2.getProbability(0.0), 0.0);
    EXPECT_DOUBLE_EQ(weibull_k3.getProbability(0.0), 0.0);
}

TEST_F(WeibullDistributionTest, ExponentialSpecialCase) {
    // Weibull(k=1, λ) should behave like Exponential(rate=1/λ)
    WeibullDistribution weibull_exp(1.0, 2.0); // k=1, λ=2
    
    // For Weibull(k=1, λ=2), this should be exponential with rate = 1/2
    // At x=0, should equal rate = 1/λ = 0.5
    EXPECT_NEAR(weibull_exp.getProbability(0.0), 0.5, 1e-10);
}

TEST_F(WeibullDistributionTest, ParameterAccessors) {
    auto dist = createParameterizedDistribution(); // k=1.5, λ=2
    EXPECT_NEAR(dist->getK(), 1.5, 1e-10);
    EXPECT_NEAR(dist->getLambda(), 2.0, 1e-10);
    
    // Test setters
    dist->setK(2.5);
    EXPECT_NEAR(dist->getK(), 2.5, 1e-10);
    
    dist->setLambda(3.0);
    EXPECT_NEAR(dist->getLambda(), 3.0, 1e-10);
}

TEST_F(WeibullDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        if (val >= 0.0) {
            double logProb = dist->getLogProbability(val);
            if (std::isfinite(logProb)) {
                EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
            }
        }
    }
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1.0)) && dist->getLogProbability(-1.0) < 0);
}

// Rayleigh Distribution Tests
class RayleighDistributionTest : public StandardizedDistributionTest<RayleighDistribution> {
protected:
    std::unique_ptr<RayleighDistribution> createDefaultDistribution() override {
        return std::make_unique<RayleighDistribution>(1.0);
    }

    std::unique_ptr<RayleighDistribution> createParameterizedDistribution() override {
        return std::make_unique<RayleighDistribution>(2.0);
    }

    std::vector<double> getValidTestValues() override {
        return {0.0, 0.5, 1.0, 2.0, 3.0};
    }

    std::vector<double> getInvalidTestValues() override {
        return {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    }

    std::vector<double> getValidFittingData() override {
        return {0.5, 1.0, 1.5, 2.0, 2.5};
    }
};

// Generate test cases for RayleighDistribution
TEST_F(RayleighDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(RayleighDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(RayleighDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(RayleighDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(RayleighDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(RayleighDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for RayleighDistribution
TEST_F(RayleighDistributionTest, ParameterValidation) {
    // Test invalid constructor parameter
    EXPECT_THROW(RayleighDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(RayleighDistribution(-1.0), std::invalid_argument);

    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return RayleighDistribution(std::numeric_limits<double>::quiet_NaN()); }(), std::invalid_argument);
    EXPECT_THROW([]() { return RayleighDistribution(std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(RayleighDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution();
    // For Rayleigh(σ=2): mean = σ * √(π/2), variance = σ² * (4-π)/2
    EXPECT_NEAR(dist->getMean(), 2.0 * sqrt(M_PI/2.0), 1e-10);
    EXPECT_NEAR(dist->getVariance(), 2.0 * 2.0 * (4.0 - M_PI) / 2.0, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), sqrt(dist->getVariance()), 1e-10);
}

TEST_F(RayleighDistributionTest, NonNegativeSupport) {
    auto dist = createDefaultDistribution();
    // Rayleigh distribution is zero for negative values
    EXPECT_DOUBLE_EQ(dist->getProbability(-0.5), 0.0);

    // Rayleigh PDF at x=0 is mathematically 0 (formula has x in numerator)
    EXPECT_DOUBLE_EQ(dist->getProbability(0.0), 0.0);
    // Positive for x > 0
    EXPECT_GT(dist->getProbability(0.5), 0.0);
    EXPECT_GT(dist->getProbability(2.0), 0.0);
}

TEST_F(RayleighDistributionTest, ModeProperty) {
    auto dist = createParameterizedDistribution();
    // Rayleigh's mode is at sigma
    EXPECT_EQ(dist->getMode(), 2.0);
}

TEST_F(RayleighDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        if (val > 0.0) { // Only test x > 0 since Rayleigh PDF is 0 at x=0
            double logProb = dist->getLogProbability(val);
            EXPECT_TRUE(std::isfinite(logProb));
            EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
        }
    }

    // Test x=0 case - should return -infinity since PDF is 0
    EXPECT_TRUE(std::isinf(dist->getLogProbability(0.0)) && dist->getLogProbability(0.0) < 0);
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1.0)) && dist->getLogProbability(-1.0) < 0);
}

// Gamma Distribution Tests
class GammaDistributionTest : public StandardizedDistributionTest<GammaDistribution> {
protected:
    std::unique_ptr<GammaDistribution> createDefaultDistribution() override {
        return std::make_unique<GammaDistribution>();
    }
    
    std::unique_ptr<GammaDistribution> createParameterizedDistribution() override {
        return std::make_unique<GammaDistribution>(2.0, 1.5);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0.5, 1.0, 2.0, 3.0, 5.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1.0, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {0.5, 1.0, 1.5, 2.0, 2.5};
    }
};

// Generate test cases for GammaDistribution
TEST_F(GammaDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(GammaDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(GammaDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(GammaDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(GammaDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(GammaDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for GammaDistribution
TEST_F(GammaDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(GammaDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(1.0, -1.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return GammaDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0); }(), std::invalid_argument);
    EXPECT_THROW([]() { return GammaDistribution(1.0, std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(GammaDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // shape=2.0, scale=1.5
    // For Gamma(k=2, θ=1.5): mean = k*θ = 3, variance = k*θ² = 4.5
    EXPECT_NEAR(dist->getMean(), 3.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 4.5, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(4.5), 1e-10);
}

TEST_F(GammaDistributionTest, ProbabilityProperties) {
    auto dist = createDefaultDistribution(); // shape=1.0, scale=1.0 (exponential)
    // Gamma distribution is zero for negative values
    EXPECT_DOUBLE_EQ(dist->getProbability(-1.0), 0.0);
    
    // Positive for positive values
    EXPECT_GT(dist->getProbability(0.5), 0.0);
    EXPECT_GT(dist->getProbability(1.0), 0.0);
    
    // Gamma(1,1) is exponential distribution - check at x=0 (may depend on implementation)
    EXPECT_GE(dist->getProbability(0.0), 0.0);
}

TEST_F(GammaDistributionTest, ShapeParameterEffects) {
    auto paramDist = createParameterizedDistribution(); // shape=2.0
    // Gamma distribution is zero at x=0 for shape > 1
    EXPECT_DOUBLE_EQ(paramDist->getProbability(0.0), 0.0);
    
    // Test that probability decreases for large x
    EXPECT_GT(paramDist->getProbability(1.0), paramDist->getProbability(5.0));
}

TEST_F(GammaDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

// Beta Distribution Tests
class BetaDistributionTest : public StandardizedDistributionTest<BetaDistribution> {
protected:
    std::unique_ptr<BetaDistribution> createDefaultDistribution() override {
        return std::make_unique<BetaDistribution>(2.0, 5.0);
    }

    std::unique_ptr<BetaDistribution> createParameterizedDistribution() override {
        return std::make_unique<BetaDistribution>(3.0, 3.0); // symmetric
    }

    std::vector<double> getValidTestValues() override {
        return {0.1, 0.25, 0.5, 0.75, 0.9};
    }

    std::vector<double> getInvalidTestValues() override {
        return {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    }

    std::vector<double> getValidFittingData() override {
        return {0.2, 0.3, 0.5, 0.7, 0.9};
    }
};

// Generate test cases for BetaDistribution
TEST_F(BetaDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(BetaDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(BetaDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(BetaDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(BetaDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(BetaDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for BetaDistribution
TEST_F(BetaDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(BetaDistribution(-1.0, 2.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(2.0, -1.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(0.0, 2.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(2.0, 0.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return BetaDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0); }(), std::invalid_argument);
    EXPECT_THROW([]() { return BetaDistribution(2.0, std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(BetaDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // α=3.0, β=3.0 (symmetric)
    // For Beta(α=3, β=3): mean = α/(α+β) = 3/6 = 0.5
    EXPECT_NEAR(dist->getMean(), 0.5, 1e-10);
    // variance = αβ/((α+β)²(α+β+1)) = 9/(36*7) = 9/252 = 1/28
    double expectedVariance = 9.0 / (36.0 * 7.0);
    EXPECT_NEAR(dist->getVariance(), expectedVariance, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(expectedVariance), 1e-10);
}

TEST_F(BetaDistributionTest, BoundedSupport) {
    auto dist = createDefaultDistribution();
    // Beta distribution is zero outside [0,1]
    EXPECT_DOUBLE_EQ(dist->getProbability(-0.1), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(1.1), 0.0);
    
    // Positive within [0,1]
    EXPECT_GT(dist->getProbability(0.2), 0.0);
    EXPECT_GT(dist->getProbability(0.5), 0.0);
    EXPECT_GT(dist->getProbability(0.8), 0.0);
}

TEST_F(BetaDistributionTest, SymmetryProperty) {
    auto dist = createParameterizedDistribution(); // α=3.0, β=3.0 (symmetric)
    // For symmetric Beta, P(x) should equal P(1-x)
    EXPECT_NEAR(dist->getProbability(0.3), dist->getProbability(0.7), 1e-10);
    EXPECT_NEAR(dist->getProbability(0.2), dist->getProbability(0.8), 1e-10);
    EXPECT_NEAR(dist->getProbability(0.1), dist->getProbability(0.9), 1e-10);
}

TEST_F(BetaDistributionTest, ParameterAccessors) {
    auto dist = createDefaultDistribution(); // α=2.0, β=5.0
    EXPECT_NEAR(dist->getAlpha(), 2.0, 1e-10);
    EXPECT_NEAR(dist->getBeta(), 5.0, 1e-10);
    
    // Test setters
    dist->setAlpha(4.0);
    EXPECT_NEAR(dist->getAlpha(), 4.0, 1e-10);
    
    dist->setBeta(6.0);
    EXPECT_NEAR(dist->getBeta(), 6.0, 1e-10);
}

TEST_F(BetaDistributionTest, SpecialCases) {
    // Beta(1,1) is uniform distribution on [0,1]
    BetaDistribution uniform(1.0, 1.0);
    EXPECT_NEAR(uniform.getProbability(0.2), 1.0, 1e-10);
    EXPECT_NEAR(uniform.getProbability(0.5), 1.0, 1e-10);
    EXPECT_NEAR(uniform.getProbability(0.8), 1.0, 1e-10);
}

TEST_F(BetaDistributionTest, GammaRelationship) {
    // Beta distribution is related to Gamma: if X~Gamma(α,1) and Y~Gamma(β,1),
    // then X/(X+Y) ~ Beta(α,β)
    auto dist = createDefaultDistribution(); // α=2.0, β=5.0
    // Mean should be α/(α+β) = 2/7
    EXPECT_NEAR(dist->getMean(), 2.0/7.0, 1e-10);
}

TEST_F(BetaDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-0.1)) && dist->getLogProbability(-0.1) < 0);
    EXPECT_TRUE(std::isinf(dist->getLogProbability(1.1)) && dist->getLogProbability(1.1) < 0);
}

// Chi-Squared Distribution Tests
class ChiSquaredDistributionTest : public StandardizedDistributionTest<ChiSquaredDistribution> {
protected:
    std::unique_ptr<ChiSquaredDistribution> createDefaultDistribution() override {
        return std::make_unique<ChiSquaredDistribution>();
    }
    
    std::unique_ptr<ChiSquaredDistribution> createParameterizedDistribution() override {
        return std::make_unique<ChiSquaredDistribution>(3.0);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0.5, 1.0, 2.0, 3.0, 5.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1.0, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {1.0, 2.0, 3.0, 4.0, 5.0};
    }
};

// Generate test cases for ChiSquaredDistribution
TEST_F(ChiSquaredDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(ChiSquaredDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(ChiSquaredDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(ChiSquaredDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(ChiSquaredDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(ChiSquaredDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for ChiSquaredDistribution
TEST_F(ChiSquaredDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(ChiSquaredDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(ChiSquaredDistribution(-1.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return ChiSquaredDistribution(std::numeric_limits<double>::quiet_NaN()); }(), std::invalid_argument);
    EXPECT_THROW([]() { return ChiSquaredDistribution(std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(ChiSquaredDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // k=3.0
    // For Chi-squared(k=3): mean = k = 3, variance = 2k = 6
    EXPECT_NEAR(dist->getMean(), 3.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 6.0, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(6.0), 1e-10);
}

TEST_F(ChiSquaredDistributionTest, NonNegativeSupport) {
    auto dist = createDefaultDistribution();
    // Chi-squared is zero for negative values
    EXPECT_DOUBLE_EQ(dist->getProbability(-1.0), 0.0);
    
    // Positive for positive values
    EXPECT_GT(dist->getProbability(0.5), 0.0);
    EXPECT_GT(dist->getProbability(1.0), 0.0);
    EXPECT_GT(dist->getProbability(2.0), 0.0);
}

TEST_F(ChiSquaredDistributionTest, ModeProperty) {
    auto dist = createParameterizedDistribution(); // k=3.0
    // For Chi-squared(k=3): mode = max(0, k-2) = max(0, 1) = 1
    EXPECT_NEAR(dist->getMode(), 1.0, 1e-10);
    
    // Test k < 2 case
    ChiSquaredDistribution chi_small(1.0);
    EXPECT_DOUBLE_EQ(chi_small.getMode(), 0.0);
}

TEST_F(ChiSquaredDistributionTest, GammaRelationship) {
    // Chi-squared(k) is Gamma(k/2, 2)
    auto chi_dist = createParameterizedDistribution(); // k=3.0
    // Mean should be the same: Chi-squared mean = k = 3, Gamma(1.5, 2) mean = 1.5 * 2 = 3
    EXPECT_NEAR(chi_dist->getMean(), 3.0, 1e-10);
    EXPECT_NEAR(chi_dist->getVariance(), 6.0, 1e-10); // Gamma(1.5, 2) variance = 1.5 * 4 = 6
}

TEST_F(ChiSquaredDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1.0)) && dist->getLogProbability(-1.0) < 0);
}

// Uniform Distribution Tests
class UniformDistributionTest : public StandardizedDistributionTest<UniformDistribution> {
protected:
    std::unique_ptr<UniformDistribution> createDefaultDistribution() override {
        return std::make_unique<UniformDistribution>();
    }
    
    std::unique_ptr<UniformDistribution> createParameterizedDistribution() override {
        return std::make_unique<UniformDistribution>(2.0, 8.0);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0.0, 0.25, 0.5, 0.75, 1.0}; // For default [0,1] distribution
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {0.1, 0.3, 0.5, 0.7, 0.9};
    }
};

// Generate test cases for UniformDistribution
TEST_F(UniformDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(UniformDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(UniformDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(UniformDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(UniformDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(UniformDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for UniformDistribution
TEST_F(UniformDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(UniformDistribution(1.0, 1.0), std::invalid_argument); // a == b
    EXPECT_THROW(UniformDistribution(2.0, 1.0), std::invalid_argument); // a > b
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return UniformDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0); }(), std::invalid_argument);
    EXPECT_THROW([]() { return UniformDistribution(0.0, std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(UniformDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // [2.0, 8.0]
    // For Uniform(a=2, b=8): mean = (a+b)/2 = 5, variance = (b-a)²/12 = 36/12 = 3
    EXPECT_NEAR(dist->getMean(), 5.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 3.0, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(3.0), 1e-10);
}

TEST_F(UniformDistributionTest, UniformProbabilityDensity) {
    auto dist = createDefaultDistribution(); // [0, 1]
    // For Uniform(0,1), PDF = 1/(1-0) = 1.0 everywhere in [0,1]
    EXPECT_NEAR(dist->getProbability(0.0), 1.0, 1e-10);
    EXPECT_NEAR(dist->getProbability(0.5), 1.0, 1e-10);
    EXPECT_NEAR(dist->getProbability(1.0), 1.0, 1e-10);
    
    // Outside the range should be 0
    EXPECT_DOUBLE_EQ(dist->getProbability(-0.1), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(1.1), 0.0);
}

TEST_F(UniformDistributionTest, BoundedSupport) {
    auto dist = createParameterizedDistribution(); // [2.0, 8.0]
    // Should be constant within bounds
    double expectedPdf = 1.0 / (8.0 - 2.0); // 1/6
    EXPECT_NEAR(dist->getProbability(3.0), expectedPdf, 1e-10);
    EXPECT_NEAR(dist->getProbability(5.0), expectedPdf, 1e-10);
    EXPECT_NEAR(dist->getProbability(7.0), expectedPdf, 1e-10);
    
    // Outside bounds should be 0
    EXPECT_DOUBLE_EQ(dist->getProbability(1.0), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(9.0), 0.0);
}

TEST_F(UniformDistributionTest, ParameterAccessors) {
    auto dist = createParameterizedDistribution(); // [2.0, 8.0]
    EXPECT_NEAR(dist->getA(), 2.0, 1e-10);
    EXPECT_NEAR(dist->getB(), 8.0, 1e-10);
    EXPECT_NEAR(dist->getMin(), 2.0, 1e-10);
    EXPECT_NEAR(dist->getMax(), 8.0, 1e-10);
}

TEST_F(UniformDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1.0)) && dist->getLogProbability(-1.0) < 0);
    EXPECT_TRUE(std::isinf(dist->getLogProbability(2.0)) && dist->getLogProbability(2.0) < 0);
}

// Pareto Distribution Tests
class ParetoDistributionTest : public StandardizedDistributionTest<ParetoDistribution> {
protected:
    std::unique_ptr<ParetoDistribution> createDefaultDistribution() override {
        return std::make_unique<ParetoDistribution>(2.0, 1.0); // k=2, x_m=1
    }

    std::unique_ptr<ParetoDistribution> createParameterizedDistribution() override {
        return std::make_unique<ParetoDistribution>(1.5, 2.0); // k=1.5, x_m=2
    }

    std::vector<double> getValidTestValues() override {
        return {1.0, 1.5, 2.0, 3.0, 5.0}; // All >= x_m for default case
    }

    std::vector<double> getInvalidTestValues() override {
        return {0.5, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    }

    std::vector<double> getValidFittingData() override {
        return {1.2, 1.5, 2.0, 2.5, 3.0}; // All >= 1.0 (minimum for default)
    }
};

// Generate test cases for ParetoDistribution
TEST_F(ParetoDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(ParetoDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(ParetoDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(ParetoDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(ParetoDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(ParetoDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for ParetoDistribution
TEST_F(ParetoDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(ParetoDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(1.0, -1.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return ParetoDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0); }(), std::invalid_argument);
    EXPECT_THROW([]() { return ParetoDistribution(1.0, std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(ParetoDistributionTest, StatisticalProperties) {
    auto dist = createDefaultDistribution(); // k=2.0, x_m=1.0
    // For Pareto(k=2, x_m=1): mean = k*x_m/(k-1) = 2*1/(2-1) = 2
    EXPECT_NEAR(dist->getMean(), 2.0, 1e-10);
    // variance = k*x_m²/((k-1)²*(k-2)) = 2*1/((1)²*(0)) = undefined for k=2
    // But our implementation should handle this gracefully
    EXPECT_GT(dist->getVariance(), 0.0);
    EXPECT_GT(dist->getStandardDeviation(), 0.0);
}

TEST_F(ParetoDistributionTest, BoundedSupport) {
    auto dist = createDefaultDistribution(); // k=2, x_m=1
    // Pareto distribution is zero for x < x_m
    EXPECT_DOUBLE_EQ(dist->getProbability(0.5), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(0.9), 0.0);
    
    // Positive for x >= x_m
    EXPECT_GT(dist->getProbability(1.0), 0.0);
    EXPECT_GT(dist->getProbability(2.0), 0.0);
    EXPECT_GT(dist->getProbability(5.0), 0.0);
}

TEST_F(ParetoDistributionTest, PowerLawProperty) {
    auto dist = createDefaultDistribution(); // k=2, x_m=1
    // Pareto distribution should follow power law: larger values have smaller probabilities
    EXPECT_GT(dist->getProbability(1.0), dist->getProbability(2.0));
    EXPECT_GT(dist->getProbability(2.0), dist->getProbability(3.0));
    EXPECT_GT(dist->getProbability(3.0), dist->getProbability(5.0));
}

TEST_F(ParetoDistributionTest, ParameterAccessors) {
    auto dist = createParameterizedDistribution(); // k=1.5, x_m=2.0
    EXPECT_NEAR(dist->getK(), 1.5, 1e-10);
    EXPECT_NEAR(dist->getXm(), 2.0, 1e-10);
    
    // Test setters
    dist->setK(2.5);
    EXPECT_NEAR(dist->getK(), 2.5, 1e-10);
    
    dist->setXm(3.0);
    EXPECT_NEAR(dist->getXm(), 3.0, 1e-10);
}

TEST_F(ParetoDistributionTest, ModeProperty) {
    auto dist = createParameterizedDistribution(); // k=1.5, x_m=2.0
    // Pareto distribution's mode is always at x_m
    EXPECT_NEAR(dist->getMode(), 2.0, 1e-10);
}

TEST_F(ParetoDistributionTest, ShapeParameterEffects) {
    // Test different shape parameters
    ParetoDistribution pareto_k1(1.5, 1.0);  // Lower k = heavier tail
    ParetoDistribution pareto_k2(3.0, 1.0);  // Higher k = lighter tail
    
    // Higher k should have higher probability at the mode (x_m = 1.0)
    EXPECT_GT(pareto_k2.getProbability(1.0), pareto_k1.getProbability(1.0));
    
    // But lower k should have higher probability for large values (heavy tail)
    EXPECT_GT(pareto_k1.getProbability(10.0), pareto_k2.getProbability(10.0));
}

TEST_F(ParetoDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(0.5)) && dist->getLogProbability(0.5) < 0);
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1.0)) && dist->getLogProbability(-1.0) < 0);
}

// Discrete Distribution Tests
class DiscreteDistributionTest : public StandardizedDistributionTest<DiscreteDistribution> {
protected:
    std::unique_ptr<DiscreteDistribution> createDefaultDistribution() override {
        return std::make_unique<DiscreteDistribution>(5); // 5 symbols
    }
    
    std::unique_ptr<DiscreteDistribution> createParameterizedDistribution() override {
        auto dist = std::make_unique<DiscreteDistribution>(3);
        dist->setProbability(0, 0.5);
        dist->setProbability(1, 0.3);
        dist->setProbability(2, 0.2);
        return dist;
    }
    
    std::vector<double> getValidTestValues() override {
        return {0, 1, 2, 3, 4};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1, 10, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {0, 0, 1, 1, 1, 2}; // 2 zeros, 3 ones, 1 two
    }
};

// Generate test cases for DiscreteDistribution
TEST_F(DiscreteDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(DiscreteDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(DiscreteDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(DiscreteDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(DiscreteDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(DiscreteDistributionTest, ParameterFitting) { testParameterFitting(); }

// Binomial Distribution Tests
class BinomialDistributionTest : public StandardizedDistributionTest<BinomialDistribution> {
protected:
    std::unique_ptr<BinomialDistribution> createDefaultDistribution() override {
        return std::make_unique<BinomialDistribution>();
    }
    
    std::unique_ptr<BinomialDistribution> createParameterizedDistribution() override {
        return std::make_unique<BinomialDistribution>(10, 0.7);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0, 1, 5, 10};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1, 11, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {5, 6, 7, 8, 9};
    }
};

// Generate test cases for BinomialDistribution
TEST_F(BinomialDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(BinomialDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(BinomialDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(BinomialDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(BinomialDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(BinomialDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for BinomialDistribution
TEST_F(BinomialDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(BinomialDistribution(0, 0.5), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(-1, 0.5), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, -0.1), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, 1.5), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return BinomialDistribution(10, std::numeric_limits<double>::quiet_NaN()); }(), std::invalid_argument);
    EXPECT_THROW([]() { return BinomialDistribution(10, std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(BinomialDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // n=10, p=0.7
    // For Binomial(n=10, p=0.7): mean = n*p = 7, variance = n*p*(1-p) = 2.1
    EXPECT_NEAR(dist->getMean(), 7.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 2.1, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(2.1), 1e-10);
}

TEST_F(BinomialDistributionTest, ProbabilityProperties) {
    auto dist = createDefaultDistribution(); // n=10, p=0.5 (symmetric)
    // For symmetric binomial, P(5) should be maximum
    double prob5 = dist->getProbability(5);
    EXPECT_GT(prob5, dist->getProbability(0));
    EXPECT_GT(prob5, dist->getProbability(10));
    
    // Test edge cases
    BinomialDistribution binomial_p0(10, 0.0);
    EXPECT_DOUBLE_EQ(binomial_p0.getProbability(0), 1.0);
    EXPECT_DOUBLE_EQ(binomial_p0.getProbability(1), 0.0);
    
    BinomialDistribution binomial_p1(10, 1.0);
    EXPECT_DOUBLE_EQ(binomial_p1.getProbability(10), 1.0);
    EXPECT_DOUBLE_EQ(binomial_p1.getProbability(9), 0.0);
}

TEST_F(BinomialDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test out of range (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1)) && dist->getLogProbability(-1) < 0);
    EXPECT_TRUE(std::isinf(dist->getLogProbability(11)) && dist->getLogProbability(11) < 0);
}

// Negative Binomial Distribution Tests
class NegativeBinomialDistributionTest : public StandardizedDistributionTest<NegativeBinomialDistribution> {
protected:
    std::unique_ptr<NegativeBinomialDistribution> createDefaultDistribution() override {
        return std::make_unique<NegativeBinomialDistribution>();
    }
    
    std::unique_ptr<NegativeBinomialDistribution> createParameterizedDistribution() override {
        return std::make_unique<NegativeBinomialDistribution>(3, 0.4);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0, 1, 5, 10};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {4, 5, 6, 7, 8};
    }
};

// Generate test cases for NegativeBinomialDistribution
TEST_F(NegativeBinomialDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(NegativeBinomialDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(NegativeBinomialDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(NegativeBinomialDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(NegativeBinomialDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(NegativeBinomialDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for NegativeBinomialDistribution
TEST_F(NegativeBinomialDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(NegativeBinomialDistribution(0.0, 0.5), std::invalid_argument);
    EXPECT_THROW(NegativeBinomialDistribution(-1.0, 0.5), std::invalid_argument);
    EXPECT_THROW(NegativeBinomialDistribution(5.0, 0.0), std::invalid_argument);
    EXPECT_THROW(NegativeBinomialDistribution(5.0, -0.1), std::invalid_argument);
    EXPECT_THROW(NegativeBinomialDistribution(5.0, 1.5), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return NegativeBinomialDistribution(std::numeric_limits<double>::quiet_NaN(), 0.5); }(), std::invalid_argument);
    EXPECT_THROW([]() { return NegativeBinomialDistribution(5.0, std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(NegativeBinomialDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // r=3, p=0.4
    // For NegBinom(r=3, p=0.4): mean = r*(1-p)/p = 3*0.6/0.4 = 4.5, variance = r*(1-p)/p² = 3*0.6/0.16 = 11.25
    EXPECT_NEAR(dist->getMean(), 4.5, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 11.25, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(11.25), 1e-10);
}

TEST_F(NegativeBinomialDistributionTest, OverDispersionProperty) {
    // Negative binomial should exhibit over-dispersion (variance > mean)
    auto dist1 = createDefaultDistribution(); // r=5, p=0.5
    auto dist2 = createParameterizedDistribution(); // r=3, p=0.4
    
    EXPECT_GT(dist1->getVariance(), dist1->getMean());
    EXPECT_GT(dist2->getVariance(), dist2->getMean());
}

TEST_F(NegativeBinomialDistributionTest, EdgeCaseProbabilities) {
    // Test edge case p = 1
    NegativeBinomialDistribution negbinom_p1(5.0, 1.0);
    EXPECT_DOUBLE_EQ(negbinom_p1.getProbability(0), 1.0);
    EXPECT_DOUBLE_EQ(negbinom_p1.getProbability(1), 0.0);
}

TEST_F(NegativeBinomialDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test out of range (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1)) && dist->getLogProbability(-1) < 0);
}

// Poisson Distribution Tests
class PoissonDistributionTest : public StandardizedDistributionTest<PoissonDistribution> {
protected:
    std::unique_ptr<PoissonDistribution> createDefaultDistribution() override {
        return std::make_unique<PoissonDistribution>();
    }
    
    std::unique_ptr<PoissonDistribution> createParameterizedDistribution() override {
        return std::make_unique<PoissonDistribution>(4.0);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0, 1, 2, 3, 4, 5};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {1, 1, 2, 3, 3, 4};
    }
};

// Generate test cases for PoissonDistribution
TEST_F(PoissonDistributionTest, BasicFunctionality) { testBasicFunctionality(); }
TEST_F(PoissonDistributionTest, ProbabilityCalculation) { testProbabilityCalculation(); }
TEST_F(PoissonDistributionTest, InvalidInputHandling) { testInvalidInputHandling(); }
TEST_F(PoissonDistributionTest, StringRepresentation) { testStringRepresentation(); }
TEST_F(PoissonDistributionTest, ResetFunctionality) { testResetFunctionality(); }
TEST_F(PoissonDistributionTest, ParameterFitting) { testParameterFitting(); }

// Additional comprehensive tests for PoissonDistribution
TEST_F(PoissonDistributionTest, ParameterValidation) {
    // Test invalid lambda values in constructor
    EXPECT_THROW(PoissonDistribution(-1.0), std::invalid_argument);
    EXPECT_THROW(PoissonDistribution(0.0), std::invalid_argument);
    
    // Test NaN and infinity using lambdas to avoid macro issues
    EXPECT_THROW([]() { return PoissonDistribution(std::numeric_limits<double>::quiet_NaN()); }(), std::invalid_argument);
    EXPECT_THROW([]() { return PoissonDistribution(std::numeric_limits<double>::infinity()); }(), std::invalid_argument);
}

TEST_F(PoissonDistributionTest, StatisticalProperties) {
    auto dist = createParameterizedDistribution(); // lambda=4.0
    // For Poisson(λ=4): mean = variance = λ = 4.0
    EXPECT_NEAR(dist->getMean(), 4.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 4.0, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), 2.0, 1e-10);
}

TEST_F(PoissonDistributionTest, KnownProbabilities) {
    PoissonDistribution poisson(2.0);
    // For λ = 2.0: P(X=0) = e^(-2) ≈ 0.1353
    double p0 = poisson.getProbability(0.0);
    EXPECT_NEAR(p0, std::exp(-2.0), 1e-10);
    
    // P(X=1) = 2 * e^(-2) ≈ 0.2707
    double p1 = poisson.getProbability(1.0);
    EXPECT_NEAR(p1, 2.0 * std::exp(-2.0), 1e-10);
    
    // P(X=2) = 2 * e^(-2) ≈ 0.2707
    double p2 = poisson.getProbability(2.0);
    EXPECT_NEAR(p2, 2.0 * std::exp(-2.0), 1e-10);
}

TEST_F(PoissonDistributionTest, NonIntegerValues) {
    auto dist = createDefaultDistribution();
    // Non-integer values should return 0
    EXPECT_DOUBLE_EQ(dist->getProbability(1.5), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(2.3), 0.0);
}

TEST_F(PoissonDistributionTest, LogProbability) {
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
    
    // Test invalid inputs (should return -infinity)
    EXPECT_TRUE(std::isinf(dist->getLogProbability(-1)) && dist->getLogProbability(-1) < 0);
    EXPECT_TRUE(std::isinf(dist->getLogProbability(2.5)) && dist->getLogProbability(2.5) < 0);
}

// Additional comprehensive tests for DiscreteDistribution
TEST_F(DiscreteDistributionTest, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW(DiscreteDistribution(0), std::invalid_argument);
    EXPECT_THROW(DiscreteDistribution(-1), std::length_error);
    
    // Test invalid probability setting
    auto dist = createDefaultDistribution();
    EXPECT_THROW(dist->setProbability(-1, 0.1), std::out_of_range);
    EXPECT_THROW(dist->setProbability(5, 0.1), std::out_of_range);
    EXPECT_THROW(dist->setProbability(0, -0.1), std::invalid_argument);
    EXPECT_THROW(dist->setProbability(0, 1.1), std::invalid_argument);
}

TEST_F(DiscreteDistributionTest, StatisticalProperties) {
    auto dist = createDefaultDistribution(); // uniform 5-symbol distribution
    // For uniform discrete distribution [0,1,2,3,4]: mean = 2.0
    EXPECT_NEAR(dist->getMean(), 2.0, 1e-10);
    EXPECT_GT(dist->getVariance(), 0.0);
    EXPECT_GT(dist->getStandardDeviation(), 0.0);
}

TEST_F(DiscreteDistributionTest, UniformDistribution) {
    auto dist = createDefaultDistribution(); // 5 symbols
    // Test valid values - should all be equal for uniform distribution
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_NEAR(prob, 0.2, 1e-10);  // 1/5 = 0.2
    }
    
    // Test probability sum equals 1
    double sum = 0.0;
    for (int i = 0; i < 5; ++i) {
        sum += dist->getProbability(i);
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST_F(DiscreteDistributionTest, CustomProbabilities) {
    auto dist = createParameterizedDistribution(); // [0.5, 0.3, 0.2]
    EXPECT_NEAR(dist->getProbability(0), 0.5, 1e-10);
    EXPECT_NEAR(dist->getProbability(1), 0.3, 1e-10);
    EXPECT_NEAR(dist->getProbability(2), 0.2, 1e-10);
    
    // Test probability sum equals 1
    double sum = dist->getProbability(0) + dist->getProbability(1) + dist->getProbability(2);
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST_F(DiscreteDistributionTest, FittingFromData) {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData(); // {0, 0, 1, 1, 1, 2}
    
    dist->fit(data);
    EXPECT_NEAR(dist->getProbability(0), 2.0/6.0, 1e-10);  // 2 zeros out of 6
    EXPECT_NEAR(dist->getProbability(1), 3.0/6.0, 1e-10);  // 3 ones out of 6
    EXPECT_NEAR(dist->getProbability(2), 1.0/6.0, 1e-10);  // 1 two out of 6
}

// Test all distributions work with the base interface
class CommonDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        distributions_.push_back(std::make_unique<GaussianDistribution>());
        distributions_.push_back(std::make_unique<StudentTDistribution>());
        distributions_.push_back(std::make_unique<LogNormalDistribution>());
        distributions_.push_back(std::make_unique<ExponentialDistribution>());
        distributions_.push_back(std::make_unique<WeibullDistribution>());
        distributions_.push_back(std::make_unique<RayleighDistribution>());
        distributions_.push_back(std::make_unique<GammaDistribution>());
        distributions_.push_back(std::make_unique<BetaDistribution>());
        distributions_.push_back(std::make_unique<ChiSquaredDistribution>());
        distributions_.push_back(std::make_unique<UniformDistribution>());
        distributions_.push_back(std::make_unique<ParetoDistribution>());
        distributions_.push_back(std::make_unique<DiscreteDistribution>(6));
        distributions_.push_back(std::make_unique<BinomialDistribution>());
        distributions_.push_back(std::make_unique<NegativeBinomialDistribution>());
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
