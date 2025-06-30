#include <gtest/gtest.h>
#include "libhmm/distributions/distributions.h"
#include <memory>
#include <vector>
#include <cmath>
#include <climits>
#include <chrono>
#include <iomanip>
#include <sstream>

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
    
    // Standard test methods that all distributions should implement
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
        EXPECT_NO_THROW(dist->getProbability(1.0)) << "Should work after reset";
    }
    
    void testParameterFitting() {
        auto dist = createDefaultDistribution();
        auto data = getValidFittingData();
        if (!data.empty()) {
            EXPECT_NO_THROW(dist->fit(data)) << "fit() should handle valid data";
        }
        
        // Test empty data
        std::vector<double> emptyData;
        EXPECT_NO_THROW(dist->fit(emptyData)) << "fit() should handle empty data gracefully";
    }
};

// Macro to generate test cases for each distribution
#define DECLARE_DISTRIBUTION_TESTS(DistributionClass, TestClassName) \
class TestClassName : public StandardizedDistributionTest<DistributionClass> { \
protected: \
    std::unique_ptr<DistributionClass> createDefaultDistribution() override; \
    std::unique_ptr<DistributionClass> createParameterizedDistribution() override; \
    std::vector<double> getValidTestValues() override; \
    std::vector<double> getInvalidTestValues() override; \
    std::vector<double> getValidFittingData() override; \
}; \
 \
TEST_F(TestClassName, BasicFunctionality) { testBasicFunctionality(); } \
TEST_F(TestClassName, ProbabilityCalculation) { testProbabilityCalculation(); } \
TEST_F(TestClassName, InvalidInputHandling) { testInvalidInputHandling(); } \
TEST_F(TestClassName, StringRepresentation) { testStringRepresentation(); } \
TEST_F(TestClassName, ResetFunctionality) { testResetFunctionality(); } \
TEST_F(TestClassName, ParameterFitting) { testParameterFitting(); }

// Gaussian Distribution - Standard Interface Tests
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
    
    std::vector<double> getInvalidFittingData() override {
        return {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    }
    
    double getExpectedMeanApprox() override { return 0.0; }
    double getExpectedVarianceApprox() override { return 1.0; }
    
    // Method declarations for standardized tests
    void runBasicFunctionalityTests() override;
    void runProbabilityTests() override;
    void runParameterValidationTests() override;
    void runFittingTests() override;
    void runStringRepresentationTests() override;
    void runStatisticalPropertiesTests() override;
    void runNumericalStabilityTests() override;
    void runInvalidInputHandlingTests() override;
    void runResetFunctionalityTests() override;
};

// Implementation of standardized test methods for GaussianDistribution
void GaussianDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getMean(), 0.0);
    EXPECT_DOUBLE_EQ(defaultDist->getStandardDeviation(), 1.0);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getMean(), 5.0);
    EXPECT_DOUBLE_EQ(paramDist->getStandardDeviation(), 2.0);
}

void GaussianDistributionTest::runProbabilityTests() {
    auto dist = createDefaultDistribution();
    
    // Test valid values
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_GT(prob, 0.0);
        EXPECT_TRUE(std::isfinite(prob));
    }
    
    // Test symmetry around mean
    EXPECT_NEAR(dist->getProbability(1.0), dist->getProbability(-1.0), 1e-10);
    
    // Maximum at mean
    EXPECT_GT(dist->getProbability(0.0), dist->getProbability(1.0));
}

void GaussianDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(GaussianDistribution(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(std::numeric_limits<double>::infinity(), 1.0), std::invalid_argument);
}

void GaussianDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_NEAR(dist->getMean(), 3.0, 1e-10);  // Mean of {1,2,3,4,5}
    EXPECT_GT(dist->getStandardDeviation(), 0.0);
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void GaussianDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Gaussian") != std::string::npos || str.find("Normal") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("5") != std::string::npos);  // Mean value
    EXPECT_TRUE(str.find("2") != std::string::npos);  // Std dev value
}

void GaussianDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution();
    
    EXPECT_NEAR(dist->getMean(), 5.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 4.0, 1e-10);  // σ² = 2² = 4
    EXPECT_NEAR(dist->getStandardDeviation(), 2.0, 1e-10);
}

void GaussianDistributionTest::runNumericalStabilityTests() {
    // Test with extreme values
    GaussianDistribution extremeDist(0.0, 1e-10);
    EXPECT_GT(extremeDist.getProbability(0.0), 0.0);
    EXPECT_TRUE(std::isfinite(extremeDist.getProbability(0.0)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

void GaussianDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void GaussianDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameters
    dist->setMean(100.0);
    dist->setStandardDeviation(50.0);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getMean(), 0.0);
    EXPECT_DOUBLE_EQ(dist->getStandardDeviation(), 1.0);
}

// TEST_F declarations for GaussianDistribution
TEST_F(GaussianDistributionTest, BasicFunctionality) {
    runBasicFunctionalityTests();
}

TEST_F(GaussianDistributionTest, ProbabilityCalculation) {
    runProbabilityTests();
}

TEST_F(GaussianDistributionTest, ParameterValidation) {
    runParameterValidationTests();
}

TEST_F(GaussianDistributionTest, ParameterFitting) {
    runFittingTests();
}

TEST_F(GaussianDistributionTest, StringRepresentation) {
    runStringRepresentationTests();
}

TEST_F(GaussianDistributionTest, StatisticalProperties) {
    runStatisticalPropertiesTests();
}

TEST_F(GaussianDistributionTest, NumericalStability) {
    runNumericalStabilityTests();
}

TEST_F(GaussianDistributionTest, InvalidInputHandling) {
    runInvalidInputHandlingTests();
}

TEST_F(GaussianDistributionTest, ResetFunctionality) {
    runResetFunctionalityTests();
}

// Discrete Distribution - Standard Interface Tests
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
    
    std::vector<double> getInvalidFittingData() override {
        return {0, 1, std::numeric_limits<double>::quiet_NaN(), 2};
    }
    
    double getExpectedMeanApprox() override { return 2.0; } // For uniform 5-symbol distribution
    double getExpectedVarianceApprox() override { return 2.0; }
};

// Implementation of standardized test methods for DiscreteDistribution
void DiscreteDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_EQ(defaultDist->getNumSymbols(), 5);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_EQ(paramDist->getNumSymbols(), 3);
    EXPECT_NEAR(paramDist->getProbability(0), 0.5, 1e-10);
    EXPECT_NEAR(paramDist->getProbability(1), 0.3, 1e-10);
    EXPECT_NEAR(paramDist->getProbability(2), 0.2, 1e-10);
}

void DiscreteDistributionTest::runProbabilityTests() {
    auto dist = createDefaultDistribution();
    
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

void DiscreteDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(DiscreteDistribution(0), std::invalid_argument);
    EXPECT_THROW(DiscreteDistribution(-1), std::invalid_argument);
    
    // Test invalid probability setting
    auto dist = createDefaultDistribution();
    EXPECT_THROW(dist->setProbability(-1, 0.1), std::out_of_range);
    EXPECT_THROW(dist->setProbability(5, 0.1), std::out_of_range);
    EXPECT_THROW(dist->setProbability(0, -0.1), std::invalid_argument);
    EXPECT_THROW(dist->setProbability(0, 1.1), std::invalid_argument);
}

void DiscreteDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_NEAR(dist->getProbability(0), 2.0/6.0, 1e-10);  // 2 zeros out of 6
    EXPECT_NEAR(dist->getProbability(1), 3.0/6.0, 1e-10);  // 3 ones out of 6
    EXPECT_NEAR(dist->getProbability(2), 1.0/6.0, 1e-10);  // 1 two out of 6
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void DiscreteDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Discrete") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("0.5") != std::string::npos);
    EXPECT_TRUE(str.find("0.3") != std::string::npos);
    EXPECT_TRUE(str.find("0.2") != std::string::npos);
}

void DiscreteDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createDefaultDistribution();
    
    // For uniform discrete distribution [0,1,2,3,4]: mean = 2.0
    EXPECT_NEAR(dist->getMean(), 2.0, 1e-10);
    EXPECT_GT(dist->getVariance(), 0.0);
    EXPECT_GT(dist->getStandardDeviation(), 0.0);
}

void DiscreteDistributionTest::runNumericalStabilityTests() {
    // Test with large symbol count
    DiscreteDistribution largeDist(1000);
    EXPECT_NEAR(largeDist.getProbability(500), 0.001, 1e-10);
    
    // Test extreme probability values
    auto dist = createDefaultDistribution();
    dist->setProbability(0, 0.99999);
    EXPECT_NEAR(dist->getProbability(0), 0.99999, 1e-10);
}

void DiscreteDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void DiscreteDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Reset and verify uniform distribution
    EXPECT_NO_THROW(dist->reset());
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(dist->getProbability(i), 1.0/3.0, 1e-10);
    }
}

// Gamma Distribution - Standard Interface Tests
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
    
    std::vector<double> getInvalidFittingData() override {
        return {0.5, -1.0, 2.0};
    }
    
    double getExpectedMeanApprox() override { return 2.0; }
    double getExpectedVarianceApprox() override { return 2.0; }
};

// Implementation of standardized test methods for GammaDistribution
void GammaDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getShape(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist->getScale(), 1.0);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getShape(), 2.0);
    EXPECT_DOUBLE_EQ(paramDist->getScale(), 1.5);
}

void GammaDistributionTest::runProbabilityTests() {
    auto dist = createDefaultDistribution();
    
    // Test valid values
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_GT(prob, 0.0);
        EXPECT_TRUE(std::isfinite(prob));
    }
    
    // Gamma distribution is zero at x=0 for shape > 1
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getProbability(0.0), 0.0);
}

void GammaDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(GammaDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(1.0, -1.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(GammaDistribution(1.0, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void GammaDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_GT(dist->getShape(), 0.0);
    EXPECT_GT(dist->getScale(), 0.0);
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void GammaDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Gamma") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("2") != std::string::npos);  // Shape value
    EXPECT_TRUE(str.find("1.5") != std::string::npos);  // Scale value
}

void GammaDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution();
    
    // For Gamma(k=2, θ=1.5): mean = k*θ = 3, variance = k*θ² = 4.5
    EXPECT_NEAR(dist->getMean(), 3.0, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 4.5, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(4.5), 1e-10);
}

void GammaDistributionTest::runNumericalStabilityTests() {
    // Test with extreme shape values
    GammaDistribution extremeDist(0.1, 1.0);
    EXPECT_GT(extremeDist.getProbability(0.1), 0.0);
    EXPECT_TRUE(std::isfinite(extremeDist.getProbability(0.1)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

void GammaDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void GammaDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameters
    dist->setShape(10.0);
    dist->setScale(5.0);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getShape(), 1.0);
    EXPECT_DOUBLE_EQ(dist->getScale(), 1.0);
}

// Exponential Distribution - Standard Interface Tests
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
    
    std::vector<double> getInvalidFittingData() override {
        return {0.5, -1.0, 2.0};
    }
    
    double getExpectedMeanApprox() override { return 1.0; }
    double getExpectedVarianceApprox() override { return 1.0; }
};

// Implementation of standardized test methods for ExponentialDistribution
void ExponentialDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getLambda(), 1.0);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getLambda(), 2.0);
}

void ExponentialDistributionTest::runProbabilityTests() {
    auto dist = createDefaultDistribution();
    
    // Test valid values
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_GE(prob, 0.0);
        EXPECT_TRUE(std::isfinite(prob));
    }
    
    // Exponential at x=0 should equal λ
    EXPECT_DOUBLE_EQ(dist->getProbability(0.0), 1.0);
    
    // Should decrease with increasing x
    EXPECT_GT(dist->getProbability(0.5), dist->getProbability(1.0));
}

void ExponentialDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(ExponentialDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(ExponentialDistribution(-1.0), std::invalid_argument);
    EXPECT_THROW(ExponentialDistribution(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(ExponentialDistribution(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void ExponentialDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_GT(dist->getLambda(), 0.0);
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void ExponentialDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Exponential") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("2") != std::string::npos);  // Lambda value
}

void ExponentialDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution();
    
    // For Exponential(λ=2): mean = 1/λ = 0.5, variance = 1/λ² = 0.25
    EXPECT_NEAR(dist->getMean(), 0.5, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 0.25, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), 0.5, 1e-10);
}

void ExponentialDistributionTest::runNumericalStabilityTests() {
    // Test with extreme lambda values
    ExponentialDistribution extremeDist(1e-6);
    EXPECT_GT(extremeDist.getProbability(0.1), 0.0);
    EXPECT_TRUE(std::isfinite(extremeDist.getProbability(0.1)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

void ExponentialDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void ExponentialDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameter
    dist->setLambda(10.0);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getLambda(), 1.0);
}

// LogNormal Distribution - Standard Interface Tests
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
        return {0.5, 1.0, 2.0, 3.0, 4.0};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {0.5, -1.0, 2.0};
    }
    
    double getExpectedMeanApprox() override { return 1.6; }
    double getExpectedVarianceApprox() override { return 1.0; }
};

// Implementation of standardized test methods for LogNormalDistribution
void LogNormalDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getMu(), 0.0);
    EXPECT_DOUBLE_EQ(defaultDist->getSigma(), 1.0);

    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getMu(), 1.0);
    EXPECT_DOUBLE_EQ(paramDist->getSigma(), 0.5);
}

void LogNormalDistributionTest::runProbabilityTests() {
    auto dist = createDefaultDistribution();

    // Test valid values
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_GT(prob, 0.0);
        EXPECT_TRUE(std::isfinite(prob));
    }

    // Should be zero at x=0
    EXPECT_DOUBLE_EQ(dist->getProbability(0.0), 0.0);

    // Should decrease for negative values
    EXPECT_DOUBLE_EQ(dist->getProbability(-1.0), 0.0);
}

void LogNormalDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(LogNormalDistribution(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(LogNormalDistribution(0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(LogNormalDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(LogNormalDistribution(1.0, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void LogNormalDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();

    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_GT(dist->getMu(), 0.0);
    EXPECT_GT(dist->getSigma(), 0.0);

    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void LogNormalDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();

    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("LogNormal") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("1") != std::string::npos);  // Mu value
    EXPECT_TRUE(str.find("0.5") != std::string::npos); // Sigma value
}

void LogNormalDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution();

    // For LogNormal with parameters (μ,σ), mean = exp(μ + (σ²/2))
    EXPECT_GT(dist->getMean(), 1.0);
    EXPECT_LT(dist->getMean(), 2.0);

    // Variance: (exp(σ²) - 1) * exp(2μ + σ²)
    EXPECT_GT(dist->getVariance(), 1.0);
    EXPECT_LT(dist->getVariance(), 10.0);
}

void LogNormalDistributionTest::runNumericalStabilityTests() {
    // Test with extreme mu and sigma values
    LogNormalDistribution extremeDist(0.0, 1e-10);
    EXPECT_GT(extremeDist.getProbability(1.0), 0.0);
    EXPECT_TRUE(std::isfinite(extremeDist.getProbability(1.0)));

    LogNormalDistribution largeDist(1e10, 1.0);
    EXPECT_GT(largeDist.getProbability(1e5), 0.0);
}

void LogNormalDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();

    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void LogNormalDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();

    // Modify parameters
    dist->setMu(-1.0);
    dist->setSigma(0.1);

    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getMu(), 0.0);
    EXPECT_DOUBLE_EQ(dist->getSigma(), 1.0);
}

// Pareto Distribution - Standard Interface Tests
class ParetoDistributionTest : public StandardizedDistributionTest<ParetoDistribution> {
protected:
    std::unique_ptr<ParetoDistribution> createDefaultDistribution() override {
        return std::make_unique<ParetoDistribution>();
    }
    
    std::unique_ptr<ParetoDistribution> createParameterizedDistribution() override {
        return std::make_unique<ParetoDistribution>(2.0, 1.5);
    }
    
    std::vector<double> getValidTestValues() override {
        return {1.5, 2.0, 3.0, 5.0, 10.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1.0, 0.5, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {1.5, 2.0, 3.0, 4.0, 5.0};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {0.5, 1.0, 2.0}; // Values below xm=1.0
    }
    
    double getExpectedMeanApprox() override { return 3.0; }
    double getExpectedVarianceApprox() override { return 1.5; }
};

// Implementation of standardized test methods for ParetoDistribution
void ParetoDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist->getXm(), 1.0);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getAlpha(), 2.0);
    EXPECT_DOUBLE_EQ(paramDist->getXm(), 1.5);
}

void ParetoDistributionTest::runProbabilityTests() {
    auto dist = createDefaultDistribution();
    
    // Test valid values (x >= xm)
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_GT(prob, 0.0);
        EXPECT_TRUE(std::isfinite(prob));
    }
    
    // Should be zero for x < xm
    EXPECT_DOUBLE_EQ(dist->getProbability(0.5), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(0.99), 0.0);
}

void ParetoDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(ParetoDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(1.0, -1.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(ParetoDistribution(1.0, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void ParetoDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_GT(dist->getAlpha(), 0.0);
    EXPECT_GT(dist->getXm(), 0.0);
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void ParetoDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Pareto") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("2") != std::string::npos);  // Alpha value
    EXPECT_TRUE(str.find("1.5") != std::string::npos);  // Xm value
}

void ParetoDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution();
    
    // For Pareto(α=2, xm=1.5): mean = α*xm/(α-1) if α > 1
    double expectedMean = 2.0 * 1.5 / (2.0 - 1.0);
    EXPECT_NEAR(dist->getMean(), expectedMean, 1e-10);
    
    // Variance exists if α > 2
    EXPECT_GT(dist->getVariance(), 0.0);
    EXPECT_GT(dist->getStandardDeviation(), 0.0);
}

void ParetoDistributionTest::runNumericalStabilityTests() {
    // Test with extreme alpha values
    ParetoDistribution extremeDist(0.1, 1.0);
    EXPECT_GT(extremeDist.getProbability(2.0), 0.0);
    EXPECT_TRUE(std::isfinite(extremeDist.getProbability(2.0)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

void ParetoDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void ParetoDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameters
    dist->setAlpha(10.0);
    dist->setXm(5.0);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(dist->getXm(), 1.0);
}

// Poisson Distribution - Standard Interface Tests
class PoissonDistributionTest : public StandardizedDistributionTest<PoissonDistribution> {
protected:
    std::unique_ptr<PoissonDistribution> createDefaultDistribution() override {
        return std::make_unique<PoissonDistribution>();
    }
    
    std::unique_ptr<PoissonDistribution> createParameterizedDistribution() override {
        return std::make_unique<PoissonDistribution>(3.5);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0, 1, 2, 3, 5, 10};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1, 1.5, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {1, 2, 2, 3, 3, 3, 4, 2, 1, 4}; // Mean = 2.5
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {1, 2, -1, 3};
    }
    
    double getExpectedMeanApprox() override { return 1.0; }
    double getExpectedVarianceApprox() override { return 1.0; }
};

// Implementation of standardized test methods for PoissonDistribution
void PoissonDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getLambda(), 1.0);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getLambda(), 3.5);
}

void PoissonDistributionTest::runProbabilityTests() {
    auto dist = createParameterizedDistribution(); // λ=3.5
    
    // Test valid integer values
    for (double val : getValidTestValues()) {
        if (val == static_cast<int>(val)) { // Only test integer values
            double prob = dist->getProbability(val);
            EXPECT_GT(prob, 0.0);
            EXPECT_TRUE(std::isfinite(prob));
        }
    }
    
    // Test known values for λ=3.5
    EXPECT_GT(dist->getProbability(0.0), 0.0);
    EXPECT_GT(dist->getProbability(3.0), 0.0);
}

void PoissonDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(PoissonDistribution(0.0), std::invalid_argument);
    EXPECT_THROW(PoissonDistribution(-1.0), std::invalid_argument);
    EXPECT_THROW(PoissonDistribution(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(PoissonDistribution(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void PoissonDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_NEAR(dist->getLambda(), 2.5, 1e-10); // Mean of data
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void PoissonDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Poisson") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("3.5") != std::string::npos);
}

void PoissonDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution();
    
    // For Poisson(λ): mean = variance = λ
    EXPECT_NEAR(dist->getMean(), 3.5, 1e-10);
    EXPECT_NEAR(dist->getVariance(), 3.5, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(3.5), 1e-10);
}

void PoissonDistributionTest::runNumericalStabilityTests() {
    // Test with large lambda
    PoissonDistribution largeDist(100.0);
    EXPECT_GT(largeDist.getProbability(100.0), 0.0);
    EXPECT_TRUE(std::isfinite(largeDist.getProbability(100.0)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        if (val == static_cast<int>(val)) {
            double logProb = dist->getLogProbability(val);
            EXPECT_TRUE(std::isfinite(logProb));
            EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
        }
    }
}

void PoissonDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void PoissonDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameter
    dist->setLambda(10.0);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getLambda(), 1.0);
}

// Beta Distribution - Standard Interface Tests
class BetaDistributionTest : public StandardizedDistributionTest<BetaDistribution> {
protected:
    std::unique_ptr<BetaDistribution> createDefaultDistribution() override {
        return std::make_unique<BetaDistribution>();
    }
    
    std::unique_ptr<BetaDistribution> createParameterizedDistribution() override {
        return std::make_unique<BetaDistribution>(2.5, 3.5);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0.1, 0.3, 0.5, 0.7, 0.9};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {0.2, 0.4, 0.6, 0.8};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {0.5, 1.5, 0.3};
    }
    
    double getExpectedMeanApprox() override { return 0.5; }
    double getExpectedVarianceApprox() override { return 0.08; }
};

// Implementation of standardized test methods for BetaDistribution
void BetaDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist->getBeta(), 1.0);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getAlpha(), 2.5);
    EXPECT_DOUBLE_EQ(paramDist->getBeta(), 3.5);
}

void BetaDistributionTest::runProbabilityTests() {
    auto dist = createDefaultDistribution();
    
    // Test valid values [0,1]
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_GT(prob, 0.0);
        EXPECT_TRUE(std::isfinite(prob));
    }
    
    // Test boundary behavior
    EXPECT_GT(dist->getProbability(0.5), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(0.0), 1.0);  // Beta(1,1) at x=0
    EXPECT_DOUBLE_EQ(dist->getProbability(1.0), 1.0);  // Beta(1,1) at x=1
}

void BetaDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(BetaDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(1.0, -1.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(BetaDistribution(1.0, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void BetaDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_GT(dist->getAlpha(), 0.0);
    EXPECT_GT(dist->getBeta(), 0.0);
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void BetaDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Beta") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("2.5") != std::string::npos);
    EXPECT_TRUE(str.find("3.5") != std::string::npos);
}

void BetaDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution();
    
    // For Beta(2.5, 3.5): mean = α/(α+β) = 2.5/6 ≈ 0.417
    double expectedMean = 2.5 / (2.5 + 3.5);
    EXPECT_NEAR(dist->getMean(), expectedMean, 1e-10);
    
    // Variance = αβ/((α+β)²(α+β+1))
    double expectedVar = (2.5 * 3.5) / (std::pow(6.0, 2) * 7.0);
    EXPECT_NEAR(dist->getVariance(), expectedVar, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
}

void BetaDistributionTest::runNumericalStabilityTests() {
    // Test with extreme alpha/beta values
    BetaDistribution extremeDist(0.1, 0.1);
    EXPECT_GT(extremeDist.getProbability(0.5), 0.0);
    EXPECT_TRUE(std::isfinite(extremeDist.getProbability(0.5)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

void BetaDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void BetaDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameters
    dist->setAlpha(10.0);
    dist->setBeta(5.0);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(dist->getBeta(), 1.0);
}

// Weibull Distribution - Standard Interface Tests
class WeibullDistributionTest : public StandardizedDistributionTest<WeibullDistribution> {
protected:
    std::unique_ptr<WeibullDistribution> createDefaultDistribution() override {
        return std::make_unique<WeibullDistribution>();
    }
    
    std::unique_ptr<WeibullDistribution> createParameterizedDistribution() override {
        return std::make_unique<WeibullDistribution>(2.5, 1.5);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0.1, 0.5, 1.0, 2.0, 5.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1.0, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {0.5, 1.0, 1.5, 2.0, 2.5};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {0.5, -1.0, 2.0};
    }
    
    double getExpectedMeanApprox() override { return 1.3; }
    double getExpectedVarianceApprox() override { return 0.8; }
};

// Implementation of standardized test methods for WeibullDistribution
void WeibullDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getK(), 1.0);
    EXPECT_DOUBLE_EQ(defaultDist->getLambda(), 1.0);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getK(), 2.5);
    EXPECT_DOUBLE_EQ(paramDist->getLambda(), 1.5);
}

void WeibullDistributionTest::runProbabilityTests() {
    auto dist = createDefaultDistribution();
    
    // Test valid values
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_GT(prob, 0.0);
        EXPECT_TRUE(std::isfinite(prob));
    }
    
    // Weibull is zero for negative values
    EXPECT_DOUBLE_EQ(dist->getProbability(-1.0), 0.0);
    EXPECT_GE(dist->getProbability(0.0), 0.0);
}

void WeibullDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(WeibullDistribution(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(-1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(1.0, -1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(WeibullDistribution(1.0, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void WeibullDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_GT(dist->getK(), 0.0);
    EXPECT_GT(dist->getLambda(), 0.0);
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void WeibullDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Weibull") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("2.5") != std::string::npos);
    EXPECT_TRUE(str.find("1.5") != std::string::npos);
}

void WeibullDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution();
    
    // For Weibull(k, λ): mean involves gamma function
    EXPECT_GT(dist->getMean(), 0.0);
    EXPECT_GT(dist->getVariance(), 0.0);
    EXPECT_GT(dist->getStandardDeviation(), 0.0);
}

void WeibullDistributionTest::runNumericalStabilityTests() {
    // Test with extreme parameters
    WeibullDistribution extremeDist(0.1, 1.0);
    EXPECT_GT(extremeDist.getProbability(0.5), 0.0);
    EXPECT_TRUE(std::isfinite(extremeDist.getProbability(0.5)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

void WeibullDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void WeibullDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameters
    dist->setK(5.0);
    dist->setLambda(3.0);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getK(), 1.0);
    EXPECT_DOUBLE_EQ(dist->getLambda(), 1.0);
}

// Uniform Distribution - Standard Interface Tests
class UniformDistributionTest : public StandardizedDistributionTest<UniformDistribution> {
protected:
    std::unique_ptr<UniformDistribution> createDefaultDistribution() override {
        return std::make_unique<UniformDistribution>();
    }
    
    std::unique_ptr<UniformDistribution> createParameterizedDistribution() override {
        return std::make_unique<UniformDistribution>(2.0, 8.0);
    }
    
    std::vector<double> getValidTestValues() override {
        return {3.0, 5.0, 7.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {1.0, 9.0, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {3.0, 4.5, 6.0, 7.5};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    }
    
    double getExpectedMeanApprox() override { return 0.5; }
    double getExpectedVarianceApprox() override { return 0.08; }
};

// Implementation of standardized test methods for UniformDistribution
void UniformDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_DOUBLE_EQ(defaultDist->getA(), 0.0);
    EXPECT_DOUBLE_EQ(defaultDist->getB(), 1.0);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_DOUBLE_EQ(paramDist->getA(), 2.0);
    EXPECT_DOUBLE_EQ(paramDist->getB(), 8.0);
}

void UniformDistributionTest::runProbabilityTests() {
    auto dist = createParameterizedDistribution(); // [2, 8]
    
    // Uniform distribution should have constant probability within interval
    double expectedPdf = 1.0 / (8.0 - 2.0); // 1/6
    
    for (double val : getValidTestValues()) {
        double prob = dist->getProbability(val);
        EXPECT_NEAR(prob, expectedPdf, 1e-10);
    }
    
    // Should be zero outside interval
    EXPECT_DOUBLE_EQ(dist->getProbability(1.0), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(9.0), 0.0);
}

void UniformDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(UniformDistribution(2.0, 1.0), std::invalid_argument); // a > b
    EXPECT_THROW(UniformDistribution(3.0, 3.0), std::invalid_argument); // a == b
    EXPECT_THROW(UniformDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(UniformDistribution(1.0, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void UniformDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_LE(dist->getA(), *std::min_element(data.begin(), data.end()));
    EXPECT_GE(dist->getB(), *std::max_element(data.begin(), data.end()));
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void UniformDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Uniform") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("2") != std::string::npos);
    EXPECT_TRUE(str.find("8") != std::string::npos);
}

void UniformDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution(); // [2, 8]
    
    // For Uniform(a, b): mean = (a+b)/2, variance = (b-a)²/12
    double expectedMean = (2.0 + 8.0) / 2.0;
    double expectedVar = std::pow(8.0 - 2.0, 2) / 12.0;
    
    EXPECT_NEAR(dist->getMean(), expectedMean, 1e-10);
    EXPECT_NEAR(dist->getVariance(), expectedVar, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
}

void UniformDistributionTest::runNumericalStabilityTests() {
    // Test with very small interval
    UniformDistribution tinyDist(0.0, 1e-6);
    EXPECT_GT(tinyDist.getProbability(5e-7), 0.0);
    EXPECT_TRUE(std::isfinite(tinyDist.getProbability(5e-7)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        double logProb = dist->getLogProbability(val);
        EXPECT_TRUE(std::isfinite(logProb));
        EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
    }
}

void UniformDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void UniformDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameters
    dist->setA(10.0);
    dist->setB(20.0);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_DOUBLE_EQ(dist->getA(), 0.0);
    EXPECT_DOUBLE_EQ(dist->getB(), 1.0);
}

// Binomial Distribution - Standard Interface Tests
class BinomialDistributionTest : public StandardizedDistributionTest<BinomialDistribution> {
protected:
    std::unique_ptr<BinomialDistribution> createDefaultDistribution() override {
        return std::make_unique<BinomialDistribution>();
    }
    
    std::unique_ptr<BinomialDistribution> createParameterizedDistribution() override {
        return std::make_unique<BinomialDistribution>(15, 0.4);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0, 3, 7, 10, 15};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1, 16, 7.5, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {3, 4, 5, 6, 7, 3, 4, 5, 6, 7};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {3, 4, -1, 5};
    }
    
    double getExpectedMeanApprox() override { return 5.0; }
    double getExpectedVarianceApprox() override { return 3.5; }
};

// Implementation of standardized test methods for BinomialDistribution
void BinomialDistributionTest::runBasicFunctionalityTests() {
    // Test default constructor
    auto defaultDist = createDefaultDistribution();
    EXPECT_EQ(defaultDist->getN(), 10);
    EXPECT_DOUBLE_EQ(defaultDist->getP(), 0.5);
    
    // Test parameterized constructor
    auto paramDist = createParameterizedDistribution();
    EXPECT_EQ(paramDist->getN(), 15);
    EXPECT_DOUBLE_EQ(paramDist->getP(), 0.4);
}

void BinomialDistributionTest::runProbabilityTests() {
    auto dist = createParameterizedDistribution(); // n=15, p=0.4
    
    // Test valid values [0, n]
    for (double val : getValidTestValues()) {
        if (val >= 0 && val <= 15 && val == static_cast<int>(val)) {
            double prob = dist->getProbability(val);
            EXPECT_GT(prob, 0.0);
            EXPECT_TRUE(std::isfinite(prob));
        }
    }
    
    // Test out of range
    EXPECT_DOUBLE_EQ(dist->getProbability(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(dist->getProbability(16.0), 0.0);
}

void BinomialDistributionTest::runParameterValidationTests() {
    // Test invalid constructor parameters
    EXPECT_THROW(BinomialDistribution(0, 0.5), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(-1, 0.5), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, -0.1), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, 1.1), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

void BinomialDistributionTest::runFittingTests() {
    auto dist = createDefaultDistribution();
    auto data = getValidFittingData();
    
    EXPECT_NO_THROW(dist->fit(data));
    EXPECT_GT(dist->getN(), 0);
    EXPECT_GE(dist->getP(), 0.0);
    EXPECT_LE(dist->getP(), 1.0);
    
    // Test empty data
    std::vector<double> emptyData;
    EXPECT_NO_THROW(dist->fit(emptyData));
}

void BinomialDistributionTest::runStringRepresentationTests() {
    auto dist = createParameterizedDistribution();
    std::string str = dist->toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_TRUE(str.find("Binomial") != std::string::npos);
    EXPECT_TRUE(str.find("Distribution") != std::string::npos);
    EXPECT_TRUE(str.find("15") != std::string::npos);
    EXPECT_TRUE(str.find("0.4") != std::string::npos);
}

void BinomialDistributionTest::runStatisticalPropertiesTests() {
    auto dist = createParameterizedDistribution(); // n=15, p=0.4
    
    // For Binomial(n, p): mean = n*p, variance = n*p*(1-p)
    double expectedMean = 15 * 0.4;
    double expectedVar = 15 * 0.4 * 0.6;
    
    EXPECT_NEAR(dist->getMean(), expectedMean, 1e-10);
    EXPECT_NEAR(dist->getVariance(), expectedVar, 1e-10);
    EXPECT_NEAR(dist->getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
}

void BinomialDistributionTest::runNumericalStabilityTests() {
    // Test with extreme parameters
    BinomialDistribution extremeDist(100, 0.01);
    EXPECT_GT(extremeDist.getProbability(1.0), 0.0);
    EXPECT_TRUE(std::isfinite(extremeDist.getProbability(1.0)));
    
    // Test log probability
    auto dist = createDefaultDistribution();
    for (double val : getValidTestValues()) {
        if (val >= 0 && val <= 10 && val == static_cast<int>(val)) {
            double logProb = dist->getLogProbability(val);
            EXPECT_TRUE(std::isfinite(logProb));
            EXPECT_NEAR(std::exp(logProb), dist->getProbability(val), 1e-10);
        }
    }
}

void BinomialDistributionTest::runInvalidInputHandlingTests() {
    auto dist = createDefaultDistribution();
    
    for (double invalidVal : getInvalidTestValues()) {
        EXPECT_DOUBLE_EQ(dist->getProbability(invalidVal), 0.0);
    }
}

void BinomialDistributionTest::runResetFunctionalityTests() {
    auto dist = createParameterizedDistribution();
    
    // Modify parameters
    dist->setN(25);
    dist->setP(0.8);
    
    // Reset and verify
    EXPECT_NO_THROW(dist->reset());
    EXPECT_EQ(dist->getN(), 10);
    EXPECT_DOUBLE_EQ(dist->getP(), 0.5);
}

// Negative Binomial Distribution - Standard Interface Tests
class NegativeBinomialDistributionTest : public StandardizedDistributionTest<NegativeBinomialDistribution> {
protected:
    std::unique_ptr<NegativeBinomialDistribution> createDefaultDistribution() override {
        return std::make_unique<NegativeBinomialDistribution>();
    }
    
    std::unique_ptr<NegativeBinomialDistribution> createParameterizedDistribution() override {
        return std::make_unique<NegativeBinomialDistribution>(6.0, 0.3);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0, 2, 5, 10, 20};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1, 5.5, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {0, 1, 2, 3, 5, 8, 10, 15, 2, 4, 7, 12};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {0, 2, -1, 5};
    }
    
    double getExpectedMeanApprox() override { return 14.0; }
    double getExpectedVarianceApprox() override { return 46.7; }
};


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

// Student's t-Distribution - Standard Interface Tests
class StudentTDistributionTest : public StandardizedDistributionTest<StudentTDistribution> {
protected:
    std::unique_ptr<StudentTDistribution> createDefaultDistribution() override {
        return std::make_unique<StudentTDistribution>();
    }
    
    std::unique_ptr<StudentTDistribution> createParameterizedDistribution() override {
        return std::make_unique<StudentTDistribution>(5.0);
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
        return {-2.0, -1.0, 0.0, 1.0, 2.0};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    }
    
    double getExpectedMeanApprox() override { return 0.0; }
    double getExpectedVarianceApprox() override { return 1.67; }
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

TEST_F(StudentTDistributionTest, ProbabilityAndLogProbabilityCalculation) {
    // t-distribution is symmetric around 0
    double logProb0 = dist_->getLogProbability(0.0);
    double logProb1 = dist_->getLogProbability(1.0);
    double logProbNeg1 = dist_->getLogProbability(-1.0);

    EXPECT_GT(logProb0, -std::numeric_limits<double>::infinity());
    EXPECT_GT(logProb1, -std::numeric_limits<double>::infinity());
    EXPECT_GT(logProbNeg1, -std::numeric_limits<double>::infinity());

    double prob0 = std::exp(logProb0);
    double prob1 = std::exp(logProb1);
    double probNeg1 = std::exp(logProbNeg1);

    // Symmetry property
    EXPECT_NEAR(prob1, probNeg1, 1e-10);

    // Maximum at x=0
    EXPECT_GT(prob0, prob1);

    // Compare log and exp probabilities
    EXPECT_NEAR(prob0, dist_->getProbability(0.0), 1e-10);
    EXPECT_NEAR(prob1, dist_->getProbability(1.0), 1e-10);
    EXPECT_NEAR(probNeg1, dist_->getProbability(-1.0), 1e-10);
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

TEST_F(StudentTDistributionTest, CumulativeProbabilityCalculation) {
    double cdf0 = dist_->getCumulativeProbability(0.0);
    double cdf1 = dist_->getCumulativeProbability(1.0);
    double cdfMinus1 = dist_->getCumulativeProbability(-1.0);

    EXPECT_GT(cdf0, 0.0);
    EXPECT_LE(cdf0, 1.0);
    EXPECT_GT(cdf1, 0.0);
    EXPECT_LE(cdf1, 1.0);
    EXPECT_GT(cdfMinus1, 0.0);

    // Ensure symmetry
    EXPECT_NEAR(1.0 - cdf1, cdfMinus1, 1e-10);

    // CDF at mean should be 0.5
    EXPECT_NEAR(cdf0, 0.5, 1e-10);
}

// Chi-squared Distribution - Standard Interface Tests
class ChiSquaredDistributionTest : public StandardizedDistributionTest<ChiSquaredDistribution> {
protected:
    std::unique_ptr<ChiSquaredDistribution> createDefaultDistribution() override {
        return std::make_unique<ChiSquaredDistribution>();
    }
    
    std::unique_ptr<ChiSquaredDistribution> createParameterizedDistribution() override {
        return std::make_unique<ChiSquaredDistribution>(4.0);
    }
    
    std::vector<double> getValidTestValues() override {
        return {0.5, 1.0, 2.0, 4.0, 8.0};
    }
    
    std::vector<double> getInvalidTestValues() override {
        return {-1.0, std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::infinity()};
    }
    
    std::vector<double> getValidFittingData() override {
        return {1.0, 2.0, 3.0, 4.0, 5.0};
    }
    
    std::vector<double> getInvalidFittingData() override {
        return {1.0, -1.0, 3.0};
    }
    
    double getExpectedMeanApprox() override { return 1.0; }
    double getExpectedVarianceApprox() override { return 2.0; }
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
    
    // Should still produce valid probabilities (can be > 1 for continuous distributions)
    double prob = narrowGauss.getProbability(0.0);
    EXPECT_GE(prob, 0.0);
    EXPECT_TRUE(std::isfinite(prob));
    
    // Test log probability should be finite
    double logProb = narrowGauss.getLogProbability(0.0);
    EXPECT_TRUE(std::isfinite(logProb));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
