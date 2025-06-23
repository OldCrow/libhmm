#include <gtest/gtest.h>
#include "libhmm/common/distribution_traits.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/poisson_distribution.h"
#include "libhmm/distributions/gamma_distribution.h"
#include "libhmm/distributions/binomial_distribution.h"

using namespace libhmm;

class DistributionTraitsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup if needed
    }
};

// Test discrete distribution trait detection
TEST_F(DistributionTraitsTest, DiscreteDistributionDetection) {
    // Discrete distributions should be detected
    EXPECT_TRUE(is_discrete_distribution_v<DiscreteDistribution>);
    EXPECT_TRUE(is_discrete_distribution_v<PoissonDistribution>);
    EXPECT_TRUE(is_discrete_distribution_v<BinomialDistribution>);
    EXPECT_TRUE(is_discrete_distribution_v<NegativeBinomialDistribution>);
    
    // Continuous distributions should not be detected as discrete
    EXPECT_FALSE(is_discrete_distribution_v<GaussianDistribution>);
    EXPECT_FALSE(is_discrete_distribution_v<GammaDistribution>);
    EXPECT_FALSE(is_discrete_distribution_v<ExponentialDistribution>);
    
    // Non-distribution types should not be detected
    EXPECT_FALSE(is_discrete_distribution_v<int>);
    EXPECT_FALSE(is_discrete_distribution_v<std::string>);
}

// Test continuous distribution trait detection
TEST_F(DistributionTraitsTest, ContinuousDistributionDetection) {
    // Continuous distributions should be detected
    EXPECT_TRUE(is_continuous_distribution_v<GaussianDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<GammaDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<ExponentialDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<LogNormalDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<ParetoDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<BetaDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<WeibullDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<UniformDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<StudentTDistribution>);
    EXPECT_TRUE(is_continuous_distribution_v<ChiSquaredDistribution>);
    
    // Discrete distributions should not be detected as continuous
    EXPECT_FALSE(is_continuous_distribution_v<DiscreteDistribution>);
    EXPECT_FALSE(is_continuous_distribution_v<PoissonDistribution>);
    EXPECT_FALSE(is_continuous_distribution_v<BinomialDistribution>);
    
    // Non-distribution types should not be detected
    EXPECT_FALSE(is_continuous_distribution_v<int>);
    EXPECT_FALSE(is_continuous_distribution_v<std::string>);
}

// Test fitting support detection
TEST_F(DistributionTraitsTest, FittingSupportDetection) {
    // Most distributions should support fitting
    EXPECT_TRUE(supports_fitting_v<DiscreteDistribution>);
    EXPECT_TRUE(supports_fitting_v<GaussianDistribution>);
    EXPECT_TRUE(supports_fitting_v<PoissonDistribution>);
    EXPECT_TRUE(supports_fitting_v<GammaDistribution>);
    
    // Non-distribution types should not support fitting
    EXPECT_FALSE(supports_fitting_v<int>);
    EXPECT_FALSE(supports_fitting_v<std::string>);
}

// Test parameter estimation detection
TEST_F(DistributionTraitsTest, ParameterEstimationDetection) {
    // Distributions with getMean() and getVariance() should be detected
    EXPECT_TRUE(has_parameter_estimation_v<GaussianDistribution>);
    EXPECT_TRUE(has_parameter_estimation_v<PoissonDistribution>);
    
    // Non-distribution types should not have parameter estimation
    EXPECT_FALSE(has_parameter_estimation_v<int>);
    EXPECT_FALSE(has_parameter_estimation_v<std::string>);
}

// Test Baum-Welch compatibility
TEST_F(DistributionTraitsTest, BaumWelchCompatibility) {
    // Discrete distributions with fitting should be Baum-Welch compatible
    EXPECT_TRUE(is_baum_welch_compatible_v<DiscreteDistribution>);
    EXPECT_TRUE(is_baum_welch_compatible_v<PoissonDistribution>);
    EXPECT_TRUE(is_baum_welch_compatible_v<BinomialDistribution>);
    
    // Continuous distributions should not be Baum-Welch compatible
    EXPECT_FALSE(is_baum_welch_compatible_v<GaussianDistribution>);
    EXPECT_FALSE(is_baum_welch_compatible_v<GammaDistribution>);
    
    // Non-distribution types should not be compatible
    EXPECT_FALSE(is_baum_welch_compatible_v<int>);
    EXPECT_FALSE(is_baum_welch_compatible_v<std::string>);
}

// Test Viterbi compatibility
TEST_F(DistributionTraitsTest, ViterbiCompatibility) {
    // Both discrete and continuous distributions should be Viterbi compatible
    EXPECT_TRUE(is_viterbi_compatible_v<DiscreteDistribution>);
    EXPECT_TRUE(is_viterbi_compatible_v<GaussianDistribution>);
    EXPECT_TRUE(is_viterbi_compatible_v<PoissonDistribution>);
    EXPECT_TRUE(is_viterbi_compatible_v<GammaDistribution>);
    
    // Non-distribution types should not be compatible
    EXPECT_FALSE(is_viterbi_compatible_v<int>);
    EXPECT_FALSE(is_viterbi_compatible_v<std::string>);
}

// Test distribution category function
TEST_F(DistributionTraitsTest, DistributionCategoryFunction) {
    // Test discrete distributions
    EXPECT_EQ(get_distribution_category<DiscreteDistribution>(), DistributionCategory::Discrete);
    EXPECT_EQ(get_distribution_category<PoissonDistribution>(), DistributionCategory::Discrete);
    
    // Test continuous distributions
    EXPECT_EQ(get_distribution_category<GaussianDistribution>(), DistributionCategory::Continuous);
    EXPECT_EQ(get_distribution_category<GammaDistribution>(), DistributionCategory::Continuous);
    
    // Test unknown types
    EXPECT_EQ(get_distribution_category<int>(), DistributionCategory::Unknown);
    EXPECT_EQ(get_distribution_category<std::string>(), DistributionCategory::Unknown);
}

// Test trainer selector
TEST_F(DistributionTraitsTest, TrainerSelector) {
    // Discrete distributions should select Baum-Welch (0)
    EXPECT_EQ(trainer_selector_v<DiscreteDistribution>, 0);
    EXPECT_EQ(trainer_selector_v<PoissonDistribution>, 0);
    
    // Continuous distributions should select Viterbi (1)
    EXPECT_EQ(trainer_selector_v<GaussianDistribution>, 1);
    EXPECT_EQ(trainer_selector_v<GammaDistribution>, 1);
}

// Test SFINAE helpers by creating template functions that only work with specific types
template<typename T, typename = enable_if_discrete_t<T>>
bool test_discrete_only(const T&) {
    return true;
}

template<typename T, typename = enable_if_continuous_t<T>>
bool test_continuous_only(const T&) {
    return true;
}

template<typename T, typename = enable_if_baum_welch_compatible_t<T>>
bool test_baum_welch_only(const T&) {
    return true;
}

template<typename T, typename = enable_if_viterbi_compatible_t<T>>
bool test_viterbi_only(const T&) {
    return true;
}

TEST_F(DistributionTraitsTest, SFINAEHelpers) {
    DiscreteDistribution discrete_dist(5);
    GaussianDistribution gaussian_dist(0.0, 1.0);
    
    // Test that SFINAE helpers work correctly
    EXPECT_TRUE(test_discrete_only(discrete_dist));
    EXPECT_TRUE(test_continuous_only(gaussian_dist));
    EXPECT_TRUE(test_baum_welch_only(discrete_dist));
    EXPECT_TRUE(test_viterbi_only(discrete_dist));
    EXPECT_TRUE(test_viterbi_only(gaussian_dist));
    
    // Note: The following should not compile if uncommented:
    // test_discrete_only(gaussian_dist);      // Should fail - Gaussian is not discrete
    // test_continuous_only(discrete_dist);    // Should fail - Discrete is not continuous
    // test_baum_welch_only(gaussian_dist);    // Should fail - Gaussian not Baum-Welch compatible
}

// Test compile-time validation
TEST_F(DistributionTraitsTest, CompileTimeValidation) {
    // These should compile without issues for valid distributions
    LIBHMM_VALIDATE_DISTRIBUTION(DiscreteDistribution);
    LIBHMM_VALIDATE_DISTRIBUTION(GaussianDistribution);
    LIBHMM_VALIDATE_DISTRIBUTION(PoissonDistribution);
    
    // Note: The following should not compile if uncommented:
    // LIBHMM_VALIDATE_DISTRIBUTION(int);        // Should fail - not a distribution
    // LIBHMM_VALIDATE_DISTRIBUTION(std::string); // Should fail - not a distribution
}

// Demonstrate practical usage with a template function
template<typename DistType>
auto create_appropriate_trainer_type() {
    if constexpr (is_baum_welch_compatible_v<DistType>) {
        return std::string("BaumWelchTrainer");
    } else if constexpr (is_viterbi_compatible_v<DistType>) {
        return std::string("ViterbiTrainer");
    } else {
        return std::string("NoSuitableTrainer");
    }
}

TEST_F(DistributionTraitsTest, PracticalUsageExample) {
    // Test that the right trainer type is selected
    EXPECT_EQ(create_appropriate_trainer_type<DiscreteDistribution>(), "BaumWelchTrainer");
    EXPECT_EQ(create_appropriate_trainer_type<GaussianDistribution>(), "ViterbiTrainer");
    EXPECT_EQ(create_appropriate_trainer_type<PoissonDistribution>(), "BaumWelchTrainer");
    EXPECT_EQ(create_appropriate_trainer_type<int>(), "NoSuitableTrainer");
}

// Test that all libhmm distributions are properly categorized
TEST_F(DistributionTraitsTest, AllDistributionsCategorized) {
    // All distributions should be either discrete or continuous, never unknown
    EXPECT_NE(get_distribution_category<DiscreteDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<GaussianDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<PoissonDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<GammaDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<ExponentialDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<LogNormalDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<ParetoDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<BetaDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<WeibullDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<UniformDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<BinomialDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<NegativeBinomialDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<StudentTDistribution>(), DistributionCategory::Unknown);
    EXPECT_NE(get_distribution_category<ChiSquaredDistribution>(), DistributionCategory::Unknown);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
