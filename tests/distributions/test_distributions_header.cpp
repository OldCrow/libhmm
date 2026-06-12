#include <gtest/gtest.h>
#include "libhmm/distributions/distributions.h"

using namespace libhmm;

class DistributionsHeaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup if needed
    }
};

// Test that all discrete distributions are available via the convenience header
TEST_F(DistributionsHeaderTest, DiscreteDistributionsAvailable) {
    // Should be able to create all discrete distributions
    EXPECT_NO_THROW({
        DiscreteDistribution discrete(5);
        BinomialDistribution binomial(10, 0.3);
        NegativeBinomialDistribution negbinom(5, 0.4);
        PoissonDistribution poisson(2.5);

        // Basic functionality test
        EXPECT_GT(discrete.getProbability(2), 0.0);
        EXPECT_GT(binomial.getProbability(3), 0.0);
        EXPECT_GT(negbinom.getProbability(2), 0.0);
        EXPECT_GT(poisson.getProbability(2), 0.0);
    });
}

// Test that all continuous distributions are available via the convenience header
TEST_F(DistributionsHeaderTest, ContinuousDistributionsAvailable) {
    // Should be able to create all continuous distributions
    EXPECT_NO_THROW({
        GaussianDistribution gaussian(0.0, 1.0);
        ExponentialDistribution exponential(1.5);
        GammaDistribution gamma(2.0, 1.5);
        LogNormalDistribution lognormal(0.0, 1.0);
        ParetoDistribution pareto(2.0, 1.0); // scale=2.0, shape=1.0
        BetaDistribution beta(2.0, 3.0);
        UniformDistribution uniform(0.0, 10.0);
        WeibullDistribution weibull(2.0, 1.0);
        StudentTDistribution studentt(5.0);
        ChiSquaredDistribution chisquared(3.0);
        RayleighDistribution rayleigh(1.0);
        VonMisesDistribution vonmises(0.0, 1.0);

        // Basic functionality test (using values in valid domains)
        EXPECT_GT(gaussian.getProbability(0.0), 0.0);
        EXPECT_GT(exponential.getProbability(1.0), 0.0);
        EXPECT_GT(gamma.getProbability(1.0), 0.0);
        EXPECT_GT(lognormal.getProbability(1.0), 0.0);
        EXPECT_GT(pareto.getProbability(3.0), 0.0); // Must be > scale parameter
        EXPECT_GT(beta.getProbability(0.5), 0.0);
        EXPECT_GT(uniform.getProbability(5.0), 0.0);
        EXPECT_GT(weibull.getProbability(1.0), 0.0);
        EXPECT_GT(studentt.getProbability(0.0), 0.0);
        EXPECT_GT(chisquared.getProbability(2.0), 0.0);
        EXPECT_GT(rayleigh.getProbability(1.0), 0.0);
        EXPECT_GT(vonmises.getProbability(0.0), 0.0);
    });
}

// Test that the base distribution interface is available
TEST_F(DistributionsHeaderTest, BaseDistributionInterfaceAvailable) {
    // Should be able to use polymorphic interface
    std::unique_ptr<EmissionDistribution> dist = std::make_unique<GaussianDistribution>(0.0, 1.0);

    double prob = dist->getProbability(0.0);
    EXPECT_GT(prob, 0.0);

    std::vector<double> data = {0.0, 1.0, -1.0};
    EXPECT_NO_THROW(dist->fit(data));
}

// Test the compile-time constants defined in the header
TEST_F(DistributionsHeaderTest, CompileTimeConstants) {
    using namespace libhmm::detail;

    EXPECT_EQ(DISCRETE_DISTRIBUTION_COUNT, 4u);
    EXPECT_EQ(CONTINUOUS_DISTRIBUTION_COUNT, 12u);
    EXPECT_EQ(MULTIVARIATE_DISTRIBUTION_COUNT, 3u);
    EXPECT_EQ(DISTRIBUTION_COUNT, 19u);
    EXPECT_EQ(DISCRETE_DISTRIBUTION_COUNT + CONTINUOUS_DISTRIBUTION_COUNT +
                  MULTIVARIATE_DISTRIBUTION_COUNT,
              DISTRIBUTION_COUNT);
}

// Test that multivariate distributions are available via the convenience header
TEST_F(DistributionsHeaderTest, MultivariateDistributionsAvailable) {
    // IndependentComponents
    {
        libhmm::IndependentComponentsDistribution d(2);
        EXPECT_EQ(d.getDimension(), 2u);
        EXPECT_FALSE(d.isDiscrete());
    }
    // DiagonalGaussian
    {
        libhmm::DiagonalGaussianDistribution d(2);
        EXPECT_EQ(d.getDimension(), 2u);
        EXPECT_EQ(d.getNumParameters(), 4u); // 2*D
    }
    // FullCovarianceGaussian
    {
        libhmm::FullCovarianceGaussianDistribution d(2);
        EXPECT_EQ(d.getDimension(), 2u);
        EXPECT_EQ(d.getNumParameters(), 5u); // D + D*(D+1)/2 = 2+3
    }
}

// Test that we can use all distributions in template contexts
template <typename DistType>
bool test_distribution_interface(DistType &dist, double test_value) {
    try {
        double prob = dist.getProbability(test_value);
        return prob >= 0.0 && prob <= 1.0;
    } catch (...) {
        return false;
    }
}

TEST_F(DistributionsHeaderTest, TemplateCompatibility) {
    // Test that all distributions work with generic template functions
    DiscreteDistribution discrete(5);
    GaussianDistribution gaussian(0.0, 1.0);
    PoissonDistribution poisson(2.5);
    ExponentialDistribution exponential(1.0);

    EXPECT_TRUE(test_distribution_interface(discrete, 2));
    EXPECT_TRUE(test_distribution_interface(gaussian, 0.0));
    EXPECT_TRUE(test_distribution_interface(poisson, 2));
    EXPECT_TRUE(test_distribution_interface(exponential, 1.0));
}

// Test that including the header doesn't cause compilation issues
TEST_F(DistributionsHeaderTest, NoCompilationIssues) {
    // If we can compile and run this test, the header works correctly
    SUCCEED();
}

// Verify that all expected namespaces and classes are accessible
TEST_F(DistributionsHeaderTest, NamespaceAccessibility) {
    // All classes should be accessible in the libhmm namespace
    EXPECT_NO_THROW({
        libhmm::GaussianDistribution gauss(0.0, 1.0);
        libhmm::DiscreteDistribution disc(5);
        libhmm::PoissonDistribution pois(2.0);
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
