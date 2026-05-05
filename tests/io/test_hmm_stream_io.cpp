#include "libhmm/hmm.h"
#include "libhmm/distributions/distributions.h"
#include <gtest/gtest.h>
#include <sstream>
#include <memory>
#include <cmath>

using namespace libhmm;

class HmmStreamIOTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HmmStreamIOTest, StudentTDistributionRoundTrip) {
    // Create HMM with Student's t distribution
    Hmm hmm(2);

    // Set up initial probabilities
    Vector pi(2);
    pi(0) = 0.6;
    pi(1) = 0.4;
    hmm.setPi(pi);

    // Set up transition matrix
    Matrix trans(2, 2);
    trans(0, 0) = 0.8;
    trans(0, 1) = 0.2;
    trans(1, 0) = 0.3;
    trans(1, 1) = 0.7;
    hmm.setTrans(trans);

    // Set up distributions
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm.setDistribution(1, std::make_unique<StudentTDistribution>(3.0, 2.0, 1.5));

    // Write HMM to string
    std::ostringstream oss;
    oss << hmm;
    std::string hmm_string = oss.str();

    std::cout << "Original HMM with Student's t:\n" << hmm_string << std::endl;

    // Read HMM back from string
    std::istringstream iss(hmm_string);
    Hmm hmm_restored(1); // Will be overwritten by >>
    iss >> hmm_restored;

    // Verify the restored HMM matches the original
    EXPECT_EQ(hmm.getNumStatesModern(), hmm_restored.getNumStatesModern());

    // Check pi vector
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(hmm.getPi()(i), hmm_restored.getPi()(i), 1e-10);
    }

    // Check transition matrix
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(hmm.getTrans()(i, j), hmm_restored.getTrans()(i, j), 1e-10);
        }
    }

    // Check distributions
    const auto *orig_dist0 = dynamic_cast<const GaussianDistribution *>(&hmm.getDistribution(0));
    const auto *rest_dist0 =
        dynamic_cast<const GaussianDistribution *>(&hmm_restored.getDistribution(0));
    ASSERT_TRUE(orig_dist0 != nullptr);
    ASSERT_TRUE(rest_dist0 != nullptr);
    EXPECT_NEAR(orig_dist0->getMean(), rest_dist0->getMean(), 1e-10);
    EXPECT_NEAR(orig_dist0->getStandardDeviation(), rest_dist0->getStandardDeviation(), 1e-10);

    const auto *orig_dist1 = dynamic_cast<const StudentTDistribution *>(&hmm.getDistribution(1));
    const auto *rest_dist1 =
        dynamic_cast<const StudentTDistribution *>(&hmm_restored.getDistribution(1));
    ASSERT_TRUE(orig_dist1 != nullptr);
    ASSERT_TRUE(rest_dist1 != nullptr);
    EXPECT_NEAR(orig_dist1->getDegreesOfFreedom(), rest_dist1->getDegreesOfFreedom(), 1e-10);
    EXPECT_NEAR(orig_dist1->getLocation(), rest_dist1->getLocation(), 1e-10);
    EXPECT_NEAR(orig_dist1->getScale(), rest_dist1->getScale(), 1e-10);
}

TEST_F(HmmStreamIOTest, ChiSquaredDistributionRoundTrip) {
    // Create HMM with Chi-squared distribution
    Hmm hmm(2);

    // Set up initial probabilities
    Vector pi(2);
    pi(0) = 0.7;
    pi(1) = 0.3;
    hmm.setPi(pi);

    // Set up transition matrix
    Matrix trans(2, 2);
    trans(0, 0) = 0.9;
    trans(0, 1) = 0.1;
    trans(1, 0) = 0.4;
    trans(1, 1) = 0.6;
    hmm.setTrans(trans);

    // Set up distributions
    hmm.setDistribution(0, std::make_unique<ChiSquaredDistribution>(5.0));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(1.0, 2.0));

    // Write HMM to string
    std::ostringstream oss;
    oss << hmm;
    std::string hmm_string = oss.str();

    std::cout << "Original HMM with Chi-squared:\n" << hmm_string << std::endl;

    // Read HMM back from string
    std::istringstream iss(hmm_string);
    Hmm hmm_restored(1); // Will be overwritten by >>
    iss >> hmm_restored;

    // Verify the restored HMM matches the original
    EXPECT_EQ(hmm.getNumStatesModern(), hmm_restored.getNumStatesModern());

    // Check pi vector
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(hmm.getPi()(i), hmm_restored.getPi()(i), 1e-10);
    }

    // Check transition matrix
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(hmm.getTrans()(i, j), hmm_restored.getTrans()(i, j), 1e-10);
        }
    }

    // Check distributions
    const auto *orig_dist0 = dynamic_cast<const ChiSquaredDistribution *>(&hmm.getDistribution(0));
    const auto *rest_dist0 =
        dynamic_cast<const ChiSquaredDistribution *>(&hmm_restored.getDistribution(0));
    ASSERT_TRUE(orig_dist0 != nullptr);
    ASSERT_TRUE(rest_dist0 != nullptr);
    EXPECT_NEAR(orig_dist0->getDegreesOfFreedom(), rest_dist0->getDegreesOfFreedom(), 1e-10);

    const auto *orig_dist1 = dynamic_cast<const GaussianDistribution *>(&hmm.getDistribution(1));
    const auto *rest_dist1 =
        dynamic_cast<const GaussianDistribution *>(&hmm_restored.getDistribution(1));
    ASSERT_TRUE(orig_dist1 != nullptr);
    ASSERT_TRUE(rest_dist1 != nullptr);
    EXPECT_NEAR(orig_dist1->getMean(), rest_dist1->getMean(), 1e-10);
    EXPECT_NEAR(orig_dist1->getStandardDeviation(), rest_dist1->getStandardDeviation(), 1e-10);
}

TEST_F(HmmStreamIOTest, BothNewDistributionsRoundTrip) {
    // Create HMM with both new distributions
    Hmm hmm(2);

    // Set up initial probabilities
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);

    // Set up transition matrix
    Matrix trans(2, 2);
    trans(0, 0) = 0.7;
    trans(0, 1) = 0.3;
    trans(1, 0) = 0.2;
    trans(1, 1) = 0.8;
    hmm.setTrans(trans);

    // Set up distributions
    hmm.setDistribution(0, std::make_unique<StudentTDistribution>(4.0, 0.0, 2.0));
    hmm.setDistribution(1, std::make_unique<ChiSquaredDistribution>(3.0));

    // Write HMM to string
    std::ostringstream oss;
    oss << hmm;
    std::string hmm_string = oss.str();

    std::cout << "Original HMM with both new distributions:\n" << hmm_string << std::endl;

    // Read HMM back from string
    std::istringstream iss(hmm_string);
    Hmm hmm_restored(1); // Will be overwritten by >>
    iss >> hmm_restored;

    // Verify the restored HMM matches the original
    EXPECT_EQ(hmm.getNumStatesModern(), hmm_restored.getNumStatesModern());

    // Check pi vector
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(hmm.getPi()(i), hmm_restored.getPi()(i), 1e-10);
    }

    // Check transition matrix
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(hmm.getTrans()(i, j), hmm_restored.getTrans()(i, j), 1e-10);
        }
    }

    // Check distributions
    const auto *orig_dist0 = dynamic_cast<const StudentTDistribution *>(&hmm.getDistribution(0));
    const auto *rest_dist0 =
        dynamic_cast<const StudentTDistribution *>(&hmm_restored.getDistribution(0));
    ASSERT_TRUE(orig_dist0 != nullptr);
    ASSERT_TRUE(rest_dist0 != nullptr);
    EXPECT_NEAR(orig_dist0->getDegreesOfFreedom(), rest_dist0->getDegreesOfFreedom(), 1e-10);
    EXPECT_NEAR(orig_dist0->getLocation(), rest_dist0->getLocation(), 1e-10);
    EXPECT_NEAR(orig_dist0->getScale(), rest_dist0->getScale(), 1e-10);

    const auto *orig_dist1 = dynamic_cast<const ChiSquaredDistribution *>(&hmm.getDistribution(1));
    const auto *rest_dist1 =
        dynamic_cast<const ChiSquaredDistribution *>(&hmm_restored.getDistribution(1));
    ASSERT_TRUE(orig_dist1 != nullptr);
    ASSERT_TRUE(rest_dist1 != nullptr);
    EXPECT_NEAR(orig_dist1->getDegreesOfFreedom(), rest_dist1->getDegreesOfFreedom(), 1e-10);
}

TEST_F(HmmStreamIOTest, UnknownDistributionTypeThrows) {
    // Create a malformed HMM string with unknown distribution type
    std::string malformed_hmm = R"(Hidden Markov Model parameters
  States: 1
  Pi: [ 1.0 ]
  Transmission matrix:
   [ 1.0 ]
  Emissions:
   State 0: UnknownDistribution
)";

    std::istringstream iss(malformed_hmm);
    Hmm hmm(1);

    // Should throw when encountering unknown distribution type
    EXPECT_THROW(iss >> hmm, std::runtime_error);
}

TEST_F(HmmStreamIOTest, StudentTDistributionParameterParsing) {
    // Test parsing of Student's t distribution with specific parameter values
    std::string hmm_string = R"(Hidden Markov Model parameters
  States: 1
  Pi: [ 1.0 ]
  Transmission matrix:
   [ 1.0 ]
  Emissions:
   State 0: StudentT Distribution:
  nu (degrees of freedom) = 5.5
  mu (location) = -1.2
  sigma (scale) = 3.7
)";

    std::istringstream iss(hmm_string);
    Hmm hmm(1);
    iss >> hmm;

    const auto *dist = dynamic_cast<const StudentTDistribution *>(&hmm.getDistribution(0));
    ASSERT_TRUE(dist != nullptr);
    EXPECT_NEAR(dist->getDegreesOfFreedom(), 5.5, 1e-10);
    EXPECT_NEAR(dist->getLocation(), -1.2, 1e-10);
    EXPECT_NEAR(dist->getScale(), 3.7, 1e-10);
}

TEST_F(HmmStreamIOTest, ChiSquaredDistributionParameterParsing) {
    // Test parsing of Chi-squared distribution with specific parameter values
    std::string hmm_string = R"(Hidden Markov Model parameters
  States: 1
  Pi: [ 1.0 ]
  Transmission matrix:
   [ 1.0 ]
  Emissions:
   State 0: ChiSquared Distribution:
  k (degrees of freedom) = 7.25
)";

    std::istringstream iss(hmm_string);
    Hmm hmm(1);
    iss >> hmm;

    const auto *dist = dynamic_cast<const ChiSquaredDistribution *>(&hmm.getDistribution(0));
    ASSERT_TRUE(dist != nullptr);
    EXPECT_NEAR(dist->getDegreesOfFreedom(), 7.25, 1e-10);
}

// =============================================================================
// All-distributions stream round-trip
//
// Verifies that every distribution type that operator<< can write can be
// correctly parsed back by operator>>. Uses EXPECT_NEAR with 1e-5 tolerance
// because toString() formats values to 6 decimal places, not max_digits10.
//
// NegativeBinomial is excluded: toString() begins with "Negative Binomial"
// (two words), so the single-token dispatch key is "Negative" which is absent
// from kStreamParsers. See NegativeBinomialStreamLimitation below.
// =============================================================================

TEST_F(HmmStreamIOTest, AllDistributionsStreamRoundTrip) {
    constexpr std::size_t N = 14; // all distributions except NegativeBinomial
    Matrix trans(N, N);
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i) {
        pi[i] = 1.0 / static_cast<double>(N);
        for (std::size_t j = 0; j < N; ++j)
            trans(i, j) = (i == j) ? 0.9 : (0.1 / 13.0);
    }

    std::vector<std::unique_ptr<EmissionDistribution>> emis(N);
    emis[0] = std::make_unique<GaussianDistribution>(1.5, 2.5);
    emis[1] = std::make_unique<ExponentialDistribution>(3.0);
    emis[2] = std::make_unique<GammaDistribution>(2.0, 1.5);
    emis[3] = std::make_unique<BetaDistribution>(2.0, 3.0);
    emis[4] = std::make_unique<WeibullDistribution>(1.5, 2.5);
    emis[5] = std::make_unique<LogNormalDistribution>(0.5, 1.2);
    emis[6] = std::make_unique<ParetoDistribution>(2.0, 0.5);
    emis[7] = std::make_unique<ChiSquaredDistribution>(4.0);
    emis[8] = std::make_unique<StudentTDistribution>(3.0, 0.5, 1.5);
    emis[9] = std::make_unique<PoissonDistribution>(2.5);
    emis[10] = std::make_unique<BinomialDistribution>(10, 0.3);
    {
        auto disc = std::make_unique<DiscreteDistribution>(4);
        disc->setProbability(0.0, 0.1);
        disc->setProbability(1.0, 0.2);
        disc->setProbability(2.0, 0.3);
        disc->setProbability(3.0, 0.4);
        emis[11] = std::move(disc);
    }
    emis[12] = std::make_unique<UniformDistribution>(1.0, 3.0);
    emis[13] = std::make_unique<RayleighDistribution>(2.0);

    Hmm original(std::move(trans), std::move(emis), std::move(pi));

    std::ostringstream oss;
    oss << original;
    const std::string s = oss.str();
    ASSERT_FALSE(s.empty());

    std::istringstream iss(s);
    Hmm restored(1);
    ASSERT_NO_THROW(iss >> restored);

    ASSERT_EQ(restored.getNumStatesModern(), N);

    // pi and trans: 6-decimal-place format means ~1e-6 precision.
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_NEAR(original.getPi()[i], restored.getPi()[i], 1e-5) << "pi[" << i << "]";
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(original.getTrans()(i, j), restored.getTrans()(i, j), 1e-5)
                << "trans(" << i << ',' << j << ')';

    auto get = [&](std::size_t i) {
        return &restored.getDistribution(i);
    };

    {
        auto *d = dynamic_cast<const GaussianDistribution *>(get(0));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getMean(), 1.5, 1e-5);
        EXPECT_NEAR(d->getStandardDeviation(), 2.5, 1e-5);
    }

    {
        auto *d = dynamic_cast<const ExponentialDistribution *>(get(1));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getLambda(), 3.0, 1e-5);
    }

    {
        auto *d = dynamic_cast<const GammaDistribution *>(get(2));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getK(), 2.0, 1e-5);
        EXPECT_NEAR(d->getTheta(), 1.5, 1e-5);
    }

    {
        auto *d = dynamic_cast<const BetaDistribution *>(get(3));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getAlpha(), 2.0, 1e-5);
        EXPECT_NEAR(d->getBeta(), 3.0, 1e-5);
    }

    {
        auto *d = dynamic_cast<const WeibullDistribution *>(get(4));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getK(), 1.5, 1e-5);
        EXPECT_NEAR(d->getLambda(), 2.5, 1e-5);
    }

    {
        auto *d = dynamic_cast<const LogNormalDistribution *>(get(5));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getMean(), 0.5, 1e-5);
        EXPECT_NEAR(d->getStandardDeviation(), 1.2, 1e-5);
    }

    {
        auto *d = dynamic_cast<const ParetoDistribution *>(get(6));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getK(), 2.0, 1e-5);
        EXPECT_NEAR(d->getXm(), 0.5, 1e-5);
    }

    {
        auto *d = dynamic_cast<const ChiSquaredDistribution *>(get(7));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getDegreesOfFreedom(), 4.0, 1e-5);
    }

    {
        auto *d = dynamic_cast<const StudentTDistribution *>(get(8));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getDegreesOfFreedom(), 3.0, 1e-5);
        EXPECT_NEAR(d->getLocation(), 0.5, 1e-5);
        EXPECT_NEAR(d->getScale(), 1.5, 1e-5);
    }

    {
        auto *d = dynamic_cast<const PoissonDistribution *>(get(9));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getLambda(), 2.5, 1e-5);
    }

    {
        auto *d = dynamic_cast<const BinomialDistribution *>(get(10));
        ASSERT_NE(d, nullptr);
        EXPECT_EQ(d->getN(), 10);
        EXPECT_NEAR(d->getP(), 0.3, 1e-5);
    }

    {
        auto *d = dynamic_cast<const DiscreteDistribution *>(get(11));
        ASSERT_NE(d, nullptr);
        EXPECT_EQ(d->getNumSymbols(), 4u);
        EXPECT_NEAR(d->getSymbolProbability(0), 0.1, 1e-5);
        EXPECT_NEAR(d->getSymbolProbability(1), 0.2, 1e-5);
        EXPECT_NEAR(d->getSymbolProbability(2), 0.3, 1e-5);
        EXPECT_NEAR(d->getSymbolProbability(3), 0.4, 1e-5);
    }

    {
        auto *d = dynamic_cast<const UniformDistribution *>(get(12));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getA(), 1.0, 1e-5);
        EXPECT_NEAR(d->getB(), 3.0, 1e-5);
    }

    {
        auto *d = dynamic_cast<const RayleighDistribution *>(get(13));
        ASSERT_NE(d, nullptr);
        EXPECT_NEAR(d->getSigma(), 2.0, 1e-5);
    }
}

// NegativeBinomial cannot round-trip through operator>>(Hmm) because
// toString() starts with "Negative Binomial" (two tokens), so the dispatch
// key read by operator>> is "Negative" — absent from kStreamParsers.
// Use JSON I/O (hmm_json.h) for HMMs with NegativeBinomial distributions.
TEST_F(HmmStreamIOTest, NegativeBinomialStreamLimitation) {
    Hmm hmm(1);
    Vector pi(1);
    pi[0] = 1.0;
    hmm.setPi(pi);
    Matrix trans(1, 1);
    trans(0, 0) = 1.0;
    hmm.setTrans(trans);
    hmm.setDistribution(0, std::make_unique<NegativeBinomialDistribution>(5.0, 0.4));

    std::ostringstream oss;
    oss << hmm;

    std::istringstream iss(oss.str());
    Hmm restored(1);
    // "Negative" is not a recognised dispatch key — must throw.
    EXPECT_THROW(iss >> restored, std::runtime_error);
}

TEST_F(HmmStreamIOTest, MultipleDistributionTypesInSameHMM) {
    // Create HMM with multiple different distribution types including the new ones
    Hmm hmm(3);

    // Set up initial probabilities
    Vector pi(3);
    pi(0) = 0.3;
    pi(1) = 0.4;
    pi(2) = 0.3;
    hmm.setPi(pi);

    // Set up transition matrix
    Matrix trans(3, 3);
    trans(0, 0) = 0.7;
    trans(0, 1) = 0.2;
    trans(0, 2) = 0.1;
    trans(1, 0) = 0.1;
    trans(1, 1) = 0.8;
    trans(1, 2) = 0.1;
    trans(2, 0) = 0.2;
    trans(2, 1) = 0.3;
    trans(2, 2) = 0.5;
    hmm.setTrans(trans);

    // Set up distributions with different types
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(1.5, 0.8));
    hmm.setDistribution(1, std::make_unique<StudentTDistribution>(2.5, 0.5, 1.2));
    hmm.setDistribution(2, std::make_unique<ChiSquaredDistribution>(4.0));

    // Write HMM to string
    std::ostringstream oss;
    oss << hmm;
    std::string hmm_string = oss.str();

    std::cout << "Original HMM with multiple distribution types:\n" << hmm_string << std::endl;

    // Read HMM back from string
    std::istringstream iss(hmm_string);
    Hmm hmm_restored(1); // Will be overwritten by >>
    iss >> hmm_restored;

    // Verify the restored HMM matches the original
    EXPECT_EQ(hmm.getNumStatesModern(), hmm_restored.getNumStatesModern());

    // Check pi vector
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(hmm.getPi()(i), hmm_restored.getPi()(i), 1e-10);
    }

    // Check transition matrix
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(hmm.getTrans()(i, j), hmm_restored.getTrans()(i, j), 1e-10);
        }
    }

    // Check distributions
    const auto *orig_gauss = dynamic_cast<const GaussianDistribution *>(&hmm.getDistribution(0));
    const auto *rest_gauss =
        dynamic_cast<const GaussianDistribution *>(&hmm_restored.getDistribution(0));
    ASSERT_TRUE(orig_gauss != nullptr);
    ASSERT_TRUE(rest_gauss != nullptr);
    EXPECT_NEAR(orig_gauss->getMean(), rest_gauss->getMean(), 1e-10);
    EXPECT_NEAR(orig_gauss->getStandardDeviation(), rest_gauss->getStandardDeviation(), 1e-10);

    const auto *orig_student = dynamic_cast<const StudentTDistribution *>(&hmm.getDistribution(1));
    const auto *rest_student =
        dynamic_cast<const StudentTDistribution *>(&hmm_restored.getDistribution(1));
    ASSERT_TRUE(orig_student != nullptr);
    ASSERT_TRUE(rest_student != nullptr);
    EXPECT_NEAR(orig_student->getDegreesOfFreedom(), rest_student->getDegreesOfFreedom(), 1e-10);
    EXPECT_NEAR(orig_student->getLocation(), rest_student->getLocation(), 1e-10);
    EXPECT_NEAR(orig_student->getScale(), rest_student->getScale(), 1e-10);

    const auto *orig_chi = dynamic_cast<const ChiSquaredDistribution *>(&hmm.getDistribution(2));
    const auto *rest_chi =
        dynamic_cast<const ChiSquaredDistribution *>(&hmm_restored.getDistribution(2));
    ASSERT_TRUE(orig_chi != nullptr);
    ASSERT_TRUE(rest_chi != nullptr);
    EXPECT_NEAR(orig_chi->getDegreesOfFreedom(), rest_chi->getDegreesOfFreedom(), 1e-10);
}
