#include "libhmm/hmm.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/student_t_distribution.h"
#include "libhmm/distributions/chi_squared_distribution.h"
#include <gtest/gtest.h>
#include <sstream>
#include <memory>

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
    trans(0, 0) = 0.8; trans(0, 1) = 0.2;
    trans(1, 0) = 0.3; trans(1, 1) = 0.7;
    hmm.setTrans(trans);
    
    // Set up distributions
    hmm.setProbabilityDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm.setProbabilityDistribution(1, std::make_unique<StudentTDistribution>(3.0, 2.0, 1.5));
    
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
    const auto* orig_dist0 = dynamic_cast<const GaussianDistribution*>(hmm.getProbabilityDistribution(0));
    const auto* rest_dist0 = dynamic_cast<const GaussianDistribution*>(hmm_restored.getProbabilityDistribution(0));
    ASSERT_TRUE(orig_dist0 != nullptr);
    ASSERT_TRUE(rest_dist0 != nullptr);
    EXPECT_NEAR(orig_dist0->getMean(), rest_dist0->getMean(), 1e-10);
    EXPECT_NEAR(orig_dist0->getStandardDeviation(), rest_dist0->getStandardDeviation(), 1e-10);
    
    const auto* orig_dist1 = dynamic_cast<const StudentTDistribution*>(hmm.getProbabilityDistribution(1));
    const auto* rest_dist1 = dynamic_cast<const StudentTDistribution*>(hmm_restored.getProbabilityDistribution(1));
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
    trans(0, 0) = 0.9; trans(0, 1) = 0.1;
    trans(1, 0) = 0.4; trans(1, 1) = 0.6;
    hmm.setTrans(trans);
    
    // Set up distributions
    hmm.setProbabilityDistribution(0, std::make_unique<ChiSquaredDistribution>(5.0));
    hmm.setProbabilityDistribution(1, std::make_unique<GaussianDistribution>(1.0, 2.0));
    
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
    const auto* orig_dist0 = dynamic_cast<const ChiSquaredDistribution*>(hmm.getProbabilityDistribution(0));
    const auto* rest_dist0 = dynamic_cast<const ChiSquaredDistribution*>(hmm_restored.getProbabilityDistribution(0));
    ASSERT_TRUE(orig_dist0 != nullptr);
    ASSERT_TRUE(rest_dist0 != nullptr);
    EXPECT_NEAR(orig_dist0->getDegreesOfFreedom(), rest_dist0->getDegreesOfFreedom(), 1e-10);
    
    const auto* orig_dist1 = dynamic_cast<const GaussianDistribution*>(hmm.getProbabilityDistribution(1));
    const auto* rest_dist1 = dynamic_cast<const GaussianDistribution*>(hmm_restored.getProbabilityDistribution(1));
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
    trans(0, 0) = 0.7; trans(0, 1) = 0.3;
    trans(1, 0) = 0.2; trans(1, 1) = 0.8;
    hmm.setTrans(trans);
    
    // Set up distributions
    hmm.setProbabilityDistribution(0, std::make_unique<StudentTDistribution>(4.0, 0.0, 2.0));
    hmm.setProbabilityDistribution(1, std::make_unique<ChiSquaredDistribution>(3.0));
    
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
    const auto* orig_dist0 = dynamic_cast<const StudentTDistribution*>(hmm.getProbabilityDistribution(0));
    const auto* rest_dist0 = dynamic_cast<const StudentTDistribution*>(hmm_restored.getProbabilityDistribution(0));
    ASSERT_TRUE(orig_dist0 != nullptr);
    ASSERT_TRUE(rest_dist0 != nullptr);
    EXPECT_NEAR(orig_dist0->getDegreesOfFreedom(), rest_dist0->getDegreesOfFreedom(), 1e-10);
    EXPECT_NEAR(orig_dist0->getLocation(), rest_dist0->getLocation(), 1e-10);
    EXPECT_NEAR(orig_dist0->getScale(), rest_dist0->getScale(), 1e-10);
    
    const auto* orig_dist1 = dynamic_cast<const ChiSquaredDistribution*>(hmm.getProbabilityDistribution(1));
    const auto* rest_dist1 = dynamic_cast<const ChiSquaredDistribution*>(hmm_restored.getProbabilityDistribution(1));
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
    
    const auto* dist = dynamic_cast<const StudentTDistribution*>(hmm.getProbabilityDistribution(0));
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
    
    const auto* dist = dynamic_cast<const ChiSquaredDistribution*>(hmm.getProbabilityDistribution(0));
    ASSERT_TRUE(dist != nullptr);
    EXPECT_NEAR(dist->getDegreesOfFreedom(), 7.25, 1e-10);
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
    trans(0, 0) = 0.7; trans(0, 1) = 0.2; trans(0, 2) = 0.1;
    trans(1, 0) = 0.1; trans(1, 1) = 0.8; trans(1, 2) = 0.1;
    trans(2, 0) = 0.2; trans(2, 1) = 0.3; trans(2, 2) = 0.5;
    hmm.setTrans(trans);
    
    // Set up distributions with different types
    hmm.setProbabilityDistribution(0, std::make_unique<GaussianDistribution>(1.5, 0.8));
    hmm.setProbabilityDistribution(1, std::make_unique<StudentTDistribution>(2.5, 0.5, 1.2));
    hmm.setProbabilityDistribution(2, std::make_unique<ChiSquaredDistribution>(4.0));
    
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
    const auto* orig_gauss = dynamic_cast<const GaussianDistribution*>(hmm.getProbabilityDistribution(0));
    const auto* rest_gauss = dynamic_cast<const GaussianDistribution*>(hmm_restored.getProbabilityDistribution(0));
    ASSERT_TRUE(orig_gauss != nullptr);
    ASSERT_TRUE(rest_gauss != nullptr);
    EXPECT_NEAR(orig_gauss->getMean(), rest_gauss->getMean(), 1e-10);
    EXPECT_NEAR(orig_gauss->getStandardDeviation(), rest_gauss->getStandardDeviation(), 1e-10);
    
    const auto* orig_student = dynamic_cast<const StudentTDistribution*>(hmm.getProbabilityDistribution(1));
    const auto* rest_student = dynamic_cast<const StudentTDistribution*>(hmm_restored.getProbabilityDistribution(1));
    ASSERT_TRUE(orig_student != nullptr);
    ASSERT_TRUE(rest_student != nullptr);
    EXPECT_NEAR(orig_student->getDegreesOfFreedom(), rest_student->getDegreesOfFreedom(), 1e-10);
    EXPECT_NEAR(orig_student->getLocation(), rest_student->getLocation(), 1e-10);
    EXPECT_NEAR(orig_student->getScale(), rest_student->getScale(), 1e-10);
    
    const auto* orig_chi = dynamic_cast<const ChiSquaredDistribution*>(hmm.getProbabilityDistribution(2));
    const auto* rest_chi = dynamic_cast<const ChiSquaredDistribution*>(hmm_restored.getProbabilityDistribution(2));
    ASSERT_TRUE(orig_chi != nullptr);
    ASSERT_TRUE(rest_chi != nullptr);
    EXPECT_NEAR(orig_chi->getDegreesOfFreedom(), rest_chi->getDegreesOfFreedom(), 1e-10);
}
