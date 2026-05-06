#include <gtest/gtest.h>
#include "libhmm/math/numerical_stability.h"
#include <limits>
#include <cmath>

using namespace libhmm;
using namespace libhmm::numerical;

class NumericalStabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        testVector_ = Vector(5);
        testVector_(0) = 1.0;
        testVector_(1) = 0.5;
        testVector_(2) = 0.3;
        testVector_(3) = 0.1;
        testVector_(4) = 0.1;
    }
    Vector testVector_;
};

TEST_F(NumericalStabilityTest, CheckFiniteAcceptsFiniteValues) {
    EXPECT_NO_THROW(NumericalSafety::checkFinite(1.0));
    EXPECT_NO_THROW(NumericalSafety::checkFinite(-1.0));
    EXPECT_NO_THROW(NumericalSafety::checkFinite(0.0));
}

TEST_F(NumericalStabilityTest, CheckFiniteRejectsNonFinite) {
    EXPECT_THROW(NumericalSafety::checkFinite(std::numeric_limits<double>::infinity()),
                 std::runtime_error);
    EXPECT_THROW(NumericalSafety::checkFinite(-std::numeric_limits<double>::infinity()),
                 std::runtime_error);
    EXPECT_THROW(NumericalSafety::checkFinite(std::numeric_limits<double>::quiet_NaN()),
                 std::runtime_error);
}

TEST_F(NumericalStabilityTest, ClampProbabilityPassesThroughValidValues) {
    EXPECT_EQ(NumericalSafety::clampProbability(0.5), 0.5);
}

TEST_F(NumericalStabilityTest, ClampProbabilityHandlesOutOfRange) {
    EXPECT_EQ(NumericalSafety::clampProbability(1.5),  NumericalConstants::MAX_PROBABILITY);
    EXPECT_EQ(NumericalSafety::clampProbability(-0.5), NumericalConstants::MIN_PROBABILITY);
    EXPECT_EQ(NumericalSafety::clampProbability(std::numeric_limits<double>::quiet_NaN()),
              NumericalConstants::MIN_PROBABILITY);
}

TEST_F(NumericalStabilityTest, NormalizeProbabilitiesSumsToOne) {
    Vector probs = testVector_;
    EXPECT_TRUE(NumericalSafety::normalizeProbabilities(probs));

    double sum = 0.0;
    for (std::size_t i = 0; i < probs.size(); ++i)
        sum += probs(i);
    EXPECT_NEAR(sum, 1.0, 1e-15);
}

TEST_F(NumericalStabilityTest, NormalizeProbabilitiesFallsBackToUniformOnZeroInput) {
    Vector zeroProbs(3);
    zeroProbs(0) = zeroProbs(1) = zeroProbs(2) = 0.0;
    EXPECT_FALSE(NumericalSafety::normalizeProbabilities(zeroProbs));

    for (std::size_t i = 0; i < zeroProbs.size(); ++i)
        EXPECT_NEAR(zeroProbs(i), 1.0 / 3.0, 1e-15);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
