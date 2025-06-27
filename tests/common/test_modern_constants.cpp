#include <gtest/gtest.h>
#include <iostream>
#include "libhmm/common/common.h"

using namespace libhmm::constants;

class ModernConstantsTest : public ::testing::Test {};

TEST_F(ModernConstantsTest, CheckConstantsExistence) {
    EXPECT_DOUBLE_EQ(precision::BW_TOLERANCE, 3.0e-7);
    EXPECT_DOUBLE_EQ(precision::ZERO, 1.0e-30);
    EXPECT_DOUBLE_EQ(precision::LIMIT_TOLERANCE, 1.0e-6);
    EXPECT_EQ(iterations::MAX_VITERBI_ITERATIONS, 500);
    EXPECT_EQ(iterations::ITMAX, 10000);
    EXPECT_DOUBLE_EQ(math::PI, 3.141592653589793238462643383279502884);
}

TEST_F(ModernConstantsTest, CheckConvergenceConstants) {
    EXPECT_DOUBLE_EQ(precision::DEFAULT_CONVERGENCE_TOLERANCE, 1.0e-8);
    EXPECT_DOUBLE_EQ(precision::HIGH_PRECISION_TOLERANCE, 1.0e-12);
    EXPECT_DOUBLE_EQ(precision::ULTRA_HIGH_PRECISION_TOLERANCE, 1.0e-15);
}

TEST_F(ModernConstantsTest, CheckProbabilityConstants) {
    EXPECT_DOUBLE_EQ(probability::MIN_PROBABILITY, 1.0e-300);
    EXPECT_DOUBLE_EQ(probability::MAX_PROBABILITY, 1.0 - 1.0e-15);
    EXPECT_DOUBLE_EQ(probability::MIN_LOG_PROBABILITY, -700.0);
    EXPECT_DOUBLE_EQ(probability::MAX_LOG_PROBABILITY, 0.0);
    EXPECT_DOUBLE_EQ(probability::SCALING_THRESHOLD, 1.0e-100);
    EXPECT_DOUBLE_EQ(probability::LOG_SCALING_THRESHOLD, -230.0);
}

TEST_F(ModernConstantsTest, CheckMathConstants) {
    EXPECT_DOUBLE_EQ(math::LN2, 0.6931471805599453094172321214581766);
    EXPECT_DOUBLE_EQ(math::E, 2.7182818284590452353602874713526625);
    EXPECT_DOUBLE_EQ(math::SQRT_2PI, 2.5066282746310005024157652848110453);
    EXPECT_DOUBLE_EQ(math::LN_2PI, 1.8378770664093454835606594728112353);
}
