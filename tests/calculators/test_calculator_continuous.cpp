#include <gtest/gtest.h>
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <cmath>
#include <memory>

using namespace libhmm;

// Two-state Gaussian HMM with clearly separated means.
// State 0: N(0, 1)   State 1: N(10, 1)
// High self-transition (0.9) and well-separated means make Viterbi
// assignments analytically predictable from the observations.
namespace {
std::unique_ptr<Hmm> make_gaussian_hmm() {
    auto hmm = std::make_unique<Hmm>(2);

    Matrix trans(2, 2);
    trans(0, 0) = 0.9;
    trans(0, 1) = 0.1;
    trans(1, 0) = 0.1;
    trans(1, 1) = 0.9;
    hmm->setTrans(trans);

    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm->setPi(pi);

    hmm->setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm->setDistribution(1, std::make_unique<GaussianDistribution>(10.0, 1.0));
    return hmm;
}
} // namespace

class ContinuousCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        hmm_ = make_gaussian_hmm();

        // Observations tightly around state 0 mean
        obs_s0_ = ObservationSet(6);
        obs_s0_(0) = 0.1;
        obs_s0_(1) = -0.2;
        obs_s0_(2) = 0.3;
        obs_s0_(3) = -0.1;
        obs_s0_(4) = 0.2;
        obs_s0_(5) = 0.0;

        // Observations tightly around state 1 mean
        obs_s1_ = ObservationSet(6);
        obs_s1_(0) = 9.9;
        obs_s1_(1) = 10.1;
        obs_s1_(2) = 9.8;
        obs_s1_(3) = 10.2;
        obs_s1_(4) = 9.7;
        obs_s1_(5) = 10.3;

        // Mixed: first half near 0, second half near 10
        obs_mix_ = ObservationSet(10);
        for (std::size_t i = 0; i < 5; ++i)
            obs_mix_(i) = static_cast<double>(i) * 0.1;
        for (std::size_t i = 5; i < 10; ++i)
            obs_mix_(i) = 10.0 + static_cast<double>(i - 5) * 0.1;
    }

    std::unique_ptr<Hmm> hmm_;
    ObservationSet obs_s0_, obs_s1_, obs_mix_;
};

// ---------------------------------------------------------------------------
// ForwardBackwardCalculator with Gaussian distributions
// ---------------------------------------------------------------------------

TEST_F(ContinuousCalculatorTest, FB_ProbabilityInRange) {
    ForwardBackwardCalculator fbc(*hmm_, obs_s0_);
    EXPECT_GT(fbc.probability(), 0.0);
    EXPECT_LE(fbc.probability(), 1.0);
}

TEST_F(ContinuousCalculatorTest, FB_LogProbFinite) {
    ForwardBackwardCalculator fbc(*hmm_, obs_s0_);
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
}

TEST_F(ContinuousCalculatorTest, FB_LogProbConsistentWithProbability) {
    ForwardBackwardCalculator fbc(*hmm_, obs_s0_);
    EXPECT_NEAR(std::exp(fbc.getLogProbability()), fbc.probability(), 1e-12);
}

TEST_F(ContinuousCalculatorTest, FB_DifferentSequencesDifferentProb) {
    ForwardBackwardCalculator fbc0(*hmm_, obs_s0_);
    ForwardBackwardCalculator fbc1(*hmm_, obs_s1_);
    EXPECT_TRUE(std::isfinite(fbc0.getLogProbability()));
    EXPECT_TRUE(std::isfinite(fbc1.getLogProbability()));
    EXPECT_NE(fbc0.getLogProbability(), fbc1.getLogProbability());
}

TEST_F(ContinuousCalculatorTest, FB_Recompute) {
    ForwardBackwardCalculator fbc(*hmm_, obs_s0_);
    const double lp0 = fbc.getLogProbability();
    fbc.compute(obs_s1_);
    EXPECT_NE(lp0, fbc.getLogProbability());
}

// ---------------------------------------------------------------------------
// ViterbiCalculator with Gaussian distributions
// ---------------------------------------------------------------------------

TEST_F(ContinuousCalculatorTest, Viterbi_PathLength) {
    ViterbiCalculator vc(*hmm_, obs_s0_);
    EXPECT_EQ(vc.decode().size(), obs_s0_.size());
}

TEST_F(ContinuousCalculatorTest, Viterbi_AllStatesValid) {
    ViterbiCalculator vc(*hmm_, obs_mix_);
    const auto seq = vc.decode();
    for (std::size_t i = 0; i < seq.size(); ++i) {
        EXPECT_GE(seq(i), 0);
        EXPECT_LT(seq(i), hmm_->getNumStates());
    }
}

TEST_F(ContinuousCalculatorTest, Viterbi_State0Observations_AssignedState0) {
    // Observations tightly around 0 and high self-transition — all steps
    // should decode to state 0.
    ViterbiCalculator vc(*hmm_, obs_s0_);
    const auto seq = vc.decode();
    for (std::size_t i = 0; i < seq.size(); ++i) {
        EXPECT_EQ(seq(i), 0) << "Expected state 0 at position " << i;
    }
}

TEST_F(ContinuousCalculatorTest, Viterbi_State1Observations_AssignedState1) {
    ViterbiCalculator vc(*hmm_, obs_s1_);
    const auto seq = vc.decode();
    for (std::size_t i = 0; i < seq.size(); ++i) {
        EXPECT_EQ(seq(i), 1) << "Expected state 1 at position " << i;
    }
}

TEST_F(ContinuousCalculatorTest, ViterbiLogProbLEForwardBackward) {
    ForwardBackwardCalculator fbc(*hmm_, obs_mix_);
    ViterbiCalculator vc(*hmm_, obs_mix_);
    // P(O|λ) >= P(O, q*|λ): FB total probability can never be less than best path.
    EXPECT_GE(fbc.getLogProbability(), vc.getLogProbability() - 1e-9);
}

// ---------------------------------------------------------------------------
// Long sequence — validates numerical stability of the SIMD batch path
// ---------------------------------------------------------------------------

TEST_F(ContinuousCalculatorTest, LongSequence_NumericallyStable) {
    ObservationSet longObs(1000);
    for (std::size_t t = 0; t < 500; ++t)
        longObs(t) = 0.0;
    for (std::size_t t = 500; t < 1000; ++t)
        longObs(t) = 10.0;

    EXPECT_NO_THROW({
        ForwardBackwardCalculator fbc(*hmm_, longObs);
        EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
        EXPECT_FALSE(std::isnan(fbc.getLogProbability()));
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
