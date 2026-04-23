#include <gtest/gtest.h>
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <cmath>
#include <memory>

using namespace libhmm;

namespace {
std::unique_ptr<Hmm> create_two_state_hmm() {
    auto hmm = std::make_unique<Hmm>(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.9; trans(0, 1) = 0.1;
    trans(1, 0) = 0.8; trans(1, 1) = 0.2;
    hmm->setTrans(trans);

    Vector pi(2);
    pi(0) = 0.75; pi(1) = 0.25;
    hmm->setPi(pi);

    auto fair = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 6; ++i) fair->setProbability(i, 1.0 / 6.0);
    hmm->setDistribution(0, std::move(fair));

    auto loaded = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 5; ++i) loaded->setProbability(i, 0.125);
    loaded->setProbability(5, 0.375);
    hmm->setDistribution(1, std::move(loaded));

    return hmm;
}
} // namespace

// ---------------------------------------------------------------------------
// Test fixture: two-state casino HMM
// ---------------------------------------------------------------------------

class CanonicalCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        hmm_ = create_two_state_hmm();

        obs5_ = ObservationSet(5);
        obs5_(0) = 0; obs5_(1) = 1; obs5_(2) = 5;
        obs5_(3) = 4; obs5_(4) = 2;

        obs1_ = ObservationSet(1);
        obs1_(0) = 3;

        obs100_ = ObservationSet(100);
        for (std::size_t i = 0; i < 100; ++i) obs100_(i) = static_cast<double>(i % 6);
    }

    std::unique_ptr<Hmm> hmm_;
    ObservationSet obs5_;
    ObservationSet obs1_;
    ObservationSet obs100_;
};

// ---------------------------------------------------------------------------
// ForwardBackwardCalculator
// ---------------------------------------------------------------------------

TEST_F(CanonicalCalculatorTest, FB_NullHmmThrows) {
    EXPECT_THROW(ForwardBackwardCalculator(nullptr, obs5_), std::invalid_argument);
}

TEST_F(CanonicalCalculatorTest, FB_EmptyObsThrows) {
    ObservationSet empty(0);
    EXPECT_THROW(ForwardBackwardCalculator(*hmm_, empty), std::invalid_argument);
}

TEST_F(CanonicalCalculatorTest, FB_ProbabilityInRange) {
    ForwardBackwardCalculator fbc(*hmm_, obs5_);
    const double p = fbc.probability();
    EXPECT_GT(p, 0.0);
    EXPECT_LE(p, 1.0);
}

TEST_F(CanonicalCalculatorTest, FB_LogProbFinite) {
    ForwardBackwardCalculator fbc(*hmm_, obs5_);
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
}

TEST_F(CanonicalCalculatorTest, FB_LogProbConsistentWithProbability) {
    ForwardBackwardCalculator fbc(*hmm_, obs5_);
    EXPECT_NEAR(std::exp(fbc.getLogProbability()), fbc.probability(), 1e-12);
}

TEST_F(CanonicalCalculatorTest, FB_MatrixDimensions) {
    ForwardBackwardCalculator fbc(*hmm_, obs5_);
    const Matrix& alpha = fbc.getLogForwardVariables();
    const Matrix& beta  = fbc.getLogBackwardVariables();
    EXPECT_EQ(alpha.size1(), obs5_.size());
    EXPECT_EQ(alpha.size2(), static_cast<std::size_t>(hmm_->getNumStates()));
    EXPECT_EQ(beta.size1(),  obs5_.size());
    EXPECT_EQ(beta.size2(),  static_cast<std::size_t>(hmm_->getNumStates()));
}

TEST_F(CanonicalCalculatorTest, FB_SingleObservation) {
    ForwardBackwardCalculator fbc(*hmm_, obs1_);
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
    EXPECT_EQ(fbc.getLogForwardVariables().size1(), 1u);
}

TEST_F(CanonicalCalculatorTest, FB_LongSequence) {
    EXPECT_NO_THROW({
        ForwardBackwardCalculator fbc(*hmm_, obs100_);
        EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
    });
}

TEST_F(CanonicalCalculatorTest, FB_RecomputeWithNewObs) {
    ForwardBackwardCalculator fbc(*hmm_, obs5_);
    const double lp1 = fbc.getLogProbability();

    fbc.compute(obs1_);
    const double lp2 = fbc.getLogProbability();

    // Different sequences should (almost certainly) give different values
    EXPECT_NE(lp1, lp2);
}

// ---------------------------------------------------------------------------
// ViterbiCalculator
// ---------------------------------------------------------------------------

TEST_F(CanonicalCalculatorTest, Viterbi_NullHmmThrows) {
    EXPECT_THROW(ViterbiCalculator(nullptr, obs5_), std::invalid_argument);
}

TEST_F(CanonicalCalculatorTest, Viterbi_EmptyObsThrows) {
    ObservationSet empty(0);
    EXPECT_THROW(ViterbiCalculator(*hmm_, empty), std::invalid_argument);
}

TEST_F(CanonicalCalculatorTest, Viterbi_SequenceLengthMatchesObs) {
    ViterbiCalculator vc(*hmm_, obs5_);
    StateSequence seq = vc.decode();
    EXPECT_EQ(seq.size(), obs5_.size());
}

TEST_F(CanonicalCalculatorTest, Viterbi_AllStatesValid) {
    ViterbiCalculator vc(*hmm_, obs5_);
    StateSequence seq = vc.decode();
    for (std::size_t i = 0; i < seq.size(); ++i) {
        EXPECT_GE(seq(i), 0);
        EXPECT_LT(seq(i), hmm_->getNumStates());
    }
}

TEST_F(CanonicalCalculatorTest, Viterbi_LogProbFinite) {
    ViterbiCalculator vc(*hmm_, obs5_);
    EXPECT_TRUE(std::isfinite(vc.getLogProbability()));
}

TEST_F(CanonicalCalculatorTest, Viterbi_DecodeMatchesStoredSequence) {
    ViterbiCalculator vc(*hmm_, obs5_);
    StateSequence decoded = vc.decode();
    const StateSequence& stored = vc.getStateSequence();
    ASSERT_EQ(decoded.size(), stored.size());
    for (std::size_t i = 0; i < decoded.size(); ++i) {
        EXPECT_EQ(decoded(i), stored(i));
    }
}

TEST_F(CanonicalCalculatorTest, Viterbi_SingleObservation) {
    ViterbiCalculator vc(*hmm_, obs1_);
    StateSequence seq = vc.decode();
    EXPECT_EQ(seq.size(), 1u);
    EXPECT_GE(seq(0), 0);
    EXPECT_LT(seq(0), hmm_->getNumStates());
}

TEST_F(CanonicalCalculatorTest, Viterbi_LongSequence) {
    EXPECT_NO_THROW({
        ViterbiCalculator vc(*hmm_, obs100_);
        EXPECT_TRUE(std::isfinite(vc.getLogProbability()));
        EXPECT_EQ(vc.getStateSequence().size(), obs100_.size());
    });
}

// ---------------------------------------------------------------------------
// Cross-calculator consistency
// ---------------------------------------------------------------------------

TEST_F(CanonicalCalculatorTest, ViterbiLogProbLEForwardBackward) {
    // The Viterbi path probability can never exceed the total probability.
    ForwardBackwardCalculator fbc(*hmm_, obs5_);
    ViterbiCalculator vc(*hmm_, obs5_);
    // log P(O|λ) >= log P(O, q*|λ)
    EXPECT_GE(fbc.getLogProbability(), vc.getLogProbability() - 1e-9);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
