#include <gtest/gtest.h>
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <cmath>
#include <limits>
#include <memory>

using namespace libhmm;

class CalculatorEdgeCasesTest : public ::testing::Test {
protected:
    // Single-state HMM — degenerate but structurally valid.
    static std::unique_ptr<Hmm> make_single_state_hmm() {
        auto hmm = std::make_unique<Hmm>(1);
        Matrix trans(1, 1); trans(0, 0) = 1.0;
        hmm->setTrans(trans);
        Vector pi(1); pi(0) = 1.0;
        hmm->setPi(pi);
        auto dist = std::make_unique<DiscreteDistribution>(4);
        for (int i = 0; i < 4; ++i) dist->setProbability(i, 0.25);
        hmm->setDistribution(0, std::move(dist));
        return hmm;
    }

    // 2-state HMM with deterministic emissions:
    //   State 0 emits symbol 0 only (P=1).
    //   State 1 emits symbol 1 only (P=1).
    // This forces Viterbi to follow the observation sequence exactly.
    static std::unique_ptr<Hmm> make_deterministic_emission_hmm() {
        auto hmm = std::make_unique<Hmm>(2);
        Matrix trans(2, 2);
        trans(0, 0) = 0.9; trans(0, 1) = 0.1;
        trans(1, 0) = 0.1; trans(1, 1) = 0.9;
        hmm->setTrans(trans);
        Vector pi(2); pi(0) = 0.5; pi(1) = 0.5;
        hmm->setPi(pi);

        auto d0 = std::make_unique<DiscreteDistribution>(4);
        d0->setProbability(0, 1.0);
        d0->setProbability(1, 0.0);
        d0->setProbability(2, 0.0);
        d0->setProbability(3, 0.0);
        hmm->setDistribution(0, std::move(d0));

        auto d1 = std::make_unique<DiscreteDistribution>(4);
        d1->setProbability(0, 0.0);
        d1->setProbability(1, 1.0);
        d1->setProbability(2, 0.0);
        d1->setProbability(3, 0.0);
        hmm->setDistribution(1, std::move(d1));
        return hmm;
    }
};

// ---------------------------------------------------------------------------
// Single-state HMM
// ---------------------------------------------------------------------------

TEST_F(CalculatorEdgeCasesTest, SingleState_FB_Finite) {
    auto hmm = make_single_state_hmm();
    ObservationSet obs(5);
    for (std::size_t i = 0; i < 5; ++i) obs(i) = i % 4;

    ForwardBackwardCalculator fbc(*hmm, obs);
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
    EXPECT_GT(fbc.probability(), 0.0);
}

TEST_F(CalculatorEdgeCasesTest, SingleState_Viterbi_OnlyState0) {
    auto hmm = make_single_state_hmm();
    ObservationSet obs(5);
    for (std::size_t i = 0; i < 5; ++i) obs(i) = i % 4;

    ViterbiCalculator vc(*hmm, obs);
    const auto seq = vc.decode();
    EXPECT_EQ(seq.size(), obs.size());
    for (std::size_t i = 0; i < seq.size(); ++i) {
        EXPECT_EQ(seq(i), 0);
    }
}

// ---------------------------------------------------------------------------
// Single-observation sequences
// ---------------------------------------------------------------------------

TEST_F(CalculatorEdgeCasesTest, SingleObs_FB) {
    auto hmm = make_single_state_hmm();
    ObservationSet obs(1); obs(0) = 2;

    ForwardBackwardCalculator fbc(*hmm, obs);
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
    EXPECT_EQ(fbc.getLogForwardVariables().size1(), 1u);
}

TEST_F(CalculatorEdgeCasesTest, SingleObs_Viterbi) {
    auto hmm = make_single_state_hmm();
    ObservationSet obs(1); obs(0) = 0;

    ViterbiCalculator vc(*hmm, obs);
    const auto seq = vc.decode();
    EXPECT_EQ(seq.size(), 1u);
    EXPECT_EQ(seq(0), 0);
}

// ---------------------------------------------------------------------------
// Deterministic emissions — Viterbi path must follow the observation symbol
// ---------------------------------------------------------------------------

TEST_F(CalculatorEdgeCasesTest, DeterministicEmissions_AllSymbol0_AssignedState0) {
    auto hmm = make_deterministic_emission_hmm();
    ObservationSet obs(6);
    for (std::size_t i = 0; i < 6; ++i) obs(i) = 0;

    ViterbiCalculator vc(*hmm, obs);
    const auto seq = vc.decode();
    for (std::size_t i = 0; i < seq.size(); ++i) {
        EXPECT_EQ(seq(i), 0) << "Position " << i << " should be state 0";
    }
}

TEST_F(CalculatorEdgeCasesTest, DeterministicEmissions_AlternatingSymbols) {
    // obs = [0, 1, 0, 1] → must map to states [0, 1, 0, 1]
    auto hmm = make_deterministic_emission_hmm();
    ObservationSet obs(4);
    obs(0) = 0; obs(1) = 1; obs(2) = 0; obs(3) = 1;

    ViterbiCalculator vc(*hmm, obs);
    const auto seq = vc.decode();
    EXPECT_EQ(seq(0), 0);
    EXPECT_EQ(seq(1), 1);
    EXPECT_EQ(seq(2), 0);
    EXPECT_EQ(seq(3), 1);
}

TEST_F(CalculatorEdgeCasesTest, DeterministicEmissions_FB_NoNaN) {
    // When one state has P=0 for the observed symbol, the total probability
    // is still computable and non-NaN (that state just contributes 0 weight).
    auto hmm = make_deterministic_emission_hmm();
    ObservationSet obs(4);
    for (std::size_t i = 0; i < 4; ++i) obs(i) = 0;  // Only state 0 can emit this

    ForwardBackwardCalculator fbc(*hmm, obs);
    EXPECT_FALSE(std::isnan(fbc.getLogProbability()));
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
    EXPECT_GT(fbc.probability(), 0.0);
}

// ---------------------------------------------------------------------------
// Long sequences — validates log-space numerical stability
// ---------------------------------------------------------------------------

TEST_F(CalculatorEdgeCasesTest, LongSequence_FB_NoUnderflow) {
    auto hmm = make_single_state_hmm();
    ObservationSet longObs(500);
    for (std::size_t i = 0; i < 500; ++i) longObs(i) = i % 4;

    ForwardBackwardCalculator fbc(*hmm, longObs);
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
    EXPECT_FALSE(std::isnan(fbc.getLogProbability()));
    // log P ≈ 500 * log(0.25) ≈ -693 — finite, not -inf
    EXPECT_LT(fbc.getLogProbability(), -100.0);
    EXPECT_GT(fbc.getLogProbability(), -std::numeric_limits<double>::infinity());
}

TEST_F(CalculatorEdgeCasesTest, LongSequence_Viterbi_NoErrors) {
    auto hmm = make_single_state_hmm();
    ObservationSet longObs(500);
    for (std::size_t i = 0; i < 500; ++i) longObs(i) = i % 4;

    EXPECT_NO_THROW({
        ViterbiCalculator vc(*hmm, longObs);
        EXPECT_EQ(vc.getStateSequence().size(), longObs.size());
    });
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
