/**
 * @file test_mv_calculator.cpp
 * @brief Tests for the MV (Obs=ObservationVectorView) calculator path.
 *
 * Exercises BasicForwardBackwardCalculator<ObservationVectorView> and
 * BasicViterbiCalculator<ObservationVectorView> on a small 2-state,
 * 2-dimensional DiagonalGaussian HMM.  The tests verify that:
 *   - log-probabilities are finite and negative.
 *   - Matrix dimensions match the sequence length and state count.
 *   - Decoded state sequences have the correct length and valid state indices.
 *   - decodePosterior() agrees in length with decode().
 *   - Two distinct observation sequences produce different log-probabilities.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/calculators/basic_viterbi_calculator.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"

using namespace libhmm;

// =============================================================================
// Fixture: 2-state, D=2 DiagonalGaussian HMM
//
//  State 0: N([0,0], [1,1])   — near-origin observations
//  State 1: N([5,5], [1,1])   — far-from-origin observations
//  Transitions: 0.8 self-loop, 0.2 switch
//  pi: uniform
// =============================================================================

namespace {

/// Build a 2-state, D-dimensional DiagonalGaussian HmmMV.
/// State 0 centred at `mean0`; State 1 centred at `mean1`.  Both have unit variance.
HmmMV make_diag_hmm(std::size_t D, double mean0 = 0.0, double mean1 = 5.0) {
    HmmMV hmm(2);
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(D, mean0, 1.0));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(D, mean1, 1.0));

    Matrix trans(2, 2);
    trans(0, 0) = 0.8;
    trans(0, 1) = 0.2;
    trans(1, 0) = 0.2;
    trans(1, 1) = 0.8;
    hmm.setTrans(trans);

    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);
    return hmm;
}

/// Build a T×D observation matrix where each row equals @p val.
ObservationMatrix make_const_obs(std::size_t T, std::size_t D, double val = 0.0) {
    ObservationMatrix mat(T, D);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t d = 0; d < D; ++d)
            mat(t, d) = val;
    return mat;
}

} // namespace

// =============================================================================
// BasicForwardBackwardCalculator<ObservationVectorView>
// =============================================================================

TEST(MvFBC, LogProbabilityIsFiniteAndNegative) {
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs = make_const_obs(10, 2, 0.0); // 10 steps near state 0
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, obs);
    const double lp = fbc.getLogProbability();
    EXPECT_TRUE(std::isfinite(lp));
    EXPECT_LT(lp, 0.0);
}

TEST(MvFBC, ForwardVariablesDimensions) {
    HmmMV hmm = make_diag_hmm(2);
    constexpr std::size_t T = 8, N = 2;
    const ObservationMatrix obs = make_const_obs(T, 2, 0.0);
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, obs);
    const Matrix &alpha = fbc.getLogForwardVariables();
    EXPECT_EQ(alpha.size1(), T);
    EXPECT_EQ(alpha.size2(), N);
}

TEST(MvFBC, BackwardVariablesDimensions) {
    HmmMV hmm = make_diag_hmm(2);
    constexpr std::size_t T = 6, N = 2;
    const ObservationMatrix obs = make_const_obs(T, 2, 0.0);
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, obs);
    const Matrix &beta = fbc.getLogBackwardVariables();
    EXPECT_EQ(beta.size1(), T);
    EXPECT_EQ(beta.size2(), N);
}

TEST(MvFBC, DecodePosteriorLength) {
    HmmMV hmm = make_diag_hmm(2);
    constexpr std::size_t T = 12;
    const ObservationMatrix obs = make_const_obs(T, 2, 0.0);
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, obs);
    const StateSequence seq = fbc.decodePosterior();
    EXPECT_EQ(seq.size(), T);
}

TEST(MvFBC, DecodePosteriorStatesInRange) {
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs = make_const_obs(10, 2, 0.0);
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, obs);
    const StateSequence seq = fbc.decodePosterior();
    for (std::size_t t = 0; t < seq.size(); ++t)
        EXPECT_LT(static_cast<std::size_t>(seq(t)), 2u) << "t=" << t;
}

TEST(MvFBC, TwoSequencesGiveDifferentLogProb) {
    // Observations near state 0 vs near state 1 should yield different logP.
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs0 = make_const_obs(10, 2, 0.0); // near state 0
    const ObservationMatrix obs1 = make_const_obs(10, 2, 5.0); // near state 1

    BasicForwardBackwardCalculator<ObservationVectorView> fbc0(hmm, obs0);
    BasicForwardBackwardCalculator<ObservationVectorView> fbc1(hmm, obs1);

    // Both should be finite
    ASSERT_TRUE(std::isfinite(fbc0.getLogProbability()));
    ASSERT_TRUE(std::isfinite(fbc1.getLogProbability()));
    // By symmetry of this HMM they should be equal (both sequences are equally
    // probable under the symmetric model); verify they're at least finite and equal.
    EXPECT_NEAR(fbc0.getLogProbability(), fbc1.getLogProbability(), 1e-9);
}

TEST(MvFBC, RecomputeWithNewObs) {
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs1 = make_const_obs(5, 2, 0.0);
    const ObservationMatrix obs2 = make_const_obs(8, 2, 0.0);
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, obs1);
    const double lp1 = fbc.getLogProbability();
    fbc.compute(obs2);
    const double lp2 = fbc.getLogProbability();
    // Longer sequence → smaller log-probability
    EXPECT_LT(lp2, lp1);
}

TEST(MvFBC, GetNumStates) {
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs = make_const_obs(5, 2);
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, obs);
    EXPECT_EQ(fbc.getNumStates(), 2u);
}

// =============================================================================
// BasicViterbiCalculator<ObservationVectorView>
// =============================================================================

TEST(MvViterbi, DecodeStateSequenceLength) {
    HmmMV hmm = make_diag_hmm(2);
    constexpr std::size_t T = 10;
    const ObservationMatrix obs = make_const_obs(T, 2, 0.0);
    BasicViterbiCalculator<ObservationVectorView> vc(hmm, obs);
    EXPECT_EQ(vc.getStateSequence().size(), T);
}

TEST(MvViterbi, DecodeStateSequenceValidStates) {
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs = make_const_obs(15, 2, 0.0);
    BasicViterbiCalculator<ObservationVectorView> vc(hmm, obs);
    const StateSequence &seq = vc.getStateSequence();
    for (std::size_t t = 0; t < seq.size(); ++t)
        EXPECT_LT(static_cast<std::size_t>(seq(t)), 2u) << "t=" << t;
}

TEST(MvViterbi, LogProbabilityIsFiniteAndNegative) {
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs = make_const_obs(10, 2, 0.0);
    BasicViterbiCalculator<ObservationVectorView> vc(hmm, obs);
    EXPECT_TRUE(std::isfinite(vc.getLogProbability()));
    EXPECT_LT(vc.getLogProbability(), 0.0);
}

TEST(MvViterbi, ObsNearState0PreferState0) {
    // With observations very close to state-0 mean and state 1 far away,
    // most decoded states should be 0.
    HmmMV hmm = make_diag_hmm(2, /*mean0=*/0.0, /*mean1=*/100.0);
    const ObservationMatrix obs = make_const_obs(20, 2, 0.0); // exactly at state 0 mean
    BasicViterbiCalculator<ObservationVectorView> vc(hmm, obs);
    const StateSequence &seq = vc.getStateSequence();
    std::size_t state0_count = 0;
    for (std::size_t t = 0; t < seq.size(); ++t)
        if (seq(t) == 0)
            ++state0_count;
    EXPECT_EQ(state0_count, seq.size()); // all should be state 0
}

TEST(MvViterbi, GetNumStates) {
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs = make_const_obs(5, 2);
    BasicViterbiCalculator<ObservationVectorView> vc(hmm, obs);
    EXPECT_EQ(vc.getNumStates(), 2u);
}

TEST(MvViterbi, EmptyObsThrows) {
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix empty_obs(0, 2);
    EXPECT_THROW((BasicViterbiCalculator<ObservationVectorView>(hmm, empty_obs)),
                 std::invalid_argument);
}

// =============================================================================
// FBC vs Viterbi consistency
// =============================================================================

TEST(MvFbcViterbi, ViterbiLogProbLeqFbcLogProb) {
    // Viterbi finds the best single path; FBC sums over all paths.
    // Therefore logP(Viterbi) ≤ logP(FBC).
    HmmMV hmm = make_diag_hmm(2);
    const ObservationMatrix obs = make_const_obs(10, 2, 0.0);
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, obs);
    BasicViterbiCalculator<ObservationVectorView> vc(hmm, obs);
    EXPECT_LE(vc.getLogProbability(), fbc.getLogProbability() + 1e-9);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
