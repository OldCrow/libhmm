#include <gtest/gtest.h>
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <cmath>
#include <memory>

using namespace libhmm;

namespace {

/// Two-state casino HMM: fair die (state 0) and loaded die (state 1).
std::unique_ptr<Hmm> make_casino_hmm() {
    auto hmm = std::make_unique<Hmm>(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.9;
    trans(0, 1) = 0.1;
    trans(1, 0) = 0.2;
    trans(1, 1) = 0.8;
    hmm->setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm->setPi(pi);

    auto fair = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 6; ++i)
        fair->setProbability(i, 1.0 / 6.0);
    hmm->setDistribution(0, std::move(fair));

    auto loaded = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 5; ++i)
        loaded->setProbability(i, 0.1);
    loaded->setProbability(5, 0.5); // strongly prefers 6
    hmm->setDistribution(1, std::move(loaded));

    return hmm;
}

/// Two-state Gaussian HMM with well-separated means.
std::unique_ptr<Hmm> make_gaussian_hmm() {
    auto hmm = std::make_unique<Hmm>(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.8;
    trans(0, 1) = 0.2;
    trans(1, 0) = 0.3;
    trans(1, 1) = 0.7;
    hmm->setTrans(trans);
    Vector pi(2);
    pi(0) = 0.6;
    pi(1) = 0.4;
    hmm->setPi(pi);
    hmm->setDistribution(0, std::make_unique<GaussianDistribution>(-5.0, 0.5));
    hmm->setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 0.5));
    return hmm;
}

} // namespace

// ---------------------------------------------------------------------------
// Basic structural checks
// ---------------------------------------------------------------------------

TEST(PosteriorDecodingTest, SequenceLengthMatchesObservations) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(10);
    for (std::size_t i = 0; i < 10; ++i)
        obs(i) = static_cast<double>(i % 6);
    ForwardBackwardCalculator fbc(*hmm, obs);
    StateSequence posterior = fbc.decodePosterior();
    EXPECT_EQ(posterior.size(), obs.size());
}

TEST(PosteriorDecodingTest, AllStatesInValidRange) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(20);
    for (std::size_t i = 0; i < 20; ++i)
        obs(i) = static_cast<double>(i % 6);
    ForwardBackwardCalculator fbc(*hmm, obs);
    StateSequence posterior = fbc.decodePosterior();
    for (std::size_t t = 0; t < posterior.size(); ++t) {
        EXPECT_GE(posterior(t), 0);
        EXPECT_LT(posterior(t), hmm->getNumStates());
    }
}

TEST(PosteriorDecodingTest, SingleObservation) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(1);
    obs(0) = 3.0;
    ForwardBackwardCalculator fbc(*hmm, obs);
    StateSequence posterior = fbc.decodePosterior();
    ASSERT_EQ(posterior.size(), 1u);
    EXPECT_GE(posterior(0), 0);
    EXPECT_LT(posterior(0), hmm->getNumStates());
}

// ---------------------------------------------------------------------------
// Semantic checks
// ---------------------------------------------------------------------------

/// For the loaded die model, a long run of 5s (index 5 = "6" on the die)
/// should be decoded mostly as state 1 (loaded).
TEST(PosteriorDecodingTest, HighValueRunDecodedAsLoadedState) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(30);
    for (std::size_t i = 0; i < 30; ++i)
        obs(i) = 5.0; // all sixes — loaded die territory
    ForwardBackwardCalculator fbc(*hmm, obs);
    StateSequence posterior = fbc.decodePosterior();
    int loadedCount = 0;
    for (std::size_t t = 0; t < posterior.size(); ++t)
        if (posterior(t) == 1)
            ++loadedCount;
    // Expect the majority decoded as loaded state
    EXPECT_GT(loadedCount, 20);
}

/// For the Gaussian model, observations near −5 should be decoded as state 0
/// and observations near +5 as state 1.
TEST(PosteriorDecodingTest, GaussianWellSeparatedMeans) {
    auto hmm = make_gaussian_hmm();

    // All observations near mean of state 0
    ObservationSet obs0(10);
    for (std::size_t i = 0; i < 10; ++i)
        obs0(i) = -5.0;
    ForwardBackwardCalculator fbc0(*hmm, obs0);
    StateSequence post0 = fbc0.decodePosterior();
    // Interior steps should clearly prefer state 0
    for (std::size_t t = 2; t < 8; ++t)
        EXPECT_EQ(post0(t), 0) << "t=" << t;

    // All observations near mean of state 1
    ObservationSet obs1(10);
    for (std::size_t i = 0; i < 10; ++i)
        obs1(i) = 5.0;
    ForwardBackwardCalculator fbc1(*hmm, obs1);
    StateSequence post1 = fbc1.decodePosterior();
    for (std::size_t t = 2; t < 8; ++t)
        EXPECT_EQ(post1(t), 1) << "t=" << t;
}

/// Posterior and Viterbi may differ. On a long alternating sequence
/// (fair / loaded observations), at least one time step should disagree.
/// This is a soft check — it may be equal on degenerate inputs, but for a
/// non-trivial HMM with mixed evidence they typically differ somewhere.
TEST(PosteriorDecodingTest, MayDifferFromViterbi) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(40);
    for (std::size_t i = 0; i < 40; ++i)
        obs(i) = (i % 2 == 0) ? 0.0 : 5.0; // alternating evidence

    ForwardBackwardCalculator fbc(*hmm, obs);
    ViterbiCalculator vc(*hmm, obs);

    StateSequence posterior = fbc.decodePosterior();
    StateSequence viterbi = vc.getStateSequence();

    ASSERT_EQ(posterior.size(), viterbi.size());
    // We do not assert that they differ — just that both are valid.
    // On non-trivial sequences they often disagree.
    bool anyDiff = false;
    for (std::size_t t = 0; t < posterior.size(); ++t) {
        if (posterior(t) != viterbi(t)) {
            anyDiff = true;
            break;
        }
    }
    // Not a hard failure — log for information only
    if (!anyDiff) {
        GTEST_LOG_(INFO) << "posterior == viterbi on this sequence (both are valid)";
    }
}

/// Recomputing with a new observation sequence should update decodePosterior.
TEST(PosteriorDecodingTest, RecomputeUpdatesResult) {
    auto hmm = make_casino_hmm();
    ObservationSet obs1(5), obs2(5);
    for (std::size_t i = 0; i < 5; ++i) {
        obs1(i) = 0.0; // all zeros
        obs2(i) = 5.0; // all fives
    }
    ForwardBackwardCalculator fbc(*hmm, obs1);
    StateSequence r1 = fbc.decodePosterior();
    fbc.compute(obs2);
    StateSequence r2 = fbc.decodePosterior();

    // The sequences are different, so at least one state should differ.
    bool differ = false;
    for (std::size_t t = 0; t < r1.size(); ++t) {
        if (r1(t) != r2(t)) {
            differ = true;
            break;
        }
    }
    EXPECT_TRUE(differ) << "Posterior decode should change when observations change";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
