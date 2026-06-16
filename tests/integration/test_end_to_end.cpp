#include <gtest/gtest.h>
#include "libhmm/hmm.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <cmath>
#include <memory>

using namespace libhmm;

// =============================================================================
// The occasionally dishonest casino (Durbin et al., 1998)
//
//   States:  0 = fair die    1 = loaded die
//   Fair:    P(i) = 1/6  for i = 0..5
//   Loaded:  P(i) = 0.1  for i = 0..4,  P(5) = 0.5
//   Trans:   P(stay fair) = 0.95,  P(stay loaded) = 0.90
//   Pi:      (0.5, 0.5)
//
// This is a canonical reference problem with well-understood structure.
// Tests here verify structural properties that can be checked analytically.
// =============================================================================

namespace {
std::unique_ptr<Hmm> make_casino_hmm() {
    auto hmm = std::make_unique<Hmm>(2);

    Matrix trans(2, 2);
    trans(0, 0) = 0.95;
    trans(0, 1) = 0.05;
    trans(1, 0) = 0.10;
    trans(1, 1) = 0.90;
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
    loaded->setProbability(5, 0.5); // Face 6 (0-indexed as 5)
    hmm->setDistribution(1, std::move(loaded));

    return hmm;
}
} // namespace

// ---------------------------------------------------------------------------
// Evaluate: P(O | λ)
// ---------------------------------------------------------------------------

TEST(CasinoEndToEnd, ShortSequence_ProbabilityBounded) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(3);
    obs(0) = 0;
    obs(1) = 1;
    obs(2) = 2;

    ForwardBackwardCalculator fbc(*hmm, obs);
    EXPECT_GT(fbc.probability(), 0.0);
    EXPECT_LE(fbc.probability(), 1.0);
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
    // Upper bound: P <= (1/6)^3 * 2 ≈ 0.009  (2 paths, uniform emissions)
    EXPECT_LT(fbc.probability(), 0.01);
}

TEST(CasinoEndToEnd, LongSequence_LogProbFinite) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(100);
    for (std::size_t i = 0; i < 100; ++i)
        obs(i) = static_cast<double>(i % 6);

    ForwardBackwardCalculator fbc(*hmm, obs);
    EXPECT_TRUE(std::isfinite(fbc.getLogProbability()));
    EXPECT_FALSE(std::isnan(fbc.getLogProbability()));
    // log P(100 steps) < 100 * log(1/6) ≈ -179
    EXPECT_LT(fbc.getLogProbability(), -100.0);
}

TEST(CasinoEndToEnd, LogConsistency) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(10);
    for (std::size_t i = 0; i < 10; ++i)
        obs(i) = static_cast<double>(i % 6);

    ForwardBackwardCalculator fbc(*hmm, obs);
    EXPECT_NEAR(std::exp(fbc.getLogProbability()), fbc.probability(), 1e-12);
}

// ---------------------------------------------------------------------------
// Decode: Viterbi best path
// ---------------------------------------------------------------------------

TEST(CasinoEndToEnd, AllSixes_PreferLoadedState) {
    // A run of all 6s strongly suggests the loaded die.
    // Loaded: P(5) = 0.50.  Fair: P(5) = 1/6 ≈ 0.167.
    // With 10 consecutive 6s the path should be predominantly state 1.
    auto hmm = make_casino_hmm();
    ObservationSet obs(10);
    for (std::size_t i = 0; i < 10; ++i)
        obs(i) = 5;

    ViterbiCalculator vc(*hmm, obs);
    const auto seq = vc.decode();
    EXPECT_EQ(seq.size(), obs.size());

    int loaded_count = 0;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        if (seq(i) == 1)
            ++loaded_count;
    }
    EXPECT_GT(loaded_count, 5); // More than half assigned to loaded
}

TEST(CasinoEndToEnd, DiverseFaces_PreferFairState) {
    // Diverse faces give approximately equal probability to all symbols —
    // no evidence for the loaded die.  Fair state should dominate.
    auto hmm = make_casino_hmm();
    ObservationSet obs(12);
    for (std::size_t i = 0; i < 12; ++i)
        obs(i) = static_cast<double>(i % 6); // One full cycle x2

    ViterbiCalculator vc(*hmm, obs);
    const auto seq = vc.decode();
    EXPECT_EQ(seq.size(), obs.size());

    int fair_count = 0;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        if (seq(i) == 0)
            ++fair_count;
    }
    EXPECT_GT(fair_count, 6); // More than half assigned to fair
}

TEST(CasinoEndToEnd, ViterbiLE_ForwardBackward) {
    auto hmm = make_casino_hmm();
    ObservationSet obs(20);
    for (std::size_t i = 0; i < 20; ++i)
        obs(i) = static_cast<double>(i % 6);

    ForwardBackwardCalculator fbc(*hmm, obs);
    ViterbiCalculator vc(*hmm, obs);
    EXPECT_GE(fbc.getLogProbability(), vc.getLogProbability() - 1e-9);
}

// ---------------------------------------------------------------------------
// Train → evaluate: BW improves log-likelihood
// ---------------------------------------------------------------------------

TEST(CasinoEndToEnd, BaumWelch_ImprovesLogLikelihood) {
    auto hmm = make_casino_hmm();

    ObservationLists obs;
    // Mix of sequences: one with many 6s (loaded) and one with diverse faces (fair)
    ObservationSet s1(30);
    for (std::size_t i = 0; i < 15; ++i)
        s1(i) = static_cast<double>(i % 6);
    for (std::size_t i = 15; i < 30; ++i)
        s1(i) = 5;
    obs.push_back(s1);
    ObservationSet s2(20);
    for (std::size_t i = 0; i < 20; ++i)
        s2(i) = static_cast<double>(i % 6);
    obs.push_back(s2);

    auto compute_total_ll = [&]() {
        double ll = 0.0;
        for (const auto &seq : obs) {
            ForwardBackwardCalculator fbc(*hmm, seq);
            ll += fbc.getLogProbability();
        }
        return ll;
    };

    const double ll_before = compute_total_ll();
    BaumWelchTrainer trainer(hmm.get(), obs);
    for (int i = 0; i < 5; ++i)
        trainer.train();
    const double ll_after = compute_total_ll();

    EXPECT_GE(ll_after, ll_before - 1e-6);
    EXPECT_NO_THROW(hmm->validate());
}

// ---------------------------------------------------------------------------
// Train → decode consistency: trained HMM still produces valid paths
// ---------------------------------------------------------------------------

TEST(CasinoEndToEnd, TrainThenDecode_ValidPath) {
    auto hmm = make_casino_hmm();

    ObservationLists obs;
    ObservationSet seq(15);
    for (std::size_t i = 0; i < 15; ++i)
        seq(i) = static_cast<double>(i % 6);
    obs.push_back(seq);

    BaumWelchTrainer trainer(hmm.get(), obs);
    trainer.train();

    ViterbiCalculator vc(*hmm, seq);
    const auto path = vc.decode();

    EXPECT_EQ(path.size(), seq.size());
    for (std::size_t i = 0; i < path.size(); ++i) {
        EXPECT_GE(path(i), 0);
        EXPECT_LT(path(i), static_cast<int>(hmm->getNumStatesModern()));
    }
}

// ---------------------------------------------------------------------------
// Gaussian end-to-end: train with BW, then evaluate
// ---------------------------------------------------------------------------

TEST(GaussianEndToEnd, TrainEvaluatePipeline) {
    Hmm hmm(2);
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
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.5));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.5));

    ObservationLists obs;
    ObservationSet seq(40);
    for (std::size_t i = 0; i < 20; ++i)
        seq(i) = static_cast<double>(i % 5) * 0.3;
    for (std::size_t i = 20; i < 40; ++i)
        seq(i) = 5.0 + static_cast<double>(i % 5) * 0.3;
    obs.push_back(seq);

    ForwardBackwardCalculator fbc_before(hmm, seq);
    const double ll_before = fbc_before.getLogProbability();

    BaumWelchTrainer trainer(&hmm, obs);
    for (int i = 0; i < 5; ++i)
        trainer.train();

    ForwardBackwardCalculator fbc_after(hmm, seq);
    const double ll_after = fbc_after.getLogProbability();

    EXPECT_TRUE(std::isfinite(ll_after));
    EXPECT_GE(ll_after, ll_before - 1e-6);
    EXPECT_NO_THROW(hmm.validate());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
