#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/topology.h"
#include "libhmm/training/baum_welch_trainer.h"

using namespace libhmm;

namespace {

void expect_row_stochastic(const Matrix &a, std::size_t i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < a.size2(); ++j) {
        EXPECT_GE(a(i, j), 0.0);
        sum += a(i, j);
    }
    EXPECT_NEAR(sum, 1.0, 1e-12) << "row " << i;
}

} // namespace

// ---------------------------------------------------------------------------
// initialize_topology — mask shapes and stochasticity
// ---------------------------------------------------------------------------

TEST(TopologyInitTest, LeftToRightIsUpperTriangularStochastic) {
    Hmm hmm(4);
    initialize_topology(hmm, HmmTopology::LeftToRight);
    const Matrix &a = hmm.getTrans();
    for (std::size_t i = 0; i < 4; ++i) {
        expect_row_stochastic(a, i);
        for (std::size_t j = 0; j < 4; ++j) {
            if (j < i)
                EXPECT_EQ(a(i, j), 0.0) << i << "," << j;
            else
                EXPECT_GT(a(i, j), 0.0) << i << "," << j;
        }
    }
    // Last state can only self-loop.
    EXPECT_EQ(a(3, 3), 1.0);
}

TEST(TopologyInitTest, LeftToRightSkipBandsForward) {
    Hmm hmm(5);
    initialize_topology(hmm, HmmTopology::LeftToRightSkip, 2);
    const Matrix &a = hmm.getTrans();
    for (std::size_t i = 0; i < 5; ++i) {
        expect_row_stochastic(a, i);
        for (std::size_t j = 0; j < 5; ++j) {
            const bool valid = j >= i && j <= i + 2;
            EXPECT_EQ(a(i, j) > 0.0, valid) << i << "," << j;
        }
    }
}

TEST(TopologyInitTest, BandedReachesNeighboursOnly) {
    Hmm hmm(5);
    initialize_topology(hmm, HmmTopology::Banded, 1);
    const Matrix &a = hmm.getTrans();
    for (std::size_t i = 0; i < 5; ++i) {
        expect_row_stochastic(a, i);
        for (std::size_t j = 0; j < 5; ++j) {
            const std::size_t dist = (j >= i) ? j - i : i - j;
            EXPECT_EQ(a(i, j) > 0.0, dist <= 1) << i << "," << j;
        }
    }
}

TEST(TopologyInitTest, ErgodicIsUniform) {
    Hmm hmm(3);
    initialize_topology(hmm, HmmTopology::Ergodic);
    const Matrix &a = hmm.getTrans();
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(a(i, j), 1.0 / 3.0, 1e-15);
}

TEST(TopologyInitTest, InvalidMaxSkipThrows) {
    Hmm hmm(3);
    EXPECT_THROW(initialize_topology(hmm, HmmTopology::LeftToRightSkip, 0), std::invalid_argument);
    EXPECT_THROW(initialize_topology(hmm, HmmTopology::Banded, -1), std::invalid_argument);
    // max_skip is ignored (not validated) for topologies that don't use it.
    EXPECT_NO_THROW(initialize_topology(hmm, HmmTopology::LeftToRight, 0));
    EXPECT_NO_THROW(initialize_topology(hmm, HmmTopology::Ergodic, 0));
}

TEST(TopologyInitTest, WorksForMultivariateHmm) {
    HmmMV hmm(3);
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(2));
    hmm.setDistribution(2, std::make_unique<DiagonalGaussianDistribution>(2));
    initialize_topology(hmm, HmmTopology::LeftToRight);
    enforce_topology(hmm, HmmTopology::LeftToRight);
    const Matrix &a = hmm.getTrans();
    EXPECT_EQ(a(1, 0), 0.0);
    EXPECT_EQ(a(2, 2), 1.0);
}

// ---------------------------------------------------------------------------
// enforce_topology — masking, renormalisation, degenerate-row repair
// ---------------------------------------------------------------------------

TEST(TopologyEnforceTest, MasksAndRenormalisesCorruptedMatrix) {
    Hmm hmm(3);
    // Adversarial: a full uniform matrix, as the M-step's unvisited-state
    // fallback produces — every row violates left-to-right.
    Matrix uniform(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            uniform(i, j) = 1.0 / 3.0;
    hmm.setTrans(uniform);

    enforce_topology(hmm, HmmTopology::LeftToRight);
    const Matrix &a = hmm.getTrans();
    for (std::size_t i = 0; i < 3; ++i) {
        expect_row_stochastic(a, i);
        for (std::size_t j = 0; j < i; ++j)
            EXPECT_EQ(a(i, j), 0.0);
    }
    // Row 0 kept its relative valid masses (all equal) -> uniform over 3.
    EXPECT_NEAR(a(0, 1), 1.0 / 3.0, 1e-12);
    // Row 2's only valid entry absorbs everything.
    EXPECT_EQ(a(2, 2), 1.0);
}

TEST(TopologyEnforceTest, DegenerateRowResetToUniformOverValid) {
    Hmm hmm(3);
    // Row 1 has mass ONLY on invalid entries: after masking its valid mass
    // is zero and it must come back uniform over the valid set, not NaN.
    Matrix t(3, 3);
    t(0, 0) = 0.5;
    t(0, 1) = 0.25;
    t(0, 2) = 0.25;
    t(1, 0) = 1.0; // back-transition only
    t(2, 2) = 1.0;
    hmm.setTrans(t);

    enforce_topology(hmm, HmmTopology::LeftToRight);
    const Matrix &a = hmm.getTrans();
    EXPECT_EQ(a(1, 0), 0.0);
    EXPECT_NEAR(a(1, 1), 0.5, 1e-12);
    EXPECT_NEAR(a(1, 2), 0.5, 1e-12);
}

TEST(TopologyEnforceTest, ErgodicIsANoOp) {
    Hmm hmm(2);
    Matrix t(2, 2);
    t(0, 0) = 0.7;
    t(0, 1) = 0.3;
    t(1, 0) = 0.6;
    t(1, 1) = 0.4;
    hmm.setTrans(t);
    enforce_topology(hmm, HmmTopology::Ergodic);
    const Matrix &a = hmm.getTrans();
    EXPECT_EQ(a(0, 0), 0.7);
    EXPECT_EQ(a(1, 0), 0.6);
}

TEST(TopologyEnforceTest, PreservesAlreadyValidMatrix) {
    Hmm hmm(3);
    initialize_topology(hmm, HmmTopology::LeftToRightSkip, 1);
    const Matrix before = hmm.getTrans();
    enforce_topology(hmm, HmmTopology::LeftToRightSkip, 1);
    const Matrix &after = hmm.getTrans();
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(after(i, j), before(i, j), 1e-15);
}

// ---------------------------------------------------------------------------
// Acceptance (issue #46): a left-to-right HMM trained on sequential data
// never back-transitions, with the constraint held through every M-step.
// ---------------------------------------------------------------------------

TEST(TopologyTrainingTest, LeftToRightTrainingNeverBackTransitions) {
    Hmm hmm(3);
    initialize_topology(hmm, HmmTopology::LeftToRight);
    Vector pi(3);
    pi(0) = 1.0; // Bakis convention: start in state 0
    pi(1) = 0.0;
    pi(2) = 0.0;
    hmm.setPi(pi);
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.5, 1.5));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(4.5, 1.5));
    hmm.setDistribution(2, std::make_unique<GaussianDistribution>(9.5, 1.5));

    // Three-phase sequential data: ~0, then ~5, then ~10, small deterministic
    // jitter. Exactly the segmentation a left-to-right chain models.
    constexpr double jitter[5] = {-0.4, -0.15, 0.0, 0.2, 0.45};
    ObservationLists obs;
    for (int s = 0; s < 3; ++s) {
        ObservationSet seq(30);
        for (std::size_t t = 0; t < 30; ++t) {
            const double centre = (t < 10) ? 0.0 : (t < 20) ? 5.0 : 10.0;
            seq(t) = centre + jitter[(t + static_cast<std::size_t>(s)) % 5];
        }
        obs.push_back(seq);
    }

    BaumWelchTrainer trainer(hmm, obs);
    double firstLogL = 0.0;
    for (int iter = 0; iter < 15; ++iter) {
        trainer.train();
        enforce_topology(hmm, HmmTopology::LeftToRight);
        if (iter == 0)
            firstLogL = trainer.getLastLogProbability();

        // The constraint must hold after EVERY iteration, not just the last.
        const Matrix &a = hmm.getTrans();
        for (std::size_t i = 0; i < 3; ++i) {
            expect_row_stochastic(a, i);
            for (std::size_t j = 0; j < i; ++j)
                EXPECT_EQ(a(i, j), 0.0) << "back-transition at iter " << iter;
        }
    }
    // Training made progress on the segmentation.
    EXPECT_GT(trainer.getLastLogProbability(), firstLogL);
}
