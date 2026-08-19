#include <gtest/gtest.h>
#include "libhmm/hmm.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <memory>
#include <stdexcept>

using namespace libhmm;

class HmmCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a basic 2-state HMM for testing
        hmm_ = std::make_unique<Hmm>(2);
    }

    std::unique_ptr<Hmm> hmm_;
};

// Constructor Tests
TEST_F(HmmCoreTest, DefaultConstructor) {
    Hmm defaultHmm;
    EXPECT_EQ(defaultHmm.getNumStatesModern(), 4u);
    EXPECT_NO_THROW(defaultHmm.validate());
}

TEST_F(HmmCoreTest, SizeConstructor) {
    Hmm hmm3(3);
    EXPECT_EQ(hmm3.getNumStatesModern(), 3u);

    Hmm hmm5(5);
    EXPECT_EQ(hmm5.getNumStatesModern(), 5u);
}

TEST_F(HmmCoreTest, ZeroStatesThrows) {
    EXPECT_THROW(Hmm(0), std::invalid_argument);
    EXPECT_THROW(Hmm(static_cast<std::size_t>(0)), std::invalid_argument);
}

TEST_F(HmmCoreTest, NegativeStatesThrows) {
    EXPECT_THROW(Hmm(-1), std::invalid_argument);
    EXPECT_THROW(Hmm(-10), std::invalid_argument);
}

// Matrix and Vector Operations
TEST_F(HmmCoreTest, SetPiVector) {
    Vector pi(2);
    pi(0) = 0.6;
    pi(1) = 0.4;

    EXPECT_NO_THROW(hmm_->setPi(pi));

    const Vector &retrievedPi = hmm_->getPi();
    EXPECT_DOUBLE_EQ(retrievedPi(0), 0.6);
    EXPECT_DOUBLE_EQ(retrievedPi(1), 0.4);
}

TEST_F(HmmCoreTest, SetPiWrongSizeThrows) {
    Vector wrongSizePi(3); // Should be size 2
    EXPECT_THROW(hmm_->setPi(wrongSizePi), std::invalid_argument);
}

TEST_F(HmmCoreTest, SetTransitionMatrix) {
    Matrix trans(2, 2);
    trans(0, 0) = 0.7;
    trans(0, 1) = 0.3;
    trans(1, 0) = 0.4;
    trans(1, 1) = 0.6;

    EXPECT_NO_THROW(hmm_->setTrans(trans));

    const Matrix &retrievedTrans = hmm_->getTrans();
    EXPECT_DOUBLE_EQ(retrievedTrans(0, 0), 0.7);
    EXPECT_DOUBLE_EQ(retrievedTrans(0, 1), 0.3);
    EXPECT_DOUBLE_EQ(retrievedTrans(1, 0), 0.4);
    EXPECT_DOUBLE_EQ(retrievedTrans(1, 1), 0.6);
}

TEST_F(HmmCoreTest, SetTransWrongSizeThrows) {
    Matrix wrongSizeTrans(3, 2); // Should be 2x2
    EXPECT_THROW(hmm_->setTrans(wrongSizeTrans), std::invalid_argument);
}

// Distribution setter/getter tests
TEST_F(HmmCoreTest, SetDistribution) {
    auto gaussDist = std::make_unique<GaussianDistribution>(1.0, 2.0);
    auto *distPtr = gaussDist.get();

    EXPECT_NO_THROW(hmm_->setDistribution(0, std::move(gaussDist)));

    const auto *retrievedDist = &hmm_->getDistribution(0);
    EXPECT_EQ(retrievedDist, distPtr);
}

TEST_F(HmmCoreTest, SetDistributionReplace) {
    // Replacing an existing distribution should succeed without memory issues.
    EXPECT_NO_THROW(hmm_->setDistribution(1, std::make_unique<GaussianDistribution>(2.0, 1.5)));
    EXPECT_NE(dynamic_cast<GaussianDistribution *>(&hmm_->getDistribution(1)), nullptr);
}

TEST_F(HmmCoreTest, SetNullDistributionThrows) {
    EXPECT_THROW(hmm_->setDistribution(0, std::unique_ptr<EmissionDistribution>{}),
                 std::invalid_argument);
}

TEST_F(HmmCoreTest, GetDistributionOutOfBoundsThrows) {
    EXPECT_THROW(hmm_->getDistribution(2), std::out_of_range);
    EXPECT_THROW(hmm_->getDistribution(10), std::out_of_range);
    // The API is size_t-only; verify a clearly out-of-range index is caught.
    EXPECT_THROW(hmm_->getDistribution(static_cast<std::size_t>(100)), std::out_of_range);
}

// Validation Tests
TEST_F(HmmCoreTest, ValidationPasses) {
    // Set up a valid HMM
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm_->setPi(pi);

    Matrix trans(2, 2);
    trans(0, 0) = 0.8;
    trans(0, 1) = 0.2;
    trans(1, 0) = 0.3;
    trans(1, 1) = 0.7;
    hmm_->setTrans(trans);

    EXPECT_NO_THROW(hmm_->validate());
}

// Move Semantics Tests
TEST_F(HmmCoreTest, MoveConstructor) {
    Hmm original(3);
    const auto originalStates = original.getNumStatesModern();

    Hmm moved = std::move(original);
    EXPECT_EQ(moved.getNumStatesModern(), originalStates);
}

TEST_F(HmmCoreTest, MoveAssignment) {
    Hmm target(2);
    Hmm source(5);

    target = std::move(source);
    EXPECT_EQ(target.getNumStatesModern(), 5u);
}

// Legacy Compatibility Tests
TEST_F(HmmCoreTest, ModernStateCountInterface) {
    EXPECT_EQ(hmm_->getNumStatesModern(), 2u);
}

TEST_F(HmmCoreTest, BoundaryConditions) {
    // Test maximum reasonable size
    EXPECT_NO_THROW(Hmm(100));

    // Test edge case of 1 state
    Hmm singleState(1);
    EXPECT_EQ(singleState.getNumStatesModern(), 1u);
    EXPECT_NO_THROW(singleState.validate());
}

TEST_F(HmmCoreTest, TypeSafetyEdgeCases) {
    EXPECT_NO_THROW(hmm_->getDistribution(static_cast<std::size_t>(0)));
    EXPECT_NO_THROW(hmm_->getDistribution(static_cast<std::size_t>(1)));
    EXPECT_THROW(hmm_->getDistribution(static_cast<std::size_t>(2)), std::out_of_range);
}

// ============================================================================
// MV HMM — null emission slot safety (regression for F2 bugfix)
// ============================================================================

TEST(MvHmmCoreTest, GetDistributionNullSlotThrows) {
    // BasicHmm<ObservationVectorView> leaves emission slots null until
    // setDistribution() is called.  getDistribution() must throw a descriptive
    // runtime_error rather than unconditionally dereferencing the null pointer.
    HmmMV hmm(2);
    // No setDistribution() called — both slots are null.
    // (void) cast silences the [[nodiscard]] warning inside EXPECT_THROW.
    EXPECT_THROW((void)hmm.getDistribution(0), std::runtime_error);
    EXPECT_THROW((void)hmm.getDistribution(1), std::runtime_error);
    // After setDistribution, the slot must no longer throw.
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2));
    EXPECT_NO_THROW((void)hmm.getDistribution(0));
    // Slot 1 is still null.
    EXPECT_THROW((void)hmm.getDistribution(1), std::runtime_error);
}

TEST(MvHmmCoreTest, GetDistributionConstNullSlotThrows) {
    HmmMV hmm(2);
    const HmmMV &chmm = hmm;
    EXPECT_THROW((void)chmm.getDistribution(0), std::runtime_error);
}

// ============================================================================
// clone() — explicit deep copy (issue #43)
// ============================================================================

TEST(HmmCloneTest, CloneIsDeepAndIndependent) {
    Hmm original(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.9;
    trans(0, 1) = 0.1;
    trans(1, 0) = 0.2;
    trans(1, 1) = 0.8;
    original.setTrans(trans);
    Vector pi(2);
    pi(0) = 0.6;
    pi(1) = 0.4;
    original.setPi(pi);
    original.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    original.setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 2.0));

    Hmm copy = original.clone();

    // The copy starts equal...
    EXPECT_EQ(copy.getNumStatesModern(), original.getNumStatesModern());
    EXPECT_DOUBLE_EQ(copy.getTrans()(0, 0), 0.9);
    EXPECT_DOUBLE_EQ(copy.getPi()(0), 0.6);
    EXPECT_DOUBLE_EQ(copy.getDistribution(0).getLogProbability(0.0),
                     original.getDistribution(0).getLogProbability(0.0));

    // ...and mutating it does not touch the original: matrices,
    Matrix trans2(2, 2);
    trans2(0, 0) = 0.5;
    trans2(0, 1) = 0.5;
    trans2(1, 0) = 0.5;
    trans2(1, 1) = 0.5;
    copy.setTrans(trans2);
    Vector pi2(2);
    pi2(0) = 0.1;
    pi2(1) = 0.9;
    copy.setPi(pi2);
    EXPECT_DOUBLE_EQ(original.getTrans()(0, 0), 0.9);
    EXPECT_DOUBLE_EQ(original.getPi()(0), 0.6);

    // ...and distributions (replace state 0's on the copy; original keeps its own object).
    copy.setDistribution(0, std::make_unique<GaussianDistribution>(100.0, 1.0));
    EXPECT_NE(&original.getDistribution(0), &copy.getDistribution(0));
    EXPECT_DOUBLE_EQ(original.getDistribution(0).getLogProbability(0.0),
                     GaussianDistribution(0.0, 1.0).getLogProbability(0.0));
}

TEST(HmmCloneTest, CloneMVDeepCopiesSetSlotsAndKeepsNullSlotsNull) {
    HmmMV original(2);
    original.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2));
    // Slot 1 deliberately left null (pre-setDistribution state).

    HmmMV copy = original.clone();

    // Set slot deep-copied: distinct objects, both usable.
    EXPECT_NO_THROW((void)copy.getDistribution(0));
    EXPECT_NE(&original.getDistribution(0), &copy.getDistribution(0));
    // Null slot stays null in the copy, and stays null in the original.
    EXPECT_THROW((void)copy.getDistribution(1), std::runtime_error);
    EXPECT_THROW((void)original.getDistribution(1), std::runtime_error);

    // clone_hmm convenience alias compiles and deep-copies for both aliases.
    HmmMV copy2 = clone_hmm(original);
    EXPECT_NE(&original.getDistribution(0), &copy2.getDistribution(0));
    Hmm scalar(2);
    Hmm scalarCopy = clone_hmm(scalar);
    EXPECT_EQ(scalarCopy.getNumStatesModern(), 2u);
}

// ============================================================================
// sample() — HMM-level sequence generation (issue #44)
// ============================================================================

TEST(HmmSampleTest, ScalarSampleShapesStatesAndStatistics) {
    Hmm hmm(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.9;
    trans(0, 1) = 0.1;
    trans(1, 0) = 0.1;
    trans(1, 1) = 0.9;
    hmm.setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);
    // Well-separated emissions so per-state empirical means are unambiguous.
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(10.0, 1.0));

    std::mt19937_64 rng(20260818);
    const std::size_t T = 1000;
    auto [obs, states] = sample(hmm, T, rng);

    ASSERT_EQ(obs.size(), T);
    ASSERT_EQ(states.size(), T);

    // State path valid, both states visited, per-state means near the truth.
    double sum[2] = {0.0, 0.0};
    std::size_t cnt[2] = {0, 0};
    for (std::size_t t = 0; t < T; ++t) {
        ASSERT_GE(states(t), 0);
        ASSERT_LT(static_cast<std::size_t>(states(t)), 2u);
        sum[states(t)] += obs(t);
        ++cnt[states(t)];
    }
    ASSERT_GT(cnt[0], 100u); // sticky symmetric chain: both states well-visited
    ASSERT_GT(cnt[1], 100u);
    // sd = 1, so mean of >=100 draws is within ~0.3 at >3 sigma margin.
    EXPECT_NEAR(sum[0] / static_cast<double>(cnt[0]), 0.0, 0.5);
    EXPECT_NEAR(sum[1] / static_cast<double>(cnt[1]), 10.0, 0.5);
}

TEST(HmmSampleTest, ScalarSampleEdgeCases) {
    Hmm hmm(2);
    // Zero-length request: empty containers, no draws, no throw.
    std::mt19937_64 rng(1);
    // Parameters unset (all-zero pi): T=0 must not touch them...
    auto [obs0, st0] = sample(hmm, 0, rng);
    EXPECT_EQ(obs0.size(), 0u);
    EXPECT_EQ(st0.size(), 0u);
    // ...but any real draw from a zeroed pi must throw, not silently pick a state.
    EXPECT_THROW((void)sample(hmm, 5, rng), std::runtime_error);
}

TEST(HmmSampleTest, MVSampleShapesAndDimensions) {
    HmmMV hmm(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.8;
    trans(0, 1) = 0.2;
    trans(1, 0) = 0.2;
    trans(1, 1) = 0.8;
    hmm.setTrans(trans);
    Vector pi(2);
    pi(0) = 1.0;
    pi(1) = 0.0;
    hmm.setPi(pi);
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(2));

    std::mt19937_64 rng(7);
    const std::size_t T = 33; // non-round length
    auto [obs, states] = sample(hmm, T, rng);

    EXPECT_EQ(obs.size1(), T);
    EXPECT_EQ(obs.size2(), 2u);
    ASSERT_EQ(states.size(), T);
    EXPECT_EQ(states(0), 0); // pi is a point mass on state 0
    for (std::size_t t = 0; t < T; ++t) {
        ASSERT_GE(states(t), 0);
        ASSERT_LT(static_cast<std::size_t>(states(t)), 2u);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
