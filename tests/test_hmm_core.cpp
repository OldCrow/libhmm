#include <gtest/gtest.h>
#include "libhmm/hmm.h"
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
    EXPECT_EQ(defaultHmm.getNumStates(), 4);
    EXPECT_NO_THROW(defaultHmm.validate());
}

TEST_F(HmmCoreTest, SizeConstructor) {
    Hmm hmm3(3);
    EXPECT_EQ(hmm3.getNumStates(), 3);
    
    Hmm hmm5(5);
    EXPECT_EQ(hmm5.getNumStates(), 5);
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
    
    const Vector& retrievedPi = hmm_->getPi();
    EXPECT_DOUBLE_EQ(retrievedPi(0), 0.6);
    EXPECT_DOUBLE_EQ(retrievedPi(1), 0.4);
}

TEST_F(HmmCoreTest, SetPiWrongSizeThrows) {
    Vector wrongSizePi(3); // Should be size 2
    EXPECT_THROW(hmm_->setPi(wrongSizePi), std::invalid_argument);
}

TEST_F(HmmCoreTest, SetTransitionMatrix) {
    Matrix trans(2, 2);
    trans(0, 0) = 0.7; trans(0, 1) = 0.3;
    trans(1, 0) = 0.4; trans(1, 1) = 0.6;
    
    EXPECT_NO_THROW(hmm_->setTrans(trans));
    
    const Matrix& retrievedTrans = hmm_->getTrans();
    EXPECT_DOUBLE_EQ(retrievedTrans(0, 0), 0.7);
    EXPECT_DOUBLE_EQ(retrievedTrans(0, 1), 0.3);
    EXPECT_DOUBLE_EQ(retrievedTrans(1, 0), 0.4);
    EXPECT_DOUBLE_EQ(retrievedTrans(1, 1), 0.6);
}

TEST_F(HmmCoreTest, SetTransWrongSizeThrows) {
    Matrix wrongSizeTrans(3, 2); // Should be 2x2
    EXPECT_THROW(hmm_->setTrans(wrongSizeTrans), std::invalid_argument);
}

// Probability Distribution Tests
TEST_F(HmmCoreTest, SetProbabilityDistributionModern) {
    auto gaussDist = std::make_unique<GaussianDistribution>(1.0, 2.0);
    auto* distPtr = gaussDist.get();
    
    EXPECT_NO_THROW(hmm_->setProbabilityDistribution(0, std::move(gaussDist)));
    
    const auto* retrievedDist = hmm_->getProbabilityDistribution(0);
    EXPECT_EQ(retrievedDist, distPtr);
}

TEST_F(HmmCoreTest, SetProbabilityDistributionLegacy) {
    auto* gaussDist = new GaussianDistribution(2.0, 1.5);
    
    EXPECT_NO_THROW(hmm_->setProbabilityDistribution(1, gaussDist));
    
    const auto* retrievedDist = hmm_->getProbabilityDistribution(1);
    EXPECT_NE(retrievedDist, nullptr);
}

TEST_F(HmmCoreTest, SetNullDistributionThrows) {
    EXPECT_THROW(hmm_->setProbabilityDistribution(0, nullptr), std::invalid_argument);
    
    ProbabilityDistribution* nullPtr = nullptr;
    EXPECT_THROW(hmm_->setProbabilityDistribution(0, nullPtr), std::invalid_argument);
}

TEST_F(HmmCoreTest, GetDistributionOutOfBoundsThrows) {
    EXPECT_THROW(hmm_->getProbabilityDistribution(2), std::out_of_range);
    EXPECT_THROW(hmm_->getProbabilityDistribution(10), std::out_of_range);
    EXPECT_THROW(hmm_->getProbabilityDistribution(-1), std::invalid_argument);
}

// Validation Tests
TEST_F(HmmCoreTest, ValidationPasses) {
    // Set up a valid HMM
    Vector pi(2);
    pi(0) = 0.5; pi(1) = 0.5;
    hmm_->setPi(pi);
    
    Matrix trans(2, 2);
    trans(0, 0) = 0.8; trans(0, 1) = 0.2;
    trans(1, 0) = 0.3; trans(1, 1) = 0.7;
    hmm_->setTrans(trans);
    
    EXPECT_NO_THROW(hmm_->validate());
}

// Move Semantics Tests
TEST_F(HmmCoreTest, MoveConstructor) {
    Hmm original(3);
    auto originalStates = original.getNumStates();
    
    Hmm moved = std::move(original);
    EXPECT_EQ(moved.getNumStates(), originalStates);
}

TEST_F(HmmCoreTest, MoveAssignment) {
    Hmm target(2);
    Hmm source(5);
    
    target = std::move(source);
    EXPECT_EQ(target.getNumStates(), 5);
}

// Legacy Compatibility Tests
TEST_F(HmmCoreTest, LegacyIntInterface) {
    EXPECT_EQ(hmm_->getNumStates(), 2);
    EXPECT_EQ(hmm_->getNumStatesModern(), 2u);
}

TEST_F(HmmCoreTest, BoundaryConditions) {
    // Test maximum reasonable size
    EXPECT_NO_THROW(Hmm(100));
    
    // Test edge case of 1 state
    Hmm singleState(1);
    EXPECT_EQ(singleState.getNumStates(), 1);
    EXPECT_NO_THROW(singleState.validate());
}

TEST_F(HmmCoreTest, TypeSafetyEdgeCases) {
    // Test size_t edge cases
    EXPECT_NO_THROW(hmm_->getProbabilityDistribution(static_cast<std::size_t>(0)));
    EXPECT_NO_THROW(hmm_->getProbabilityDistribution(static_cast<std::size_t>(1)));
    EXPECT_THROW(hmm_->getProbabilityDistribution(static_cast<std::size_t>(2)), std::out_of_range);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
