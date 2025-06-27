#include <gtest/gtest.h>
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/calculators/scaled_simd_viterbi_calculator.h"
#include "libhmm/calculators/log_simd_viterbi_calculator.h"
#include "libhmm/two_state_hmm.h"
#include <memory>
#include <cmath>

using namespace libhmm;

class ViterbiCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a standard two-state HMM for testing
        hmm_ = createTwoStateHmm();
        
        // Create test observation sequence
        observations_ = ObservationSet(5);
        observations_(0) = 0; // fair die outcomes
        observations_(1) = 1;
        observations_(2) = 5; // loaded die outcome
        observations_(3) = 4;
        observations_(4) = 2;
    }

    std::unique_ptr<Hmm> hmm_;
    ObservationSet observations_;
};

// Standard Viterbi Calculator Tests
TEST_F(ViterbiCalculatorTest, ViterbiBasicFunctionality) {
    ViterbiCalculator vc(hmm_.get(), observations_);
    
    // Test decoding
    StateSequence sequence = vc.decode();
    EXPECT_EQ(sequence.size(), observations_.size());
    
    // Test that all states are valid
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_GE(sequence(i), 0);
        EXPECT_LT(sequence(i), hmm_->getNumStates());
    }
    
    // Test log probability
    double logProb = vc.getLogProbability();
    EXPECT_TRUE(std::isfinite(logProb)); // Should be finite
    
    // Test that we can get the sequence again
    StateSequence sequence2 = vc.getStateSequence();
    EXPECT_EQ(sequence.size(), sequence2.size());
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_EQ(sequence(i), sequence2(i));
    }
}

TEST_F(ViterbiCalculatorTest, ViterbiNullHmmThrows) {
    EXPECT_THROW(ViterbiCalculator(nullptr, observations_), std::invalid_argument);
}

// Scaled SIMD Viterbi Calculator Tests
TEST_F(ViterbiCalculatorTest, ScaledSIMDViterbiCalculatorFunctionality) {
    ScaledSIMDViterbiCalculator svc(hmm_.get(), observations_);
    
    // Test decoding
    StateSequence sequence = svc.decode();
    EXPECT_EQ(sequence.size(), observations_.size());
    
    // Test that all states are valid
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_GE(sequence(i), 0);
        EXPECT_LT(sequence(i), hmm_->getNumStates());
    }
    
    // Test log probability
    double logProb = svc.getLogProbability();
    EXPECT_TRUE(std::isfinite(logProb)); // Should be finite
    
    // Compare with standard Viterbi for consistency
    ViterbiCalculator vc(hmm_.get(), observations_);
    StateSequence standardSequence = vc.decode();
    double standardLogProb = vc.getLogProbability();
    
    // Should give approximately the same result
    EXPECT_NEAR(logProb, standardLogProb, 1e-10);
    
    // Sequences should be identical (deterministic algorithm)
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_EQ(sequence(i), standardSequence(i));
    }
}

// Log SIMD Viterbi Calculator Tests
TEST_F(ViterbiCalculatorTest, LogSIMDViterbiCalculatorFunctionality) {
    LogSIMDViterbiCalculator lvc(hmm_.get(), observations_);
    
    // Test decoding
    StateSequence sequence = lvc.decode();
    EXPECT_EQ(sequence.size(), observations_.size());
    
    // Test that all states are valid
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_GE(sequence(i), 0);
        EXPECT_LT(sequence(i), hmm_->getNumStates());
    }
    
    // Test log probability
    double logProb = lvc.getLogProbability();
    EXPECT_TRUE(std::isfinite(logProb)); // Should be finite
    
    // Compare with standard Viterbi for consistency
    ViterbiCalculator vc(hmm_.get(), observations_);
    StateSequence standardSequence = vc.decode();
    double standardLogProb = vc.getLogProbability();
    
    // Should give approximately the same result
    EXPECT_NEAR(logProb, standardLogProb, 1e-10);
    
    // Sequences should be identical (deterministic algorithm)
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_EQ(sequence(i), standardSequence(i));
    }
}

// Cross-Calculator Consistency Tests for Viterbi
TEST_F(ViterbiCalculatorTest, ViterbiCalculatorConsistency) {
    ViterbiCalculator vc(hmm_.get(), observations_);
    ScaledSIMDViterbiCalculator svc(hmm_.get(), observations_);
    LogSIMDViterbiCalculator lvc(hmm_.get(), observations_);
    
    StateSequence standardSeq = vc.decode();
    StateSequence scaledSeq = svc.decode();
    StateSequence logSeq = lvc.decode();
    
    double standardLogProb = vc.getLogProbability();
    double scaledLogProb = svc.getLogProbability();
    double logLogProb = lvc.getLogProbability();
    
    // All should give approximately the same result
    EXPECT_NEAR(standardLogProb, scaledLogProb, 1e-10);
    EXPECT_NEAR(standardLogProb, logLogProb, 1e-10);
    
    // Sequences should be identical (deterministic algorithm)
    for (std::size_t i = 0; i < standardSeq.size(); ++i) {
        EXPECT_EQ(standardSeq(i), scaledSeq(i));
        EXPECT_EQ(standardSeq(i), logSeq(i));
    }
}

// Edge Case Tests for Viterbi
TEST_F(ViterbiCalculatorTest, SingleObservation) {
    ObservationSet singleObs(1);
    singleObs(0) = 2;
    
    ViterbiCalculator vc(hmm_.get(), singleObs);
    StateSequence seq = vc.decode();
    EXPECT_EQ(seq.size(), 1u);
    
    // Test SIMD variants as well
    ScaledSIMDViterbiCalculator svc(hmm_.get(), singleObs);
    StateSequence scaledSeq = svc.decode();
    EXPECT_EQ(scaledSeq.size(), 1u);
    EXPECT_EQ(seq(0), scaledSeq(0));
    
    LogSIMDViterbiCalculator lvc(hmm_.get(), singleObs);
    StateSequence logSeq = lvc.decode();
    EXPECT_EQ(logSeq.size(), 1u);
    EXPECT_EQ(seq(0), logSeq(0));
}

TEST_F(ViterbiCalculatorTest, DISABLED_EmptyObservationSequence) {
    ObservationSet emptyObs(0);
    
    // For empty sequences, test that Viterbi calculators handle them appropriately
    try {
        ViterbiCalculator vc(hmm_.get(), emptyObs);
        StateSequence seq = vc.decode();
        EXPECT_EQ(seq.size(), 0u);
    } catch (const std::exception&) {
        // It's also valid to throw for empty sequences
        SUCCEED();
    }
}

// Performance and Numerical Stability Tests for Viterbi
TEST_F(ViterbiCalculatorTest, LongSequenceStability) {
    // Create a longer sequence that might cause numerical issues
    ObservationSet longObs(100);
    for (std::size_t i = 0; i < longObs.size(); ++i) {
        longObs(i) = i % 6; // Cycle through dice outcomes
    }
    
    // SIMD versions should handle this well
    EXPECT_NO_THROW(ScaledSIMDViterbiCalculator(hmm_.get(), longObs));
    EXPECT_NO_THROW(LogSIMDViterbiCalculator(hmm_.get(), longObs));
    
    // And they should give consistent results
    ViterbiCalculator vc(hmm_.get(), longObs);
    ScaledSIMDViterbiCalculator svc(hmm_.get(), longObs);
    LogSIMDViterbiCalculator lvc(hmm_.get(), longObs);
    
    StateSequence standardSeq = vc.decode();
    StateSequence scaledSeq = svc.decode();
    StateSequence logSeq = lvc.decode();
    
    // Sequences should be identical for this deterministic algorithm
    for (std::size_t i = 0; i < standardSeq.size(); ++i) {
        EXPECT_EQ(standardSeq(i), scaledSeq(i));
        EXPECT_EQ(standardSeq(i), logSeq(i));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
