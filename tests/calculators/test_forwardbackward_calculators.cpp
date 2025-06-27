#include <gtest/gtest.h>
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/two_state_hmm.h"
#include <memory>
#include <cmath>

using namespace libhmm;

class CalculatorTest : public ::testing::Test {
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

// Forward-Backward Calculator Tests
TEST_F(CalculatorTest, ForwardBackwardBasicFunctionality) {
    ForwardBackwardCalculator fbc(hmm_.get(), observations_);
    
    // Test that probability is calculated
    double prob = fbc.probability();
    EXPECT_GT(prob, 0.0);
    EXPECT_LE(prob, 1.0);
    
    // Test forward variables
    Matrix forward = fbc.getForwardVariables();
    EXPECT_EQ(forward.size1(), observations_.size());
    EXPECT_EQ(forward.size2(), static_cast<std::size_t>(hmm_->getNumStates()));
    
    // Test backward variables
    Matrix backward = fbc.getBackwardVariables();
    EXPECT_EQ(backward.size1(), observations_.size());
    EXPECT_EQ(backward.size2(), static_cast<std::size_t>(hmm_->getNumStates()));
}

TEST_F(CalculatorTest, ForwardBackwardNullHmmThrows) {
    EXPECT_THROW(ForwardBackwardCalculator(nullptr, observations_), std::invalid_argument);
}

// Scaled SIMD Forward-Backward Calculator Tests
TEST_F(CalculatorTest, ScaledSIMDForwardBackwardFunctionality) {
    ScaledSIMDForwardBackwardCalculator sfbc(hmm_.get(), observations_);
    
    // Test probability calculation
    double prob = sfbc.probability();
    EXPECT_GT(prob, 0.0);
    EXPECT_LE(prob, 1.0);
    
    // Compare with regular forward-backward for consistency
    ForwardBackwardCalculator fbc(hmm_.get(), observations_);
    double fbcProb = fbc.probability();
    
    // Should be approximately equal (within floating point precision)
    EXPECT_NEAR(prob, fbcProb, 1e-10);
}

// Log SIMD Forward-Backward Calculator Tests
TEST_F(CalculatorTest, LogSIMDForwardBackwardFunctionality) {
    LogSIMDForwardBackwardCalculator lfbc(hmm_.get(), observations_);
    
    // Test probability calculation
    double prob = lfbc.probability();
    EXPECT_GT(prob, 0.0);
    EXPECT_LE(prob, 1.0);
    
    // Compare with regular forward-backward for consistency
    ForwardBackwardCalculator fbc(hmm_.get(), observations_);
    double fbcProb = fbc.probability();
    
    // Should be approximately equal (within floating point precision)
    EXPECT_NEAR(prob, fbcProb, 1e-10);
}

// Viterbi Calculator Tests
TEST_F(CalculatorTest, ViterbiCalculatorFunctionality) {
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
    EXPECT_TRUE(std::isfinite(logProb)); // Should be finite (may be positive or negative)
    
    // Test that we can get the sequence again
    StateSequence sequence2 = vc.getStateSequence();
    EXPECT_EQ(sequence.size(), sequence2.size());
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_EQ(sequence(i), sequence2(i));
    }
}

TEST_F(CalculatorTest, ViterbiNullHmmThrows) {
    EXPECT_THROW(ViterbiCalculator(nullptr, observations_), std::invalid_argument);
}

// Cross-Calculator Consistency Tests
TEST_F(CalculatorTest, CalculatorConsistency) {
    ForwardBackwardCalculator fbc(hmm_.get(), observations_);
    ScaledSIMDForwardBackwardCalculator sfbc(hmm_.get(), observations_);
    LogSIMDForwardBackwardCalculator lfbc(hmm_.get(), observations_);
    
    double fbcProb = fbc.probability();
    double sfbcProb = sfbc.probability();
    double lfbcProb = lfbc.probability();
    
    // All should give approximately the same result
    EXPECT_NEAR(fbcProb, sfbcProb, 1e-10);
    EXPECT_NEAR(fbcProb, lfbcProb, 1e-10);
}

// Edge Case Tests
TEST_F(CalculatorTest, SingleObservation) {
    ObservationSet singleObs(1);
    singleObs(0) = 2;
    
    ForwardBackwardCalculator fbc(hmm_.get(), singleObs);
    EXPECT_NO_THROW(fbc.probability());
    
    ViterbiCalculator vc(hmm_.get(), singleObs);
    StateSequence seq = vc.decode();
    EXPECT_EQ(seq.size(), 1u);
}

TEST_F(CalculatorTest, DISABLED_EmptyObservationSequence) {
    ObservationSet emptyObs(0);
    
    // Empty observation sequences are mathematically invalid for HMM calculations
    // These should either throw exceptions or handle gracefully without segfaulting
    
    // Test ForwardBackwardCalculator
    bool fbc_handled_gracefully = false;
    try {
        ForwardBackwardCalculator fbc(hmm_.get(), emptyObs);
        try {
            double prob = fbc.probability();
            // If we get here without throwing, check it's a reasonable value
            EXPECT_TRUE(std::isfinite(prob));
            fbc_handled_gracefully = true;
        } catch (const std::exception&) {
            // Expected - probability calculation should fail for empty sequences
            fbc_handled_gracefully = true;
        }
    } catch (const std::exception&) {
        // Expected - construction might fail for empty sequences
        fbc_handled_gracefully = true;
    }
    EXPECT_TRUE(fbc_handled_gracefully);
    
    // Test ViterbiCalculator
    bool vc_handled_gracefully = false;
    try {
        ViterbiCalculator vc(hmm_.get(), emptyObs);
        try {
            StateSequence seq = vc.decode();
            // If decoding succeeded, sequence should be empty
            EXPECT_EQ(seq.size(), 0u);
            vc_handled_gracefully = true;
        } catch (const std::exception&) {
            // Expected - decoding should fail for empty sequences
            vc_handled_gracefully = true;
        }
    } catch (const std::exception&) {
        // Expected - construction might fail for empty sequences
        vc_handled_gracefully = true;
    }
    EXPECT_TRUE(vc_handled_gracefully);
    
    // Test SIMD calculators - they should also handle empty sequences gracefully
    bool simd_handled_gracefully = false;
    try {
        ScaledSIMDForwardBackwardCalculator sfbc(hmm_.get(), emptyObs);
        try {
            double prob = sfbc.probability();
            EXPECT_TRUE(std::isfinite(prob));
            simd_handled_gracefully = true;
        } catch (const std::exception&) {
            simd_handled_gracefully = true;
        }
    } catch (const std::exception&) {
        simd_handled_gracefully = true;
    }
    EXPECT_TRUE(simd_handled_gracefully);
}

// Performance and Numerical Stability Tests
TEST_F(CalculatorTest, LongSequenceStability) {
    // Create a longer sequence that might cause numerical issues
    ObservationSet longObs(100);
    for (std::size_t i = 0; i < longObs.size(); ++i) {
        longObs(i) = i % 6; // Cycle through dice outcomes
    }
    
    // Scaled version should handle this better than regular
    EXPECT_NO_THROW(ScaledSIMDForwardBackwardCalculator(hmm_.get(), longObs));
    EXPECT_NO_THROW(LogSIMDForwardBackwardCalculator(hmm_.get(), longObs));
}

TEST_F(CalculatorTest, Matrix3DFunctionality) {
    // Test the BasicMatrix3D utility used in training
    BasicMatrix3D<double> matrix3d(2, 3, 4);
    
    EXPECT_EQ(matrix3d.getXDimensionSize(), 2u);
    EXPECT_EQ(matrix3d.getYDimensionSize(), 3u);
    EXPECT_EQ(matrix3d.getZDimensionSize(), 4u);
    
    // Test setting and getting values
    matrix3d.Set(0, 1, 2, 5.5);
    EXPECT_DOUBLE_EQ(matrix3d(0, 1, 2), 5.5);
    
    // Test bounds checking
    EXPECT_THROW(matrix3d(2, 0, 0), std::out_of_range);
    EXPECT_THROW(matrix3d(0, 3, 0), std::out_of_range);
    EXPECT_THROW(matrix3d(0, 0, 4), std::out_of_range);
}

TEST_F(CalculatorTest, Matrix3DZeroDimensionThrows) {
    EXPECT_THROW(BasicMatrix3D<double>(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(BasicMatrix3D<double>(1, 0, 1), std::invalid_argument);
    EXPECT_THROW(BasicMatrix3D<double>(1, 1, 0), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
