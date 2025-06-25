#include <gtest/gtest.h>
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/uniform_distribution.h"
#include <memory>
#include <cmath>

using namespace libhmm;

class SIMDForwardBackwardCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 2-state HMM with Gaussian distributions
        createGaussianHmm();
        
        // Create a simple 2-state HMM with discrete distributions
        createDiscreteHmm();
        
        // Create test observations
        createTestObservations();
    }
    
    void createGaussianHmm() {
        gaussianHmm_ = std::make_unique<Hmm>(2);
        
        // Set transition matrix
        Matrix trans(2, 2);
        trans(0, 0) = 0.7; trans(0, 1) = 0.3;
        trans(1, 0) = 0.4; trans(1, 1) = 0.6;
        gaussianHmm_->setTrans(trans);
        
        // Set initial state probabilities
        Vector pi(2);
        pi(0) = 0.6; pi(1) = 0.4;
        gaussianHmm_->setPi(pi);
        
        // Set Gaussian emission distributions
        auto gauss0 = std::make_unique<GaussianDistribution>(0.0, 1.0);
        auto gauss1 = std::make_unique<GaussianDistribution>(3.0, 1.5);
        gaussianHmm_->setProbabilityDistribution(0, std::move(gauss0));
        gaussianHmm_->setProbabilityDistribution(1, std::move(gauss1));
    }
    
    void createDiscreteHmm() {
        discreteHmm_ = std::make_unique<Hmm>(2);
        
        // Set transition matrix
        Matrix trans(2, 2);
        trans(0, 0) = 0.8; trans(0, 1) = 0.2;
        trans(1, 0) = 0.3; trans(1, 1) = 0.7;
        discreteHmm_->setTrans(trans);
        
        // Set initial state probabilities
        Vector pi(2);
        pi(0) = 0.5; pi(1) = 0.5;
        discreteHmm_->setPi(pi);
        
        // Set discrete emission distributions
        auto disc0 = std::make_unique<DiscreteDistribution>(3);
        disc0->setProbability(0, 0.6);
        disc0->setProbability(1, 0.3);
        disc0->setProbability(2, 0.1);
        
        auto disc1 = std::make_unique<DiscreteDistribution>(3);
        disc1->setProbability(0, 0.1);
        disc1->setProbability(1, 0.4);
        disc1->setProbability(2, 0.5);
        
        discreteHmm_->setProbabilityDistribution(0, std::move(disc0));
        discreteHmm_->setProbabilityDistribution(1, std::move(disc1));
    }
    
    void createTestObservations() {
        // Continuous observations for Gaussian HMM
        continuousObs_.resize(10);
        continuousObs_(0) = 0.1;  // Likely state 0
        continuousObs_(1) = 2.8;  // Likely state 1
        continuousObs_(2) = -0.5; // Likely state 0
        continuousObs_(3) = 3.2;  // Likely state 1
        continuousObs_(4) = 0.3;  // Likely state 0
        continuousObs_(5) = 2.9;  // Likely state 1
        continuousObs_(6) = -0.2; // Likely state 0
        continuousObs_(7) = 3.1;  // Likely state 1
        continuousObs_(8) = 0.0;  // Likely state 0
        continuousObs_(9) = 2.7;  // Likely state 1
        
        // Discrete observations for discrete HMM
        discreteObs_.resize(8);
        discreteObs_(0) = 0; // Likely state 0
        discreteObs_(1) = 2; // Likely state 1
        discreteObs_(2) = 0; // Likely state 0
        discreteObs_(3) = 2; // Likely state 1
        discreteObs_(4) = 1; // Neutral
        discreteObs_(5) = 2; // Likely state 1
        discreteObs_(6) = 0; // Likely state 0
        discreteObs_(7) = 1; // Neutral
    }
    
    std::unique_ptr<Hmm> gaussianHmm_;
    std::unique_ptr<Hmm> discreteHmm_;
    ObservationSet continuousObs_;
    ObservationSet discreteObs_;
};

// Test basic functionality of ScaledSIMDForwardBackwardCalculator
TEST_F(SIMDForwardBackwardCalculatorTest, ScaledSIMDForwardBackwardBasicFunctionality) {
    ScaledSIMDForwardBackwardCalculator calc(gaussianHmm_.get(), continuousObs_);
    
    // Compute forward and backward variables
    calc.compute();
    
    // Get probability
    double prob = calc.getProbability();
    EXPECT_GT(prob, 0.0);
    EXPECT_LE(prob, 1.0);
    
    // Check log probability
    double logProb = calc.getLogProbability();
    EXPECT_TRUE(std::isfinite(logProb));
    EXPECT_LE(logProb, 0.0);
    
    // Test that we can get forward and backward variables
    Matrix forwardVars = calc.getForwardVariables();
    Matrix backwardVars = calc.getBackwardVariables();
    
    EXPECT_EQ(forwardVars.size1(), continuousObs_.size());
    EXPECT_EQ(forwardVars.size2(), gaussianHmm_->getNumStates());
    EXPECT_EQ(backwardVars.size1(), continuousObs_.size());
    EXPECT_EQ(backwardVars.size2(), gaussianHmm_->getNumStates());
}

// Test basic functionality of LogSIMDForwardBackwardCalculator
TEST_F(SIMDForwardBackwardCalculatorTest, LogSIMDForwardBackwardBasicFunctionality) {
    LogSIMDForwardBackwardCalculator calc(discreteHmm_.get(), discreteObs_);
    
    // Compute forward and backward variables
    calc.compute();
    
    // Get probability
    double prob = calc.getProbability();
    EXPECT_GT(prob, 0.0);
    EXPECT_LE(prob, 1.0);
    
    // Check log probability
    double logProb = calc.getLogProbability();
    EXPECT_TRUE(std::isfinite(logProb));
    EXPECT_LE(logProb, 0.0);
    
    // Test that we can get log forward and backward variables
    Matrix logForwardVars = calc.getLogForwardVariables();
    Matrix logBackwardVars = calc.getLogBackwardVariables();
    
    EXPECT_EQ(logForwardVars.size1(), discreteObs_.size());
    EXPECT_EQ(logForwardVars.size2(), discreteHmm_->getNumStates());
    EXPECT_EQ(logBackwardVars.size1(), discreteObs_.size());
    EXPECT_EQ(logBackwardVars.size2(), discreteHmm_->getNumStates());
}

// Test consistency between calculators using a hybrid approach
TEST_F(SIMDForwardBackwardCalculatorTest, CalculatorConsistency) {
    // Part 1: Test SIMD calculator consistency with each other on challenging Gaussian problem
    // (Both SIMD calculators should handle numerical stability well)
    {
        ScaledSIMDForwardBackwardCalculator scaledSIMDCalc(gaussianHmm_.get(), continuousObs_);
        LogSIMDForwardBackwardCalculator logSIMDCalc(gaussianHmm_.get(), continuousObs_);
        
        // Compute SIMD probabilities
        scaledSIMDCalc.compute();
        logSIMDCalc.compute();
        
        double scaledSIMDProb = scaledSIMDCalc.getProbability();
        double logSIMDProb = logSIMDCalc.getProbability();
        double scaledSIMDLogProb = scaledSIMDCalc.getLogProbability();
        double logSIMDLogProb = logSIMDCalc.getLogProbability();
        
        // SIMD calculators should agree with each other on probabilities
        EXPECT_NEAR(scaledSIMDProb, logSIMDProb, 1e-10) 
            << "SIMD calculators should produce consistent probabilities";
        EXPECT_NEAR(scaledSIMDLogProb, logSIMDLogProb, 1e-10)
            << "SIMD calculators should produce consistent log probabilities";
    }
    
    // Part 2: Test standard calculator vs SIMD calculators on numerically stable discrete problem
    // (Discrete HMM with short sequence should be stable for all calculators)
    {
        ForwardBackwardCalculator standardCalc(discreteHmm_.get(), discreteObs_);
        ScaledSIMDForwardBackwardCalculator scaledSIMDCalc(discreteHmm_.get(), discreteObs_);
        LogSIMDForwardBackwardCalculator logSIMDCalc(discreteHmm_.get(), discreteObs_);
        
        // Compute probabilities
        double standardProb = standardCalc.probability();
        
        scaledSIMDCalc.compute();
        logSIMDCalc.compute();
        
        double scaledSIMDProb = scaledSIMDCalc.getProbability();
        double logSIMDProb = logSIMDCalc.getProbability();
        
        // All calculators should agree on discrete problems
        EXPECT_NEAR(standardProb, scaledSIMDProb, 1e-10)
            << "Standard and Scaled-SIMD should agree on stable discrete problem";
        EXPECT_NEAR(standardProb, logSIMDProb, 1e-10)
            << "Standard and Log-SIMD should agree on stable discrete problem";
        
        // Check log probabilities
        double standardLogProb = std::log(standardProb);
        double scaledSIMDLogProb = scaledSIMDCalc.getLogProbability();
        double logSIMDLogProb = logSIMDCalc.getLogProbability();
        
        EXPECT_NEAR(standardLogProb, scaledSIMDLogProb, 1e-10)
            << "Log probabilities should match for stable discrete problem";
        EXPECT_NEAR(standardLogProb, logSIMDLogProb, 1e-10)
            << "Log probabilities should match for stable discrete problem";
    }
}

// Test with discrete distributions
TEST_F(SIMDForwardBackwardCalculatorTest, DiscreteDistributionSupport) {
    ScaledSIMDForwardBackwardCalculator scaledCalc(discreteHmm_.get(), discreteObs_);
    LogSIMDForwardBackwardCalculator logCalc(discreteHmm_.get(), discreteObs_);
    
    // Compute probabilities
    scaledCalc.compute();
    logCalc.compute();
    
    // Both should compute probabilities
    double scaledProb = scaledCalc.getProbability();
    double logProb = logCalc.getProbability();

    EXPECT_GT(scaledProb, 0.0);
    EXPECT_LE(scaledProb, 1.0);

    EXPECT_GT(logProb, 0.0);
    EXPECT_LE(logProb, 1.0);
}

// Check if SIMD is available
TEST_F(SIMDForwardBackwardCalculatorTest, SIMDAvailabilityReporting) {
    EXPECT_TRUE(ScaledSIMDForwardBackwardCalculator::isSIMDEnabled());
    EXPECT_TRUE(LogSIMDForwardBackwardCalculator::isSIMDEnabled());
}

