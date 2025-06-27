#include <gtest/gtest.h>
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/calculators/scaled_simd_viterbi_calculator.h"
#include "libhmm/calculators/log_simd_viterbi_calculator.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <memory>
#include <cmath>

using namespace libhmm;

class SIMDViterbiCalculatorTest : public ::testing::Test {
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

// Test basic functionality of ScaledSIMDViterbiCalculator
TEST_F(SIMDViterbiCalculatorTest, ScaledSIMDViterbiBasicFunctionality) {
    ScaledSIMDViterbiCalculator calc(gaussianHmm_.get(), continuousObs_);
    
    // Decode should not throw and return a sequence
    StateSequence sequence = calc.decode();
    
    EXPECT_EQ(sequence.size(), continuousObs_.size());
    
    // Check that all states are valid
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_GE(sequence(i), 0);
        EXPECT_LT(sequence(i), gaussianHmm_->getNumStates());
    }
    
    // Check that log probability is reasonable
    double logProb = calc.getLogProbability();
    EXPECT_TRUE(std::isfinite(logProb));
    EXPECT_LT(logProb, 0.0); // Log probability should be negative
}

// Test basic functionality of LogSIMDViterbiCalculator
TEST_F(SIMDViterbiCalculatorTest, LogSIMDViterbiBasicFunctionality) {
    LogSIMDViterbiCalculator calc(gaussianHmm_.get(), continuousObs_);
    
    // Decode should not throw and return a sequence
    StateSequence sequence = calc.decode();
    
    EXPECT_EQ(sequence.size(), continuousObs_.size());
    
    // Check that all states are valid
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        EXPECT_GE(sequence(i), 0);
        EXPECT_LT(sequence(i), gaussianHmm_->getNumStates());
    }
    
    // Check that log probability is reasonable
    double logProb = calc.getLogProbability();
    EXPECT_TRUE(std::isfinite(logProb));
    EXPECT_LT(logProb, 0.0); // Log probability should be negative
}

// Test consistency between standard and SIMD Viterbi implementations
TEST_F(SIMDViterbiCalculatorTest, CalculatorConsistency) {
    ViterbiCalculator standardCalc(gaussianHmm_.get(), continuousObs_);
    ScaledSIMDViterbiCalculator scaledSIMDCalc(gaussianHmm_.get(), continuousObs_);
    LogSIMDViterbiCalculator logSIMDCalc(gaussianHmm_.get(), continuousObs_);
    
    // Get results from all calculators
    StateSequence standardSeq = standardCalc.decode();
    StateSequence scaledSIMDSeq = scaledSIMDCalc.decode();
    StateSequence logSIMDSeq = logSIMDCalc.decode();
    
    double standardLogProb = standardCalc.getLogProbability();
    double scaledSIMDLogProb = scaledSIMDCalc.getLogProbability();
    double logSIMDLogProb = logSIMDCalc.getLogProbability();
    
    // All sequences should have the same length
    EXPECT_EQ(standardSeq.size(), scaledSIMDSeq.size());
    EXPECT_EQ(standardSeq.size(), logSIMDSeq.size());
    
    // State sequences should be the same (or very similar for this simple case)
    // Note: Due to numerical differences, we allow some variation
    int differences = 0;
    for (std::size_t i = 0; i < standardSeq.size(); ++i) {
        if (standardSeq(i) != scaledSIMDSeq(i)) differences++;
    }
    EXPECT_LE(differences, 2); // Allow at most 2 differences
    
    differences = 0;
    for (std::size_t i = 0; i < standardSeq.size(); ++i) {
        if (standardSeq(i) != logSIMDSeq(i)) differences++;
    }
    EXPECT_LE(differences, 2); // Allow at most 2 differences
    
    // Log probabilities should be close
    EXPECT_NEAR(standardLogProb, scaledSIMDLogProb, 1e-6);
    EXPECT_NEAR(standardLogProb, logSIMDLogProb, 1e-6);
}

// Test with discrete distributions
TEST_F(SIMDViterbiCalculatorTest, DiscreteDistributionSupport) {
    ScaledSIMDViterbiCalculator scaledCalc(discreteHmm_.get(), discreteObs_);
    LogSIMDViterbiCalculator logCalc(discreteHmm_.get(), discreteObs_);
    
    // Both should work with discrete distributions
    StateSequence scaledSeq = scaledCalc.decode();
    StateSequence logSeq = logCalc.decode();
    
    EXPECT_EQ(scaledSeq.size(), discreteObs_.size());
    EXPECT_EQ(logSeq.size(), discreteObs_.size());
    
    // Verify sequences are reasonable
    for (std::size_t i = 0; i < scaledSeq.size(); ++i) {
        EXPECT_GE(scaledSeq(i), 0);
        EXPECT_LT(scaledSeq(i), discreteHmm_->getNumStates());
        EXPECT_GE(logSeq(i), 0);
        EXPECT_LT(logSeq(i), discreteHmm_->getNumStates());
    }
}

// Test SIMD availability reporting
TEST_F(SIMDViterbiCalculatorTest, SIMDAvailabilityReporting) {
    bool scaledSIMDEnabled = ScaledSIMDViterbiCalculator::isSIMDEnabled();
    bool logSIMDEnabled = LogSIMDViterbiCalculator::isSIMDEnabled();
    
    // Both should report the same SIMD availability
    EXPECT_EQ(scaledSIMDEnabled, logSIMDEnabled);
    
    // On modern x86_64 systems, SIMD should be available
    // On other platforms, it may or may not be available
    std::cout << "SIMD Support Available: " << (scaledSIMDEnabled ? "Yes" : "No") << std::endl;
}

// Test scaling factors (ScaledSIMDViterbiCalculator specific)
TEST_F(SIMDViterbiCalculatorTest, ScalingFactorsRetrieval) {
    ScaledSIMDViterbiCalculator calc(gaussianHmm_.get(), continuousObs_);
    
    calc.decode();
    
    std::vector<double> scalingFactors = calc.getScalingFactors();
    EXPECT_EQ(scalingFactors.size(), continuousObs_.size());
    
    // All scaling factors should be positive
    for (double factor : scalingFactors) {
        EXPECT_GT(factor, 0.0);
    }
}

// Test LogSIMD calculator specific features
TEST_F(SIMDViterbiCalculatorTest, LogSIMDSpecificFeatures) {
    double logZero = LogSIMDViterbiCalculator::getLogZero();
    // Should be a very negative number, consistent with consolidated constants
    EXPECT_LT(logZero, -500.0); // Updated to match MIN_LOG_PROBABILITY (-700)
    EXPECT_TRUE(std::isfinite(logZero));
    // Verify it matches our consolidated constant
    EXPECT_EQ(logZero, constants::probability::MIN_LOG_PROBABILITY);
}

// Test error handling
TEST_F(SIMDViterbiCalculatorTest, ErrorHandling) {
    // Test with null HMM
    EXPECT_THROW(ScaledSIMDViterbiCalculator(nullptr, continuousObs_), std::invalid_argument);
    EXPECT_THROW(LogSIMDViterbiCalculator(nullptr, continuousObs_), std::invalid_argument);
    
    // Test with empty observations
    ObservationSet emptyObs(0);
    EXPECT_THROW(ScaledSIMDViterbiCalculator(gaussianHmm_.get(), emptyObs), std::invalid_argument);
    EXPECT_THROW(LogSIMDViterbiCalculator(gaussianHmm_.get(), emptyObs), std::invalid_argument);
}

// Performance comparison test (informational)
TEST_F(SIMDViterbiCalculatorTest, PerformanceComparison) {
    const int numIterations = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Standard Viterbi
    for (int i = 0; i < numIterations; ++i) {
        ViterbiCalculator calc(gaussianHmm_.get(), continuousObs_);
        calc.decode();
    }
    
    auto standardEnd = std::chrono::high_resolution_clock::now();
    
    // Scaled SIMD Viterbi
    for (int i = 0; i < numIterations; ++i) {
        ScaledSIMDViterbiCalculator calc(gaussianHmm_.get(), continuousObs_);
        calc.decode();
    }
    
    auto scaledSIMDEnd = std::chrono::high_resolution_clock::now();
    
    // Log SIMD Viterbi
    for (int i = 0; i < numIterations; ++i) {
        LogSIMDViterbiCalculator calc(gaussianHmm_.get(), continuousObs_);
        calc.decode();
    }
    
    auto logSIMDEnd = std::chrono::high_resolution_clock::now();
    
    auto standardTime = std::chrono::duration_cast<std::chrono::microseconds>(standardEnd - start);
    auto scaledSIMDTime = std::chrono::duration_cast<std::chrono::microseconds>(scaledSIMDEnd - standardEnd);
    auto logSIMDTime = std::chrono::duration_cast<std::chrono::microseconds>(logSIMDEnd - scaledSIMDEnd);
    
    std::cout << "\nPerformance Comparison (" << numIterations << " iterations):" << std::endl;
    std::cout << "Standard Viterbi:     " << standardTime.count() << " μs" << std::endl;
    std::cout << "Scaled SIMD Viterbi:  " << scaledSIMDTime.count() << " μs" << std::endl;
    std::cout << "Log SIMD Viterbi:     " << logSIMDTime.count() << " μs" << std::endl;
    
    // All methods should complete in reasonable time
    EXPECT_LT(standardTime.count(), 100000); // Less than 100ms total
    EXPECT_LT(scaledSIMDTime.count(), 100000);
    EXPECT_LT(logSIMDTime.count(), 100000);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
