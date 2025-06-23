#include <gtest/gtest.h>
#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/scaled_baum_welch_trainer.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <memory>
#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>

using namespace libhmm;

class TrainingEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 3-state HMM with Gaussian distributions for edge case testing
        gaussianHmm_ = std::make_unique<Hmm>(3);
        setupGaussianHmm();
        
        // Create a 2-state HMM with discrete distributions
        discreteHmm_ = std::make_unique<Hmm>(2);
        setupDiscreteHmm();
    }
    
    void setupGaussianHmm() {
        // Set up transition matrix
        Matrix trans(3, 3);
        trans(0, 0) = 0.5; trans(0, 1) = 0.3; trans(0, 2) = 0.2;
        trans(1, 0) = 0.2; trans(1, 1) = 0.6; trans(1, 2) = 0.2;
        trans(2, 0) = 0.1; trans(2, 1) = 0.1; trans(2, 2) = 0.8;
        gaussianHmm_->setTrans(trans);
        
        // Set up pi vector
        Vector pi(3);
        pi(0) = 0.4; pi(1) = 0.3; pi(2) = 0.3;
        gaussianHmm_->setPi(pi);
        
        // Set up Gaussian emission distributions with different means
        gaussianHmm_->setProbabilityDistribution(0, std::make_unique<GaussianDistribution>(1.0, 0.5));
        gaussianHmm_->setProbabilityDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.0));
        gaussianHmm_->setProbabilityDistribution(2, std::make_unique<GaussianDistribution>(10.0, 2.0));
    }
    
    void setupDiscreteHmm() {
        // Set up transition matrix
        Matrix trans(2, 2);
        trans(0, 0) = 0.7; trans(0, 1) = 0.3;
        trans(1, 0) = 0.4; trans(1, 1) = 0.6;
        discreteHmm_->setTrans(trans);
        
        // Set up pi vector
        Vector pi(2);
        pi(0) = 0.6; pi(1) = 0.4;
        discreteHmm_->setPi(pi);
        
        // Set up discrete emission distributions
        auto dist0 = std::make_unique<DiscreteDistribution>(3);
        auto dist1 = std::make_unique<DiscreteDistribution>(3);
        
        dist0->setProbability(0, 0.7);
        dist0->setProbability(1, 0.2);
        dist0->setProbability(2, 0.1);
        
        dist1->setProbability(0, 0.1);
        dist1->setProbability(1, 0.3);
        dist1->setProbability(2, 0.6);
        
        discreteHmm_->setProbabilityDistribution(0, std::move(dist0));
        discreteHmm_->setProbabilityDistribution(1, std::move(dist1));
    }

    std::unique_ptr<Hmm> gaussianHmm_;
    std::unique_ptr<Hmm> discreteHmm_;
};

// Test empty cluster handling in ViterbiTrainer
TEST_F(TrainingEdgeCasesTest, ViterbiTrainerEmptyClusterHandling) {
    // Create sparse observation data that might lead to empty clusters
    ObservationLists sparseObsLists;
    
    // Create a sequence with only 2 distinct values for 3 states
    // This increases the chance that one state/cluster will be empty
    ObservationSet seq1(10);
    for (std::size_t i = 0; i < 5; ++i) {
        seq1(i) = 1.0; // All around state 0's mean
    }
    for (std::size_t i = 5; i < 10; ++i) {
        seq1(i) = 10.0; // All around state 2's mean
    }
    sparseObsLists.push_back(seq1);
    
    // Add another sequence with the same pattern
    ObservationSet seq2(8);
    for (std::size_t i = 0; i < 4; ++i) {
        seq2(i) = 0.8;
    }
    for (std::size_t i = 4; i < 8; ++i) {
        seq2(i) = 9.8;
    }
    sparseObsLists.push_back(seq2);
    
    ViterbiTrainer trainer(gaussianHmm_.get(), sparseObsLists);
    
    // Capture output to check for warnings
    std::ostringstream captured_output;
    std::streambuf* old_cerr = std::cerr.rdbuf(captured_output.rdbuf());
    
    // Should not throw even if clusters become empty
    EXPECT_NO_THROW(trainer.train());
    
    // Restore cerr
    std::cerr.rdbuf(old_cerr);
    
    // Check if warning messages were printed
    std::string output = captured_output.str();
    // The output might contain warnings about empty clusters
    // This is acceptable behavior - we just want to ensure no crashes
}

// Test maximum iteration limiting in ViterbiTrainer
TEST_F(TrainingEdgeCasesTest, ViterbiTrainerMaxIterations) {
    // Create observation data that might cause slow convergence
    ObservationLists slowConvergenceObs;
    
    ObservationSet seq1(20);
    for (std::size_t i = 0; i < seq1.size(); ++i) {
        // Create noisy data around the boundaries between distributions
        seq1(i) = 2.5 + (static_cast<double>(i % 3) * 0.1); // Values around boundaries
    }
    slowConvergenceObs.push_back(seq1);
    
    ViterbiTrainer trainer(gaussianHmm_.get(), slowConvergenceObs);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Should complete within reasonable time due to iteration limit
    EXPECT_NO_THROW(trainer.train());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    
    // Should not take extremely long due to MAX_VITERBI_ITERATIONS limit
    EXPECT_LT(duration.count(), 300); // Should complete within 5 minutes
}

// Test very short observation sequences
TEST_F(TrainingEdgeCasesTest, BaumWelchTrainerShortSequences) {
    ObservationLists shortSequences;
    
    // Add sequence with only 1 observation
    ObservationSet seq1(1);
    seq1(0) = 0;
    shortSequences.push_back(seq1);
    
    // Add sequence with 2 observations
    ObservationSet seq2(2);
    seq2(0) = 1;
    seq2(1) = 2;
    shortSequences.push_back(seq2);
    
    BaumWelchTrainer trainer(discreteHmm_.get(), shortSequences);
    
    // Should handle short sequences gracefully
    EXPECT_NO_THROW(trainer.train());
}

// Test very short observation sequences with scaled version
TEST_F(TrainingEdgeCasesTest, ScaledBaumWelchTrainerShortSequences) {
    ObservationLists shortSequences;
    
    // Add sequence with only 1 observation
    ObservationSet seq1(1);
    seq1(0) = 0;
    shortSequences.push_back(seq1);
    
    ScaledBaumWelchTrainer trainer(discreteHmm_.get(), shortSequences);
    
    // Should handle short sequences gracefully
    EXPECT_NO_THROW(trainer.train());
}

// Test with zero probabilities in observations
TEST_F(TrainingEdgeCasesTest, BaumWelchTrainerZeroProbabilities) {
    ObservationLists obsWithZeros;
    
    // Create sequences that might result in zero emission probabilities
    ObservationSet seq1(5);
    seq1(0) = 0; seq1(1) = 1; seq1(2) = 2; seq1(3) = 0; seq1(4) = 1;
    obsWithZeros.push_back(seq1);
    
    // Modify discrete distribution to have some zero probabilities
    auto dist0 = std::make_unique<DiscreteDistribution>(3);
    dist0->setProbability(0, 0.0); // Zero probability for symbol 0
    dist0->setProbability(1, 0.7);
    dist0->setProbability(2, 0.3);
    discreteHmm_->setProbabilityDistribution(0, std::move(dist0));
    
    BaumWelchTrainer trainer(discreteHmm_.get(), obsWithZeros);
    
    // Should handle zero probabilities gracefully
    EXPECT_NO_THROW(trainer.train());
}

// Test with NaN and infinite values (scaled version)
TEST_F(TrainingEdgeCasesTest, ScaledBaumWelchTrainerNaNHandling) {
    ObservationLists normalObs;
    
    ObservationSet seq1(5);
    seq1(0) = 0; seq1(1) = 1; seq1(2) = 2; seq1(3) = 0; seq1(4) = 1;
    normalObs.push_back(seq1);
    
    // Set up distributions that might cause numerical issues
    auto dist0 = std::make_unique<DiscreteDistribution>(3);
    dist0->setProbability(0, 1e-100); // Very small probability
    dist0->setProbability(1, 1.0 - 2e-100);
    dist0->setProbability(2, 1e-100);
    discreteHmm_->setProbabilityDistribution(0, std::move(dist0));
    
    ScaledBaumWelchTrainer trainer(discreteHmm_.get(), normalObs);
    
    // Should handle numerical edge cases gracefully
    EXPECT_NO_THROW(trainer.train());
}

// Test type safety improvements
TEST_F(TrainingEdgeCasesTest, TypeSafetyValidation) {
    ObservationLists normalObs;
    
    ObservationSet seq1(10);
    for (std::size_t i = 0; i < seq1.size(); ++i) {
        seq1(i) = static_cast<double>(i % 3);
    }
    normalObs.push_back(seq1);
    
    // Test null HMM handling
    EXPECT_THROW(ViterbiTrainer(nullptr, normalObs), std::invalid_argument);
    EXPECT_THROW(BaumWelchTrainer(nullptr, normalObs), std::invalid_argument);
    EXPECT_THROW(ScaledBaumWelchTrainer(nullptr, normalObs), std::invalid_argument);
    
    // Test empty observation lists
    ObservationLists emptyObs;
    EXPECT_THROW(ViterbiTrainer(gaussianHmm_.get(), emptyObs), std::invalid_argument);
    EXPECT_THROW(BaumWelchTrainer(discreteHmm_.get(), emptyObs), std::invalid_argument);
    EXPECT_THROW(ScaledBaumWelchTrainer(discreteHmm_.get(), emptyObs), std::invalid_argument);
}

// Test distribution compatibility checking
TEST_F(TrainingEdgeCasesTest, DistributionCompatibilityValidation) {
    ObservationLists normalObs;
    
    ObservationSet seq1(5);
    seq1(0) = 0; seq1(1) = 1; seq1(2) = 2; seq1(3) = 0; seq1(4) = 1;
    normalObs.push_back(seq1);
    
    // Baum-Welch trainers should work with discrete distributions
    EXPECT_NO_THROW(BaumWelchTrainer(discreteHmm_.get(), normalObs));
    EXPECT_NO_THROW(ScaledBaumWelchTrainer(discreteHmm_.get(), normalObs));
    
    // Baum-Welch trainers should fail during training (not construction) with Gaussian distributions
    // This is because they specifically require discrete distributions for emission probability updates
    BaumWelchTrainer bwTrainer(gaussianHmm_.get(), normalObs);
    EXPECT_THROW(bwTrainer.train(), std::runtime_error);
    
    ScaledBaumWelchTrainer sbwTrainer(gaussianHmm_.get(), normalObs);
    EXPECT_THROW(sbwTrainer.train(), std::runtime_error);
    
    // ViterbiTrainer should work with both discrete and continuous distributions
    EXPECT_NO_THROW(ViterbiTrainer(gaussianHmm_.get(), normalObs));
    EXPECT_NO_THROW(ViterbiTrainer(discreteHmm_.get(), normalObs));
}

// Test convergence behavior
TEST_F(TrainingEdgeCasesTest, ViterbiTrainerConvergenceBehavior) {
    // Create highly clustered data that should converge quickly
    ObservationLists clusteredObs;
    
    ObservationSet seq1(15);
    for (std::size_t i = 0; i < 5; ++i) {
        seq1(i) = 1.0; // Clear cluster around state 0
    }
    for (std::size_t i = 5; i < 10; ++i) {
        seq1(i) = 5.0; // Clear cluster around state 1
    }
    for (std::size_t i = 10; i < 15; ++i) {
        seq1(i) = 10.0; // Clear cluster around state 2
    }
    clusteredObs.push_back(seq1);
    
    ViterbiTrainer trainer(gaussianHmm_.get(), clusteredObs);
    
    // Should converge without hitting iteration limit
    EXPECT_NO_THROW(trainer.train());
    
    // After training, the distributions should be somewhat close to the data clusters
    auto* dist0 = gaussianHmm_->getProbabilityDistribution(0);
    auto* dist1 = gaussianHmm_->getProbabilityDistribution(1);
    auto* dist2 = gaussianHmm_->getProbabilityDistribution(2);
    
    // The distributions should exist and be valid
    EXPECT_NE(dist0, nullptr);
    EXPECT_NE(dist1, nullptr);
    EXPECT_NE(dist2, nullptr);
}

// Test memory safety with RAII
TEST_F(TrainingEdgeCasesTest, MemorySafetyRAII) {
    {
        ObservationLists normalObs;
        ObservationSet seq1(10);
        for (std::size_t i = 0; i < seq1.size(); ++i) {
            seq1(i) = static_cast<double>(i);
        }
        normalObs.push_back(seq1);
        
        // Create trainer in a scope that will be destroyed
        auto trainer = std::make_unique<ViterbiTrainer>(gaussianHmm_.get(), normalObs);
        EXPECT_NO_THROW(trainer->getHmm());
        
        // trainer should be automatically cleaned up when scope ends
    }
    
    // HMM should still be valid after trainer destruction
    EXPECT_NO_THROW(gaussianHmm_->validate());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
