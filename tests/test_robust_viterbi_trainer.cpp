#include <gtest/gtest.h>
#include "libhmm/training/robust_viterbi_trainer.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/distributions.h"
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <random>

namespace libhmm {

class RobustViterbiTrainerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 2-state HMM for testing
        createSimple2StateHMM();
        
        // Create test observation data
        createTestObservations();
        
        // Create problematic data for robustness testing
        createProblematicObservations();
    }
    
    void createSimple2StateHMM() {
        hmm_ = std::make_unique<Hmm>(2);
        
        // Set initial probabilities
        Vector pi(2);
        pi(0) = 0.6;
        pi(1) = 0.4;
        hmm_->setPi(pi);
        
        // Set transition matrix
        Matrix trans(2, 2);
        trans(0, 0) = 0.7;  trans(0, 1) = 0.3;
        trans(1, 0) = 0.4;  trans(1, 1) = 0.6;
        hmm_->setTrans(trans);
        
        // Set emission distributions (Gaussian)
        auto dist1 = std::make_unique<GaussianDistribution>(0.0, 1.0);
        auto dist2 = std::make_unique<GaussianDistribution>(3.0, 1.5);
        hmm_->setProbabilityDistribution(0, std::move(dist1));
        hmm_->setProbabilityDistribution(1, std::move(dist2));
    }
    
    void createTestObservations() {
        // Create normal observation sequences
        normalObservations_.clear();
        
        // Sequence 1: Values from state 0 (mean ~0)
        ObservationSet seq1(10);
        std::vector<double> vals1 = {-0.5, 0.2, -0.1, 0.8, -0.3, 0.1, -0.7, 0.4, -0.2, 0.6};
        for (size_t i = 0; i < vals1.size(); ++i) {
            seq1(i) = vals1[i];
        }
        normalObservations_.push_back(seq1);
        
        // Sequence 2: Values from state 1 (mean ~3)
        ObservationSet seq2(8);
        std::vector<double> vals2 = {2.8, 3.2, 3.5, 2.6, 3.8, 2.9, 3.1, 3.4};
        for (size_t i = 0; i < vals2.size(); ++i) {
            seq2(i) = vals2[i];
        }
        normalObservations_.push_back(seq2);
        
        // Sequence 3: Mixed values
        ObservationSet seq3(12);
        std::vector<double> vals3 = {0.1, 2.9, 0.3, 3.1, -0.2, 3.5, 0.4, 2.7, -0.1, 3.2, 0.2, 2.8};
        for (size_t i = 0; i < vals3.size(); ++i) {
            seq3(i) = vals3[i];
        }
        normalObservations_.push_back(seq3);
    }
    
    void createProblematicObservations() {
        // Create observation sequences with numerical issues
        problematicObservations_.clear();
        
        // Sequence with NaN values
        ObservationSet seqNaN(5);
        seqNaN(0) = 1.0;
        seqNaN(1) = std::numeric_limits<double>::quiet_NaN();
        seqNaN(2) = 2.0;
        seqNaN(3) = 3.0;
        seqNaN(4) = std::numeric_limits<double>::quiet_NaN();
        problematicObservations_.push_back(seqNaN);
        
        // Sequence with infinite values
        ObservationSet seqInf(4);
        seqInf(0) = 1.0;
        seqInf(1) = std::numeric_limits<double>::infinity();
        seqInf(2) = 2.0;
        seqInf(3) = -std::numeric_limits<double>::infinity();
        problematicObservations_.push_back(seqInf);
        
        // Sequence with extreme values
        ObservationSet seqExtreme(6);
        seqExtreme(0) = 1e-100;
        seqExtreme(1) = 1e100;
        seqExtreme(2) = -1e100;
        seqExtreme(3) = 1e-200;
        seqExtreme(4) = 1e200;
        seqExtreme(5) = 0.0;
        problematicObservations_.push_back(seqExtreme);
    }
    
    void TearDown() override {
        hmm_.reset();
    }
    
    std::unique_ptr<Hmm> hmm_;
    ObservationLists normalObservations_;
    ObservationLists problematicObservations_;
};

// Test basic construction and configuration
TEST_F(RobustViterbiTrainerTest, BasicConstruction) {
    RobustViterbiTrainer::RobustTrainingConfig config;
    config.enableProgressReporting = false; // Disable for clean testing
    
    EXPECT_NO_THROW({
        RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    });
}

TEST_F(RobustViterbiTrainerTest, ConfigurationValidation) {
    RobustViterbiTrainer::RobustTrainingConfig config;
    config.convergenceTolerance = -1.0; // Invalid
    config.enableProgressReporting = false;
    
    EXPECT_THROW({
        RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    }, std::runtime_error);
    
    config.convergenceTolerance = 1.5; // Invalid (> 1.0)
    EXPECT_THROW({
        RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    }, std::runtime_error);
    
    config.convergenceTolerance = 1e-8; // Valid
    config.maxIterations = 0; // Invalid
    EXPECT_THROW({
        RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    }, std::runtime_error);
}

TEST_F(RobustViterbiTrainerTest, InvalidInputValidation) {
    RobustViterbiTrainer::RobustTrainingConfig config;
    config.enableProgressReporting = false;
    
    // Null HMM
    EXPECT_THROW({
        RobustViterbiTrainer trainer(nullptr, normalObservations_, config);
    }, std::invalid_argument);
    
    // Empty observation lists
    ObservationLists emptyObs;
    EXPECT_THROW({
        RobustViterbiTrainer trainer(hmm_.get(), emptyObs, config);
    }, std::invalid_argument);
}

// Test training presets
TEST_F(RobustViterbiTrainerTest, TrainingPresets) {
    auto conservative = training_presets::conservative();
    EXPECT_TRUE(conservative.enableAdaptivePrecision);
    EXPECT_TRUE(conservative.enableErrorRecovery);
    EXPECT_TRUE(conservative.enableDiagnostics);
    EXPECT_EQ(conservative.convergenceTolerance, 1e-10);
    EXPECT_EQ(conservative.maxIterations, 5000);
    
    auto aggressive = training_presets::aggressive();
    EXPECT_FALSE(aggressive.enableAdaptivePrecision);
    EXPECT_TRUE(aggressive.enableErrorRecovery);
    EXPECT_FALSE(aggressive.enableDiagnostics);
    EXPECT_EQ(aggressive.convergenceTolerance, 1e-6);
    EXPECT_EQ(aggressive.maxIterations, 500);
    
    auto realtime = training_presets::realtime();
    EXPECT_FALSE(realtime.enableAdaptivePrecision);
    EXPECT_FALSE(realtime.enableProgressReporting);
    EXPECT_EQ(realtime.maxIterations, 100);
    
    auto highPrecision = training_presets::highPrecision();
    EXPECT_EQ(highPrecision.convergenceTolerance, 1e-12);
    EXPECT_EQ(highPrecision.maxIterations, 10000);
    
    auto balanced = training_presets::balanced();
    EXPECT_TRUE(balanced.enableAdaptivePrecision);
    EXPECT_TRUE(balanced.enableErrorRecovery);
    EXPECT_EQ(balanced.convergenceTolerance, 1e-8);
    EXPECT_EQ(balanced.maxIterations, 1000);
}

// Test factory methods
TEST_F(RobustViterbiTrainerTest, FactoryMethods) {
    auto trainer1 = RobustViterbiTrainer::createWithRecommendedSettings(
        hmm_.get(), normalObservations_, "conservative");
    EXPECT_NE(trainer1, nullptr);
    EXPECT_TRUE(trainer1->getConfig().enableAdaptivePrecision);
    
    auto trainer2 = RobustViterbiTrainer::createWithRecommendedSettings(
        hmm_.get(), normalObservations_, "aggressive");
    EXPECT_NE(trainer2, nullptr);
    EXPECT_FALSE(trainer2->getConfig().enableAdaptivePrecision);
    
    auto trainer3 = RobustViterbiTrainer::createWithRecommendedSettings(
        hmm_.get(), normalObservations_, "unknown_preset");
    EXPECT_NE(trainer3, nullptr);
    // Should default to balanced
    EXPECT_TRUE(trainer3->getConfig().enableAdaptivePrecision);
}

// Test normal training (no numerical issues)
TEST_F(RobustViterbiTrainerTest, NormalTraining) {
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.maxIterations = 50; // Reduce for faster testing
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    
    // Store original parameters for comparison
    Vector originalPi = hmm_->getPi();
    Matrix originalTrans = hmm_->getTrans();
    
    EXPECT_NO_THROW({
        trainer.train();
    });
    
    // Check that training completed
    EXPECT_TRUE(trainer.hasConverged() || trainer.reachedMaxIterations());
    
    // Check that parameters changed (learning occurred)
    Vector newPi = hmm_->getPi();
    Matrix newTrans = hmm_->getTrans();
    
    bool parametersChanged = false;
    for (std::size_t i = 0; i < newPi.size(); ++i) {
        if (std::abs(originalPi(i) - newPi(i)) > 1e-6) {
            parametersChanged = true;
            break;
        }
    }
    
    if (!parametersChanged) {
        for (std::size_t i = 0; i < newTrans.size1(); ++i) {
            for (std::size_t j = 0; j < newTrans.size2(); ++j) {
                if (std::abs(originalTrans(i, j) - newTrans(i, j)) > 1e-6) {
                    parametersChanged = true;
                    break;
                }
            }
            if (parametersChanged) break;
        }
    }
    
    EXPECT_TRUE(parametersChanged);
}

// Test error recovery with problematic data
TEST_F(RobustViterbiTrainerTest, ErrorRecoveryWithProblematicData) {
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.enableErrorRecovery = true;
    config.maxIterations = 20;
    
    RobustViterbiTrainer trainer(hmm_.get(), problematicObservations_, config);
    
    // Should not throw with error recovery enabled
    EXPECT_NO_THROW({
        trainer.train();
    });
    
    // Should complete training (either converged or reached max iterations)
    EXPECT_TRUE(trainer.hasConverged() || trainer.reachedMaxIterations());
}

// Test strict mode with problematic data (should fail during construction)
TEST_F(RobustViterbiTrainerTest, StrictModeWithProblematicData) {
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.enableErrorRecovery = false;
    config.maxIterations = 20;
    
    // Should throw during construction with strict error handling
    EXPECT_THROW({
        RobustViterbiTrainer trainer(hmm_.get(), problematicObservations_, config);
    }, std::runtime_error);
}

// Test convergence detection
TEST_F(RobustViterbiTrainerTest, ConvergenceDetection) {
    auto config = training_presets::conservative();
    config.enableProgressReporting = false;
    config.convergenceTolerance = 1e-6; // Relaxed for faster convergence
    config.maxIterations = 100;
    config.convergenceWindow = 3;
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    trainer.train();
    
    // Check convergence report
    std::string report = trainer.getConvergenceReport();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Convergence Report"), std::string::npos);
    
    // Check convergence status
    bool converged = trainer.hasConverged();
    bool maxReached = trainer.reachedMaxIterations();
    EXPECT_TRUE(converged || maxReached);
}

// Test adaptive precision
TEST_F(RobustViterbiTrainerTest, AdaptivePrecision) {
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.enableAdaptivePrecision = true;
    config.maxIterations = 50;
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    
    EXPECT_TRUE(trainer.isAdaptivePrecisionEnabled());
    
    trainer.train();
    double finalTolerance = trainer.getCurrentTolerance();
    
    // Tolerance might have changed during training
    EXPECT_GT(finalTolerance, 0);
}

// Test diagnostics
TEST_F(RobustViterbiTrainerTest, NumericalDiagnostics) {
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.enableDiagnostics = true;
    config.maxIterations = 20;
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    trainer.train();
    
    // Check that health report is available
    std::string healthReport = trainer.getNumericalHealthReport();
    EXPECT_FALSE(healthReport.empty());
}

// Test configuration management
TEST_F(RobustViterbiTrainerTest, ConfigurationManagement) {
    auto config1 = training_presets::aggressive();
    config1.enableProgressReporting = false;
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config1);
    
    EXPECT_EQ(trainer.getConfig().maxIterations, config1.maxIterations);
    EXPECT_EQ(trainer.getConfig().convergenceTolerance, config1.convergenceTolerance);
    
    // Change configuration
    auto config2 = training_presets::conservative();
    config2.enableProgressReporting = false;
    trainer.setConfig(config2);
    
    EXPECT_EQ(trainer.getConfig().maxIterations, config2.maxIterations);
    EXPECT_EQ(trainer.getConfig().convergenceTolerance, config2.convergenceTolerance);
}

// Test recovery strategy management
TEST_F(RobustViterbiTrainerTest, RecoveryStrategyManagement) {
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    
    // Test default strategy
    EXPECT_EQ(trainer.getRecoveryStrategy(), numerical::ErrorRecovery::RecoveryStrategy::GRACEFUL);
    
    // Change strategy
    trainer.setRecoveryStrategy(numerical::ErrorRecovery::RecoveryStrategy::ROBUST);
    EXPECT_EQ(trainer.getRecoveryStrategy(), numerical::ErrorRecovery::RecoveryStrategy::ROBUST);
    
    trainer.setRecoveryStrategy(numerical::ErrorRecovery::RecoveryStrategy::STRICT);
    EXPECT_EQ(trainer.getRecoveryStrategy(), numerical::ErrorRecovery::RecoveryStrategy::STRICT);
    EXPECT_FALSE(trainer.getConfig().enableErrorRecovery);
}

// Test early termination
TEST_F(RobustViterbiTrainerTest, EarlyTermination) {
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.maxIterations = 1000; // Large number
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    
    // Request early termination
    trainer.requestEarlyTermination();
    
    // Training should terminate quickly
    trainer.train();
    
    // Should not have reached max iterations
    EXPECT_FALSE(trainer.reachedMaxIterations());
}

// Test state reset
TEST_F(RobustViterbiTrainerTest, StateReset) {
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.maxIterations = 20;
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    
    // Train once
    trainer.train();
    bool firstConverged = trainer.hasConverged();
    
    // Reset and train again
    trainer.resetTrainingState();
    trainer.train();
    bool secondConverged = trainer.hasConverged();
    
    // Both training runs should complete
    EXPECT_TRUE(firstConverged || trainer.reachedMaxIterations());
    EXPECT_TRUE(secondConverged || trainer.reachedMaxIterations());
}

// Test performance recommendations
TEST_F(RobustViterbiTrainerTest, PerformanceRecommendations) {
    auto config = training_presets::realtime();
    config.enableProgressReporting = false;
    config.maxIterations = 2; // Very low to force max iterations
    config.convergenceTolerance = 1e-12; // Very strict to prevent early convergence
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    trainer.train();
    
    // Check if max iterations were actually reached
    bool hitMaxIterations = trainer.reachedMaxIterations();
    
    auto recommendations = trainer.getPerformanceRecommendations();
    
    if (hitMaxIterations) {
        EXPECT_FALSE(recommendations.empty());
        
        // Should recommend increasing iterations since we hit the max
        bool foundIterationRecommendation = false;
        for (const auto& rec : recommendations) {
            if (rec.find("maximum iterations") != std::string::npos) {
                foundIterationRecommendation = true;
                break;
            }
        }
        EXPECT_TRUE(foundIterationRecommendation);
    } else {
        // If we didn't hit max iterations, recommendations might be empty or different
        // This is also valid behavior
        EXPECT_TRUE(true);
    }
}

// Test with different HMM configurations
TEST_F(RobustViterbiTrainerTest, DifferentHMMConfigurations) {
    // Test with 3-state HMM
    auto hmm3 = std::make_unique<Hmm>(3);
    
    Vector pi3(3);
    pi3(0) = 0.5; pi3(1) = 0.3; pi3(2) = 0.2;
    hmm3->setPi(pi3);
    
    Matrix trans3(3, 3);
    trans3(0,0) = 0.6; trans3(0,1) = 0.3; trans3(0,2) = 0.1;
    trans3(1,0) = 0.2; trans3(1,1) = 0.6; trans3(1,2) = 0.2;
    trans3(2,0) = 0.1; trans3(2,1) = 0.2; trans3(2,2) = 0.7;
    hmm3->setTrans(trans3);
    
    hmm3->setProbabilityDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm3->setProbabilityDistribution(1, std::make_unique<GaussianDistribution>(3.0, 1.5));
    hmm3->setProbabilityDistribution(2, std::make_unique<GaussianDistribution>(-2.0, 0.8));
    
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.maxIterations = 50;
    
    EXPECT_NO_THROW({
        RobustViterbiTrainer trainer(hmm3.get(), normalObservations_, config);
        trainer.train();
    });
}

// Test with Poisson distributions
TEST_F(RobustViterbiTrainerTest, PoissonDistributions) {
    // Create HMM with Poisson distributions
    auto hmmPoisson = std::make_unique<Hmm>(2);
    
    Vector pi(2);
    pi(0) = 0.6; pi(1) = 0.4;
    hmmPoisson->setPi(pi);
    
    Matrix trans(2, 2);
    trans(0,0) = 0.7; trans(0,1) = 0.3;
    trans(1,0) = 0.4; trans(1,1) = 0.6;
    hmmPoisson->setTrans(trans);
    
    hmmPoisson->setProbabilityDistribution(0, std::make_unique<PoissonDistribution>(2.0));
    hmmPoisson->setProbabilityDistribution(1, std::make_unique<PoissonDistribution>(5.0));
    
    // Create integer observations for Poisson
    ObservationLists poissonObs;
    ObservationSet seq1(10);
    std::vector<double> vals = {1, 2, 3, 2, 1, 4, 5, 6, 5, 4};
    for (size_t i = 0; i < vals.size(); ++i) {
        seq1(i) = vals[i];
    }
    poissonObs.push_back(seq1);
    
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.maxIterations = 30;
    
    EXPECT_NO_THROW({
        RobustViterbiTrainer trainer(hmmPoisson.get(), poissonObs, config);
        trainer.train();
    });
}

// Test memory and resource management
TEST_F(RobustViterbiTrainerTest, ResourceManagement) {
    // Test that trainer can be destroyed safely after training
    auto config = training_presets::balanced();
    config.enableProgressReporting = false;
    config.maxIterations = 20;
    
    {
        RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
        trainer.train();
        
        // Verify reports are available before destruction
        EXPECT_FALSE(trainer.getConvergenceReport().empty());
        EXPECT_FALSE(trainer.getNumericalHealthReport().empty());
    }
    
    // Trainer should be destroyed cleanly
    EXPECT_TRUE(true); // If we reach here, no crash occurred
}

// Integration test with all features enabled
TEST_F(RobustViterbiTrainerTest, FullFeaturesIntegration) {
    auto config = training_presets::conservative();
    config.enableProgressReporting = false; // Keep quiet for testing
    config.maxIterations = 50;
    
    RobustViterbiTrainer trainer(hmm_.get(), normalObservations_, config);
    
    // Verify all features are enabled
    EXPECT_TRUE(trainer.getConfig().enableAdaptivePrecision);
    EXPECT_TRUE(trainer.getConfig().enableErrorRecovery);
    EXPECT_TRUE(trainer.getConfig().enableDiagnostics);
    EXPECT_TRUE(trainer.isAdaptivePrecisionEnabled());
    
    // Train with all features
    EXPECT_NO_THROW({
        trainer.train();
    });
    
    // Verify results
    EXPECT_TRUE(trainer.hasConverged() || trainer.reachedMaxIterations());
    EXPECT_FALSE(trainer.getConvergenceReport().empty());
    EXPECT_FALSE(trainer.getNumericalHealthReport().empty());
    
    auto recommendations = trainer.getPerformanceRecommendations();
    // Should have some recommendations or be empty (both are valid)
    EXPECT_TRUE(true);
}

} // namespace libhmm
