#include <gtest/gtest.h>
#include "libhmm/training/trainer_traits.h"
#include "libhmm/two_state_hmm.h"
#include "libhmm/distributions/distributions.h"
#include <memory>

using namespace libhmm;
using namespace libhmm::trainer_traits;

class TrainerTraitsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create Gaussian (continuous) HMM
        gaussianHmm_ = std::make_unique<Hmm>(2);
        auto gaussDist0 = std::make_unique<GaussianDistribution>(0.0, 1.0);
        auto gaussDist1 = std::make_unique<GaussianDistribution>(3.0, 1.5);
        gaussianHmm_->setProbabilityDistribution(0, std::move(gaussDist0));
        gaussianHmm_->setProbabilityDistribution(1, std::move(gaussDist1));
        
        // Create discrete HMM
        discreteHmm_ = std::make_unique<Hmm>(2);
        auto dist0 = std::make_unique<DiscreteDistribution>(3);
        dist0->setProbability(0, 0.5);
        dist0->setProbability(1, 0.3);
        dist0->setProbability(2, 0.2);
        discreteHmm_->setProbabilityDistribution(0, std::move(dist0));
        
        auto dist1 = std::make_unique<DiscreteDistribution>(3);
        dist1->setProbability(0, 0.2);
        dist1->setProbability(1, 0.4);
        dist1->setProbability(2, 0.4);
        discreteHmm_->setProbabilityDistribution(1, std::move(dist1));
        
        // Create test observation data
        createTestObservations();
    }
    
    void createTestObservations() {
        // Create continuous observations
        continuousObs_.clear();
        ObservationSet seq1(10);
        for (std::size_t i = 0; i < seq1.size(); ++i) {
            seq1(i) = static_cast<double>(i % 3) + 0.1 * i; // Continuous values
        }
        continuousObs_.push_back(seq1);
        
        ObservationSet seq2(15);
        for (std::size_t i = 0; i < seq2.size(); ++i) {
            seq2(i) = static_cast<double>((i + 5) % 4) + 0.05 * i;
        }
        continuousObs_.push_back(seq2);
        
        // Create discrete observations
        discreteObs_.clear();
        ObservationSet dseq1(8);
        for (std::size_t i = 0; i < dseq1.size(); ++i) {
            dseq1(i) = static_cast<double>(i % 3); // Discrete values
        }
        discreteObs_.push_back(dseq1);
        
        ObservationSet dseq2(12);
        for (std::size_t i = 0; i < dseq2.size(); ++i) {
            dseq2(i) = static_cast<double>((i + 1) % 3);
        }
        discreteObs_.push_back(dseq2);
        
        // Create complex observations (larger problem)
        complexObs_.clear();
        for (int seqIdx = 0; seqIdx < 20; ++seqIdx) {
            ObservationSet complexSeq(50);
            for (std::size_t i = 0; i < complexSeq.size(); ++i) {
                complexSeq(i) = static_cast<double>(seqIdx % 5) + 0.1 * i + 10.0 * seqIdx;
            }
            complexObs_.push_back(complexSeq);
        }
    }
    
    std::unique_ptr<Hmm> gaussianHmm_;
    std::unique_ptr<Hmm> discreteHmm_;
    ObservationLists continuousObs_;
    ObservationLists discreteObs_;
    ObservationLists complexObs_;
};

// Test DataCharacteristics analysis
TEST_F(TrainerTraitsTest, DataCharacteristicsAnalysis) {
    // Test continuous data characteristics
    DataCharacteristics contChars(continuousObs_, gaussianHmm_.get());
    
    EXPECT_EQ(contChars.numSequences, 2);
    EXPECT_EQ(contChars.numStates, 2);
    EXPECT_GT(contChars.avgSequenceLength, 0);
    EXPECT_FALSE(contChars.isDiscrete);
    EXPECT_GT(contChars.dataRange, 0.0);
    EXPECT_EQ(contChars.complexity, ProblemComplexity::SIMPLE);
    
    // Test discrete data characteristics
    DataCharacteristics discChars(discreteObs_, discreteHmm_.get());
    
    EXPECT_EQ(discChars.numSequences, 2);
    EXPECT_EQ(discChars.numStates, 2);
    EXPECT_TRUE(discChars.isDiscrete);
    EXPECT_EQ(discChars.complexity, ProblemComplexity::SIMPLE);
    
    // Test complex data characteristics
    DataCharacteristics complexChars(complexObs_, gaussianHmm_.get());
    
    EXPECT_EQ(complexChars.numSequences, 20);
    EXPECT_GT(complexChars.dataRange, 100.0); // Should have large range
    EXPECT_GE(complexChars.complexity, ProblemComplexity::MODERATE);
}

// Test TrainerCapabilities
TEST_F(TrainerTraitsTest, TrainerCapabilities) {
    // Test Viterbi capabilities
    auto viterbiCaps = TrainerTraits::getCapabilities(TrainerType::VITERBI);
    
    EXPECT_EQ(viterbiCaps.type, TrainerType::VITERBI);
    EXPECT_FALSE(viterbiCaps.supportsDiscreteDistributions);
    EXPECT_TRUE(viterbiCaps.supportsContinuousDistributions);
    EXPECT_TRUE(viterbiCaps.handlesEmptyClusters); // After Phase 1 modernization
    EXPECT_FALSE(viterbiCaps.isNumericallyStable);
    EXPECT_GT(viterbiCaps.speedFactor, 0.0);
    
    // Test Baum-Welch capabilities
    auto bwCaps = TrainerTraits::getCapabilities(TrainerType::BAUM_WELCH);
    
    EXPECT_EQ(bwCaps.type, TrainerType::BAUM_WELCH);
    EXPECT_TRUE(bwCaps.supportsDiscreteDistributions);
    EXPECT_FALSE(bwCaps.supportsContinuousDistributions);
    EXPECT_FALSE(bwCaps.handlesEmptyClusters);
    
    // Test Scaled Baum-Welch capabilities
    auto scaledCaps = TrainerTraits::getCapabilities(TrainerType::SCALED_BAUM_WELCH);
    
    EXPECT_EQ(scaledCaps.type, TrainerType::SCALED_BAUM_WELCH);
    EXPECT_TRUE(scaledCaps.supportsDiscreteDistributions);
    EXPECT_TRUE(scaledCaps.isNumericallyStable);
    
    // Test Robust Viterbi capabilities
    auto robustCaps = TrainerTraits::getCapabilities(TrainerType::ROBUST_VITERBI);
    
    EXPECT_EQ(robustCaps.type, TrainerType::ROBUST_VITERBI);
    EXPECT_TRUE(robustCaps.supportsContinuousDistributions);
    EXPECT_TRUE(robustCaps.isNumericallyStable);
    EXPECT_TRUE(robustCaps.handlesEmptyClusters);
    EXPECT_TRUE(robustCaps.providesConvergenceInfo);
}

// Test distribution type support
TEST_F(TrainerTraitsTest, DistributionTypeSupport) {
    // Viterbi should support continuous but not discrete
    EXPECT_FALSE(TrainerTraits::supportsDistributionType(TrainerType::VITERBI, true));
    EXPECT_TRUE(TrainerTraits::supportsDistributionType(TrainerType::VITERBI, false));
    
    // Baum-Welch should support discrete but not continuous
    EXPECT_TRUE(TrainerTraits::supportsDistributionType(TrainerType::BAUM_WELCH, true));
    EXPECT_FALSE(TrainerTraits::supportsDistributionType(TrainerType::BAUM_WELCH, false));
    
    // Robust Viterbi should support continuous but not discrete
    EXPECT_FALSE(TrainerTraits::supportsDistributionType(TrainerType::ROBUST_VITERBI, true));
    EXPECT_TRUE(TrainerTraits::supportsDistributionType(TrainerType::ROBUST_VITERBI, false));
}

// Test trainer recommendations
TEST_F(TrainerTraitsTest, TrainerRecommendations) {
    // Test recommendations for continuous data
    DataCharacteristics contChars(continuousObs_, gaussianHmm_.get());
    
    auto speedRecs = TrainerTraits::getRecommendedTrainers(contChars, TrainingObjective::SPEED);
    EXPECT_FALSE(speedRecs.empty());
    
    auto accuracyRecs = TrainerTraits::getRecommendedTrainers(contChars, TrainingObjective::ACCURACY);
    EXPECT_FALSE(accuracyRecs.empty());
    
    auto robustnessRecs = TrainerTraits::getRecommendedTrainers(contChars, TrainingObjective::ROBUSTNESS);
    EXPECT_FALSE(robustnessRecs.empty());
    
    // All recommendations should support continuous distributions
    for (auto type : speedRecs) {
        EXPECT_TRUE(TrainerTraits::supportsDistributionType(type, false));
    }
    
    // Test recommendations for discrete data
    DataCharacteristics discChars(discreteObs_, discreteHmm_.get());
    
    auto discRecs = TrainerTraits::getRecommendedTrainers(discChars, TrainingObjective::BALANCED);
    EXPECT_FALSE(discRecs.empty());
    
    // All recommendations should support discrete distributions
    for (auto type : discRecs) {
        EXPECT_TRUE(TrainerTraits::supportsDistributionType(type, true));
    }
}

// Test performance prediction
TEST_F(TrainerTraitsTest, PerformancePrediction) {
    DataCharacteristics chars(continuousObs_, gaussianHmm_.get());
    
    // Test prediction for compatible trainer
    auto prediction = TrainerTraits::predictPerformance(TrainerType::VITERBI, chars, TrainingObjective::BALANCED);
    
    EXPECT_EQ(prediction.trainerType, TrainerType::VITERBI);
    EXPECT_GT(prediction.expectedTrainingTime, 0.0);
    EXPECT_GT(prediction.expectedAccuracy, 0.0);
    EXPECT_LE(prediction.expectedAccuracy, 1.0);
    EXPECT_GT(prediction.expectedRobustness, 0.0);
    EXPECT_LE(prediction.expectedRobustness, 1.0);
    EXPECT_GT(prediction.confidenceScore, 0.0);
    EXPECT_LE(prediction.confidenceScore, 1.0);
    EXPECT_FALSE(prediction.rationale.empty());
    
    // Test prediction for incompatible trainer
    auto badPrediction = TrainerTraits::predictPerformance(TrainerType::BAUM_WELCH, chars, TrainingObjective::BALANCED);
    
    EXPECT_EQ(badPrediction.confidenceScore, 0.0);
    EXPECT_FALSE(badPrediction.warnings.empty());
}

// Test optimal trainer selection
TEST_F(TrainerTraitsTest, OptimalTrainerSelection) {
    // Test selection for continuous data
    DataCharacteristics contChars(continuousObs_, gaussianHmm_.get());
    
    auto speedOptimal = TrainerTraits::selectOptimalTrainer(contChars, TrainingObjective::SPEED);
    EXPECT_TRUE(TrainerTraits::supportsDistributionType(speedOptimal, false));
    
    auto robustnessOptimal = TrainerTraits::selectOptimalTrainer(contChars, TrainingObjective::ROBUSTNESS);
    EXPECT_TRUE(TrainerTraits::supportsDistributionType(robustnessOptimal, false));
    
    // Test selection for discrete data
    DataCharacteristics discChars(discreteObs_, discreteHmm_.get());
    
    auto discOptimal = TrainerTraits::selectOptimalTrainer(discChars, TrainingObjective::BALANCED);
    EXPECT_TRUE(TrainerTraits::supportsDistributionType(discOptimal, true));
}

// Test compatibility checking
TEST_F(TrainerTraitsTest, CompatibilityChecking) {
    // Test compatibility with Gaussian HMM (continuous distributions)
    EXPECT_TRUE(TrainerTraits::isCompatible(TrainerType::VITERBI, gaussianHmm_.get()));
    EXPECT_FALSE(TrainerTraits::isCompatible(TrainerType::BAUM_WELCH, gaussianHmm_.get()));
    
    // Test compatibility with discrete HMM
    EXPECT_FALSE(TrainerTraits::isCompatible(TrainerType::VITERBI, discreteHmm_.get()));
    EXPECT_TRUE(TrainerTraits::isCompatible(TrainerType::BAUM_WELCH, discreteHmm_.get()));
    
    // Test with null HMM
    EXPECT_FALSE(TrainerTraits::isCompatible(TrainerType::VITERBI, nullptr));
}

// Test TrainingConfiguration
TEST_F(TrainerTraitsTest, TrainingConfiguration) {
    DataCharacteristics chars(continuousObs_, gaussianHmm_.get());
    
    // Test configuration for different objectives
    TrainingConfiguration speedConfig(chars, TrainingObjective::SPEED);
    EXPECT_EQ(speedConfig.objective, TrainingObjective::SPEED);
    EXPECT_FALSE(speedConfig.enableProgressReporting);
    
    TrainingConfiguration accuracyConfig(chars, TrainingObjective::ACCURACY);
    EXPECT_EQ(accuracyConfig.objective, TrainingObjective::ACCURACY);
    EXPECT_TRUE(accuracyConfig.enableAdaptivePrecision);
    
    TrainingConfiguration robustnessConfig(chars, TrainingObjective::ROBUSTNESS);
    EXPECT_EQ(robustnessConfig.objective, TrainingObjective::ROBUSTNESS);
    EXPECT_TRUE(robustnessConfig.enableRobustErrorRecovery);
    
    // Test default configuration
    TrainingConfiguration defaultConfig;
    EXPECT_EQ(defaultConfig.objective, TrainingObjective::BALANCED);
}

// Test TrainerFactory
TEST_F(TrainerTraitsTest, TrainerFactory) {
    // Test creating optimal trainer for continuous data
    auto optimalTrainer = TrainerFactory::createOptimalTrainer(
        gaussianHmm_.get(), continuousObs_, TrainingObjective::BALANCED);
    
    EXPECT_NE(optimalTrainer, nullptr);
    
    // Test creating specific trainer
    auto viterbiTrainer = TrainerFactory::createTrainer(
        TrainerType::VITERBI, gaussianHmm_.get(), continuousObs_);
    
    EXPECT_NE(viterbiTrainer, nullptr);
    
    // Test creating trainer for discrete data
    auto bwTrainer = TrainerFactory::createTrainer(
        TrainerType::BAUM_WELCH, discreteHmm_.get(), discreteObs_);
    
    EXPECT_NE(bwTrainer, nullptr);
    
    // Test error handling
    EXPECT_THROW(TrainerFactory::createTrainer(
        TrainerType::VITERBI, nullptr, continuousObs_), std::invalid_argument);
    
    ObservationLists emptyObs;
    EXPECT_THROW(TrainerFactory::createTrainer(
        TrainerType::VITERBI, gaussianHmm_.get(), emptyObs), std::invalid_argument);
}

// Test AutoTrainer
TEST_F(TrainerTraitsTest, AutoTrainer) {
    // Test AutoTrainer with continuous data
    AutoTrainer autoTrainer(gaussianHmm_.get(), continuousObs_, TrainingObjective::BALANCED);
    
    EXPECT_NE(autoTrainer.getSelectedType(), TrainerType::AUTO);
    EXPECT_NE(autoTrainer.getTrainer(), nullptr);
    EXPECT_FALSE(autoTrainer.getSelectionRationale().empty());
    
    // Test training
    bool success = autoTrainer.train();
    EXPECT_TRUE(success);
    EXPECT_GE(autoTrainer.getTrainingDuration(), 0.0); // Can be 0 for very fast training
    EXPECT_TRUE(autoTrainer.isTrainingSuccessful());
    
    // Test performance report
    std::string report = autoTrainer.getPerformanceReport();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Training Performance Report"), std::string::npos);
}

// Test performance comparison report
TEST_F(TrainerTraitsTest, PerformanceComparisonReport) {
    DataCharacteristics chars(continuousObs_, gaussianHmm_.get());
    
    std::string report = TrainerTraits::generatePerformanceComparison(chars, TrainingObjective::BALANCED);
    
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Trainer Performance Comparison"), std::string::npos);
    EXPECT_NE(report.find("Problem Characteristics"), std::string::npos);
    EXPECT_NE(report.find("Expected Time"), std::string::npos);
    EXPECT_NE(report.find("Expected Accuracy"), std::string::npos);
    EXPECT_NE(report.find("Expected Robustness"), std::string::npos);
}

// Test configuration presets
TEST_F(TrainerTraitsTest, ConfigurationPresets) {
    auto realtimeConfig = presets::realtime();
    EXPECT_EQ(realtimeConfig.objective, TrainingObjective::SPEED);
    EXPECT_LE(realtimeConfig.maxIterations, 500);
    EXPECT_FALSE(realtimeConfig.enableAdaptivePrecision);
    
    auto accuracyConfig = presets::highAccuracy();
    EXPECT_EQ(accuracyConfig.objective, TrainingObjective::ACCURACY);
    EXPECT_LT(accuracyConfig.convergenceTolerance, 1e-10);
    EXPECT_TRUE(accuracyConfig.enableAdaptivePrecision);
    
    auto robustnessConfig = presets::maxRobustness();
    EXPECT_EQ(robustnessConfig.objective, TrainingObjective::ROBUSTNESS);
    EXPECT_TRUE(robustnessConfig.enableRobustErrorRecovery);
    EXPECT_EQ(robustnessConfig.recoveryStrategy, numerical::ErrorRecovery::RecoveryStrategy::ROBUST);
    
    auto balancedConfig = presets::balanced();
    EXPECT_EQ(balancedConfig.objective, TrainingObjective::BALANCED);
    EXPECT_TRUE(balancedConfig.enableAdaptivePrecision);
    EXPECT_TRUE(balancedConfig.enableRobustErrorRecovery);
    
    auto simpleConfig = presets::simple();
    EXPECT_FALSE(simpleConfig.enableAdaptivePrecision);
    EXPECT_FALSE(simpleConfig.enableRobustErrorRecovery);
    
    auto largeScaleConfig = presets::largScale();
    EXPECT_FALSE(largeScaleConfig.enableProgressReporting); // Reduce overhead
}

// Test edge cases
TEST_F(TrainerTraitsTest, EdgeCases) {
    // Test with empty observation lists
    ObservationLists emptyObs;
    DataCharacteristics emptyChars(emptyObs, gaussianHmm_.get());
    
    EXPECT_EQ(emptyChars.numSequences, 0);
    EXPECT_EQ(emptyChars.avgSequenceLength, 0);
    
    // Test with null HMM
    DataCharacteristics nullHmmChars(continuousObs_, nullptr);
    EXPECT_EQ(nullHmmChars.numStates, 0);
    
    // Test getting capabilities for AUTO type
    auto autoCaps = TrainerTraits::getCapabilities(TrainerType::AUTO);
    EXPECT_EQ(autoCaps.type, TrainerType::AUTO);
    EXPECT_EQ(autoCaps.name, "Automatic Selection");
    
    // Test recommendations with empty data
    auto emptyRecs = TrainerTraits::getRecommendedTrainers(emptyChars, TrainingObjective::BALANCED);
    // Should still return some recommendations, filtered by compatibility
}

// Test complex scenarios
TEST_F(TrainerTraitsTest, ComplexScenarios) {
    // Test with complex data
    DataCharacteristics complexChars(complexObs_, gaussianHmm_.get());
    
    EXPECT_EQ(complexChars.numSequences, 20);
    EXPECT_GT(complexChars.avgSequenceLength, 30);
    EXPECT_GE(complexChars.complexity, ProblemComplexity::MODERATE);
    
    // Test that complex problems prefer robust trainers
    auto robustnessRecs = TrainerTraits::getRecommendedTrainers(complexChars, TrainingObjective::ROBUSTNESS);
    EXPECT_FALSE(robustnessRecs.empty());
    
    // Test performance prediction on complex data
    auto complexPrediction = TrainerTraits::predictPerformance(
        TrainerType::ROBUST_VITERBI, complexChars, TrainingObjective::ROBUSTNESS);
    
    EXPECT_GT(complexPrediction.expectedTrainingTime, 0.0);
    EXPECT_GT(complexPrediction.confidenceScore, 0.0);
    
    // Test configuration adaptation to complexity
    TrainingConfiguration complexConfig(complexChars, TrainingObjective::BALANCED);
    EXPECT_TRUE(complexConfig.enableAdaptivePrecision);
    EXPECT_GE(complexConfig.maxIterations, 1000);
}

// Test all capabilities retrieval
TEST_F(TrainerTraitsTest, AllCapabilities) {
    auto allCaps = TrainerTraits::getAllCapabilities();
    
    EXPECT_FALSE(allCaps.empty());
    EXPECT_TRUE(allCaps.find(TrainerType::VITERBI) != allCaps.end());
    EXPECT_TRUE(allCaps.find(TrainerType::BAUM_WELCH) != allCaps.end());
    EXPECT_TRUE(allCaps.find(TrainerType::SCALED_BAUM_WELCH) != allCaps.end());
    EXPECT_TRUE(allCaps.find(TrainerType::ROBUST_VITERBI) != allCaps.end());
    
    // AUTO should not be in the map as it's not a concrete trainer
    EXPECT_TRUE(allCaps.find(TrainerType::AUTO) == allCaps.end());
    
    // Verify each capability has proper values
    for (const auto& [type, caps] : allCaps) {
        EXPECT_FALSE(caps.name.empty());
        EXPECT_GT(caps.speedFactor, 0.0);
        EXPECT_GT(caps.accuracyFactor, 0.0);
        EXPECT_GT(caps.robustnessFactor, 0.0);
        EXPECT_GT(caps.memoryOverhead, 0.0);
        EXPECT_GT(caps.maxIterations, 0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
