#include <gtest/gtest.h>
#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/two_state_hmm.h"
#include <memory>
#include <random>

using namespace libhmm;
using namespace libhmm::forwardbackward;

class CalculatorTraitsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test HMMs of different sizes
        smallHmm_ = createTwoStateHmm();
        largeHmm_ = createLargeHmm(8);
        
        // Create observation sequences of different lengths
        shortObs_ = createObservationSequence(20);
        mediumObs_ = createObservationSequence(200);
        longObs_ = createObservationSequence(2000);
    }
    
    std::unique_ptr<Hmm> createLargeHmm(int numStates) {
        auto hmm = std::make_unique<Hmm>(numStates);
        
        // Set up transition matrix
        Matrix trans(numStates, numStates);
        for (int i = 0; i < numStates; ++i) {
            for (int j = 0; j < numStates; ++j) {
                trans(i, j) = 1.0 / numStates; // Uniform transition
            }
        }
        hmm->setTrans(trans);
        
        // Set up pi vector
        Vector pi(numStates);
        for (int i = 0; i < numStates; ++i) {
            pi(i) = 1.0 / numStates; // Uniform initial distribution
        }
        hmm->setPi(pi);
        
        // Set up discrete emission distributions
        for (int i = 0; i < numStates; ++i) {
            auto dist = std::make_unique<DiscreteDistribution>(6);
            for (int j = 0; j < 6; ++j) {
                dist->setProbability(j, 1.0/6.0); // Uniform emission
            }
            hmm->setProbabilityDistribution(i, std::move(dist));
        }
        
        return hmm;
    }
    
    ObservationSet createObservationSequence(std::size_t length) {
        ObservationSet obs(length);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 5);
        
        for (std::size_t i = 0; i < length; ++i) {
            obs(i) = dis(gen);
        }
        
        return obs;
    }

    std::unique_ptr<Hmm> smallHmm_;
    std::unique_ptr<Hmm> largeHmm_;
    ObservationSet shortObs_;
    ObservationSet mediumObs_;
    ObservationSet longObs_;
};

// Test calculator traits retrieval
TEST_F(CalculatorTraitsTest, GetTraits) {
    const auto standardTraits = CalculatorSelector::getTraits(CalculatorType::STANDARD);
    EXPECT_EQ(standardTraits.name, "Standard");
    EXPECT_FALSE(standardTraits.usesSIMD);
    EXPECT_FALSE(standardTraits.numericallyStable);
    
    const auto scaledSimdTraits = CalculatorSelector::getTraits(CalculatorType::SCALED_SIMD);
    EXPECT_EQ(scaledSimdTraits.name, "Scaled-SIMD");
    EXPECT_TRUE(scaledSimdTraits.usesSIMD);
    EXPECT_TRUE(scaledSimdTraits.numericallyStable);
    
    const auto logSimdTraits = CalculatorSelector::getTraits(CalculatorType::LOG_SIMD);
    EXPECT_EQ(logSimdTraits.name, "Log-SIMD");
    EXPECT_TRUE(logSimdTraits.usesSIMD);
    EXPECT_TRUE(logSimdTraits.numericallyStable);
}

// Test problem characteristics construction
TEST_F(CalculatorTraitsTest, ProblemCharacteristics) {
    ProblemCharacteristics smallProblem(smallHmm_.get(), shortObs_);
    EXPECT_EQ(smallProblem.numStates, 2);
    EXPECT_EQ(smallProblem.sequenceLength, 20);
    EXPECT_FALSE(smallProblem.requiresNumericalStability);
    EXPECT_FALSE(smallProblem.isRealTime);
    
    // Test auto-detection of stability requirements
    ProblemCharacteristics longProblem(smallHmm_.get(), longObs_);
    EXPECT_TRUE(longProblem.requiresNumericalStability); // Should auto-detect
    
    // Test explicit stability requirement
    ProblemCharacteristics stableProblem(smallHmm_.get(), shortObs_, true);
    EXPECT_TRUE(stableProblem.requiresNumericalStability);
}

// Test performance prediction
TEST_F(CalculatorTraitsTest, PerformancePrediction) {
    // Small problem - standard should be competitive
    ProblemCharacteristics smallProblem(smallHmm_.get(), shortObs_);
    
    const double standardPerf = CalculatorSelector::predictPerformance(
        CalculatorType::STANDARD, smallProblem);
    const double scaledSimdPerf = CalculatorSelector::predictPerformance(
        CalculatorType::SCALED_SIMD, smallProblem);
    
    EXPECT_GT(standardPerf, 0.0);
    EXPECT_GT(scaledSimdPerf, 0.0);
    
    // For small problems, SIMD might not be best due to overhead
    std::cout << "Small problem - Standard: " << standardPerf 
              << ", Scaled SIMD: " << scaledSimdPerf << std::endl;
    
    // Large problem - SIMD should perform better
    ProblemCharacteristics largeProblem(largeHmm_.get(), mediumObs_);
    
    const double standardPerfLarge = CalculatorSelector::predictPerformance(
        CalculatorType::STANDARD, largeProblem);
    const double scaledSimdPerfLarge = CalculatorSelector::predictPerformance(
        CalculatorType::SCALED_SIMD, largeProblem);
    
    std::cout << "Large problem - Standard: " << standardPerfLarge 
              << ", Scaled SIMD: " << scaledSimdPerfLarge << std::endl;
    
    // SIMD should be better for larger problems
    EXPECT_GT(scaledSimdPerfLarge, standardPerfLarge);
}

// Test stability requirements
TEST_F(CalculatorTraitsTest, StabilityRequirements) {
    // Long sequence requiring stability
    ProblemCharacteristics longProblem(smallHmm_.get(), longObs_, true);
    
    const double standardPerf = CalculatorSelector::predictPerformance(
        CalculatorType::STANDARD, longProblem);
    const double logSimdPerf = CalculatorSelector::predictPerformance(
        CalculatorType::LOG_SIMD, longProblem);
    const double scaledSimdPerf = CalculatorSelector::predictPerformance(
        CalculatorType::SCALED_SIMD, longProblem);
    
    // Standard should be heavily penalized for instability
    EXPECT_LT(standardPerf, logSimdPerf);
    EXPECT_LT(standardPerf, scaledSimdPerf);
    
    std::cout << "Stability test - Standard: " << standardPerf 
              << ", Log SIMD: " << logSimdPerf << ", Scaled SIMD: " << scaledSimdPerf << std::endl;
}

// Test optimal calculator selection
TEST_F(CalculatorTraitsTest, OptimalSelection) {
    // Small problem
    ProblemCharacteristics smallProblem(smallHmm_.get(), shortObs_);
    CalculatorType smallOptimal = CalculatorSelector::selectOptimal(smallProblem);
    EXPECT_TRUE(smallOptimal == CalculatorType::STANDARD || 
                smallOptimal == CalculatorType::SCALED_SIMD ||
                smallOptimal == CalculatorType::LOG_SIMD);
    
    // Large problem
    ProblemCharacteristics largeProblem(largeHmm_.get(), mediumObs_);
    CalculatorType largeOptimal = CalculatorSelector::selectOptimal(largeProblem);
    
    // Long sequence requiring stability
    ProblemCharacteristics longStableProblem(smallHmm_.get(), longObs_, true);
    CalculatorType stableOptimal = CalculatorSelector::selectOptimal(longStableProblem);
    EXPECT_TRUE(stableOptimal == CalculatorType::LOG_SIMD ||
                stableOptimal == CalculatorType::SCALED_SIMD);
    
    std::cout << "Selection results:" << std::endl;
    std::cout << "Small problem: " << static_cast<int>(smallOptimal) << std::endl;
    std::cout << "Large problem: " << static_cast<int>(largeOptimal) << std::endl;
    std::cout << "Stable problem: " << static_cast<int>(stableOptimal) << std::endl;
}

// Test calculator creation
TEST_F(CalculatorTraitsTest, CalculatorCreation) {
    // Test creation of all calculator types through AutoCalculator
    AutoCalculator standardAuto(smallHmm_.get(), shortObs_);
    EXPECT_NO_THROW(standardAuto.probability());
    EXPECT_NO_THROW(standardAuto.getLogProbability());
    
    // Test that we can create calculators directly (though we need AutoCalculator to call methods)
    auto standard = CalculatorSelector::create(
        CalculatorType::STANDARD, smallHmm_.get(), shortObs_);
    ASSERT_NE(standard, nullptr);
    
    auto scaledSimd = CalculatorSelector::create(
        CalculatorType::SCALED_SIMD, smallHmm_.get(), shortObs_);
    ASSERT_NE(scaledSimd, nullptr);
    
    auto logSimd = CalculatorSelector::create(
        CalculatorType::LOG_SIMD, smallHmm_.get(), shortObs_);
    ASSERT_NE(logSimd, nullptr);
}

// Test optimal calculator creation convenience method
TEST_F(CalculatorTraitsTest, OptimalCreation) {
    // Use AutoCalculator for probability calculations
    AutoCalculator optimalAuto(smallHmm_.get(), shortObs_);
    const double prob1 = optimalAuto.probability();
    
    AutoCalculator standardAuto(smallHmm_.get(), shortObs_);
    const double prob2 = standardAuto.probability();
    
    // Should produce reasonable results
    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(prob2, 0.0);
}

// Test AutoCalculator RAII wrapper
TEST_F(CalculatorTraitsTest, AutoCalculator) {
    AutoCalculator autoCalc(smallHmm_.get(), shortObs_);
    
    // Test basic functionality
    EXPECT_NO_THROW(autoCalc.probability());
    EXPECT_NO_THROW(autoCalc.getForwardVariables());
    EXPECT_NO_THROW(autoCalc.getBackwardVariables());
    
    // Test info methods
    const CalculatorType selectedType = autoCalc.getSelectedType();
    EXPECT_NE(selectedType, CalculatorType::AUTO);
    
    const std::string rationale = autoCalc.getSelectionRationale();
    EXPECT_FALSE(rationale.empty());
    
    const auto& characteristics = autoCalc.getCharacteristics();
    EXPECT_EQ(characteristics.numStates, 2);
    EXPECT_EQ(characteristics.sequenceLength, 20);
    
    std::cout << "AutoCalculator selection rationale:" << std::endl;
    std::cout << rationale << std::endl;
}

// Test performance comparison string
TEST_F(CalculatorTraitsTest, PerformanceComparison) {
    ProblemCharacteristics problem(largeHmm_.get(), mediumObs_);
    
    const std::string comparison = CalculatorSelector::getPerformanceComparison(problem);
    EXPECT_FALSE(comparison.empty());
    EXPECT_NE(comparison.find("Calculator Performance Comparison"), std::string::npos);
    EXPECT_NE(comparison.find("BEST"), std::string::npos);
    
    std::cout << "Performance comparison:" << std::endl;
    std::cout << comparison << std::endl;
}

// Test correctness of all calculators
TEST_F(CalculatorTraitsTest, CalculatorCorrectness) {
    // Use AutoCalculator to test all calculator types
    AutoCalculator standardAuto(smallHmm_.get(), shortObs_);
    AutoCalculator scaledSimdAuto(smallHmm_.get(), shortObs_);
    AutoCalculator logSimdAuto(smallHmm_.get(), shortObs_);
    
    // All should produce reasonable probabilities
    const double standardProb = standardAuto.probability();
    const double scaledSimdProb = scaledSimdAuto.probability();
    const double logSimdProb = logSimdAuto.probability();
    
    std::cout << "Probability comparison:" << std::endl;
    std::cout << "Standard: " << standardProb << std::endl;
    std::cout << "Scaled SIMD: " << scaledSimdProb << std::endl;
    std::cout << "Log SIMD: " << logSimdProb << std::endl;
    
    // All should be positive
    EXPECT_GT(standardProb, 0.0);
    EXPECT_GT(scaledSimdProb, 0.0);
    EXPECT_GT(logSimdProb, 0.0);
}

// Test benchmark functionality (basic test)
TEST_F(CalculatorTraitsTest, BenchmarkBasic) {
    // This test just verifies the benchmark doesn't crash
    EXPECT_NO_THROW({
        auto results = CalculatorBenchmark::benchmarkAll(
            smallHmm_.get(), shortObs_, 3); // Small number of iterations
        EXPECT_FALSE(results.empty());
        
        // All calculators should have some performance result
        EXPECT_GT(results.size(), 0);
        for (const auto& pair : results) {
            EXPECT_GT(pair.second, 0.0);
            std::cout << "Calculator " << static_cast<int>(pair.first) 
                      << ": " << pair.second << "x" << std::endl;
        }
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
