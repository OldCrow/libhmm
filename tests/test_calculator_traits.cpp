#include <gtest/gtest.h>
#include "libhmm/calculators/calculator_traits.h"
#include "libhmm/two_state_hmm.h"
#include <memory>
#include <random>

using namespace libhmm;
using namespace libhmm::calculators;

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
    
    const auto optimizedTraits = CalculatorSelector::getTraits(CalculatorType::OPTIMIZED);
    EXPECT_EQ(optimizedTraits.name, "SIMD-Optimized");
    EXPECT_TRUE(optimizedTraits.usesSIMD);
    EXPECT_TRUE(optimizedTraits.supportsParallel);
    
    const auto logTraits = CalculatorSelector::getTraits(CalculatorType::LOG_SPACE);
    EXPECT_EQ(logTraits.name, "LogSpace");
    EXPECT_TRUE(logTraits.numericallyStable);
    
    const auto scaledTraits = CalculatorSelector::getTraits(CalculatorType::SCALED);
    EXPECT_EQ(scaledTraits.name, "Scaled");
    EXPECT_TRUE(scaledTraits.numericallyStable);
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
    const double optimizedPerf = CalculatorSelector::predictPerformance(
        CalculatorType::OPTIMIZED, smallProblem);
    
    EXPECT_GT(standardPerf, 0.0);
    EXPECT_GT(optimizedPerf, 0.0);
    
    // For small problems, optimized might not be best due to overhead
    std::cout << "Small problem - Standard: " << standardPerf 
              << ", Optimized: " << optimizedPerf << std::endl;
    
    // Large problem - optimized should perform better
    ProblemCharacteristics largeProblem(largeHmm_.get(), mediumObs_);
    
    const double standardPerfLarge = CalculatorSelector::predictPerformance(
        CalculatorType::STANDARD, largeProblem);
    const double optimizedPerfLarge = CalculatorSelector::predictPerformance(
        CalculatorType::OPTIMIZED, largeProblem);
    
    std::cout << "Large problem - Standard: " << standardPerfLarge 
              << ", Optimized: " << optimizedPerfLarge << std::endl;
    
    // Optimized should be better for larger problems
    EXPECT_GT(optimizedPerfLarge, standardPerfLarge);
}

// Test stability requirements
TEST_F(CalculatorTraitsTest, StabilityRequirements) {
    // Long sequence requiring stability
    ProblemCharacteristics longProblem(smallHmm_.get(), longObs_, true);
    
    const double standardPerf = CalculatorSelector::predictPerformance(
        CalculatorType::STANDARD, longProblem);
    const double logPerf = CalculatorSelector::predictPerformance(
        CalculatorType::LOG_SPACE, longProblem);
    const double scaledPerf = CalculatorSelector::predictPerformance(
        CalculatorType::SCALED, longProblem);
    
    // Standard should be heavily penalized for instability
    EXPECT_LT(standardPerf, logPerf);
    EXPECT_LT(standardPerf, scaledPerf);
    
    std::cout << "Stability test - Standard: " << standardPerf 
              << ", Log: " << logPerf << ", Scaled: " << scaledPerf << std::endl;
}

// Test optimal calculator selection
TEST_F(CalculatorTraitsTest, OptimalSelection) {
    // Small problem
    ProblemCharacteristics smallProblem(smallHmm_.get(), shortObs_);
    CalculatorType smallOptimal = CalculatorSelector::selectOptimal(smallProblem);
    EXPECT_TRUE(smallOptimal == CalculatorType::STANDARD || 
                smallOptimal == CalculatorType::OPTIMIZED);
    
    // Large problem
    ProblemCharacteristics largeProblem(largeHmm_.get(), mediumObs_);
    CalculatorType largeOptimal = CalculatorSelector::selectOptimal(largeProblem);
    
    // Long sequence requiring stability
    ProblemCharacteristics longStableProblem(smallHmm_.get(), longObs_, true);
    CalculatorType stableOptimal = CalculatorSelector::selectOptimal(longStableProblem);
    EXPECT_TRUE(stableOptimal == CalculatorType::LOG_SPACE || 
                stableOptimal == CalculatorType::SCALED);
    
    std::cout << "Selection results:" << std::endl;
    std::cout << "Small problem: " << static_cast<int>(smallOptimal) << std::endl;
    std::cout << "Large problem: " << static_cast<int>(largeOptimal) << std::endl;
    std::cout << "Stable problem: " << static_cast<int>(stableOptimal) << std::endl;
}

// Test calculator creation
TEST_F(CalculatorTraitsTest, CalculatorCreation) {
    // Test creation of all calculator types
    auto standard = CalculatorSelector::create(
        CalculatorType::STANDARD, smallHmm_.get(), shortObs_);
    ASSERT_NE(standard, nullptr);
    EXPECT_NO_THROW(standard->probability());
    
    auto scaled = CalculatorSelector::create(
        CalculatorType::SCALED, smallHmm_.get(), shortObs_);
    ASSERT_NE(scaled, nullptr);
    EXPECT_NO_THROW(scaled->probability());
    
    auto logSpace = CalculatorSelector::create(
        CalculatorType::LOG_SPACE, smallHmm_.get(), shortObs_);
    ASSERT_NE(logSpace, nullptr);
    EXPECT_NO_THROW(logSpace->probability());
    
    auto optimized = CalculatorSelector::create(
        CalculatorType::OPTIMIZED, smallHmm_.get(), shortObs_);
    ASSERT_NE(optimized, nullptr);
    EXPECT_NO_THROW(optimized->probability());
    
    // Test AUTO type
    auto autoCalc = CalculatorSelector::create(
        CalculatorType::AUTO, smallHmm_.get(), shortObs_);
    ASSERT_NE(autoCalc, nullptr);
    EXPECT_NO_THROW(autoCalc->probability());
}

// Test optimal calculator creation convenience method
TEST_F(CalculatorTraitsTest, OptimalCreation) {
    auto optimal = CalculatorSelector::createOptimal(smallHmm_.get(), shortObs_);
    ASSERT_NE(optimal, nullptr);
    
    const double prob1 = optimal->probability();
    
    // Create standard calculator for comparison
    auto standard = CalculatorSelector::create(
        CalculatorType::STANDARD, smallHmm_.get(), shortObs_);
    const double prob2 = standard->probability();
    
    // Should produce same results
    EXPECT_NEAR(prob1, prob2, 1e-10);
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
    // Create all calculator types
    auto standard = CalculatorSelector::create(
        CalculatorType::STANDARD, smallHmm_.get(), shortObs_);
    auto scaled = CalculatorSelector::create(
        CalculatorType::SCALED, smallHmm_.get(), shortObs_);
    auto logSpace = CalculatorSelector::create(
        CalculatorType::LOG_SPACE, smallHmm_.get(), shortObs_);
    auto optimized = CalculatorSelector::create(
        CalculatorType::OPTIMIZED, smallHmm_.get(), shortObs_);
    
    // All should produce similar probabilities
    const double standardProb = standard->probability();
    const double scaledProb = scaled->probability();
    const double logProb = logSpace->probability();
    const double optimizedProb = optimized->probability();
    
    std::cout << "Probability comparison:" << std::endl;
    std::cout << "Standard: " << standardProb << std::endl;
    std::cout << "Scaled: " << scaledProb << std::endl;
    std::cout << "Log: " << logProb << std::endl;
    std::cout << "Optimized: " << optimizedProb << std::endl;
    
    // All should be reasonably close (allowing for numerical differences)
    EXPECT_NEAR(standardProb, optimizedProb, 1e-10);
    EXPECT_NEAR(standardProb, scaledProb, 1e-8);   // Scaled may have slight differences
    EXPECT_NEAR(standardProb, logProb, 1e-8);      // Log may have slight differences
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
