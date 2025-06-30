#include <gtest/gtest.h>
#include "libhmm/libhmm.h"
#include <memory>

using namespace libhmm;

class ModernizedAPITest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple HMM
        hmm_ = std::make_unique<Hmm>(2);
        
        // Initialize with random probabilities
        Vector pi(2);
        Matrix trans(2, 2);
        
        for (size_t i = 0; i < 2; ++i) {
            pi(i) = 0.5;  // Equal probabilities
            for (size_t j = 0; j < 2; ++j) {
                trans(i, j) = 0.5;  // Equal probabilities
            }
        }
        
        hmm_->setPi(pi);
        hmm_->setTrans(trans);
        
        // Set discrete emission distributions
        for (size_t i = 0; i < 2; ++i) {
            auto dist = std::make_unique<DiscreteDistribution>(3);
            for (size_t j = 0; j < 3; ++j) {
                dist->setProbability(j, 1.0/3.0);  // Equal probabilities
            }
            hmm_->setProbabilityDistribution(i, std::move(dist));
        }
        
        // Create observations
        obs_.resize(5);
        for (size_t i = 0; i < 5; ++i) {
            obs_(i) = i % 3;
        }
    }
    
    std::unique_ptr<Hmm> hmm_;
    ObservationSet obs_;
};

// Test modern const reference constructors
TEST_F(ModernizedAPITest, ModernConstReferenceConstructors) {
    // Test ForwardBackwardCalculator with modern constructor
    EXPECT_NO_THROW({
        ForwardBackwardCalculator fbModern(*hmm_, obs_);
        double prob = fbModern.probability();
        EXPECT_GT(prob, 0.0);
        EXPECT_LE(prob, 1.0);
    });
    
    // Test ViterbiCalculator with modern constructor
    EXPECT_NO_THROW({
        ViterbiCalculator viterbiModern(*hmm_, obs_);
        StateSequence seq = viterbiModern.decode();
        double logProb = viterbiModern.getLogProbability();
        
        EXPECT_EQ(seq.size(), obs_.size());
        EXPECT_TRUE(std::isfinite(logProb));
    });
}

// Test legacy pointer constructors (with deprecation warnings suppressed)
TEST_F(ModernizedAPITest, LegacyPointerConstructors) {
    // Test ForwardBackwardCalculator with legacy constructor
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_NO_THROW({
        ForwardBackwardCalculator fbLegacy(hmm_.get(), obs_);
        double prob = fbLegacy.probability();
        EXPECT_GT(prob, 0.0);
        EXPECT_LE(prob, 1.0);
    });
    #pragma clang diagnostic pop
    
    // Test ViterbiCalculator with legacy constructor
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_NO_THROW({
        ViterbiCalculator viterbiLegacy(hmm_.get(), obs_);
        StateSequence seq = viterbiLegacy.decode();
        double logProb = viterbiLegacy.getLogProbability();
        
        EXPECT_EQ(seq.size(), obs_.size());
        EXPECT_TRUE(std::isfinite(logProb));
    });
    #pragma clang diagnostic pop
}

// Test consistency between modern and legacy constructors
TEST_F(ModernizedAPITest, ModernVsLegacyConsistency) {
    // Test ForwardBackwardCalculator consistency
    ForwardBackwardCalculator fbModern(*hmm_, obs_);
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    ForwardBackwardCalculator fbLegacy(hmm_.get(), obs_);
    #pragma clang diagnostic pop
    
    double probModern = fbModern.probability();
    double probLegacy = fbLegacy.probability();
    
    EXPECT_NEAR(probModern, probLegacy, 1e-15);
    
    // Test ViterbiCalculator consistency
    ViterbiCalculator viterbiModern(*hmm_, obs_);
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    ViterbiCalculator viterbiLegacy(hmm_.get(), obs_);
    #pragma clang diagnostic pop
    
    StateSequence seqModern = viterbiModern.decode();
    StateSequence seqLegacy = viterbiLegacy.decode();
    double logProbModern = viterbiModern.getLogProbability();
    double logProbLegacy = viterbiLegacy.getLogProbability();
    
    EXPECT_EQ(seqModern.size(), seqLegacy.size());
    EXPECT_NEAR(logProbModern, logProbLegacy, 1e-15);
    
    // Sequences should be identical
    for (size_t i = 0; i < seqModern.size(); ++i) {
        EXPECT_EQ(seqModern(i), seqLegacy(i));
    }
}

// Test null pointer rejection
TEST_F(ModernizedAPITest, NullPointerRejection) {
    // Test that null pointers are properly rejected
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_THROW({
        ForwardBackwardCalculator fbNull(nullptr, obs_);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        ViterbiCalculator viterbiNull(nullptr, obs_);
    }, std::invalid_argument);
    #pragma clang diagnostic pop
}

// Test Calculator traits system integration
TEST_F(ModernizedAPITest, TraitsSystemIntegration) {
    // Test forward-backward traits
    {
        using namespace libhmm::forwardbackward;
        ProblemCharacteristics fbCharacteristics(hmm_.get(), obs_);
        CalculatorType fbOptimal = CalculatorSelector::selectOptimal(fbCharacteristics);
        
        // Should select a valid calculator type
        EXPECT_TRUE(fbOptimal == CalculatorType::STANDARD ||
                    fbOptimal == CalculatorType::SCALED_SIMD ||
                    fbOptimal == CalculatorType::LOG_SIMD ||
                    fbOptimal == CalculatorType::ADVANCED_LOG_SIMD);
        
        // Test AutoCalculator
        AutoCalculator fbAuto(hmm_.get(), obs_);
        double fbAutoProb = fbAuto.probability();
        EXPECT_GT(fbAutoProb, 0.0);
        EXPECT_LE(fbAutoProb, 1.0);
    }
    
    // Test viterbi traits (separate scope to avoid namespace collision)
    {
        using namespace libhmm::viterbi;
        ProblemCharacteristics viterbiCharacteristics(hmm_.get(), obs_);
        CalculatorType viterbiOptimal = CalculatorSelector::selectOptimal(viterbiCharacteristics);
        
        // Should select a valid calculator type
        EXPECT_TRUE(viterbiOptimal == CalculatorType::STANDARD ||
                    viterbiOptimal == CalculatorType::SCALED_SIMD ||
                    viterbiOptimal == CalculatorType::LOG_SIMD ||
                    viterbiOptimal == CalculatorType::ADVANCED_LOG_SIMD);
        
        // Test AutoCalculator
        AutoCalculator viterbiAuto(hmm_.get(), obs_);
        StateSequence viterbiAutoSeq = viterbiAuto.decode();
        double viterbiAutoLogProb = viterbiAuto.getLogProbability();
        
        EXPECT_EQ(viterbiAutoSeq.size(), obs_.size());
        EXPECT_TRUE(std::isfinite(viterbiAutoLogProb));
    }
}

// Test getHmmRef() method works correctly
TEST_F(ModernizedAPITest, GetHmmRefFunctionality) {
    ForwardBackwardCalculator fbModern(*hmm_, obs_);
    ViterbiCalculator viterbiModern(*hmm_, obs_);
    
    // Test that getHmmRef() returns the same HMM
    const Hmm& hmmRefFB = fbModern.getHmmRef();
    const Hmm& hmmRefViterbi = viterbiModern.getHmmRef();
    
    EXPECT_EQ(hmmRefFB.getNumStates(), hmm_->getNumStates());
    EXPECT_EQ(hmmRefViterbi.getNumStates(), hmm_->getNumStates());
    
    // Test that we can use the reference to access HMM properties
    EXPECT_EQ(hmmRefFB.getPi().size(), 2);
    EXPECT_EQ(hmmRefViterbi.getTrans().size1(), 2);
    EXPECT_EQ(hmmRefViterbi.getTrans().size2(), 2);
}

// Test smart pointer integration
TEST_F(ModernizedAPITest, SmartPointerIntegration) {
    // Test with shared_ptr (properly constructed)
    std::shared_ptr<const Hmm> sharedHmm(hmm_.get(), [](const Hmm*){});
    
    EXPECT_NO_THROW({
        ForwardBackwardCalculator fbShared(*sharedHmm, obs_);
        ViterbiCalculator viterbiShared(*sharedHmm, obs_);
        
        double prob = fbShared.probability();
        StateSequence seq = viterbiShared.decode();
        
        EXPECT_GT(prob, 0.0);
        EXPECT_EQ(seq.size(), obs_.size());
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
