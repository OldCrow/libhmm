#include <gtest/gtest.h>
#include "libhmm/libhmm.h"
#include <chrono>
#include <memory>

using namespace libhmm;

class BasicCalculatorPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a larger HMM for meaningful performance test
        hmm_ = std::make_unique<Hmm>(8);
        
        // Initialize with random probabilities
        Vector pi(8);
        Matrix trans(8, 8);
        
        for (size_t i = 0; i < 8; ++i) {
            pi(i) = 1.0/8.0;  // Equal probabilities
            for (size_t j = 0; j < 8; ++j) {
                trans(i, j) = 1.0/8.0;  // Equal probabilities
            }
        }
        
        hmm_->setPi(pi);
        hmm_->setTrans(trans);
        
        // Set discrete emission distributions
        for (size_t i = 0; i < 8; ++i) {
            auto dist = std::make_unique<DiscreteDistribution>(6);
            for (size_t j = 0; j < 6; ++j) {
                dist->setProbability(j, 1.0/6.0);  // Equal probabilities
            }
            hmm_->setProbabilityDistribution(i, std::move(dist));
        }
        
        // Create longer observation sequence
        longObs_.resize(1000);
        for (size_t i = 0; i < 1000; ++i) {
            longObs_(i) = i % 6;
        }
        
        // Create simple HMM for accuracy tests
        simpleHmm_ = std::make_unique<Hmm>(2);
        Vector simplePi(2);
        Matrix simpleTrans(2, 2);
        
        simplePi(0) = 0.6; simplePi(1) = 0.4;
        simpleTrans(0, 0) = 0.7; simpleTrans(0, 1) = 0.3;
        simpleTrans(1, 0) = 0.4; simpleTrans(1, 1) = 0.6;
        
        simpleHmm_->setPi(simplePi);
        simpleHmm_->setTrans(simpleTrans);
        
        for (size_t i = 0; i < 2; ++i) {
            auto dist = std::make_unique<DiscreteDistribution>(3);
            dist->setProbability(0, 0.5);
            dist->setProbability(1, 0.3);
            dist->setProbability(2, 0.2);
            simpleHmm_->setProbabilityDistribution(i, std::move(dist));
        }
        
        simpleObs_.resize(5);
        for (size_t i = 0; i < 5; ++i) {
            simpleObs_(i) = i % 3;
        }
    }
    
    std::unique_ptr<Hmm> hmm_;
    std::unique_ptr<Hmm> simpleHmm_;
    ObservationSet longObs_;
    ObservationSet simpleObs_;
};

TEST_F(BasicCalculatorPerformanceTest, ForwardBackwardPerformance) {
    std::cout << "\nTesting ForwardBackwardCalculator performance:\n";
    
    // Test multiple iterations for timing
    const int iterations = 20;  // Reduced for CI/test environments
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        ForwardBackwardCalculator calc(*hmm_, longObs_);
        double prob = calc.probability();
        double logProb = calc.getLogProbability();
        
        // Use values to prevent optimization
        volatile double vProb = prob;
        volatile double vLogProb = logProb;
        (void)vProb; (void)vLogProb;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  " << iterations << " iterations completed in " 
              << duration.count() / 1000.0 << " ms\n";
    std::cout << "  Average per calculation: " 
              << (duration.count() / iterations) / 1000.0 << " ms\n";
              
    // Basic performance expectation - should complete in reasonable time
    EXPECT_LT(duration.count() / iterations, 100000);  // Less than 100ms per iteration
}

TEST_F(BasicCalculatorPerformanceTest, ViterbiPerformance) {
    std::cout << "\nTesting ViterbiCalculator performance:\n";
    
    const int iterations = 20;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        ViterbiCalculator calc(*hmm_, longObs_);
        StateSequence seq = calc.decode();
        double logProb = calc.getLogProbability();
        
        // Use values to prevent optimization
        volatile double vLogProb = logProb;
        volatile int vSeqSize = seq.size();
        (void)vLogProb; (void)vSeqSize;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  " << iterations << " iterations completed in " 
              << duration.count() / 1000.0 << " ms\n";
    std::cout << "  Average per calculation: " 
              << (duration.count() / iterations) / 1000.0 << " ms\n";
    
    // Basic performance expectation
    EXPECT_LT(duration.count() / iterations, 50000);  // Less than 50ms per iteration
}

TEST_F(BasicCalculatorPerformanceTest, LogProbabilityConsistency) {
    std::cout << "\nTesting log probability consistency:\n";
    
    ForwardBackwardCalculator fbCalc(*simpleHmm_, simpleObs_);
    ViterbiCalculator viterbiCalc(*simpleHmm_, simpleObs_);
    
    double fbProb = fbCalc.probability();
    double fbLogProb = fbCalc.getLogProbability();
    StateSequence viterbiSeq = viterbiCalc.decode();
    double viterbiLogProb = viterbiCalc.getLogProbability();
    
    std::cout << "  Forward-Backward probability: " << fbProb << "\n";
    std::cout << "  Forward-Backward log probability: " << fbLogProb << "\n";
    std::cout << "  Viterbi log probability: " << viterbiLogProb << "\n";
    std::cout << "  Viterbi sequence length: " << viterbiSeq.size() << "\n";
    
    // Verify log probability consistency
    double expectedLogProb = std::log(fbProb);
    double logProbDiff = std::abs(fbLogProb - expectedLogProb);
    
    std::cout << "  Log probability consistency check: ";
    if (logProbDiff < 1e-10) {
        std::cout << "PASSED (diff: " << logProbDiff << ")\n";
    } else {
        std::cout << "FAILED (diff: " << logProbDiff << ")\n";
    }
    
    // Test assertions
    EXPECT_GT(fbProb, 0.0);
    EXPECT_LE(fbProb, 1.0);
    EXPECT_TRUE(std::isfinite(fbLogProb));
    EXPECT_LE(fbLogProb, 0.0);  // Log of probability <= 1 should be <= 0
    EXPECT_EQ(viterbiSeq.size(), simpleObs_.size());
    EXPECT_TRUE(std::isfinite(viterbiLogProb));
    EXPECT_LT(logProbDiff, 1e-10);  // Consistency check
}

TEST_F(BasicCalculatorPerformanceTest, OptimizationCaching) {
    // Test that the caching optimizations work correctly
    ForwardBackwardCalculator calc1(*simpleHmm_, simpleObs_);
    ForwardBackwardCalculator calc2(*simpleHmm_, simpleObs_);
    
    double prob1 = calc1.probability();
    double prob2 = calc2.probability();
    
    // Should produce identical results
    EXPECT_DOUBLE_EQ(prob1, prob2);
    
    // Test that isComputed() works correctly
    EXPECT_TRUE(calc1.isComputed());
    EXPECT_TRUE(calc2.isComputed());
    
    // Test getLogProbability consistency
    double logProb1 = calc1.getLogProbability();
    double logProb2 = calc2.getLogProbability();
    EXPECT_NEAR(logProb1, logProb2, 1e-15);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
