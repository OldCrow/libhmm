#include <gtest/gtest.h>
#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/distributions.h"
#include <memory>
#include <vector>
#include <iostream>
#include <limits>
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
        trans(0, 0) = 0.5;
        trans(0, 1) = 0.3;
        trans(0, 2) = 0.2;
        trans(1, 0) = 0.2;
        trans(1, 1) = 0.6;
        trans(1, 2) = 0.2;
        trans(2, 0) = 0.1;
        trans(2, 1) = 0.1;
        trans(2, 2) = 0.8;
        gaussianHmm_->setTrans(trans);

        // Set up pi vector
        Vector pi(3);
        pi(0) = 0.4;
        pi(1) = 0.3;
        pi(2) = 0.3;
        gaussianHmm_->setPi(pi);

        // Set up Gaussian emission distributions with different means
        gaussianHmm_->setDistribution(0, std::make_unique<GaussianDistribution>(1.0, 0.5));
        gaussianHmm_->setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.0));
        gaussianHmm_->setDistribution(2, std::make_unique<GaussianDistribution>(10.0, 2.0));
    }

    void setupDiscreteHmm() {
        // Set up transition matrix
        Matrix trans(2, 2);
        trans(0, 0) = 0.7;
        trans(0, 1) = 0.3;
        trans(1, 0) = 0.4;
        trans(1, 1) = 0.6;
        discreteHmm_->setTrans(trans);

        // Set up pi vector
        Vector pi(2);
        pi(0) = 0.6;
        pi(1) = 0.4;
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

        discreteHmm_->setDistribution(0, std::move(dist0));
        discreteHmm_->setDistribution(1, std::move(dist1));
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
    std::streambuf *old_cerr = std::cerr.rdbuf(captured_output.rdbuf());

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

// ScaledBaumWelchTrainer was removed; test short sequences via canonical BaumWelchTrainer
TEST_F(TrainingEdgeCasesTest, BaumWelchTrainerSingleObservationSequence) {
    ObservationLists shortSequences;
    ObservationSet seq1(1);
    seq1(0) = 0;
    shortSequences.push_back(seq1);

    BaumWelchTrainer trainer(discreteHmm_.get(), shortSequences);
    EXPECT_NO_THROW(trainer.train());
}

// Test with zero probabilities in observations
TEST_F(TrainingEdgeCasesTest, BaumWelchTrainerZeroProbabilities) {
    ObservationLists obsWithZeros;

    // Create sequences that might result in zero emission probabilities
    ObservationSet seq1(5);
    seq1(0) = 0;
    seq1(1) = 1;
    seq1(2) = 2;
    seq1(3) = 0;
    seq1(4) = 1;
    obsWithZeros.push_back(seq1);

    // Modify discrete distribution to have some zero probabilities
    auto dist0 = std::make_unique<DiscreteDistribution>(3);
    dist0->setProbability(0, 0.0); // Zero probability for symbol 0
    dist0->setProbability(1, 0.7);
    dist0->setProbability(2, 0.3);
    discreteHmm_->setDistribution(0, std::move(dist0));

    BaumWelchTrainer trainer(discreteHmm_.get(), obsWithZeros);

    // Should handle zero probabilities gracefully
    EXPECT_NO_THROW(trainer.train());
}

// Numerical edge cases: very small probabilities (log-space BW handles these robustly)
TEST_F(TrainingEdgeCasesTest, BaumWelchTrainerNearZeroProbabilities) {
    ObservationLists normalObs;
    ObservationSet seq1(5);
    seq1(0) = 0;
    seq1(1) = 1;
    seq1(2) = 2;
    seq1(3) = 0;
    seq1(4) = 1;
    normalObs.push_back(seq1);

    auto dist0 = std::make_unique<DiscreteDistribution>(3);
    dist0->setProbability(0, 1e-100); // Very small — should not underflow in log-space
    dist0->setProbability(1, 1.0 - 2e-100);
    dist0->setProbability(2, 1e-100);
    discreteHmm_->setDistribution(0, std::move(dist0));

    BaumWelchTrainer trainer(discreteHmm_.get(), normalObs);
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

    // Test empty observation lists
    ObservationLists emptyObs;
    EXPECT_THROW(ViterbiTrainer(gaussianHmm_.get(), emptyObs), std::invalid_argument);
    EXPECT_THROW(BaumWelchTrainer(discreteHmm_.get(), emptyObs), std::invalid_argument);
}

// Test distribution compatibility checking
TEST_F(TrainingEdgeCasesTest, DistributionCompatibilityValidation) {
    ObservationLists normalObs;

    ObservationSet seq1(5);
    seq1(0) = 0;
    seq1(1) = 1;
    seq1(2) = 2;
    seq1(3) = 0;
    seq1(4) = 1;
    normalObs.push_back(seq1);

    // BaumWelchTrainer works with any EmissionDistribution via weighted fit()
    EXPECT_NO_THROW(BaumWelchTrainer(discreteHmm_.get(), normalObs));

    // Canonical BaumWelchTrainer works with both discrete and Gaussian distributions
    // (old ScaledBaumWelchTrainer was discrete-only; removed in Phase 4)
    BaumWelchTrainer bwTrainer(gaussianHmm_.get(), normalObs);
    EXPECT_NO_THROW(bwTrainer.train());

    // ViterbiTrainer works with both discrete and continuous distributions
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
    // The distributions should exist and be accessible (getDistribution returns a reference)
    EXPECT_NO_THROW(gaussianHmm_->getDistribution(0));
    EXPECT_NO_THROW(gaussianHmm_->getDistribution(1));
    EXPECT_NO_THROW(gaussianHmm_->getDistribution(2));
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
        EXPECT_EQ(&trainer->getHmm(), gaussianHmm_.get());

        // trainer should be automatically cleaned up when scope ends
    }

    // HMM should still be valid after trainer destruction
    EXPECT_NO_THROW(gaussianHmm_->validate());
}

// Tr-1: ViterbiTrainer::runIteration must return -inf (not a stale log-prob)
// when every sequence has zero probability under the current model.  A stale
// return value would fool the convergence check into declaring success.
TEST_F(TrainingEdgeCasesTest, ViterbiTrainerAllInvalidSequencesReturnsNegInf) {
    // Discrete HMM with alphabet {0,1,2} — feed symbol 99 (outside alphabet)
    // so every sequence gets log-probability = -inf.
    ObservationLists badObs;
    ObservationSet seq(3);
    seq(0) = 99.0;
    seq(1) = 99.0;
    seq(2) = 99.0;
    badObs.push_back(seq);

    // Clone the discrete HMM so we don't mutate the fixture for later tests.
    Hmm testHmm(2);
    {
        Matrix trans(2, 2);
        trans(0, 0) = 0.7;
        trans(0, 1) = 0.3;
        trans(1, 0) = 0.4;
        trans(1, 1) = 0.6;
        testHmm.setTrans(trans);
        Vector pi(2);
        pi(0) = 0.6;
        pi(1) = 0.4;
        testHmm.setPi(pi);
        auto d0 = std::make_unique<DiscreteDistribution>(3);
        d0->setProbability(0, 0.7);
        d0->setProbability(1, 0.2);
        d0->setProbability(2, 0.1);
        auto d1 = std::make_unique<DiscreteDistribution>(3);
        d1->setProbability(0, 0.1);
        d1->setProbability(1, 0.3);
        d1->setProbability(2, 0.6);
        testHmm.setDistribution(0, std::move(d0));
        testHmm.setDistribution(1, std::move(d1));
    }

    ViterbiTrainer trainer(testHmm, badObs);
    // train() will converge (the window fills with -inf values) but the final
    // log-probability must signal failure (not a pre-training stale value).
    ASSERT_NO_THROW(trainer.train());
    EXPECT_EQ(trainer.getLastLogProbability(), -std::numeric_limits<double>::infinity());
}
// Tr-1b: BaumWelchTrainer should expose -inf when training fails because every
// sequence has zero probability under the current model.
TEST_F(TrainingEdgeCasesTest, BaumWelchTrainerAllInvalidSequencesLeavesNegInf) {
    ObservationLists badObs;
    ObservationSet seq(3);
    seq(0) = 99.0;
    seq(1) = 99.0;
    seq(2) = 99.0;
    badObs.push_back(seq);

    BaumWelchTrainer trainer(discreteHmm_.get(), badObs);
    EXPECT_EQ(trainer.getLastLogProbability(), -std::numeric_limits<double>::infinity());

    EXPECT_THROW(trainer.train(), std::runtime_error);
    EXPECT_EQ(trainer.getLastLogProbability(), -std::numeric_limits<double>::infinity());
}

// Tr-2: BaumWelchTrainer must emit a diagnostic to stderr when all valid
// sequences have length 1 and no transition statistics can be accumulated.
TEST_F(TrainingEdgeCasesTest, BaumWelchTrainerAllLength1EmitsDiagnostic) {
    ObservationLists length1Obs;
    ObservationSet s0(1);
    s0(0) = 0.0;
    ObservationLists::value_type s1(1);
    s1(0) = 1.0;
    ObservationLists::value_type s2(1);
    s2(0) = 2.0;
    length1Obs.push_back(s0);
    length1Obs.push_back(s1);
    length1Obs.push_back(s2);

    BaumWelchTrainer trainer(discreteHmm_.get(), length1Obs);

    // Redirect stderr to capture the diagnostic.
    std::ostringstream captured;
    std::streambuf *old_cerr = std::cerr.rdbuf(captured.rdbuf());
    ASSERT_NO_THROW(trainer.train());
    std::cerr.rdbuf(old_cerr);

    // Verify the diagnostic message was emitted.
    EXPECT_NE(captured.str().find("length 1"), std::string::npos)
        << "Expected length-1 diagnostic in stderr, got: " << captured.str();

    // Parameters must be valid (no NaN) despite the degenerate input.
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isfinite(discreteHmm_->getPi()(i)));
        for (std::size_t j = 0; j < 2; ++j)
            EXPECT_TRUE(std::isfinite(discreteHmm_->getTrans()(i, j)));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
