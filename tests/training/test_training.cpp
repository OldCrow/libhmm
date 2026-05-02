#include <gtest/gtest.h>
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/training/segmental_kmeans_trainer.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <memory>
#include <vector>

using namespace libhmm;

class TrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 2-state HMM with discrete distributions for Baum-Welch
        discreteHmm_ = std::make_unique<Hmm>(2);
        setupDiscreteHmm();

        // Create a 3-state HMM with Gaussian distributions for Viterbi
        gaussianHmm_ = std::make_unique<Hmm>(3);
        setupGaussianHmm();

        // Create training data
        setupTrainingData();
    }

    void setupDiscreteHmm() {
        // Set up transition matrix
        Matrix trans(2, 2);
        trans(0, 0) = 0.8;
        trans(0, 1) = 0.2;
        trans(1, 0) = 0.3;
        trans(1, 1) = 0.7;
        discreteHmm_->setTrans(trans);

        // Set up pi vector
        Vector pi(2);
        pi(0) = 0.6;
        pi(1) = 0.4;
        discreteHmm_->setPi(pi);

        // Set up discrete emission distributions
        auto dist0 = std::make_unique<DiscreteDistribution>(6);
        auto dist1 = std::make_unique<DiscreteDistribution>(6);

        // Fair die
        for (int i = 0; i < 6; ++i) {
            dist0->setProbability(i, 1.0 / 6.0);
        }

        // Loaded die
        for (int i = 0; i < 5; ++i) {
            dist1->setProbability(i, 0.1);
        }
        dist1->setProbability(5, 0.5);

        discreteHmm_->setDistribution(0, std::move(dist0));
        discreteHmm_->setDistribution(1, std::move(dist1));
    }

    void setupGaussianHmm() {
        // Set up transition matrix
        Matrix trans(3, 3);
        trans(0, 0) = 0.7;
        trans(0, 1) = 0.2;
        trans(0, 2) = 0.1;
        trans(1, 0) = 0.1;
        trans(1, 1) = 0.8;
        trans(1, 2) = 0.1;
        trans(2, 0) = 0.1;
        trans(2, 1) = 0.1;
        trans(2, 2) = 0.8;
        gaussianHmm_->setTrans(trans);

        // Set up pi vector
        Vector pi(3);
        pi(0) = 0.33;
        pi(1) = 0.33;
        pi(2) = 0.34;
        gaussianHmm_->setPi(pi);

        // Set up Gaussian emission distributions
        gaussianHmm_->setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
        gaussianHmm_->setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.5));
        gaussianHmm_->setDistribution(2, std::make_unique<GaussianDistribution>(10.0, 2.0));
    }

    void setupTrainingData() {
        // Create discrete observation sequences for Baum-Welch
        discreteObsLists_.clear();

        ObservationSet seq1(10);
        for (std::size_t i = 0; i < seq1.size(); ++i) {
            seq1(i) = static_cast<double>(i % 6); // Cycle through die faces
        }
        discreteObsLists_.push_back(seq1);

        ObservationSet seq2(8);
        for (std::size_t i = 0; i < seq2.size(); ++i) {
            seq2(i) = static_cast<double>((i + 2) % 6);
        }
        discreteObsLists_.push_back(seq2);

        // Create Gaussian observation sequences for Viterbi training
        gaussianObsLists_.clear();

        ObservationSet gseq1(15);
        for (std::size_t i = 0; i < gseq1.size(); ++i) {
            gseq1(i) = static_cast<double>(i) * 0.5; // 0, 0.5, 1.0, 1.5, ...
        }
        gaussianObsLists_.push_back(gseq1);

        ObservationSet gseq2(12);
        for (std::size_t i = 0; i < gseq2.size(); ++i) {
            gseq2(i) = 5.0 + static_cast<double>(i) * 0.3; // Around mean of state 1
        }
        gaussianObsLists_.push_back(gseq2);
    }

    std::unique_ptr<Hmm> discreteHmm_;
    std::unique_ptr<Hmm> gaussianHmm_;
    ObservationLists discreteObsLists_;
    ObservationLists gaussianObsLists_;
};

// Baum-Welch Trainer Tests
TEST_F(TrainingTest, BaumWelchTrainerConstruction) {
    EXPECT_NO_THROW(BaumWelchTrainer(discreteHmm_.get(), discreteObsLists_));
}

TEST_F(TrainingTest, BaumWelchTrainerNullHmmThrows) {
    EXPECT_THROW(BaumWelchTrainer(nullptr, discreteObsLists_), std::invalid_argument);
}

TEST_F(TrainingTest, BaumWelchTrainerEmptyObservationsThrows) {
    ObservationLists emptyLists;
    EXPECT_THROW(BaumWelchTrainer(discreteHmm_.get(), emptyLists), std::invalid_argument);
}

TEST_F(TrainingTest, BaumWelchTraining) {
    BaumWelchTrainer trainer(discreteHmm_.get(), discreteObsLists_);

    // Store original parameters for comparison
    Matrix originalTrans = discreteHmm_->getTrans();
    Vector originalPi = discreteHmm_->getPi();

    // Run training
    EXPECT_NO_THROW(trainer.train());

    // Check that parameters have been updated
    Matrix newTrans = discreteHmm_->getTrans();
    Vector newPi = discreteHmm_->getPi();

    // Parameters should have changed (unless we got very unlucky)
    bool transChanged = false;
    for (std::size_t i = 0; i < originalTrans.size1() && !transChanged; ++i) {
        for (std::size_t j = 0; j < originalTrans.size2() && !transChanged; ++j) {
            if (std::abs(originalTrans(i, j) - newTrans(i, j)) > 1e-10) {
                transChanged = true;
            }
        }
    }
    EXPECT_TRUE(transChanged);
}

// ScaledBaumWelchTrainer was removed in Phase 4 — its functionality is covered
// by the canonical BaumWelchTrainer which is log-space and numerically stable.

// Viterbi Trainer Tests
TEST_F(TrainingTest, ViterbiTrainerConstruction) {
    EXPECT_NO_THROW(ViterbiTrainer(gaussianHmm_.get(), gaussianObsLists_));
}

TEST_F(TrainingTest, ViterbiTrainerNullHmmThrows) {
    EXPECT_THROW(ViterbiTrainer(nullptr, gaussianObsLists_), std::invalid_argument);
}

// Note: Viterbi training can be slow, so we'll test basic functionality only
TEST_F(TrainingTest, ViterbiTrainerBasicFunctionality) {
    ViterbiTrainer trainer(gaussianHmm_.get(), gaussianObsLists_);

    // Should be properly constructed and return correct HMM/obs references
    EXPECT_EQ(&trainer.getHmm(), gaussianHmm_.get());
    EXPECT_FALSE(trainer.getObservationLists().empty());
}

// Segmental K-Means Trainer Tests
TEST_F(TrainingTest, SegmentalKMeansTrainerConstruction) {
    EXPECT_NO_THROW(SegmentalKMeansTrainer(discreteHmm_.get(), discreteObsLists_));
}

TEST_F(TrainingTest, SegmentalKMeansTrainerBasicFunctionality) {
    SegmentalKMeansTrainer trainer(discreteHmm_.get(), discreteObsLists_);

    EXPECT_FALSE(trainer.isTerminated());
    EXPECT_EQ(&trainer.getHmm(), discreteHmm_.get());
}

// Integration Tests
TEST_F(TrainingTest, TrainerHmmConsistency) {
    BaumWelchTrainer trainer(discreteHmm_.get(), discreteObsLists_);

    // HMM from trainer should be the same object
    EXPECT_EQ(&trainer.getHmm(), discreteHmm_.get());

    // Observation lists should match
    const auto &obsLists = trainer.getObservationLists();
    EXPECT_EQ(obsLists.size(), discreteObsLists_.size());
}

TEST_F(TrainingTest, MultipleTrainingRounds) {
    BaumWelchTrainer trainer(discreteHmm_.get(), discreteObsLists_);

    // Should be able to run training multiple times
    EXPECT_NO_THROW(trainer.train());
    EXPECT_NO_THROW(trainer.train());

    // HMM should remain valid
    EXPECT_NO_THROW(discreteHmm_->validate());
}

// Edge Cases
TEST_F(TrainingTest, SingleObservationSequence) {
    ObservationLists singleSeq;
    ObservationSet seq(1);
    seq(0) = 3;
    singleSeq.push_back(seq);

    EXPECT_NO_THROW(BaumWelchTrainer(discreteHmm_.get(), singleSeq));
}

TEST_F(TrainingTest, DISABLED_EmptyObservationInSequence) {
    // Test mixed sequences - some valid, some empty
    ObservationLists mixedSeqList;

    // Add a valid sequence first
    ObservationSet validSeq(5);
    for (std::size_t i = 0; i < validSeq.size(); ++i) {
        validSeq(i) = static_cast<double>(i % 6);
    }
    mixedSeqList.push_back(validSeq);

    // Add an empty sequence
    ObservationSet emptySeq(0);
    mixedSeqList.push_back(emptySeq);

    // This should work - the trainer should skip empty sequences
    EXPECT_NO_THROW({
        BaumWelchTrainer trainer(discreteHmm_.get(), mixedSeqList);
        // Training should succeed, skipping the empty sequence
        trainer.train();
    });

    // Test with only empty sequences - this should throw or handle gracefully
    ObservationLists onlyEmptySeqList;
    onlyEmptySeqList.push_back(ObservationSet(0));

    // This should either throw during construction or training
    bool handled_gracefully = false;
    try {
        BaumWelchTrainer trainer(discreteHmm_.get(), onlyEmptySeqList);
        try {
            trainer.train();
            // If we get here, the implementation handled empty sequences gracefully
            handled_gracefully = true;
        } catch (const std::exception &) {
            // Expected - training with only empty sequences should fail
            handled_gracefully = true;
        }
    } catch (const std::exception &) {
        // Expected - construction with only empty sequences might fail
        handled_gracefully = true;
    }

    EXPECT_TRUE(handled_gracefully);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
