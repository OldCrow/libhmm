#include <gtest/gtest.h>
#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/training/segmental_kmeans_trainer.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <cmath>
#include <limits>
#include <memory>

using namespace libhmm;

// ---------------------------------------------------------------------------
// Helper: 2-state discrete HMM (fair/loaded die)
// ---------------------------------------------------------------------------

static std::unique_ptr<Hmm> makeDiscreteHmm() {
    auto hmm = std::make_unique<Hmm>(2);

    Matrix trans(2, 2);
    trans(0, 0) = 0.8;
    trans(0, 1) = 0.2;
    trans(1, 0) = 0.3;
    trans(1, 1) = 0.7;
    hmm->setTrans(trans);

    Vector pi(2);
    pi(0) = 0.6;
    pi(1) = 0.4;
    hmm->setPi(pi);

    auto d0 = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 6; ++i)
        d0->setProbability(i, 1.0 / 6.0);
    hmm->setDistribution(0, std::move(d0));

    auto d1 = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 5; ++i)
        d1->setProbability(i, 0.1);
    d1->setProbability(5, 0.5);
    hmm->setDistribution(1, std::move(d1));

    return hmm;
}

// ---------------------------------------------------------------------------
// Helper: 3-state Gaussian HMM
// ---------------------------------------------------------------------------

static std::unique_ptr<Hmm> makeGaussianHmm() {
    auto hmm = std::make_unique<Hmm>(3);

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
    hmm->setTrans(trans);

    Vector pi(3);
    pi(0) = 0.33;
    pi(1) = 0.33;
    pi(2) = 0.34;
    hmm->setPi(pi);

    hmm->setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm->setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.5));
    hmm->setDistribution(2, std::make_unique<GaussianDistribution>(10.0, 2.0));

    return hmm;
}

// ---------------------------------------------------------------------------
// Helper: discrete observation sequences
// ---------------------------------------------------------------------------

static ObservationLists makeDiscreteObs() {
    ObservationLists lists;
    ObservationSet s1(10);
    for (std::size_t i = 0; i < 10; ++i)
        s1(i) = static_cast<double>(i % 6);
    lists.push_back(s1);
    ObservationSet s2(8);
    for (std::size_t i = 0; i < 8; ++i)
        s2(i) = static_cast<double>((i + 2) % 6);
    lists.push_back(s2);
    return lists;
}

// ---------------------------------------------------------------------------
// Helper: Gaussian observation sequences
// ---------------------------------------------------------------------------

static ObservationLists makeGaussianObs() {
    ObservationLists lists;
    ObservationSet s1(15);
    for (std::size_t i = 0; i < 15; ++i)
        s1(i) = static_cast<double>(i) * 0.5;
    lists.push_back(s1);
    ObservationSet s2(12);
    for (std::size_t i = 0; i < 12; ++i)
        s2(i) = 5.0 + static_cast<double>(i) * 0.3;
    lists.push_back(s2);
    return lists;
}

// ===========================================================================
// BaumWelchTrainer
// ===========================================================================

TEST(BaumWelchTrainerTest, Construction) {
    auto hmm = makeDiscreteHmm();
    auto obs = makeDiscreteObs();
    EXPECT_NO_THROW(BaumWelchTrainer(hmm.get(), obs));
}

TEST(BaumWelchTrainerTest, NullHmmThrows) {
    auto obs = makeDiscreteObs();
    EXPECT_THROW(BaumWelchTrainer(nullptr, obs), std::invalid_argument);
}

TEST(BaumWelchTrainerTest, EmptyObsThrows) {
    auto hmm = makeDiscreteHmm();
    ObservationLists empty;
    EXPECT_THROW(BaumWelchTrainer(hmm.get(), empty), std::invalid_argument);
}

TEST(BaumWelchTrainerTest, GetHmmReturnsCorrectHmm) {
    auto hmm = makeDiscreteHmm();
    auto obs = makeDiscreteObs();
    BaumWelchTrainer trainer(hmm.get(), obs);
    EXPECT_EQ(&trainer.getHmm(), hmm.get());
}

TEST(BaumWelchTrainerTest, TrainUpdatesParameters) {
    auto hmm = makeDiscreteHmm();
    auto obs = makeDiscreteObs();
    Matrix origTrans = hmm->getTrans();

    BaumWelchTrainer trainer(hmm.get(), obs);
    EXPECT_NO_THROW(trainer.train());

    // Parameters should have changed
    Matrix newTrans = hmm->getTrans();
    bool changed = false;
    for (std::size_t i = 0; i < 2 && !changed; ++i)
        for (std::size_t j = 0; j < 2 && !changed; ++j)
            if (std::abs(origTrans(i, j) - newTrans(i, j)) > 1e-10)
                changed = true;
    EXPECT_TRUE(changed);
}

TEST(BaumWelchTrainerTest, TrainPreservesHmmValidity) {
    auto hmm = makeDiscreteHmm();
    auto obs = makeDiscreteObs();
    BaumWelchTrainer trainer(hmm.get(), obs);
    trainer.train();
    EXPECT_NO_THROW(hmm->validate());
}
TEST(BaumWelchTrainerTest, ExposesLastLogProbability) {
    auto hmm = makeDiscreteHmm();
    auto obs = makeDiscreteObs();
    double expectedLogProb = 0.0;
    for (const auto &seq : obs) {
        BasicForwardBackwardCalculator<double> fbc(*hmm, seq);
        expectedLogProb += fbc.getLogProbability();
    }

    BaumWelchTrainer trainer(hmm.get(), obs);
    EXPECT_EQ(trainer.getLastLogProbability(), -std::numeric_limits<double>::infinity());

    ASSERT_NO_THROW(trainer.train());
    EXPECT_NEAR(trainer.getLastLogProbability(), expectedLogProb, 1e-10);
}

TEST(BaumWelchTrainerTest, MultipleRoundsStayValid) {
    auto hmm = makeDiscreteHmm();
    auto obs = makeDiscreteObs();
    BaumWelchTrainer trainer(hmm.get(), obs);
    EXPECT_NO_THROW(trainer.train());
    EXPECT_NO_THROW(trainer.train());
    EXPECT_NO_THROW(hmm->validate());
}

TEST(BaumWelchTrainerTest, WorksWithGaussianDistributions) {
    // New design: BaumWelch works with any EmissionDistribution via weighted fit
    auto hmm = makeGaussianHmm();
    auto obs = makeGaussianObs();
    BaumWelchTrainer trainer(hmm.get(), obs);
    EXPECT_NO_THROW(trainer.train());
    EXPECT_NO_THROW(hmm->validate());
}

// ===========================================================================
// ViterbiTrainer
// ===========================================================================

TEST(ViterbiTrainerTest, Construction) {
    auto hmm = makeGaussianHmm();
    auto obs = makeGaussianObs();
    EXPECT_NO_THROW(ViterbiTrainer(hmm.get(), obs));
}

TEST(ViterbiTrainerTest, NullHmmThrows) {
    auto obs = makeGaussianObs();
    EXPECT_THROW(ViterbiTrainer(nullptr, obs), std::invalid_argument);
}

TEST(ViterbiTrainerTest, EmptyObsThrows) {
    auto hmm = makeGaussianHmm();
    ObservationLists empty;
    EXPECT_THROW(ViterbiTrainer(hmm.get(), empty), std::invalid_argument);
}

TEST(ViterbiTrainerTest, GetHmmReturnsCorrectHmm) {
    auto hmm = makeGaussianHmm();
    auto obs = makeGaussianObs();
    ViterbiTrainer trainer(hmm.get(), obs);
    EXPECT_EQ(&trainer.getHmm(), hmm.get());
}

TEST(ViterbiTrainerTest, TrainUpdatesParameters) {
    auto hmm = makeGaussianHmm();
    auto obs = makeGaussianObs();
    Matrix origTrans = hmm->getTrans();

    ViterbiTrainer trainer(hmm.get(), obs);
    EXPECT_NO_THROW(trainer.train());

    Matrix newTrans = hmm->getTrans();
    bool changed = false;
    for (std::size_t i = 0; i < 3 && !changed; ++i)
        for (std::size_t j = 0; j < 3 && !changed; ++j)
            if (std::abs(origTrans(i, j) - newTrans(i, j)) > 1e-10)
                changed = true;
    EXPECT_TRUE(changed);
}

TEST(ViterbiTrainerTest, TrainSetsConvergenceFlags) {
    auto hmm = makeGaussianHmm();
    auto obs = makeGaussianObs();
    // Use a small iteration cap to guarantee one flag gets set
    ViterbiTrainer trainer(hmm.get(), obs, {1e-6, 200, 3, false});
    trainer.train();
    EXPECT_TRUE(trainer.hasConverged() || trainer.reachedMaxIterations());
}

TEST(ViterbiTrainerTest, PreservesHmmValidity) {
    auto hmm = makeGaussianHmm();
    auto obs = makeGaussianObs();
    ViterbiTrainer trainer(hmm.get(), obs, {1e-6, 50, 3, false});
    trainer.train();
    EXPECT_NO_THROW(hmm->validate());
}

TEST(ViterbiTrainerTest, Presets) {
    EXPECT_LT(training_presets::precise().convergenceTolerance,
              training_presets::balanced().convergenceTolerance);
    EXPECT_LT(training_presets::balanced().convergenceTolerance,
              training_presets::fast().convergenceTolerance);
    EXPECT_GT(training_presets::precise().maxIterations, training_presets::fast().maxIterations);
}

// ===========================================================================
// SegmentalKMeansTrainer
// ===========================================================================

TEST(SegmentalKMeansTrainerTest, Construction) {
    auto hmm = makeDiscreteHmm();
    auto obs = makeDiscreteObs();
    EXPECT_NO_THROW(SegmentalKMeansTrainer(hmm.get(), obs));
}

TEST(SegmentalKMeansTrainerTest, NullHmmThrows) {
    auto obs = makeDiscreteObs();
    EXPECT_THROW(SegmentalKMeansTrainer(nullptr, obs), std::invalid_argument);
}

TEST(SegmentalKMeansTrainerTest, EmptyObsThrows) {
    auto hmm = makeDiscreteHmm();
    ObservationLists empty;
    EXPECT_THROW(SegmentalKMeansTrainer(hmm.get(), empty), std::invalid_argument);
}

TEST(SegmentalKMeansTrainerTest, AcceptsAnyScalarDistribution) {
    // The discrete-only restriction was removed in favour of the generic fit()
    // M-step. SegmentalKMeansTrainer now works with any scalar EmissionDistribution,
    // including Gaussian. Construction must not throw.
    auto hmm = makeGaussianHmm();
    auto obs = makeGaussianObs();
    EXPECT_NO_THROW(SegmentalKMeansTrainer(hmm.get(), obs));
}

TEST(SegmentalKMeansTrainerTest, InitialState) {
    auto hmm = makeDiscreteHmm();
    auto obs = makeDiscreteObs();
    SegmentalKMeansTrainer trainer(hmm.get(), obs);
    EXPECT_FALSE(trainer.isTerminated());
    EXPECT_EQ(&trainer.getHmm(), hmm.get());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
