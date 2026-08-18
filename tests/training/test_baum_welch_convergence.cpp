#include <gtest/gtest.h>
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <cmath>
#include <limits>
#include <memory>

using namespace libhmm;

// Sum log P(O_k | λ) over all sequences.
// This is the quantity Baum-Welch maximises at each iteration.
static double total_log_likelihood(const Hmm &hmm, const ObservationLists &seqs) {
    double total = 0.0;
    for (const auto &seq : seqs) {
        if (seq.size() == 0)
            continue;
        ForwardBackwardCalculator fbc(hmm, seq);
        total += fbc.getLogProbability();
    }
    return total;
}

// ---------------------------------------------------------------------------
// Discrete HMM fixture (fair/loaded die)
// ---------------------------------------------------------------------------

class BaumWelchConvergenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        hmm_ = std::make_unique<Hmm>(2);
        Matrix trans(2, 2);
        trans(0, 0) = 0.8;
        trans(0, 1) = 0.2;
        trans(1, 0) = 0.3;
        trans(1, 1) = 0.7;
        hmm_->setTrans(trans);
        Vector pi(2);
        pi(0) = 0.6;
        pi(1) = 0.4;
        hmm_->setPi(pi);

        auto d0 = std::make_unique<DiscreteDistribution>(6);
        for (int i = 0; i < 6; ++i)
            d0->setProbability(i, 1.0 / 6.0);
        hmm_->setDistribution(0, std::move(d0));

        auto d1 = std::make_unique<DiscreteDistribution>(6);
        for (int i = 0; i < 5; ++i)
            d1->setProbability(i, 0.1);
        d1->setProbability(5, 0.5);
        hmm_->setDistribution(1, std::move(d1));

        // Three observation sequences of varying length
        obs_.clear();
        ObservationSet s1(20);
        for (std::size_t i = 0; i < 20; ++i)
            s1(i) = static_cast<double>(i % 6);
        obs_.push_back(s1);
        ObservationSet s2(15);
        for (std::size_t i = 0; i < 15; ++i)
            s2(i) = static_cast<double>((i * 2 + 1) % 6);
        obs_.push_back(s2);
        ObservationSet s3(12);
        for (std::size_t i = 0; i < 12; ++i)
            s3(i) = static_cast<double>((i + 3) % 6);
        obs_.push_back(s3);
    }

    std::unique_ptr<Hmm> hmm_;
    ObservationLists obs_;
};

// ---------------------------------------------------------------------------
// Core EM monotonicity property
// ---------------------------------------------------------------------------

TEST_F(BaumWelchConvergenceTest, LogLikelihoodNonDecreasing) {
    // Each Baum-Welch iteration must not decrease total log-likelihood.
    // A small tolerance (1e-6) absorbs floating-point rounding in the
    // M-step normalisation.
    BaumWelchTrainer trainer(hmm_.get(), obs_);

    double prev_ll = total_log_likelihood(*hmm_, obs_);
    ASSERT_TRUE(std::isfinite(prev_ll)) << "Initial log-likelihood must be finite";

    for (int iter = 0; iter < 10; ++iter) {
        trainer.train();
        const double curr_ll = total_log_likelihood(*hmm_, obs_);
        EXPECT_GE(curr_ll, prev_ll - 1e-6) << "Log-likelihood decreased at iteration " << iter + 1
                                           << ": " << prev_ll << " -> " << curr_ll;
        prev_ll = curr_ll;
    }
}

TEST_F(BaumWelchConvergenceTest, LogLikelihoodImproves) {
    // After sufficient iterations the log-likelihood should be strictly
    // higher than the untrained starting point.
    const double initial_ll = total_log_likelihood(*hmm_, obs_);

    BaumWelchTrainer trainer(hmm_.get(), obs_);
    for (int i = 0; i < 10; ++i)
        trainer.train();

    const double final_ll = total_log_likelihood(*hmm_, obs_);
    EXPECT_GT(final_ll, initial_ll);
}

// ---------------------------------------------------------------------------
// clone() under training (issue #43 acceptance): two clones of one initial
// model, trained independently, diverge — and training one leaves the other
// (and the original) untouched.
// ---------------------------------------------------------------------------

TEST_F(BaumWelchConvergenceTest, ClonesTrainIndependentlyAndCanDiverge) {
    Hmm a = hmm_->clone();
    Hmm b = hmm_->clone();

    // Train a on the fixture data; train b on a differently-shaped set
    // (heavy sixes), so the two EM runs pull the parameters apart.
    ObservationLists heavySixes;
    ObservationSet s(24);
    for (std::size_t i = 0; i < 24; ++i)
        s(i) = (i % 4 == 0) ? static_cast<double>(i % 3) : 5.0;
    heavySixes.push_back(s);

    BaumWelchTrainer ta(&a, obs_);
    BaumWelchTrainer tb(&b, heavySixes);
    for (int i = 0; i < 5; ++i) {
        ta.train();
        tb.train();
    }

    // The original is untouched by either run.
    EXPECT_DOUBLE_EQ(hmm_->getTrans()(0, 0), 0.8);
    EXPECT_DOUBLE_EQ(hmm_->getPi()(0), 0.6);

    // The two trained clones diverged from each other.
    bool diverged = false;
    for (std::size_t i = 0; i < 2 && !diverged; ++i)
        for (std::size_t j = 0; j < 2 && !diverged; ++j)
            if (std::abs(a.getTrans()(i, j) - b.getTrans()(i, j)) > 1e-6)
                diverged = true;
    EXPECT_TRUE(diverged) << "independent training runs produced identical transition matrices";

    // Both trained models remain valid.
    EXPECT_NO_THROW(a.validate());
    EXPECT_NO_THROW(b.validate());
}

TEST_F(BaumWelchConvergenceTest, HmmRemainsValidAfterTraining) {
    BaumWelchTrainer trainer(hmm_.get(), obs_);
    for (int i = 0; i < 5; ++i) {
        trainer.train();
        EXPECT_NO_THROW(hmm_->validate());
    }
}

// ---------------------------------------------------------------------------
// Gaussian distributions — EM monotonicity also holds for continuous case
// ---------------------------------------------------------------------------

TEST(BaumWelchConvergenceGaussianTest, LogLikelihoodNonDecreasing) {
    Hmm hmm(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.7;
    trans(0, 1) = 0.3;
    trans(1, 0) = 0.3;
    trans(1, 1) = 0.7;
    hmm.setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 2.0));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 2.0));

    ObservationLists obs;
    ObservationSet seq(30);
    for (std::size_t i = 0; i < 15; ++i)
        seq(i) = static_cast<double>(i % 3) * 0.5;
    for (std::size_t i = 15; i < 30; ++i)
        seq(i) = 5.0 + static_cast<double>(i % 3) * 0.5;
    obs.push_back(seq);

    BaumWelchTrainer trainer(&hmm, obs);
    double prev_ll = total_log_likelihood(hmm, obs);
    ASSERT_TRUE(std::isfinite(prev_ll));

    for (int iter = 0; iter < 5; ++iter) {
        trainer.train();
        const double curr_ll = total_log_likelihood(hmm, obs);
        EXPECT_GE(curr_ll, prev_ll - 1e-6)
            << "Gaussian BW: log-likelihood decreased at iteration " << iter + 1;
        prev_ll = curr_ll;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
