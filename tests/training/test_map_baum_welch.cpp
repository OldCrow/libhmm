#include <gtest/gtest.h>
#include "libhmm/training/map_baum_welch_trainer.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <cmath>
#include <memory>

using namespace libhmm;

namespace {

/// 2-state, 6-symbol casino HMM with uniform initial parameters.
std::unique_ptr<Hmm> make_uniform_hmm() {
    auto hmm = std::make_unique<Hmm>(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.5;
    trans(0, 1) = 0.5;
    trans(1, 0) = 0.5;
    trans(1, 1) = 0.5;
    hmm->setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm->setPi(pi);
    for (int s = 0; s < 2; ++s)
        hmm->setDistribution(s, std::make_unique<DiscreteDistribution>(6));
    return hmm;
}

/// Observation sequences: first half low symbols (fair), second half sixes (loaded).
ObservationLists make_casino_obs() {
    ObservationLists obs;
    for (int s = 0; s < 5; ++s) {
        ObservationSet seq(20);
        for (int t = 0; t < 10; ++t)
            seq(t) = static_cast<double>(t % 4); // 0-3, fair-like
        for (int t = 10; t < 20; ++t)
            seq(t) = 5.0; // all sixes, loaded-like
        obs.push_back(seq);
    }
    return obs;
}

/// Total log-likelihood across all sequences.
double total_logL(const Hmm &hmm, const ObservationLists &seqs) {
    double ll = 0.0;
    for (const auto &seq : seqs)
        ll += ForwardBackwardCalculator(hmm, seq).getLogProbability();
    return ll;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction and validation
// ---------------------------------------------------------------------------

TEST(MapBaumWelchTest, NegativePseudoCountThrows) {
    auto hmm = make_uniform_hmm();
    auto obs = make_casino_obs();
    EXPECT_THROW(MapBaumWelchTrainer(*hmm, obs, -0.1), std::invalid_argument);
}

TEST(MapBaumWelchTest, NullHmmThrows) {
    auto obs = make_casino_obs();
    EXPECT_THROW(MapBaumWelchTrainer(nullptr, obs, 1.0), std::invalid_argument);
}

TEST(MapBaumWelchTest, EmptyObsThrows) {
    auto hmm = make_uniform_hmm();
    EXPECT_THROW(MapBaumWelchTrainer(*hmm, {}, 1.0), std::invalid_argument);
}

TEST(MapBaumWelchTest, SetPseudoCountNegativeThrows) {
    auto hmm = make_uniform_hmm();
    MapBaumWelchTrainer trainer(*hmm, make_casino_obs(), 1.0);
    EXPECT_THROW(trainer.setPseudoCount(-0.5), std::invalid_argument);
}

TEST(MapBaumWelchTest, GetPseudoCount) {
    auto hmm = make_uniform_hmm();
    MapBaumWelchTrainer trainer(*hmm, make_casino_obs(), 2.5);
    EXPECT_DOUBLE_EQ(trainer.getPseudoCount(), 2.5);
    trainer.setPseudoCount(0.0);
    EXPECT_DOUBLE_EQ(trainer.getPseudoCount(), 0.0);
}

// ---------------------------------------------------------------------------
// c = 0 recovers standard Baum-Welch exactly
// ---------------------------------------------------------------------------

TEST(MapBaumWelchTest, ZeroPseudoCountMatchesBaumWelch) {
    auto obs = make_casino_obs();

    auto hmmBW = make_uniform_hmm();
    auto hmmMAP = make_uniform_hmm();

    BaumWelchTrainer bw(*hmmBW, obs);
    MapBaumWelchTrainer map(*hmmMAP, obs, 0.0);

    bw.train();
    map.train();

    // Transitions must match to floating-point precision.
    const Matrix &tbw = hmmBW->getTrans();
    const Matrix &tmap = hmmMAP->getTrans();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(tbw(i, j), tmap(i, j), 1e-12)
                << "A(" << i << "," << j << ") mismatch at c=0";

    // Pi must match.
    const Vector &pbw = hmmBW->getPi();
    const Vector &pmap = hmmMAP->getPi();
    for (int i = 0; i < 2; ++i)
        EXPECT_NEAR(pbw(i), pmap(i), 1e-12) << "pi(" << i << ") mismatch at c=0";
}

// ---------------------------------------------------------------------------
// c > 0 produces strictly positive transitions and smoothed emissions
// ---------------------------------------------------------------------------

TEST(MapBaumWelchTest, PositivePseudoCountSmooths) {
    auto hmm = make_uniform_hmm();
    auto obs = make_casino_obs();
    MapBaumWelchTrainer trainer(*hmm, obs, 1.0);
    trainer.train();

    const Matrix &A = hmm->getTrans();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_GT(A(i, j), 0.0) << "A(" << i << "," << j << ") should be > 0 with c=1";

    const Vector &pi = hmm->getPi();
    for (int i = 0; i < 2; ++i)
        EXPECT_GT(pi(i), 0.0) << "pi(" << i << ") should be > 0 with c=1";
}

TEST(MapBaumWelchTest, DiscreteEmissionsSmoothed) {
    auto hmm = make_uniform_hmm();
    auto obs = make_casino_obs();
    MapBaumWelchTrainer trainer(*hmm, obs, 1.0);
    trainer.train();

    // After smoothing with c=1 and K=6, every symbol should have positive probability.
    for (int s = 0; s < 2; ++s) {
        const auto &dd = static_cast<const DiscreteDistribution &>(hmm->getDistribution(s));
        for (std::size_t k = 0; k < dd.getNumSymbols(); ++k)
            EXPECT_GT(dd.getSymbolProbability(k), 0.0)
                << "B(" << s << "," << k << ") should be > 0 with c=1";
    }
}

// ---------------------------------------------------------------------------
// computeLogPrior
// ---------------------------------------------------------------------------

TEST(MapBaumWelchTest, LogPriorZeroWhenCIsZero) {
    auto hmm = make_uniform_hmm();
    MapBaumWelchTrainer trainer(*hmm, make_casino_obs(), 0.0);
    EXPECT_DOUBLE_EQ(trainer.computeLogPrior(), 0.0);
}

TEST(MapBaumWelchTest, LogPriorFiniteAndNonPositive) {
    auto hmm = make_uniform_hmm();
    MapBaumWelchTrainer trainer(*hmm, make_casino_obs(), 1.0);
    trainer.train();
    const double lp = trainer.computeLogPrior();
    EXPECT_TRUE(std::isfinite(lp));
    EXPECT_LE(lp, 0.0); // log of probabilities ≤ 1, scaled by c > 0
}

// ---------------------------------------------------------------------------
// MAP objective is monotone over multiple iterations
// ---------------------------------------------------------------------------

TEST(MapBaumWelchTest, MapObjectiveMonotone) {
    auto hmm = make_uniform_hmm();
    auto obs = make_casino_obs();
    MapBaumWelchTrainer trainer(*hmm, obs, 1.0);

    double prevObjective = -std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < 8; ++iter) {
        trainer.train();
        const double logL = total_logL(*hmm, obs);
        const double logPr = trainer.computeLogPrior();
        const double mapObj = logL + logPr;
        EXPECT_GE(mapObj, prevObjective - 1e-6) // allow tiny floating-point slack
            << "MAP objective decreased at iteration " << iter;
        prevObjective = mapObj;
    }
}

// ---------------------------------------------------------------------------
// Continuous emissions use MLE (unaffected by pseudo-count)
// ---------------------------------------------------------------------------

TEST(MapBaumWelchTest, ContinuousEmissionsUnchangedByPseudoCount) {
    // Build a 2-state Gaussian HMM.
    auto hmmMLE = std::make_unique<Hmm>(2);
    auto hmmMAP = std::make_unique<Hmm>(2);
    for (auto *h : {hmmMLE.get(), hmmMAP.get()}) {
        Matrix trans(2, 2);
        trans(0, 0) = 0.8;
        trans(0, 1) = 0.2;
        trans(1, 0) = 0.3;
        trans(1, 1) = 0.7;
        h->setTrans(trans);
        Vector pi(2);
        pi(0) = 0.5;
        pi(1) = 0.5;
        h->setPi(pi);
        h->setDistribution(0, std::make_unique<GaussianDistribution>(-2.0, 1.0));
        h->setDistribution(1, std::make_unique<GaussianDistribution>(2.0, 1.0));
    }

    ObservationLists obs;
    ObservationSet seq(20);
    for (int t = 0; t < 10; ++t)
        seq(t) = -2.0 + 0.1 * (t % 3);
    for (int t = 10; t < 20; ++t)
        seq(t) = 2.0 + 0.1 * (t % 3);
    obs.push_back(seq);

    BaumWelchTrainer bw(*hmmMLE, obs);
    MapBaumWelchTrainer map(*hmmMAP, obs, 2.0);
    bw.train();
    map.train();

    // Gaussian means should be equal — c only affects transitions and π.
    const auto &g0mle = static_cast<const GaussianDistribution &>(hmmMLE->getDistribution(0));
    const auto &g0map = static_cast<const GaussianDistribution &>(hmmMAP->getDistribution(0));
    EXPECT_NEAR(g0mle.getMean(), g0map.getMean(), 1e-10);
    EXPECT_NEAR(g0mle.getStandardDeviation(), g0map.getStandardDeviation(), 1e-10);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
