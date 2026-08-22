#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>

#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/training/fit_best_of_n.h"

using namespace libhmm;

// ---------------------------------------------------------------------------
// Deterministic bimodal scalar data: two Gaussian-like modes at ±5, built
// without an RNG so every platform sees identical bytes.  The jitter cycles
// through a fixed pattern spanning roughly ±1 around each mode.
// ---------------------------------------------------------------------------
static ObservationLists make_bimodal_scalar_data() {
    constexpr double jitter[7] = {-0.9, -0.5, -0.15, 0.0, 0.2, 0.55, 0.85};
    ObservationLists lists;
    for (int s = 0; s < 3; ++s) {
        ObservationSet seq(60);
        for (std::size_t t = 0; t < 60; ++t) {
            const double centre = ((t + static_cast<std::size_t>(s)) % 2 == 0) ? -5.0 : 5.0;
            seq(t) = centre + jitter[(t * 3 + static_cast<std::size_t>(s)) % 7];
        }
        lists.push_back(seq);
    }
    return lists;
}

// Adversarial start: both states carry IDENTICAL emissions, so plain EM is
// trapped at a symmetric fixed point — the states can never differentiate.
static Hmm make_symmetric_start() {
    Hmm hmm(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.5;
    trans(0, 1) = 0.5;
    trans(1, 0) = 0.5;
    trans(1, 1) = 0.5;
    hmm.setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 4.0));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(0.0, 4.0));
    return hmm;
}

// ---------------------------------------------------------------------------
// Acceptance invariant (issue #45): best-of-n is never worse than a single
// run.  This holds BY CONSTRUCTION — restart 0 trains from the caller's
// current parameters unrandomised, so the single run is one of the candidates
// — which makes this test deterministic on every platform rather than a
// statistical 90%-of-seeds check.
// ---------------------------------------------------------------------------
TEST(FitBestOfNTest, NeverWorseThanSingleRun) {
    const ObservationLists obs = make_bimodal_scalar_data();

    Hmm single = make_symmetric_start();
    std::mt19937_64 rngA(20260819);
    const double singleLogL = fit_best_of_n(single, obs, 1, rngA);
    ASSERT_TRUE(std::isfinite(singleLogL));

    Hmm multi = make_symmetric_start();
    std::mt19937_64 rngB(20260819);
    const double bestLogL = fit_best_of_n(multi, obs, 10, rngB);

    EXPECT_GE(bestLogL, singleLogL - 1e-9);
}

// ---------------------------------------------------------------------------
// Multimodal recovery: from the symmetric trap, 10 random restarts must find
// the two-mode solution.  The modes sit 10 sigma apart, so each randomised
// restart recovers with probability near 1; the margins below hold for any
// conforming RNG stream, keeping this non-flaky across CI platforms.
// ---------------------------------------------------------------------------
TEST(FitBestOfNTest, RecoversFromAdversarialStartOnMultimodalData) {
    const ObservationLists obs = make_bimodal_scalar_data();

    Hmm single = make_symmetric_start();
    std::mt19937_64 rngA(20260819);
    const double singleLogL = fit_best_of_n(single, obs, 1, rngA);

    Hmm multi = make_symmetric_start();
    std::mt19937_64 rngB(20260819);
    const double bestLogL = fit_best_of_n(multi, obs, 10, rngB);

    // Splitting ±5 modes lumped under one sigma≈5 Gaussian into two sigma≈0.5
    // components gains several nats per observation; demand a fraction of it.
    EXPECT_GT(bestLogL, singleLogL + 10.0);

    // The winning model must place one state on each mode (in either order).
    const auto &d0 = dynamic_cast<const GaussianDistribution &>(multi.getDistribution(0));
    const auto &d1 = dynamic_cast<const GaussianDistribution &>(multi.getDistribution(1));
    const double lo = std::min(d0.getMean(), d1.getMean());
    const double hi = std::max(d0.getMean(), d1.getMean());
    EXPECT_LT(lo, -2.0);
    EXPECT_GT(hi, 2.0);
}

// ---------------------------------------------------------------------------
// Multivariate path: kmeans_init-based restarts, same by-construction
// invariant, on two well-separated 2-D clusters.
// ---------------------------------------------------------------------------
TEST(FitBestOfNTest, MultivariateNeverWorseThanSingleRun) {
    constexpr double jitter[5] = {-0.4, -0.1, 0.0, 0.15, 0.35};
    MultiObservationLists obs;
    for (int s = 0; s < 2; ++s) {
        ObservationMatrix m(40, 2);
        for (std::size_t t = 0; t < 40; ++t) {
            const double cx = (t % 2 == 0) ? -4.0 : 4.0;
            m(t, 0) = cx + jitter[(t + static_cast<std::size_t>(s)) % 5];
            m(t, 1) = -cx + jitter[(t * 2 + static_cast<std::size_t>(s)) % 5];
        }
        obs.push_back(m);
    }

    auto make_start = [] {
        HmmMV hmm(2);
        // HmmMV zero-initialises pi and trans; they must be set explicitly.
        Matrix trans(2, 2);
        trans(0, 0) = 0.5;
        trans(0, 1) = 0.5;
        trans(1, 0) = 0.5;
        trans(1, 1) = 0.5;
        hmm.setTrans(trans);
        Vector pi(2);
        pi(0) = 0.5;
        pi(1) = 0.5;
        hmm.setPi(pi);
        hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2, 0.0, 9.0));
        hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(2, 0.0, 9.0));
        return hmm;
    };

    HmmMV single = make_start();
    std::mt19937_64 rngA(20260819);
    const double singleLogL = fit_best_of_n(single, obs, 1, rngA);
    ASSERT_TRUE(std::isfinite(singleLogL));

    HmmMV multi = make_start();
    std::mt19937_64 rngB(20260819);
    const double bestLogL = fit_best_of_n(multi, obs, 5, rngB);

    EXPECT_GE(bestLogL, singleLogL - 1e-9);
    EXPECT_TRUE(std::isfinite(bestLogL));
}

// ---------------------------------------------------------------------------
// Argument validation
// ---------------------------------------------------------------------------
TEST(FitBestOfNTest, ZeroRestartsThrows) {
    const ObservationLists obs = make_bimodal_scalar_data();
    Hmm hmm = make_symmetric_start();
    std::mt19937_64 rng(1);
    EXPECT_THROW((void)fit_best_of_n(hmm, obs, 0, rng), std::invalid_argument);
}

TEST(FitBestOfNTest, EmptyObservationListsThrows) {
    const ObservationLists obs;
    Hmm hmm = make_symmetric_start();
    std::mt19937_64 rng(1);
    EXPECT_THROW((void)fit_best_of_n(hmm, obs, 3, rng), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// A restart whose final M-step leaves a non-finite parameter scores NaN. It
// must be discarded, not installed: before the isnan() guard restart 0's NaN
// won via `!haveBest` and was never displaced (logL > NaN is false), so
// fit_best_of_n returned NaN and a poisoned model.
// ---------------------------------------------------------------------------
namespace {
// Emission whose FIRST weighted fit() in the process poisons that instance:
// afterwards every log-probability it reports is NaN, imitating an M-step
// that produced a non-finite parameter.
class NaNOnceGaussian : public GaussianDistribution {
public:
    using GaussianDistribution::GaussianDistribution;
    inline static int fitCalls = 0;
    bool poisoned = false;

    void fit(std::span<const double> data, std::span<const double> weights) override {
        GaussianDistribution::fit(data, weights);
        if (fitCalls++ == 0)
            poisoned = true;
    }
    [[nodiscard]] double getLogProbability(double x) const noexcept override {
        return poisoned ? std::numeric_limits<double>::quiet_NaN()
                        : GaussianDistribution::getLogProbability(x);
    }
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override {
        if (poisoned) {
            std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
            return;
        }
        GaussianDistribution::getBatchLogProbabilities(observations, out);
    }
    [[nodiscard]] std::unique_ptr<EmissionDistribution> clone() const override {
        return std::make_unique<NaNOnceGaussian>(*this);
    }
};
} // namespace

TEST(FitBestOfNTest, NaNLogLikelihoodRestartIsDiscarded) {
    NaNOnceGaussian::fitCalls = 0;
    const auto data = make_bimodal_scalar_data();
    Hmm hmm = make_symmetric_start();
    hmm.setDistribution(0, std::make_unique<NaNOnceGaussian>(0.0, 4.0));
    hmm.setDistribution(1, std::make_unique<NaNOnceGaussian>(0.0, 4.0));
    std::mt19937_64 rng(7);
    // max_iters = 1: the restart loop exits right after the M-step that
    // poisoned state 0, so restart 0's final log-likelihood is NaN and the
    // trainer never gets a second pass in which to throw on it.
    const double logL = fit_best_of_n(hmm, data, 3, rng, 1);
    EXPECT_TRUE(std::isfinite(logL)) << "best-of-n returned " << logL;
    for (std::size_t s = 0; s < 2; ++s) {
        const auto *g = dynamic_cast<const GaussianDistribution *>(&hmm.getDistribution(s));
        ASSERT_NE(g, nullptr);
        EXPECT_TRUE(std::isfinite(g->getMean()));
    }
}
