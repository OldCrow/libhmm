#include <gtest/gtest.h>

#include "libhmm/calculators/fb_recurrence_policy.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>

using namespace libhmm;

namespace {

constexpr double kAbsTol = 1e-9;
constexpr double kRelTol = 1e-12;

void expectClose(double a, double b, double absTol = kAbsTol, double relTol = kRelTol) {
    if (std::isnan(a) || std::isnan(b)) {
        FAIL() << "Unexpected NaN: a=" << a << " b=" << b;
    }
    if (a == b) {
        return;
    }
    const double diff = std::abs(a - b);
    if (diff <= absTol) {
        return;
    }
    const double largest = std::max(std::abs(a), std::abs(b));
    EXPECT_LE(diff, relTol * largest)
        << "values differ beyond tolerance: a=" << a << " b=" << b << " diff=" << diff;
}

void expectMatricesClose(const Matrix &a, const Matrix &b) {
    ASSERT_EQ(a.size1(), b.size1());
    ASSERT_EQ(a.size2(), b.size2());
    for (std::size_t i = 0; i < a.size1(); ++i) {
        for (std::size_t j = 0; j < a.size2(); ++j) {
            const double av = a(i, j);
            const double bv = b(i, j);
            // -inf is a valid log-space value; require an exact match in that
            // case so the kernels do not silently disagree on which transitions
            // are infeasible.
            if (std::isinf(av) || std::isinf(bv)) {
                EXPECT_EQ(av, bv) << "log-zero mismatch at (" << i << "," << j << ")";
                continue;
            }
            expectClose(av, bv);
        }
    }
}

std::unique_ptr<Hmm> makeDiscreteCasinoHmm(std::size_t numStates) {
    auto hmm = std::make_unique<Hmm>(static_cast<int>(numStates));

    Matrix trans(numStates, numStates);
    for (std::size_t i = 0; i < numStates; ++i) {
        double rowSum = 0.0;
        for (std::size_t j = 0; j < numStates; ++j) {
            const double w = 0.1 + 0.5 * static_cast<double>((i + j + 1) % 7);
            trans(i, j) = w;
            rowSum += w;
        }
        for (std::size_t j = 0; j < numStates; ++j) {
            trans(i, j) /= rowSum;
        }
    }
    hmm->setTrans(trans);

    Vector pi(numStates);
    for (std::size_t i = 0; i < numStates; ++i) {
        pi(i) = 1.0 / static_cast<double>(numStates);
    }
    hmm->setPi(pi);

    constexpr std::size_t kAlphabet = 6;
    for (std::size_t i = 0; i < numStates; ++i) {
        auto dist = std::make_unique<DiscreteDistribution>(kAlphabet);
        std::array<double, kAlphabet> weights{};
        double sum = 0.0;
        for (std::size_t s = 0; s < kAlphabet; ++s) {
            const double w = 0.05 + 0.2 * static_cast<double>((i * 11 + s * 3 + 1) % 5);
            weights[s] = w;
            sum += w;
        }
        for (std::size_t s = 0; s < kAlphabet; ++s) {
            dist->setProbability(static_cast<double>(s), weights[s] / sum);
        }
        hmm->setDistribution(i, std::move(dist));
    }
    return hmm;
}

ObservationSet makeDeterministicObs(std::size_t length, std::size_t alphabet) {
    ObservationSet obs(length);
    for (std::size_t t = 0; t < length; ++t) {
        obs(t) = static_cast<double>((t * 7 + 3) % alphabet);
    }
    return obs;
}

std::unique_ptr<Hmm> makeContinuousGaussianHmm(std::size_t numStates) {
    auto hmm = std::make_unique<Hmm>(static_cast<int>(numStates));

    Matrix trans(numStates, numStates);
    for (std::size_t i = 0; i < numStates; ++i) {
        double rowSum = 0.0;
        for (std::size_t j = 0; j < numStates; ++j) {
            const double w = 0.1 + 0.4 * std::sin(0.7 * static_cast<double>(i) +
                                                  1.3 * static_cast<double>(j));
            const double clamped = std::max(w, 0.05);
            trans(i, j) = clamped;
            rowSum += clamped;
        }
        for (std::size_t j = 0; j < numStates; ++j) {
            trans(i, j) /= rowSum;
        }
    }
    hmm->setTrans(trans);

    Vector pi(numStates);
    for (std::size_t i = 0; i < numStates; ++i) {
        pi(i) = 1.0 / static_cast<double>(numStates);
    }
    hmm->setPi(pi);

    for (std::size_t i = 0; i < numStates; ++i) {
        const double mean = 2.0 * static_cast<double>(i);
        const double sigma = 1.0;
        hmm->setDistribution(i, std::make_unique<GaussianDistribution>(mean, sigma));
    }
    return hmm;
}

ObservationSet makeContinuousObs(std::size_t length, std::size_t numStates) {
    ObservationSet obs(length);
    for (std::size_t t = 0; t < length; ++t) {
        obs(t) =
            std::sin(0.1 * static_cast<double>(t)) * static_cast<double>(numStates);
    }
    return obs;
}

void runParityCheck(const Hmm &hmm, const ObservationSet &obs) {
    ForwardBackwardCalculator pair(hmm, obs);
    pair.setRecurrenceModeOverride(FbRecurrenceMode::Pairwise);
    pair.compute();

    ForwardBackwardCalculator maxr(hmm, obs);
    maxr.setRecurrenceModeOverride(FbRecurrenceMode::MaxReduce);
    maxr.compute();

    ASSERT_EQ(pair.getRecurrenceMode(), FbRecurrenceMode::Pairwise);
    ASSERT_EQ(maxr.getRecurrenceMode(), FbRecurrenceMode::MaxReduce);

    expectClose(pair.getLogProbability(), maxr.getLogProbability());
    expectMatricesClose(pair.getLogForwardVariables(), maxr.getLogForwardVariables());
    expectMatricesClose(pair.getLogBackwardVariables(), maxr.getLogBackwardVariables());
}

} // namespace

// ---------------------------------------------------------------------------
// Discrete coverage across N=2..8 with a fixed-length sequence
// ---------------------------------------------------------------------------

class FbModeParityDiscreteTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(FbModeParityDiscreteTest, KernelsAgreeOnDiscreteHmm) {
    const std::size_t numStates = GetParam();
    auto hmm = makeDiscreteCasinoHmm(numStates);
    const ObservationSet obs = makeDeterministicObs(200, 6);
    runParityCheck(*hmm, obs);
}

INSTANTIATE_TEST_SUITE_P(N2to8, FbModeParityDiscreteTest,
                         ::testing::Values<std::size_t>(2, 3, 4, 5, 6, 7, 8));

// ---------------------------------------------------------------------------
// Continuous (Gaussian) coverage at the medium-N regime
// ---------------------------------------------------------------------------

class FbModeParityContinuousTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(FbModeParityContinuousTest, KernelsAgreeOnContinuousHmm) {
    const std::size_t numStates = GetParam();
    auto hmm = makeContinuousGaussianHmm(numStates);
    const ObservationSet obs = makeContinuousObs(500, numStates);
    runParityCheck(*hmm, obs);
}

INSTANTIATE_TEST_SUITE_P(N4_8_16, FbModeParityContinuousTest,
                         ::testing::Values<std::size_t>(4, 8, 16));

// ---------------------------------------------------------------------------
// Override accessor sanity
// ---------------------------------------------------------------------------

TEST(FbModeParityOverride, OverrideSurfacesViaGetter) {
    auto hmm = makeDiscreteCasinoHmm(4);
    const ObservationSet obs = makeDeterministicObs(50, 6);

    ForwardBackwardCalculator fbc(*hmm, obs);
    EXPECT_FALSE(fbc.getRecurrenceModeOverride().has_value());

    fbc.setRecurrenceModeOverride(FbRecurrenceMode::MaxReduce);
    ASSERT_TRUE(fbc.getRecurrenceModeOverride().has_value());
    EXPECT_EQ(*fbc.getRecurrenceModeOverride(), FbRecurrenceMode::MaxReduce);
    fbc.compute();
    EXPECT_EQ(fbc.getRecurrenceMode(), FbRecurrenceMode::MaxReduce);

    fbc.setRecurrenceModeOverride(std::nullopt);
    EXPECT_FALSE(fbc.getRecurrenceModeOverride().has_value());
}
