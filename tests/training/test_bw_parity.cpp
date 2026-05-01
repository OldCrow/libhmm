#include <gtest/gtest.h>

#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/training/baum_welch_trainer.h"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

using namespace libhmm;

namespace {

constexpr double kBitExactTol = 0.0;
constexpr double kRelTol = 1e-12;
constexpr double kAbsTol = 1e-14;

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

void expectMatricesEqual(const Matrix &a, const Matrix &b, double absTol) {
    ASSERT_EQ(a.size1(), b.size1());
    ASSERT_EQ(a.size2(), b.size2());
    for (std::size_t i = 0; i < a.size1(); ++i) {
        for (std::size_t j = 0; j < a.size2(); ++j) {
            if (absTol == kBitExactTol) {
                EXPECT_EQ(a(i, j), b(i, j))
                    << "mismatch at (" << i << "," << j << ")";
            } else {
                expectClose(a(i, j), b(i, j), absTol);
            }
        }
    }
}

void expectVectorsEqual(const Vector &a, const Vector &b, double absTol) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (absTol == kBitExactTol) {
            EXPECT_EQ(a(i), b(i)) << "mismatch at (" << i << ")";
        } else {
            expectClose(a(i), b(i), absTol);
        }
    }
}

std::unique_ptr<Hmm> makeDiscreteCasino(std::size_t numStates, std::size_t alphabet) {
    auto hmm = std::make_unique<Hmm>(static_cast<int>(numStates));

    Matrix trans(numStates, numStates);
    for (std::size_t i = 0; i < numStates; ++i) {
        double rowSum = 0.0;
        for (std::size_t j = 0; j < numStates; ++j) {
            const double w = 0.1 + 0.4 * static_cast<double>((i + j + 1) % 5);
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

    for (std::size_t i = 0; i < numStates; ++i) {
        auto dist = std::make_unique<DiscreteDistribution>(static_cast<int>(alphabet));
        std::vector<double> weights(alphabet);
        double sum = 0.0;
        for (std::size_t s = 0; s < alphabet; ++s) {
            const double w = 0.05 + 0.2 * static_cast<double>((i * 11 + s * 3 + 1) % 5);
            weights[s] = w;
            sum += w;
        }
        for (std::size_t s = 0; s < alphabet; ++s) {
            dist->setProbability(static_cast<double>(s), weights[s] / sum);
        }
        hmm->setDistribution(i, std::move(dist));
    }
    return hmm;
}

ObservationLists makeDiscreteSequences() {
    ObservationLists out;
    constexpr std::size_t kAlphabet = 6;
    constexpr std::array<std::size_t, 4> kLengths{50, 75, 30, 100};
    for (std::size_t s = 0; s < kLengths.size(); ++s) {
        ObservationSet seq(kLengths[s]);
        for (std::size_t t = 0; t < kLengths[s]; ++t) {
            seq(t) = static_cast<double>((t * 7 + s * 13 + 3) % kAlphabet);
        }
        out.push_back(seq);
    }
    return out;
}

double scoreSequencesUnderModel(const Hmm &hmm, const ObservationLists &seqs) {
    double total = 0.0;
    for (const auto &seq : seqs) {
        if (seq.size() == 0) {
            continue;
        }
        ForwardBackwardCalculator fbc(hmm, seq);
        const double lp = fbc.getLogProbability();
        if (std::isfinite(lp)) {
            total += lp;
        }
    }
    return total;
}

} // namespace

// ---------------------------------------------------------------------------
// Determinism: two independent BW runs from the same starting point on the
// same input must produce bit-exact identical updated parameters.
// ---------------------------------------------------------------------------

TEST(BaumWelchParity, OneStepDeterministic_DiscreteN3) {
    auto hmmA = makeDiscreteCasino(3, 6);
    auto hmmB = makeDiscreteCasino(3, 6);
    const ObservationLists seqs = makeDiscreteSequences();

    BaumWelchTrainer trainerA(*hmmA, seqs);
    BaumWelchTrainer trainerB(*hmmB, seqs);
    trainerA.train();
    trainerB.train();

    expectVectorsEqual(hmmA->getPi(), hmmB->getPi(), kBitExactTol);
    expectMatricesEqual(hmmA->getTrans(), hmmB->getTrans(), kBitExactTol);
    for (int i = 0; i < hmmA->getNumStates(); ++i) {
        const auto *distA = dynamic_cast<const DiscreteDistribution *>(&hmmA->getDistribution(i));
        const auto *distB = dynamic_cast<const DiscreteDistribution *>(&hmmB->getDistribution(i));
        ASSERT_NE(distA, nullptr);
        ASSERT_NE(distB, nullptr);
        ASSERT_EQ(distA->getNumSymbols(), distB->getNumSymbols());
        for (std::size_t s = 0; s < distA->getNumSymbols(); ++s) {
            EXPECT_EQ(distA->getSymbolProbability(s), distB->getSymbolProbability(s))
                << "state " << i << " symbol " << s;
        }
    }
}

TEST(BaumWelchParity, OneStepDeterministic_DiscreteN5) {
    auto hmmA = makeDiscreteCasino(5, 6);
    auto hmmB = makeDiscreteCasino(5, 6);
    const ObservationLists seqs = makeDiscreteSequences();

    BaumWelchTrainer trainerA(*hmmA, seqs);
    BaumWelchTrainer trainerB(*hmmB, seqs);
    trainerA.train();
    trainerB.train();

    expectVectorsEqual(hmmA->getPi(), hmmB->getPi(), kBitExactTol);
    expectMatricesEqual(hmmA->getTrans(), hmmB->getTrans(), kBitExactTol);
}

// ---------------------------------------------------------------------------
// EM monotonicity: a single train() step on the supplied sequences must not
// reduce the total observation log-probability under the model.
// ---------------------------------------------------------------------------

TEST(BaumWelchParity, OneStepMonotonic_Discrete) {
    auto hmm = makeDiscreteCasino(3, 6);
    const ObservationLists seqs = makeDiscreteSequences();

    const double scoreBefore = scoreSequencesUnderModel(*hmm, seqs);
    BaumWelchTrainer trainer(*hmm, seqs);
    trainer.train();
    const double scoreAfter = scoreSequencesUnderModel(*hmm, seqs);

    EXPECT_TRUE(std::isfinite(scoreBefore));
    EXPECT_TRUE(std::isfinite(scoreAfter));
    // Allow a small tolerance for floating-point noise around stationary points.
    EXPECT_GE(scoreAfter, scoreBefore - 1e-9)
        << "BW step should not decrease log-likelihood: before=" << scoreBefore
        << " after=" << scoreAfter;
}

// ---------------------------------------------------------------------------
// Invariants: post-step pi sums to 1, transition rows sum to 1, no NaN/inf.
// ---------------------------------------------------------------------------

TEST(BaumWelchParity, OneStepInvariants_Discrete) {
    auto hmm = makeDiscreteCasino(4, 6);
    const ObservationLists seqs = makeDiscreteSequences();

    BaumWelchTrainer trainer(*hmm, seqs);
    trainer.train();

    const Vector &pi = hmm->getPi();
    double piSum = 0.0;
    for (std::size_t i = 0; i < pi.size(); ++i) {
        EXPECT_TRUE(std::isfinite(pi(i)));
        EXPECT_GE(pi(i), 0.0);
        EXPECT_LE(pi(i), 1.0);
        piSum += pi(i);
    }
    EXPECT_NEAR(piSum, 1.0, 1e-12);

    const Matrix &trans = hmm->getTrans();
    for (std::size_t i = 0; i < trans.size1(); ++i) {
        double rowSum = 0.0;
        for (std::size_t j = 0; j < trans.size2(); ++j) {
            const double v = trans(i, j);
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_GE(v, 0.0);
            EXPECT_LE(v, 1.0);
            rowSum += v;
        }
        EXPECT_NEAR(rowSum, 1.0, 1e-12);
    }
}
