#include <gtest/gtest.h>
#include "libhmm/training/model_selection.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/exponential_distribution.h"
#include "libhmm/distributions/student_t_distribution.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include <cmath>
#include <limits>
#include <memory>

using namespace libhmm;

namespace {

/// N-state HMM with K-symbol discrete emissions.
std::unique_ptr<Hmm> make_discrete_hmm(int numStates, int numSymbols) {
    auto hmm = std::make_unique<Hmm>(numStates);
    // Uniform transition matrix
    Matrix trans(numStates, numStates);
    for (int i = 0; i < numStates; ++i)
        for (int j = 0; j < numStates; ++j)
            trans(i, j) = 1.0 / numStates;
    hmm->setTrans(trans);
    Vector pi(numStates);
    for (int i = 0; i < numStates; ++i)
        pi(i) = 1.0 / numStates;
    hmm->setPi(pi);
    for (int i = 0; i < numStates; ++i)
        hmm->setDistribution(i, std::make_unique<DiscreteDistribution>(numSymbols));
    return hmm;
}

} // namespace

// ---------------------------------------------------------------------------
// getNumParameters() for individual distributions
// ---------------------------------------------------------------------------

TEST(ModelSelectionTest, GaussianNumParameters) {
    GaussianDistribution g;
    EXPECT_EQ(g.getNumParameters(), 2u);
}

TEST(ModelSelectionTest, ExponentialNumParameters) {
    ExponentialDistribution e;
    EXPECT_EQ(e.getNumParameters(), 1u);
}

TEST(ModelSelectionTest, StudentTNumParameters) {
    StudentTDistribution t;
    EXPECT_EQ(t.getNumParameters(), 3u);
}

TEST(ModelSelectionTest, DiscreteNumParametersK6) {
    DiscreteDistribution d(6);
    EXPECT_EQ(d.getNumParameters(), 5u); // K-1 = 5
}

TEST(ModelSelectionTest, DiscreteNumParametersK2) {
    DiscreteDistribution d(2);
    EXPECT_EQ(d.getNumParameters(), 1u); // K-1 = 1
}

TEST(ModelSelectionTest, DiscreteNumParametersK1) {
    DiscreteDistribution d(1);
    EXPECT_EQ(d.getNumParameters(), 0u); // K-1 = 0 (trivially determined)
}

// ---------------------------------------------------------------------------
// count_free_parameters
// ---------------------------------------------------------------------------

/// 2-state HMM, 6-symbol discrete emissions.
/// Free params: 2*(2-1) + (2-1) + 2*(6-1) = 2 + 1 + 10 = 13.
TEST(ModelSelectionTest, CountFreeParamsDiscrete2State6Symbol) {
    auto hmm = make_discrete_hmm(2, 6);
    const std::size_t expected = 2u * (2u - 1u)    // transitions
                                 + (2u - 1u)       // initial distribution
                                 + 2u * (6u - 1u); // emissions
    EXPECT_EQ(count_free_parameters(*hmm), expected);
}

/// 3-state HMM, 4-symbol discrete emissions.
/// Free params: 3*(3-1) + (3-1) + 3*(4-1) = 6 + 2 + 9 = 17.
TEST(ModelSelectionTest, CountFreeParamsDiscrete3State4Symbol) {
    auto hmm = make_discrete_hmm(3, 4);
    const std::size_t expected = 3u * (3u - 1u) + (3u - 1u) + 3u * (4u - 1u);
    EXPECT_EQ(count_free_parameters(*hmm), expected);
}

/// 2-state HMM with Gaussian emissions (2 params each).
/// Free params: 2*(2-1) + (2-1) + 2*2 = 2 + 1 + 4 = 7.
TEST(ModelSelectionTest, CountFreeParamsGaussian2State) {
    auto hmm = std::make_unique<Hmm>(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.8;
    trans(0, 1) = 0.2;
    trans(1, 0) = 0.3;
    trans(1, 1) = 0.7;
    hmm->setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm->setPi(pi);
    hmm->setDistribution(0, std::make_unique<GaussianDistribution>());
    hmm->setDistribution(1, std::make_unique<GaussianDistribution>());
    EXPECT_EQ(count_free_parameters(*hmm), 7u);
}

// ---------------------------------------------------------------------------
// compute_aic
// ---------------------------------------------------------------------------

TEST(ModelSelectionTest, AicFormula) {
    // AIC = 2k - 2*logL
    const double logL = -50.0;
    const std::size_t k = 5;
    EXPECT_DOUBLE_EQ(compute_aic(logL, k), 2.0 * 5.0 - 2.0 * (-50.0));
}

TEST(ModelSelectionTest, AicZeroParams) {
    EXPECT_DOUBLE_EQ(compute_aic(-10.0, 0), 20.0);
}

// ---------------------------------------------------------------------------
// compute_bic
// ---------------------------------------------------------------------------

TEST(ModelSelectionTest, BicFormula) {
    // BIC = k*ln(n) - 2*logL
    const double logL = -50.0;
    const std::size_t k = 5;
    const std::size_t n = 100;
    const double expected = 5.0 * std::log(100.0) - 2.0 * (-50.0);
    EXPECT_DOUBLE_EQ(compute_bic(logL, k, n), expected);
}

TEST(ModelSelectionTest, BicGreaterThanAicForLargeN) {
    // For large n, BIC penalizes complexity more than AIC.
    const double logL = -100.0;
    const std::size_t k = 10;
    const std::size_t n = 10000;
    EXPECT_GT(compute_bic(logL, k, n), compute_aic(logL, k));
}

// ---------------------------------------------------------------------------
// compute_aicc
// ---------------------------------------------------------------------------

TEST(ModelSelectionTest, AiccExceedsAicForSmallN) {
    const double logL = -20.0;
    const std::size_t k = 5;
    const std::size_t n = 20; // n > k+1, so correction is finite
    EXPECT_GT(compute_aicc(logL, k, n), compute_aic(logL, k));
}

TEST(ModelSelectionTest, AiccEqualsAicPlusCorrectionTerm) {
    const double logL = -20.0;
    const std::size_t k = 3;
    const std::size_t n = 50;
    const double kd = 3.0, nd = 50.0;
    const double expected = compute_aic(logL, k) + 2.0 * kd * (kd + 1.0) / (nd - kd - 1.0);
    EXPECT_DOUBLE_EQ(compute_aicc(logL, k, n), expected);
}

TEST(ModelSelectionTest, AiccInfiniteWhenNLEKPlusOne) {
    // n = k+1: denominator is zero
    EXPECT_TRUE(std::isinf(compute_aicc(-10.0, 5, 6)));
    // n < k+1: also infinite
    EXPECT_TRUE(std::isinf(compute_aicc(-10.0, 5, 4)));
}

TEST(ModelSelectionTest, AiccConvergesToAicForLargeN) {
    const double logL = -100.0;
    const std::size_t k = 5;
    const std::size_t n = 1000000;
    const double aic = compute_aic(logL, k);
    const double aicc = compute_aicc(logL, k, n);
    EXPECT_NEAR(aic, aicc, 1e-3);
}

// ---------------------------------------------------------------------------
// evaluate_model
// ---------------------------------------------------------------------------

TEST(ModelSelectionTest, EvaluateModelReturnsConsistentCriteria) {
    auto hmm = make_discrete_hmm(2, 6);
    const double logL = -80.0;
    const std::size_t n = 100;

    HmmModelCriteria c = evaluate_model(*hmm, logL, n);
    const std::size_t k = count_free_parameters(*hmm);

    EXPECT_DOUBLE_EQ(c.aic, compute_aic(logL, k));
    EXPECT_DOUBLE_EQ(c.bic, compute_bic(logL, k, n));
    EXPECT_DOUBLE_EQ(c.aicc, compute_aicc(logL, k, n));
}

TEST(ModelSelectionTest, AicAlwaysLessThanBicForReasonableSequence) {
    // BIC penalises more than AIC for n >= 8 (since ln(n) > 2 for n >= 8).
    auto hmm = make_discrete_hmm(2, 6);
    const double logL = -80.0;
    HmmModelCriteria c = evaluate_model(*hmm, logL, 1000);
    EXPECT_LT(c.aic, c.bic);
}

/// Integration: run FB calculator on a real sequence and feed logL into criteria.
TEST(ModelSelectionTest, IntegrationWithForwardBackward) {
    auto hmm = make_discrete_hmm(2, 6);
    ObservationSet obs(50);
    for (std::size_t i = 0; i < 50; ++i)
        obs(i) = static_cast<double>(i % 6);

    ForwardBackwardCalculator fbc(*hmm, obs);
    const double logL = fbc.getLogProbability();
    ASSERT_TRUE(std::isfinite(logL));

    HmmModelCriteria c = evaluate_model(*hmm, logL, obs.size());
    EXPECT_TRUE(std::isfinite(c.aic));
    EXPECT_TRUE(std::isfinite(c.bic));
    EXPECT_TRUE(std::isfinite(c.aicc));
    // All criteria should be positive (logL is very negative for a small HMM)
    EXPECT_GT(c.aic, 0.0);
    EXPECT_GT(c.bic, 0.0);
    EXPECT_GT(c.aicc, 0.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
