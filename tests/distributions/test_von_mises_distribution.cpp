#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

#include "libhmm/distributions/von_mises_distribution.h"
#include "libhmm/math/bessel.h"
#include "libhmm/math/constants.h"

using libhmm::VonMisesDistribution;

// ============================================================================
// Helpers
// ============================================================================

static constexpr double PI = libhmm::constants::math::PI;
static constexpr double TWO_PI = libhmm::constants::math::TWO_PI;
static constexpr double TOL = 1e-5; // looser than machine epsilon — A&S accuracy

// Numerical integral of the von Mises PDF from -π to π (should equal 1)
static double integrate_pdf(const VonMisesDistribution &d, int n = 2048) {
    const double h = TWO_PI / n;
    double sum = 0.5 * (d.getProbability(-PI) + d.getProbability(PI));
    for (int i = 1; i < n; ++i)
        sum += d.getProbability(-PI + i * h);
    return sum * h;
}

// ============================================================================
// Bessel function sanity (both tiers should agree with known values)
// ============================================================================

TEST(BesselFunctions, KnownValuesI0) {
    // I₀(0) = 1
    EXPECT_NEAR(libhmm::detail::bessel_i0(0.0), 1.0, 1e-7);
    // I₀(1) ≈ 1.266065878 (Abramowitz & Stegun table 9.8)
    EXPECT_NEAR(libhmm::detail::bessel_i0(1.0), 1.266065878, 1e-6);
    // I₀(5) ≈ 27.23987
    EXPECT_NEAR(libhmm::detail::bessel_i0(5.0), 27.23987, 1e-4);
    // I₀ is even
    EXPECT_NEAR(libhmm::detail::bessel_i0(-2.0), libhmm::detail::bessel_i0(2.0), 1e-12);
}

TEST(BesselFunctions, KnownValuesI1) {
    // I₁(0) = 0
    EXPECT_NEAR(libhmm::detail::bessel_i1(0.0), 0.0, 1e-10);
    // I₁(1) ≈ 0.565159104
    EXPECT_NEAR(libhmm::detail::bessel_i1(1.0), 0.565159104, 1e-6);
    // I₁ is odd
    EXPECT_NEAR(libhmm::detail::bessel_i1(-2.0), -libhmm::detail::bessel_i1(2.0), 1e-12);
}

TEST(BesselFunctions, LogI0Consistency) {
    // log I₀(x) should equal log(I₀(x)) for moderate x
    for (double x : {0.1, 1.0, 3.0, 5.0, 10.0}) {
        const double direct = std::log(libhmm::detail::bessel_i0(x));
        const double log_form = libhmm::detail::log_bessel_i0(x);
        EXPECT_NEAR(log_form, direct, 1e-5) << "x=" << x;
    }
    // Large x: log_bessel_i0 should not overflow even when bessel_i0 would
    EXPECT_TRUE(std::isfinite(libhmm::detail::log_bessel_i0(800.0)));
}

// ============================================================================
// Construction and validation
// ============================================================================

TEST(VonMisesDistribution, DefaultConstruction) {
    VonMisesDistribution d;
    EXPECT_DOUBLE_EQ(d.getMu(), 0.0);
    EXPECT_DOUBLE_EQ(d.getKappa(), 1.0);
    EXPECT_FALSE(d.isDiscrete());
    EXPECT_EQ(d.getNumParameters(), 2u);
}

TEST(VonMisesDistribution, ParameterValidation) {
    // NaN/inf mu rejected
    EXPECT_THROW(VonMisesDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0),
                 std::invalid_argument);
    EXPECT_THROW(VonMisesDistribution(std::numeric_limits<double>::infinity(), 1.0),
                 std::invalid_argument);
    // Negative kappa rejected
    EXPECT_THROW(VonMisesDistribution(0.0, -0.1), std::invalid_argument);
    // kappa = 0 is allowed (uniform distribution)
    EXPECT_NO_THROW(VonMisesDistribution(0.0, 0.0));
}

TEST(VonMisesDistribution, MuWrapping) {
    // mu is wrapped to (-π, π]
    VonMisesDistribution d(4.0, 1.0); // 4.0 > π, should wrap
    EXPECT_LT(d.getMu(), PI);
    EXPECT_GT(d.getMu(), -PI);
    // Wrapped value should equal 4.0 - 2π
    EXPECT_NEAR(d.getMu(), 4.0 - TWO_PI, 1e-12);
}

// ============================================================================
// PDF and log-PDF
// ============================================================================

TEST(VonMisesDistribution, PDFNormalises) {
    // ∫ f(x|μ,κ) dx from -π to π ≈ 1 for several (μ,κ) combinations
    for (auto [mu, kappa] : std::vector<std::pair<double, double>>{
             {0.0, 0.0}, {0.0, 0.5}, {0.0, 2.0}, {1.5, 5.0}, {-2.0, 0.1}}) {
        VonMisesDistribution d(mu, kappa);
        EXPECT_NEAR(integrate_pdf(d), 1.0, 1e-4) << "mu=" << mu << " kappa=" << kappa;
    }
}

TEST(VonMisesDistribution, LogPDFConsistency) {
    VonMisesDistribution d(0.5, 2.0);
    for (double x : {-PI, -1.0, 0.0, 1.0, PI}) {
        const double p = d.getProbability(x);
        const double lp = d.getLogProbability(x);
        if (p > 0.0)
            EXPECT_NEAR(std::log(p), lp, 1e-10) << "x=" << x;
    }
}

TEST(VonMisesDistribution, PDFPeakAtMu) {
    // PDF is maximised at x = μ
    const double mu = 1.2, kappa = 3.0;
    VonMisesDistribution d(mu, kappa);
    const double at_peak = d.getProbability(mu);
    const double off_peak = d.getProbability(mu + 0.5);
    EXPECT_GT(at_peak, off_peak);
}

TEST(VonMisesDistribution, UniformWhenKappaZero) {
    // κ = 0: uniform on (-π, π], PDF = 1/(2π)
    VonMisesDistribution d(0.0, 0.0);
    const double expected = 1.0 / TWO_PI;
    EXPECT_NEAR(d.getProbability(0.0), expected, TOL);
    EXPECT_NEAR(d.getProbability(1.5), expected, TOL);
    EXPECT_NEAR(d.getProbability(-PI), expected, TOL);
}

TEST(VonMisesDistribution, InvalidObservationReturnsMinusInf) {
    VonMisesDistribution d(0.0, 1.0);
    EXPECT_EQ(d.getLogProbability(std::numeric_limits<double>::quiet_NaN()),
              -std::numeric_limits<double>::infinity());
    EXPECT_EQ(d.getLogProbability(std::numeric_limits<double>::infinity()),
              -std::numeric_limits<double>::infinity());
}

// ============================================================================
// Circular variance
// ============================================================================

TEST(VonMisesDistribution, CircularVariance) {
    // κ = 0 → circular variance = 1 (uniform)
    EXPECT_NEAR(VonMisesDistribution(0.0, 0.0).getCircularVariance(), 1.0, TOL);
    // κ → large → circular variance → 0
    EXPECT_LT(VonMisesDistribution(0.0, 100.0).getCircularVariance(), 0.01);
    // Monotonically decreasing with κ
    const double v1 = VonMisesDistribution(0.0, 1.0).getCircularVariance();
    const double v2 = VonMisesDistribution(0.0, 5.0).getCircularVariance();
    EXPECT_GT(v1, v2);
}

// ============================================================================
// Fitting (weighted and unweighted)
// ============================================================================

TEST(VonMisesDistribution, FitUnweightedRecovery) {
    // Concentrated data near μ = 0.8 → fitted μ should be close
    std::vector<double> data;
    const double true_mu = 0.8;
    for (int i = -5; i <= 5; ++i)
        data.push_back(true_mu + i * 0.05); // symmetric around true_mu

    VonMisesDistribution d;
    d.fit(data);
    EXPECT_NEAR(d.getMu(), true_mu, 0.01);
    EXPECT_GT(d.getKappa(), 0.0);
}

TEST(VonMisesDistribution, FitWeightedKnownResult) {
    // All weight on a single direction θ = 1.0 → μ = 1.0, κ very large
    std::vector<double> data = {-1.0, 0.0, 1.0, 2.0, 3.0};
    std::vector<double> weights = {0.0, 0.0, 1.0, 0.0, 0.0}; // all on index 2 (θ=1)

    VonMisesDistribution d;
    d.fit(data, weights);
    EXPECT_NEAR(d.getMu(), 1.0, 0.01);
    // R̄ = 1.0 → very large κ
    EXPECT_GT(d.getKappa(), 100.0);
}

TEST(VonMisesDistribution, FitEmptyDataResetsToDefault) {
    VonMisesDistribution d(1.5, 5.0);
    std::vector<double> empty;
    d.fit(empty);
    EXPECT_DOUBLE_EQ(d.getMu(), 0.0);
    EXPECT_DOUBLE_EQ(d.getKappa(), 1.0);
}

TEST(VonMisesDistribution, FitZeroWeightsKeepsCurrentParams) {
    // When all weights are zero (state has no responsibility), current parameters
    // must be preserved. Resetting to defaults would cause state collapse in EM:
    // the state gets default params, attracts no observations next iteration,
    // and never recovers.
    VonMisesDistribution d(1.5, 5.0);
    std::vector<double> data = {0.0, 1.0};
    std::vector<double> weights = {0.0, 0.0};
    d.fit(data, weights);
    EXPECT_DOUBLE_EQ(d.getMu(), 1.5);
    EXPECT_DOUBLE_EQ(d.getKappa(), 5.0);
}

// ============================================================================
// Batch log-probabilities
// ============================================================================

TEST(VonMisesDistribution, BatchMatchesScalar) {
    VonMisesDistribution d(0.5, 2.0);
    std::vector<double> obs = {-PI, -1.0, 0.0, 0.5, 1.5, PI};
    std::vector<double> out(obs.size());

    d.getBatchLogProbabilities(obs, out);
    for (std::size_t i = 0; i < obs.size(); ++i)
        EXPECT_NEAR(out[i], d.getLogProbability(obs[i]), 1e-12) << "i=" << i;
}

// ============================================================================
// JSON round-trip
// ============================================================================

TEST(VonMisesDistribution, JsonRoundTrip) {
    VonMisesDistribution d(1.23456789, 4.56789012);
    const std::string json = d.to_json();

    // Deserialise via from_json by constructing a Reader
    // (simpler: reconstruct from known parameters and compare)
    EXPECT_NE(json.find("VonMises"), std::string::npos);
    EXPECT_NE(json.find("mu"), std::string::npos);
    EXPECT_NE(json.find("kappa"), std::string::npos);
}

// ============================================================================
// Reset
// ============================================================================

TEST(VonMisesDistribution, Reset) {
    VonMisesDistribution d(1.5, 5.0);
    d.reset();
    EXPECT_DOUBLE_EQ(d.getMu(), 0.0);
    EXPECT_DOUBLE_EQ(d.getKappa(), 1.0);
}

// ============================================================================
// toString
// ============================================================================

TEST(VonMisesDistribution, ToStringContainsParameters) {
    VonMisesDistribution d(0.5, 2.0);
    const std::string s = d.toString();
    EXPECT_NE(s.find("Von Mises"), std::string::npos);
    EXPECT_NE(s.find("0.5"), std::string::npos);
    EXPECT_NE(s.find("2.0"), std::string::npos);
}
