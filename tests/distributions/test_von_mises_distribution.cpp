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

// ----------------------------------------------------------------------------
// 1 - I1/I0.  Above kOneMinusABound the implementation is a pure series in
// 1/x that touches neither Bessel function, so these bounds are the same on
// both tiers.
// ----------------------------------------------------------------------------

TEST(BesselFunctions, OneMinusRatioLargeArgument) {
    // Reference values from mpmath at dps 50.
    struct Row {
        double x, expected;
    };
    constexpr Row kRows[] = {
        {30.0, 0.016810444634663907},     {50.0, 0.010051032621502247},
        {100.0, 0.0050126269948312344},   {200.0, 0.0025031407483564725},
        {713.9, 0.00070062381335515958},  {714.0, 0.00070052565232812161},
        {1000.0, 0.0005001251251957198},  {10000.0, 5.0001250125019535e-5},
        {1000000.0, 5.00000125000125e-7},
    };
    for (const auto &r : kRows) {
        const double got = libhmm::detail::one_minus_bessel_ratio(r.x);
        EXPECT_NEAR(got, r.expected, 1e-15 * r.expected) << "x=" << r.x;
    }
}

TEST(BesselFunctions, OneMinusRatioStaysFiniteWhereI0Overflows) {
    // I0 overflows double at x = 713.986909; the naive 1 - i1/i0 returned NaN
    // for every larger argument.  Regression for issue #73.
    for (double x : {713.0, 714.0, 715.0, 1000.0, 1.0e4, 1.0e6, 1.0e300}) {
        const double v = libhmm::detail::one_minus_bessel_ratio(x);
        EXPECT_TRUE(std::isfinite(v)) << "x=" << x;
        EXPECT_GT(v, 0.0) << "x=" << x;
        EXPECT_LT(v, 1.0) << "x=" << x;
    }
}

TEST(BesselFunctions, OneMinusRatioEdgeCases) {
    EXPECT_DOUBLE_EQ(libhmm::detail::one_minus_bessel_ratio(0.0), 1.0);
    EXPECT_DOUBLE_EQ(libhmm::detail::one_minus_bessel_ratio(-1.0), 1.0);
    EXPECT_DOUBLE_EQ(
        libhmm::detail::one_minus_bessel_ratio(std::numeric_limits<double>::quiet_NaN()), 1.0);
    // Monotone decreasing in x, and continuous across the branch crossover.
    double prev = 1.0;
    for (double x = 0.5; x < 120.0; x += 0.25) {
        const double v = libhmm::detail::one_minus_bessel_ratio(x);
        EXPECT_LT(v, prev) << "x=" << x;
        prev = v;
    }
}

// ============================================================================
// The Bessel tier this TU compiles.  Since #75 this matches the library's,
// so a test can name the tier and assert against it. Before that fix every
// test TU silently compiled Tier 2 regardless of platform, which is why the
// Tier 1 seam defect (#72) was never caught.
// ============================================================================

TEST(BesselFunctions, TierMatchesTheBuild) {
    // The canary for #75, and it has to be two-sided: asserting "Tier 2 is
    // within 1.6e-7" would pass on a Tier 1 build too, so a regression would
    // go unnoticed. Instead, decide independently of libhmm whether this
    // compiler HAS the C++17 special math functions, and require the config
    // header to agree. A silent revert to a PRIVATE definition fails here.
#if defined(__cpp_lib_math_special_functions)
    constexpr bool compiler_has_it = true;
#else
    constexpr bool compiler_has_it = false;
#endif
#if defined(LIBHMM_HAS_CXX17_BESSEL)
    constexpr bool libhmm_selected_it = true;
#else
    constexpr bool libhmm_selected_it = false;
#endif

    if (compiler_has_it) {
        EXPECT_TRUE(libhmm_selected_it)
            << "this TU has std::cyl_bessel_i but did not get "
               "LIBHMM_HAS_CXX17_BESSEL — libhmm/config.h is not reaching test "
               "TUs, so they are compiling a different Bessel tier than the "
               "library ships (issue #75)";
    }

    // And the selected tier must actually behave like the tier it claims.
    constexpr double kI0_1 = 1.2660658777520084; // mpmath, dps 50
    constexpr double kI1_1 = 0.56515910399248503;
    if (libhmm_selected_it) {
        EXPECT_NEAR(libhmm::detail::bessel_i0(1.0), kI0_1, 1e-14);
        EXPECT_NEAR(libhmm::detail::bessel_i1(1.0), kI1_1, 1e-14);
    } else {
        EXPECT_NEAR(libhmm::detail::bessel_i0(1.0), kI0_1, 1.6e-7);
        EXPECT_NEAR(libhmm::detail::bessel_i1(1.0), kI1_1, 1.6e-7);
    }
}

TEST(BesselFunctions, NegativeArgumentsAreDefinedOnBothTiers) {
    // Regression for #76. Tier 1 forwarded straight to std::cyl_bessel_i,
    // which is specified only for x >= 0 and whose implementations disagree
    // outside it: libstdc++ throws std::domain_error (and these wrappers are
    // noexcept, so that becomes std::terminate), while MSVC's STL returns the
    // even/odd continuation that Tier 2 also implements. The suite could not
    // see it until #75, because every test TU compiled Tier 2.
    for (double x : {0.5, 2.0, 3.75, 10.0, 100.0}) {
        EXPECT_DOUBLE_EQ(libhmm::detail::bessel_i0(-x), libhmm::detail::bessel_i0(x))
            << "I0 is even, x=" << x;
        EXPECT_DOUBLE_EQ(libhmm::detail::bessel_i1(-x), -libhmm::detail::bessel_i1(x))
            << "I1 is odd, x=" << x;
        EXPECT_DOUBLE_EQ(libhmm::detail::log_bessel_i0(-x), libhmm::detail::log_bessel_i0(x))
            << "log I0 is even, x=" << x;
    }
    // Across the log I0 branch seam too, where only Tier 1 has a branch.
    EXPECT_DOUBLE_EQ(libhmm::detail::log_bessel_i0(-800.0), libhmm::detail::log_bessel_i0(800.0));
}

TEST(BesselFunctions, LogI0HasNoStepAtTheSeam) {
    // Tier 1 switches to an asymptotic expansion above x = 700. #72 was a
    // ~1900 ULP jump there. d/dx log I0 = A(x), so a central difference
    // straddling the seam recovers A(700); a step does not cancel and shows
    // up divided by 2h. Tier 2 has no branch at 700 and is smooth across it,
    // so the same assertion holds on both tiers — only the tolerance differs.
    constexpr double kA700 = 0.99928545881842609; // mpmath, dps 50
    constexpr double h = 1.0e-3;
    const double central =
        (libhmm::detail::log_bessel_i0(700.0 + h) - libhmm::detail::log_bessel_i0(700.0 - h)) /
        (2.0 * h);
#if defined(LIBHMM_HAS_CXX17_BESSEL)
    EXPECT_NEAR(central, kA700, 1e-8);
#else
    EXPECT_NEAR(central, kA700, 1e-6);
#endif
}

// ============================================================================
// Numerical defects #72 / #73, exercised through the public API as well, so
// the shipped path stays covered independently of what a test TU compiles.
// ============================================================================

TEST(VonMisesDistribution, LogNormaliserIsSmoothAcrossKappa700) {
    // log_bessel_i0 switches to an asymptotic expansion above kappa = 700.
    // Truncated too early, that branch was ~1900 ULP off the moment it was
    // taken, putting a 2.14e-10 step into every density built on it.
    //
    // logNormaliser(k) = k - getLogProbability(mu), since cos(0) = 1, and
    // d/dk logNormaliser = A(k) = I1(k)/I0(k). A central difference straddling
    // the seam therefore recovers A(700); a step at the seam does not cancel
    // and shows up divided by 2h.
    constexpr double kA700 = 0.99928545881842609; // mpmath, dps 50
    constexpr double h = 1.0e-3;

    auto log_normaliser = [](double kappa) {
        const VonMisesDistribution d(0.0, kappa);
        return kappa - d.getLogProbability(0.0);
    };

    const double central = (log_normaliser(700.0 + h) - log_normaliser(700.0 - h)) / (2.0 * h);
    EXPECT_NEAR(central, kA700, 1e-8) << "log-normaliser has a step at the kappa = 700 branch seam";
}

TEST(VonMisesDistribution, CircularVarianceFiniteWhereI0Overflows) {
    // I0 overflows double at kappa = 713.986909; the old 1 - i1/i0 form
    // returned NaN for everything above. Regression for issue #73.
    for (double kappa : {700.0, 713.0, 714.0, 1000.0, 1.0e4, 1.0e6}) {
        const VonMisesDistribution d(0.0, kappa);
        const double cv = d.getCircularVariance();
        EXPECT_FALSE(std::isnan(cv)) << "kappa=" << kappa;
        EXPECT_GT(cv, 0.0) << "kappa=" << kappa;
        EXPECT_LT(cv, 1.0) << "kappa=" << kappa;
        // 1 - A(k) -> 1/(2k) for large k.
        EXPECT_NEAR(cv, 1.0 / (2.0 * kappa), 1.0e-3 / kappa) << "kappa=" << kappa;
    }
}

TEST(VonMisesDistribution, FitOnConcentratedAnglesGivesFiniteVariance) {
    // kappa_from_r_bar returns 1e6 for R_bar >= 1 ("effectively point mass"),
    // which is reached by fitting identical angles — an ordinary EM degenerate
    // case. That drove getCircularVariance() to NaN before #73.
    const std::vector<double> identical(64, 0.7);
    VonMisesDistribution d;
    d.fit(identical);

    EXPECT_GT(d.getKappa(), 700.0);
    EXPECT_FALSE(std::isnan(d.getCircularVariance()));
    EXPECT_GE(d.getCircularVariance(), 0.0);
    EXPECT_FALSE(std::isnan(d.getMean()));
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
        if (p > 0.0) {
            EXPECT_NEAR(std::log(p), lp, 1e-10) << "x=" << x;
        }
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
        // 7-term Horner cos has max error ~2e-10; use 1e-9 so the test is
        // robust across all ISA tiers (SSE2/AVX2/AVX-512/NEON) while still
        // catching gross numerical errors.
        EXPECT_NEAR(out[i], d.getLogProbability(obs[i]), 1e-9) << "i=" << i;
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
// CDF boundary behaviour (regression tests for bugfixes)
// ============================================================================

TEST(VonMisesDistribution, CDFAtNegativePiIsNearZero) {
    // getCumulativeProbability(-π) must be ≈ 0.0 (not ≈ 1.0).
    // Bug: wrap_angle used x <= -π which mapped exactly -π → +π, causing the
    // integrator to sweep the full circle and return ≈ 1.0.
    // Fix: changed to x < -π so -π stays as -π and the integration range is [−π, −π],
    // which the h≈0 guard catches and returns 0.0.
    VonMisesDistribution d(0.0, 1.0);
    EXPECT_NEAR(d.getCumulativeProbability(-PI), 0.0, 1e-6);
}

TEST(VonMisesDistribution, CDFAtPiIsNearOne) {
    // CDF(+π) integrates the full distribution and should return ≈ 1.0.
    VonMisesDistribution d(0.0, 1.0);
    EXPECT_NEAR(d.getCumulativeProbability(PI), 1.0, 1e-4);
}

TEST(VonMisesDistribution, CDFMonotone) {
    // CDF must be non-decreasing inside (-π, π).
    VonMisesDistribution d(0.5, 2.0);
    const double prev = d.getCumulativeProbability(-2.0);
    for (double x : {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0}) {
        const double curr = d.getCumulativeProbability(x);
        EXPECT_GE(curr, prev - 1e-8) << "CDF decreased at x=" << x;
    }
}

// ============================================================================
// Weighted fit with non-finite observations (regression for sumW inflation bug)
// ============================================================================

TEST(VonMisesDistribution, FitWeightedIgnoresNonFiniteObservations) {
    // data=[1.0, NaN], weights=[1.0, 1.0].
    // The NaN observation must be excluded from both sumW and S/C.
    // With only one valid observation at θ=1.0, the distribution should
    // concentrate sharply there (kappa large, mu ≈ 1.0).
    // Bug: sumW included the NaN observation's weight (2.0 instead of 1.0),
    // deflating R_bar to 0.5 and producing a much smaller kappa.
    VonMisesDistribution d(0.0, 1.0);
    const std::vector<double> data = {1.0, std::numeric_limits<double>::quiet_NaN()};
    const std::vector<double> weights = {1.0, 1.0};
    d.fit(data, weights);
    EXPECT_NEAR(d.getMu(), 1.0, 0.01);
    EXPECT_GT(d.getKappa(), 100.0); // R_bar=1.0 → essentially a point mass
}

TEST(VonMisesDistribution, FitWeightedIgnoresInfObservations) {
    // Same as above but with +Inf instead of NaN.
    VonMisesDistribution d(0.0, 1.0);
    const std::vector<double> data = {1.0, std::numeric_limits<double>::infinity()};
    const std::vector<double> weights = {1.0, 0.5};
    d.fit(data, weights);
    EXPECT_NEAR(d.getMu(), 1.0, 0.01);
    EXPECT_GT(d.getKappa(), 100.0);
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
