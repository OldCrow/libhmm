#pragma once

/**
 * @file bessel.h
 * @brief Modified Bessel functions of the first kind for VonMisesDistribution.
 *
 * Provides I₀(x), I₁(x), and log I₀(x) via two implementation tiers:
 *
 *   Tier 1 (LIBHMM_HAS_CXX17_BESSEL defined):
 *     Delegates to std::cyl_bessel_i(ν, x) from <cmath> (C++17 §29.9.3).
 *     Available on GCC 6.1+, MSVC 2017 15.5+. Not available on AppleClang /
 *     macOS libc++ (not implemented as of Xcode 15 / macOS 14).
 *
 *   Tier 2 (fallback):
 *     Polynomial approximations from Abramowitz & Stegun §9.8.1–9.8.4.
 *     Accurate to ε < 1.6×10⁻⁷. For log I₀(x) at large x the asymptotic
 *     expansion is used directly to avoid exp() overflow.
 *
 * Also provides one_minus_bessel_ratio(x) = 1 − I₁(x)/I₀(x), which is NOT a
 * composition of the above: computing it that way overflows to NaN above
 * x = 713.99 and cancels ~log₂(2x) bits below it. Its large-x branch is
 * tier-independent, so it holds the same bound on both tiers.
 *
 * CMakeLists.txt sets LIBHMM_HAS_CXX17_BESSEL via check_cxx_source_compiles.
 * Source files that use these helpers are compiled under LIBHMM_BEST_SIMD_FLAGS.
 */

#include <cmath>

#include "libhmm/math/constants.h"

namespace libhmm::detail {

// ---------------------------------------------------------------------------
// Coefficients of t¹…t⁴ in the Hankel expansion of I₀ (A&S 9.7.1, ν = 0),
//   I₀(x) ~ e^x/√(2πx) · [1 + 1/(8x) + 9/(128x²) + 75/(1024x³) + …]
// used by log_bessel_i0 above its overflow threshold.
//
// Truncating after t² — as this did before — leaves out 0.0732421875·t³, which
// at x = 700 is 2.13e-10 against a result of ~697 whose ULP is 1.14e-13.  That
// put the branch ~1900 ULP off the instant it was taken, giving log I₀ a step
// discontinuity at exactly x = 700 that any density built on it inherited.
// Carrying through t⁴ measures ≤ 0.79 ULP over x ∈ [700, 20000] against mpmath
// at dps 60; a fifth term measures identical and is not carried.
// ---------------------------------------------------------------------------
inline constexpr double kLogI0Bound = 700.0; // exp(710) ≈ DBL_MAX
inline constexpr double kI0HankelCoeffs[] = {
    0.125,             // t^1   1/8
    0.0703125,         // t^2   9/128
    0.0732421875,      // t^3   75/1024
    0.112152099609375, // t^4   3675/32768
};

/// Bracket of the I₀ Hankel expansion, less its leading 1, for x ≥ kLogI0Bound.
[[nodiscard]] inline double i0_hankel_bracket_m1(double x) noexcept {
    const double t = 1.0 / x;
    double s = 0.0;
    for (int j = static_cast<int>(sizeof(kI0HankelCoeffs) / sizeof(double)) - 1; j >= 0; --j)
        s = (s + kI0HankelCoeffs[j]) * t;
    return s;
}

#if defined(LIBHMM_HAS_CXX17_BESSEL)

// ---------------------------------------------------------------------------
// Tier 1: delegate to C++17 <cmath> special functions
// ---------------------------------------------------------------------------

[[nodiscard]] inline double bessel_i0(double x) noexcept {
    return std::cyl_bessel_i(0.0, x);
}

[[nodiscard]] inline double bessel_i1(double x) noexcept {
    return std::cyl_bessel_i(1.0, x);
}

[[nodiscard]] inline double log_bessel_i0(double x) noexcept {
    // For large x, I₀(x) overflows double; use the asymptotic form instead.
    //   log I₀(x) = x − ½log(2π) − ½log(x) + log1p(bracket − 1)
    // Split as ½log(2π) + ½log(x) rather than ½log(2πx): the product form
    // overflows to +inf for x > DBL_MAX/2π, which would return −inf here.
    if (x > kLogI0Bound) {
        return x - constants::math::HALF_LN_2PI - 0.5 * std::log(x) +
               std::log1p(i0_hankel_bracket_m1(x));
    }
    return std::log(std::cyl_bessel_i(0.0, x));
}

#else

// ---------------------------------------------------------------------------
// Tier 2: A&S polynomial approximations (portable fallback)
//
// I₀(x), A&S 9.8.1 / 9.8.2
// I₁(x), A&S 9.8.3 / 9.8.4
//
// Numerical precision: error < 1.6×10⁻⁷ in the polynomial region.
// ---------------------------------------------------------------------------

[[nodiscard]] inline double bessel_i0(double x) noexcept {
    // A&S 9.8.1 / 9.8.2
    const double ax = std::fabs(x);
    if (ax <= 3.75) {
        const double t = (ax / 3.75) * (ax / 3.75);
        return 1.0 +
               t * (3.5156229 +
                    t * (3.0899424 +
                         t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
    } else {
        const double t = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) *
               (0.39894228 +
                t * (0.01328592 +
                     t * (0.00225319 +
                          t * (-0.00157565 +
                               t * (0.00916281 + t * (-0.02057706 +
                                                      t * (0.02635537 + t * (-0.01647633 +
                                                                             t * 0.00392377))))))));
    }
}

[[nodiscard]] inline double bessel_i1(double x) noexcept {
    // A&S 9.8.3 / 9.8.4
    const double ax = std::fabs(x);
    double result;
    if (ax <= 3.75) {
        const double t = (ax / 3.75) * (ax / 3.75);
        result =
            ax *
            (0.5 +
             t * (0.87890594 +
                  t * (0.51498869 +
                       t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
    } else {
        const double t = 3.75 / ax;
        result =
            (std::exp(ax) / std::sqrt(ax)) *
            (0.39894228 +
             t * (-0.03988024 +
                  t * (-0.00362018 +
                       t * (0.00163801 +
                            t * (-0.01031555 +
                                 t * (0.02282967 + t * (-0.02895312 +
                                                        t * (0.01787654 + t * (-0.00420059)))))))));
    }
    return (x < 0.0) ? -result : result;
}

[[nodiscard]] inline double log_bessel_i0(double x) noexcept {
    // For x > 3.75: use the factored form to avoid exp() overflow.
    //   log I₀(x) = x - 0.5·log(x) + log(P(3.75/x))
    // where P is the A&S 9.8.2 polynomial (without the exp/sqrt factor).
    const double ax = std::fabs(x);
    if (ax <= 3.75) {
        // bessel_i0 ≥ 1 for all x, so log ≥ 0; no underflow risk.
        return std::log(bessel_i0(ax));
    } else {
        const double t = 3.75 / ax;
        const double poly =
            0.39894228 +
            t * (0.01328592 +
                 t * (0.00225319 +
                      t * (-0.00157565 +
                           t * (0.00916281 +
                                t * (-0.02057706 +
                                     t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
        // log(exp(x)/sqrt(x) * poly) = x - 0.5*log(x) + log(poly)
        return ax - 0.5 * std::log(ax) + std::log(poly);
    }
}

#endif // LIBHMM_HAS_CXX17_BESSEL

// ---------------------------------------------------------------------------
// 1 − A(x), where A(x) = I₁(x)/I₀(x) is the von Mises mean resultant length.
//
// Computing this as `1.0 - bessel_i1(x)/bessel_i0(x)` fails two ways:
//
//   1. I₀(x) overflows double at x = 713.986909, above which both Bessel calls
//      return +inf and `1 - inf/inf` is NaN.  That range is reachable: fitting
//      a von Mises to concentrated angles yields κ = 1e6 (kappa_from_r_bar's
//      point-mass branch).
//   2. A(x) → 1 − 1/(2x), so the subtraction cancels ~log₂(2x) bits however
//      accurate I₀ and I₁ are — relative error is amplified by 2x.
//
// The quantity itself is well conditioned (d(1−A)/dx · x/(1−A) = 1), so both
// are method defects rather than intrinsic limits.  Above kOneMinusABound the
// Hankel expansions of I₀ and I₁ are divided as series and 1 − A is evaluated
// directly, which touches neither Bessel function and so cannot overflow or
// cancel.  Below it the direct form is used, where the amplification is small.
//
// Accuracy, measured against mpmath at dps 60 over κ ∈ [1, 1e5]:
//   - asymptotic branch (κ ≥ 30, 17 terms): ≤ 2 ULP
//   - direct branch (κ < 30):               ≤ 52 ULP, worst near κ ≈ 24
// The direct branch is what bounds the whole function; its error is the 2κ
// amplification acting on correctly-rounded inputs, so it cannot be improved
// without lowering the crossover, and the series diverges below κ ≈ 25.  Under
// Tier 2 that branch inherits the A&S 1.6e-7 error amplified by 2κ; the
// asymptotic branch is tier-independent and so is exact to the bound above on
// every platform.
//
// Series coefficients: the ratio of the two Hankel asymptotic expansions
// (A&S 9.7.1, ν = 0 and ν = 1) divided as power series in t = 1/x.  Leading
// terms 1/2, 1/8, 1/8, 25/128, 13/32 agree with the published expansion.
// ---------------------------------------------------------------------------

/// Crossover between the direct and asymptotic forms of one_minus_bessel_ratio.
inline constexpr double kOneMinusABound = 30.0;

/// Coefficients of t¹…t¹⁷ in the expansion of 1 − I₁/I₀, t = 1/x.
inline constexpr double kOneMinusACoeffs[] = {
    0.5,                // t^1   1/2
    0.125,              // t^2   1/8
    0.125,              // t^3   1/8
    0.1953125,          // t^4   25/128
    0.40625,            // t^5   13/32
    1.0478515625,       // t^6   1073/1024
    3.21875,            // t^7   103/32
    11.466461181640625, // t^8   375733/32768
    46.478515625,       // t^9   23797/512
    211.27614974975586, // t^10  55384775/262144
    1064.67822265625,   // t^11  2180461/2048
    5892.0457146167755, // t^12  24713030909/4194304
    35528.87744140625,  // t^13  72763141/2048
    231884.6359563172,  // t^14  7780757249041/33554432
    1628749.4532470703, // t^15  13342715521/8192
    12251067.63286615,  // t^16  26308967412122125/2147483648
    98252781.81546783,  // t^17  12878188618117/131072
};

/// 1 − I₁(x)/I₀(x) for x ≥ 0. Returns 1 at x = 0 (I₁(0) = 0, I₀(0) = 1).
[[nodiscard]] inline double one_minus_bessel_ratio(double x) noexcept {
    if (!(x > 0.0)) // also catches NaN
        return 1.0;

    if (x >= kOneMinusABound) {
        const double t = 1.0 / x;
        double s = 0.0;
        for (int j = static_cast<int>(sizeof(kOneMinusACoeffs) / sizeof(double)) - 1; j >= 0; --j)
            s = (s + kOneMinusACoeffs[j]) * t;
        return s;
    }

    const double i0 = bessel_i0(x);
    const double i1 = bessel_i1(x);
    return (i0 > 0.0) ? 1.0 - i1 / i0 : 1.0;
}

} // namespace libhmm::detail
