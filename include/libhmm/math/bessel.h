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
 * CMakeLists.txt sets LIBHMM_HAS_CXX17_BESSEL via check_cxx_source_compiles.
 * Source files that use these helpers are compiled under LIBHMM_BEST_SIMD_FLAGS.
 */

#include <cmath>

namespace libhmm::detail {

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
    // I₀(x) ≈ exp(x)/√(2πx) · [1 + 1/(8x) + 9/(128x²) + ...]
    // log I₀(x) ≈ x - 0.5·log(2πx) + log(1 + 1/(8x) + ...)
    if (x > 700.0) { // exp(710) ≈ DBL_MAX
        const double t = 1.0 / x;
        return x - 0.5 * std::log(2.0 * M_PI * x) + std::log1p(0.125 * t + 0.0703125 * t * t);
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

} // namespace libhmm::detail
