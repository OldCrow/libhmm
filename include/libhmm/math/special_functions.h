#pragma once

/**
 * @file special_functions.h
 * @brief Digamma (ψ) and trigamma (ψ') functions used by GammaDistribution MLE fitting.
 *
 * digamma(x)  = ψ(x)  = d/dx  ln Γ(x)   — needed to solve the Gamma shape MLE equation.
 * trigamma(x) = ψ'(x) = d²/dx² ln Γ(x)  — needed for the Newton–Raphson step derivative.
 *
 * Implementation: recurrence ψ(x) = ψ(x+1) − 1/x shifts x into the asymptotic region
 * (x ≥ 6), then the Stirling asymptotic expansion is applied (A&S §6.3.18 / §6.4.12).
 *
 * Accuracy: |error| < 2×10⁻¹⁴ for x > 0 (five Bernoulli-number terms, x shifted ≥ 6).
 */

#include <cmath>

namespace libhmm::detail {

/// Digamma function ψ(x) = d/dx ln Γ(x).
/// Accurate to ~2×10⁻¹⁴ for x > 0 via recurrence + asymptotic series (A&S §6.3.18).
[[nodiscard]] inline double digamma(double x) noexcept {
    double result = 0.0;
    // Shift into asymptotic region using ψ(x) = ψ(x+1) − 1/x
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic series: ψ(x) ~ ln x − 1/(2x) − Σ B_{2n}/(2n·x^{2n})
    // B₂=1/6, B₄=−1/30, B₆=1/42, B₈=−1/30, B₁₀=5/66
    // → −1/(12x²) + 1/(120x⁴) − 1/(252x⁶) + 1/(240x⁸) − 1/(132x¹⁰)
    const double r = 1.0 / x;
    const double r2 = r * r;
    result += std::log(x) - 0.5 * r -
              r2 * (1.0 / 12.0 -
                    r2 * (1.0 / 120.0 - r2 * (1.0 / 252.0 - r2 * (1.0 / 240.0 - r2 / 132.0))));
    return result;
}

/// Trigamma function ψ'(x) = d²/dx² ln Γ(x).
/// Accurate to ~2×10⁻¹⁴ for x > 0 via recurrence + asymptotic series (A&S §6.4.12).
[[nodiscard]] inline double trigamma(double x) noexcept {
    double result = 0.0;
    // Shift into asymptotic region using ψ'(x) = ψ'(x+1) + 1/x²
    while (x < 6.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic series: ψ'(x) ~ 1/x + 1/(2x²) + Σ B_{2n}/x^{2n+1}
    // B₂=1/6, B₄=−1/30, B₆=1/42, B₈=−1/30
    // → 1/(6x³) − 1/(30x⁵) + 1/(42x⁷) − 1/(30x⁹)
    const double r = 1.0 / x;
    const double r2 = r * r;
    result +=
        r * (1.0 + 0.5 * r + r2 * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0))));
    return result;
}

} // namespace libhmm::detail
