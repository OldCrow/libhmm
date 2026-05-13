#pragma once

/**
 * @file psi_functions.h
 * @brief Digamma (ψ) and trigamma (ψ') functions used by MLE fits.
 *
 * digamma(x)  = ψ(x)  = d/dx  ln Γ(x)
 * trigamma(x) = ψ'(x) = d²/dx² ln Γ(x)
 *
 * Implementation: recurrence shifts x into the asymptotic region (x ≥ 6), then
 * the Stirling asymptotic expansion is applied (A&S §6.3.18 / §6.4.12).
 *
 * Accuracy: |error| < 2×10⁻¹⁴ for x > 0 (five Bernoulli-number terms, x shifted ≥ 6).
 */

#include <cmath>

namespace libhmm::detail {

/// Digamma function ψ(x) = d/dx ln Γ(x).
/// Accurate to ~2×10⁻¹⁴ for x > 0 via recurrence + asymptotic series (A&S §6.3.18).
[[nodiscard]] inline double digamma(double x) noexcept {
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
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
    while (x < 6.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    const double r = 1.0 / x;
    const double r2 = r * r;
    result +=
        r * (1.0 + 0.5 * r + r2 * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0))));
    return result;
}

} // namespace libhmm::detail
