#pragma once
// include/libhmm/detail/log_utils.h
//
// Internal header — NOT part of the public API.
// Not installed to CMAKE_INSTALL_PREFIX/include (detail/ is excluded).
//
// Canonical log-space utilities shared across ForwardBackward, BaumWelch,
// MapBaumWelch, and Viterbi.  Centralising these prevents silent divergence
// between per-class duplicates.

#include <cmath>
#include <limits>

namespace libhmm {
namespace detail {

/// Sentinel value for log(0): -∞.
/// All log-space algorithm classes reference this single definition.
inline constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

/// Numerically stable log(exp(a) + exp(b)).
///
/// Uses the identity:
///   log(exp(a) + exp(b)) = max(a,b) + log1p(exp(min(a,b) - max(a,b)))
///
/// Special-cases LOG_ZERO inputs to avoid exp(-inf) = 0 producing 0 + log1p(0)
/// instead of the correct max value.
[[nodiscard]] inline double logSumExp(double a, double b) noexcept {
    if (a == LOG_ZERO)
        return b;
    if (b == LOG_ZERO)
        return a;
    if (a > b)
        return a + std::log1p(std::exp(b - a));
    return b + std::log1p(std::exp(a - b));
}

} // namespace detail
} // namespace libhmm
