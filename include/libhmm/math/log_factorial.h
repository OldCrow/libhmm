#pragma once

/**
 * @file log_factorial.h
 * @brief Shared log-factorial table, log(k!) for integer k.
 *
 * Discrete distributions need log(k!) per observation, and k is an integer
 * count — so this is a table lookup, not a call into lgamma. The distinction
 * matters beyond speed: it is why Poisson and Binomial do NOT depend on a
 * vectorized lgamma to reach tier 2, contrary to what the per-distribution
 * comments used to claim. Only NegativeBinomial's log Γ(k + r) needs a real
 * lgamma, r being a continuous parameter.
 *
 * One shared table rather than a copy per distribution instance: the values
 * depend on nothing but k, so a per-instance cache rebuilt in updateCache()
 * (as Poisson's used to be) recomputes constants every time a parameter moves.
 *
 * Accuracy, measured against mpmath at dps 50:
 *   - k ≤ 18: EXACT (0.0 ULP). k! is representable in double up to 18!, and
 *     each product in the loop below is therefore exact, so the only rounding
 *     is the single std::log.
 *   - k > 18: ≤ 1 ULP, from std::lgamma at integer arguments.
 * Keeping the exact branch is not just for accuracy — it means Poisson's
 * k ≤ 12 results are bit-identical to what it produced before this table
 * existed, since that path was already log() of an exact factorial.
 *
 * The table is built once on first use, not at static-initialisation time, so
 * there is no initialisation-order dependency to reason about.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace libhmm::detail {

/// Largest k for which k! is exactly representable in double (18! < 2^53).
inline constexpr int kExactFactorialMax = 18;

/// Largest k held in the table. Above this, log_factorial falls back to
/// lgamma. Log-factorials do not overflow, so this bound is a coverage/size
/// trade-off (8 KB) and NOT the 170 factorial-overflow limit that governs a
/// table of factorials themselves.
inline constexpr int kLogFactorialMax = 1023;

/// log(k!) for k in [0, kLogFactorialMax]. Built once on first use.
[[nodiscard]] inline const std::array<double, kLogFactorialMax + 1> &
log_factorial_table() noexcept {
    static const std::array<double, kLogFactorialMax + 1> kTable = [] {
        std::array<double, kLogFactorialMax + 1> t{};
        double exact = 1.0; // 0! = 1! = 1
        for (int k = 0; k <= kExactFactorialMax; ++k) {
            if (k > 1)
                exact *= static_cast<double>(k); // exact while k! < 2^53
            t[static_cast<std::size_t>(k)] = std::log(exact);
        }
        for (int k = kExactFactorialMax + 1; k <= kLogFactorialMax; ++k)
            t[static_cast<std::size_t>(k)] = std::lgamma(static_cast<double>(k) + 1.0);
        return t;
    }();
    return kTable;
}

/// log(k!). Returns +inf for k < 0, matching log Γ(k+1) at the poles of Γ.
[[nodiscard]] inline double log_factorial(int k) noexcept {
    if (k < 0)
        return std::numeric_limits<double>::infinity();
    if (k <= kLogFactorialMax)
        return log_factorial_table()[static_cast<std::size_t>(k)];
    return std::lgamma(static_cast<double>(k) + 1.0);
}

} // namespace libhmm::detail
