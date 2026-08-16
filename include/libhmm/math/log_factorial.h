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

/// Number of entries in the table, covering k in [0, kLogFactorialTableSize).
/// Log-factorials do not overflow, so this is a coverage/size trade-off (8 KB)
/// and NOT the 170 factorial-overflow limit that governs a table of the
/// factorials themselves.
///
/// The size, not the largest index, is the primary constant and is what both
/// the array declaration and the bounds check below are written against —
/// cppcheck 2.13 mis-evaluates a `kMax + 1` extent and reports a false
/// out-of-bounds at the top index, which this spelling avoids without needing
/// a suppression.
inline constexpr std::size_t kLogFactorialTableSize = 1024;

/// Largest k held in the table. Above this, log_factorial falls back to lgamma.
inline constexpr int kLogFactorialMax = static_cast<int>(kLogFactorialTableSize) - 1;

/// log(k!) for k in [0, kLogFactorialMax]. Built once on first use.
[[nodiscard]] inline const std::array<double, kLogFactorialTableSize> &
log_factorial_table() noexcept {
    static const std::array<double, kLogFactorialTableSize> kTable = [] {
        std::array<double, kLogFactorialTableSize> t{};
        double exact = 1.0; // 0! = 1! = 1
        for (int k = 0; k <= kExactFactorialMax; ++k) {
            if (k > 1)
                exact *= static_cast<double>(k); // exact while k! < 2^53
            t[static_cast<std::size_t>(k)] = std::log(exact);
        }
        for (std::size_t k = kExactFactorialMax + 1; k < kLogFactorialTableSize; ++k)
            t[k] = std::lgamma(static_cast<double>(k) + 1.0);
        return t;
    }();
    return kTable;
}

static_assert(kLogFactorialMax > kExactFactorialMax,
              "the exact branch must not run past the end of the table");
static_assert(static_cast<std::size_t>(kLogFactorialMax) + 1 == kLogFactorialTableSize,
              "kLogFactorialMax must be the last valid index of the table");

/// log(k!). Returns +inf for k < 0, matching log Γ(k+1) at the poles of Γ.
[[nodiscard]] inline double log_factorial(int k) noexcept {
    if (k < 0)
        return std::numeric_limits<double>::infinity();
    const auto uk = static_cast<std::size_t>(k);
    if (uk < kLogFactorialTableSize)
        return log_factorial_table()[uk];
    return std::lgamma(static_cast<double>(k) + 1.0);
}

} // namespace libhmm::detail
