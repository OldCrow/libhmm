#pragma once

#include <cmath>
#include <optional>
#include <span>

namespace libhmm::detail {

/// Result of a two-pass weighted statistics computation.
struct WeightedStats {
    double mean{0.0};     ///< Weighted mean.
    double variance{0.0}; ///< Weighted variance (biased: sum(wi*(xi-mean)^2) / sumW).
};

/// Computes weighted mean and variance via a two-pass algorithm.
/// No data filter is applied — all elements contribute.
/// Returns nullopt if the total weight is non-positive or NaN.
[[nodiscard]] std::optional<WeightedStats>
compute_weighted_stats(std::span<const double> data,
                       std::span<const double> weights) noexcept;

/// Computes the weighted mean only (single pass: sumWX / sumW).
/// Returns nullopt if the total weight is non-positive or NaN.
[[nodiscard]] std::optional<double>
compute_weighted_mean(std::span<const double> data,
                      std::span<const double> weights) noexcept;

} // namespace libhmm::detail
