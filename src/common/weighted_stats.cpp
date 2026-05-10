#include "libhmm/math/weighted_stats.h"
#include "libhmm/math/constants.h"
#include <functional>
#include <numeric>

using namespace libhmm::constants;

namespace libhmm::detail {

std::optional<WeightedStats> compute_weighted_stats(std::span<const double> data,
                                                    std::span<const double> weights) noexcept {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW < precision::ZERO || std::isnan(sumW))
        return std::nullopt;

    // Pass 1 — weighted mean: Σ(w_i · x_i) / Σ(w_i)
    const double mean =
        std::inner_product(weights.begin(), weights.end(), data.begin(), 0.0) / sumW;

    // Pass 2 — weighted variance (biased): Σ(w_i · (x_i − μ)²) / Σ(w_i)
    const double var =
        std::transform_reduce(data.begin(), data.end(), weights.begin(), 0.0, std::plus<>{},
                              [mean](double x, double w) {
                                  const double d = x - mean;
                                  return w * d * d;
                              }) /
        sumW;

    return WeightedStats{mean, var};
}

std::optional<double> compute_weighted_mean(std::span<const double> data,
                                            std::span<const double> weights) noexcept {
    // Single-pass accumulation of both Σw_i and Σ(w_i·x_i) in one loop.
    // NOTE: index loop kept deliberately — there is no std::views::zip in C++20
    // (C++23 only), so a clean range-based alternative does not exist without
    // introducing a second pass or an ad-hoc struct accumulator.
    double sumW = 0.0, sumWX = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        sumW += weights[i];
        sumWX += weights[i] * data[i];
    }
    if (sumW < precision::ZERO || std::isnan(sumW))
        return std::nullopt;
    return sumWX / sumW;
}

} // namespace libhmm::detail
