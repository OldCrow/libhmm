#include "libhmm/math/weighted_stats.h"
#include "libhmm/math/constants.h"
#include <numeric>

using namespace libhmm::constants;

namespace libhmm::detail {

std::optional<WeightedStats> compute_weighted_stats(std::span<const double> data,
                                                    std::span<const double> weights) noexcept {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW < precision::ZERO || std::isnan(sumW))
        return std::nullopt;

    // Pass 1 — weighted mean
    double mean = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        mean += weights[i] * data[i];
    mean /= sumW;

    // Pass 2 — weighted variance (biased)
    double var = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double d = data[i] - mean;
        var += weights[i] * d * d;
    }
    var /= sumW;

    return WeightedStats{mean, var};
}

std::optional<double> compute_weighted_mean(std::span<const double> data,
                                            std::span<const double> weights) noexcept {
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
