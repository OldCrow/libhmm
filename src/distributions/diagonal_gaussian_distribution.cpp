#include "libhmm/distributions/diagonal_gaussian_distribution.h"

#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace libhmm {

// =============================================================================
// Construction
// =============================================================================

DiagonalGaussianDistribution::DiagonalGaussianDistribution(std::size_t dim,
                                                             double mean,
                                                             double var)
    : dim_{dim}
    , mean_(dim, mean)
    , var_(dim, std::max(var, kMinVar))
{
    if (dim == 0) {
        throw std::invalid_argument("DiagonalGaussianDistribution: dim must be > 0");
    }
    if (var <= 0.0) {
        throw std::invalid_argument(
            "DiagonalGaussianDistribution: initial variance must be > 0");
    }
    log_var_.resize(dim);
    inv_var_.resize(dim);
    updateCache();
}

// =============================================================================
// Evaluation
// =============================================================================

double DiagonalGaussianDistribution::getLogProbability(
    const ObservationVectorView& x) const noexcept
{
    if (x.size() != dim_) return -std::numeric_limits<double>::infinity();
    if (!isCacheValid()) updateCache();

    double sum = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) {
        const double z = x[d] - mean_[d];
        sum += log_var_[d] + z * z * inv_var_[d];
    }
    return -log_normalizer_ - 0.5 * sum;
}

double DiagonalGaussianDistribution::getProbability(
    const ObservationVectorView& x) const
{
    return std::exp(getLogProbability(x));
}

// =============================================================================
// Fitting — unweighted MLE
// =============================================================================

void DiagonalGaussianDistribution::fit(
    std::span<const ObservationVectorView> data)
{
    if (data.size() < 2) { reset(); return; }
    const double n  = static_cast<double>(data.size());
    const double in = 1.0 / n;

    // Compute mean
    for (std::size_t d = 0; d < dim_; ++d) {
        double s = 0.0;
        for (const auto& x : data) { s += x[d]; }
        mean_[d] = s * in;
    }
    // Compute variance (MLE: N denominator)
    for (std::size_t d = 0; d < dim_; ++d) {
        double s2 = 0.0;
        for (const auto& x : data) {
            const double z = x[d] - mean_[d];
            s2 += z * z;
        }
        var_[d] = std::max(s2 * in, kMinVar);
    }
    invalidateCache();
}

// =============================================================================
// Fitting — weighted MLE (Baum-Welch M-step)
// =============================================================================

void DiagonalGaussianDistribution::fit(
    std::span<const ObservationVectorView> data,
    std::span<const double> weights)
{
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW <= 0.0 || data.empty()) return;

    const double inv_sumW = 1.0 / sumW;
    const std::size_t n   = data.size();

    for (std::size_t d = 0; d < dim_; ++d) {
        // Weighted mean
        double sw = 0.0;
        for (std::size_t i = 0; i < n; ++i) { sw += weights[i] * data[i][d]; }
        mean_[d] = sw * inv_sumW;

        // Weighted variance
        double sv = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double z = data[i][d] - mean_[d];
            sv += weights[i] * z * z;
        }
        var_[d] = std::max(sv * inv_sumW, kMinVar);
    }
    invalidateCache();
}

void DiagonalGaussianDistribution::reset() noexcept
{
    std::fill(mean_.begin(), mean_.end(), 0.0);
    std::fill(var_.begin(),  var_.end(),  1.0);
    invalidateCache();
}

// =============================================================================
// Sampling
// =============================================================================

double DiagonalGaussianDistribution::sample(std::mt19937_64&) const
{
    throw std::logic_error(
        "DiagonalGaussianDistribution::sample() returns double which is not "
        "meaningful for D-dimensional distributions. Use sample_mv().");
}

std::vector<double> DiagonalGaussianDistribution::sample_mv(
    std::mt19937_64& rng) const
{
    if (!isCacheValid()) updateCache();
    std::vector<double> result(dim_);
    for (std::size_t d = 0; d < dim_; ++d) {
        std::normal_distribution<double> dist(mean_[d], std::sqrt(var_[d]));
        result[d] = dist(rng);
    }
    return result;
}

// =============================================================================
// Metadata
// =============================================================================

std::string DiagonalGaussianDistribution::to_json() const
{
    std::ostringstream oss;
    oss << "{\"type\":\"DiagonalGaussian\",\"dim\":" << dim_ << "}";
    return oss.str();
}

std::string DiagonalGaussianDistribution::toString() const
{
    std::ostringstream oss;
    oss << std::fixed;
    oss << "DiagonalGaussian Distribution (D=" << dim_ << "):\n";
    for (std::size_t d = 0; d < dim_; ++d) {
        oss << "  [" << d << "] mu=" << mean_[d]
            << "  var=" << var_[d] << "\n";
    }
    return oss.str();
}

} // namespace libhmm
