#include "libhmm/distributions/rayleigh_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/performance/simd_double_ops.h" // runtime dispatch
// Header already includes: <iostream>, <cmath>, <cassert>, <stdexcept>, <sstream>, <iomanip> via common.h
#include <limits>  // For std::numeric_limits (not in common.h)
#include <numeric> // For std::accumulate

namespace libhmm {

using namespace constants;

/**
 * Computes the probability density function for the Rayleigh distribution.
 *
 * PDF: f(x) = (x/σ²) * exp(-x²/(2σ²)) for x ≥ 0
 *
 * @param value The value at which to evaluate the PDF
 * @return Probability density
 */
double RayleighDistribution::getProbability(double value) const {
    if (value < constants::math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value))
        return constants::math::ZERO_DOUBLE;
    ensureCache();

    // PDF calculation
    return (value * invSigmaSquared_) * std::exp(negHalfInvSigmaSquared_ * value * value);
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 *
 * For Rayleigh distribution: log(f(x)) = log(x) - 2*log(σ) - x²/(2σ²) for x > 0
 *
 * @param value The value at which to evaluate the log-PDF
 * @return Natural logarithm of the probability density, or -∞ for invalid values
 */
double RayleighDistribution::getLogProbability(double value) const noexcept {
    if (value <= constants::math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    ensureCache();
    return std::log(value) - constants::math::TWO * logSigma_ +
           negHalfInvSigmaSquared_ * value * value;
}

double RayleighDistribution::getCumulativeProbability(double value) const noexcept {
    if (value < constants::math::ZERO_DOUBLE)
        return constants::math::ZERO_DOUBLE;
    ensureCache();

    return constants::math::ONE - std::exp(negHalfInvSigmaSquared_ * value * value);
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 * This method is efficient as it requires only a single pass through the data
 * to compute the sum of squares.
 *
 * @param values Vector of observed data
 */
double RayleighDistribution::sample(std::mt19937_64 &rng) const {
    // Inverse-CDF method: F(x) = 1 - exp(-x^2 / (2*sigma^2))
    // => x = sigma * sqrt(-2 * log(U)), U ~ U(0,1).
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::min(), 1.0);
    return sigma_ * std::sqrt(-2.0 * std::log(dist(rng)));
}

void RayleighDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }
    double sumSq = 0.0;
    for (const double val : data) {
        if (val <= constants::math::ZERO_DOUBLE) {
            reset();
            return;
        }
        sumSq += val * val;
    }
    sigma_ = std::sqrt(sumSq / (constants::math::TWO * static_cast<double>(data.size())));
    invalidateCache();
}

void RayleighDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;
    double sumWSq = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            sumWSq += weights[i] * data[i] * data[i];
    const double sigmaEst = std::sqrt(sumWSq / (constants::math::TWO * sumW));
    if (!std::isfinite(sigmaEst) || sigmaEst <= 0.0) {
        reset();
        return;
    }
    sigma_ = sigmaEst;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (σ = 1.0).
 */
void RayleighDistribution::reset() noexcept {
    sigma_ = constants::math::ONE;
    invalidateCache();
}

std::string RayleighDistribution::toString() const {
    std::ostringstream oss{};
    oss << std::fixed << std::setprecision(6);
    oss << "Rayleigh Distribution:\n";
    oss << "      σ (scale parameter) = " << sigma_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    oss << "      Median = " << getMedian() << "\n";
    oss << "      Mode = " << getMode() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const RayleighDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   Rayleigh Distribution:
//     \u03c3 (scale parameter) = VALUE
//     Mean = VALUE
//     Variance = VALUE
//     Median = VALUE
//     Mode = VALUE
std::istream &operator>>(std::istream &is, RayleighDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "Rayleigh" "Distribution:"
        is >> s >> s >> s >> s >> t; // "\u03c3" "(scale" "parameter)" "=" VALUE
        const double sigma = std::stod(t);
        // skip Mean, Variance, Median, Mode
        is >> s >> s >> t;
        is >> s >> s >> t;
        is >> s >> s >> t;
        is >> s >> s >> t;
        if (is.good())
            distribution.setSigma(sigma);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

void RayleighDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                    std::span<double> out) const {
    ensureCache();
    // inv2sigma_sq = 1/(2σ²) = -negHalfInvSigmaSquared_ (negate cached negative value)
    // log_norm = -2*log(σ)
    performance::get_double_vec_ops().rayleigh_batch(observations.data(), out.data(),
                                                     observations.size(), -negHalfInvSigmaSquared_,
                                                     -2.0 * logSigma_);
}

std::string RayleighDistribution::to_json() const {
    return json::write_distribution("Rayleigh", {{"sigma", sigma_}});
}
std::unique_ptr<EmissionDistribution> RayleighDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double sigma = r.read_double();
    r.consume('}');
    return std::make_unique<RayleighDistribution>(sigma);
}

} // namespace libhmm
