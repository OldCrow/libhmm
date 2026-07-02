#include "libhmm/distributions/pareto_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/performance/simd_double_ops.h" // runtime dispatch
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>   // For std::accumulate (not in common.h)
#include <algorithm> // For std::min_element (exists in common.h, included for clarity)
#include <cfloat>    // For FLT_* constants (not in common.h)

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Pareto distribution.
 *
 * For Pareto distribution: f(x) = (k * x_m^k) / x^(k+1) for x ≥ x_m
 *
 * Uses direct PDF calculation for optimal performance, avoiding expensive CDF differences.
 *
 * @param x The value at which to evaluate the probability density
 * @return Probability density for the given value
 */
double ParetoDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x) || x < xm_)
        return math::ZERO_DOUBLE;
    ensureCache();

    // Direct PDF calculation: f(x) = (k * x_m^k) / x^(k+1)
    // Using cached kXmPowK_ = k * x_m^k and kPlus1_ = k + 1 for efficiency
    const double pdf = kXmPowK_ / std::pow(x, kPlus1_);

    // Ensure numerical stability
    if (std::isnan(pdf) || pdf < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    return pdf;
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 *
 * For Pareto distribution: log(f(x)) = log(k) + k*log(x_m) - (k+1)*log(x) for x ≥ x_m
 *
 * @param value The value at which to evaluate the log-PDF
 * @return Natural logarithm of the probability density, or -∞ for invalid values
 */
double ParetoDistribution::getLogProbability(double value) const noexcept {
    if (std::isnan(value) || std::isinf(value) || value < xm_) {
        return -std::numeric_limits<double>::infinity();
    }

    ensureCache();
    return logK_ + kLogXm_ - kPlus1_ * std::log(value);
}

double ParetoDistribution::getCumulativeProbability(double value) const noexcept {
    if (std::isnan(value) || value < xm_)
        return math::ZERO_DOUBLE;
    ensureCache();

    return math::ONE - std::pow(xm_ / value, k_);
}

/**
 * Fits the distribution parameters
 *
 * For Pareto distribution, the MLE estimators are:
 * x_m = min(x_i) for all i
 * k = n / Σ(ln(x_i) - ln(x_m)) for i = 1 to n
 *
 * @param values Vector of observed data
 */
double ParetoDistribution::sample(std::mt19937_64 &rng) const {
    // Inverse-CDF method: F(x) = 1 - (xm/x)^k => x = xm * U^(-1/k), U ~ U(0,1).
    // Avoid U=0 by using the half-open interval (min, 1).
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::min(), 1.0);
    return xm_ * std::pow(dist(rng), -1.0 / k_);
}

void ParetoDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }
    double minVal = *std::min_element(data.begin(), data.end());
    if (minVal <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    double sumLog = 0.0;
    for (const double val : data)
        if (val > math::ZERO_DOUBLE)
            sumLog += std::log(val) - std::log(minVal);
    if (sumLog <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    xm_ = minVal;
    k_ = static_cast<double>(data.size()) / sumLog;
    if (!std::isfinite(k_) || k_ <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    invalidateCache();
}

void ParetoDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;
    double minVal = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            minVal = std::min(minVal, data[i]);
    if (minVal <= math::ZERO_DOUBLE || !std::isfinite(minVal)) {
        reset();
        return;
    }
    double sumWLog = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            sumWLog += weights[i] * (std::log(data[i]) - std::log(minVal));
    if (sumWLog <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    xm_ = minVal;
    k_ = sumW / sumWLog;
    if (!std::isfinite(k_) || k_ <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (k = 1.0, x_m = 1.0).
 * This corresponds to a standard Pareto distribution.
 */
void ParetoDistribution::reset() noexcept {
    k_ = math::ONE;
    xm_ = math::ONE;
    invalidateCache();
}

std::string ParetoDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Pareto Distribution:\n";
    oss << "      k (shape parameter) = " << k_ << "\n";
    oss << "      x_m (scale parameter) = " << xm_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::ParetoDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   Pareto Distribution:
//     k (shape parameter) = VALUE
//     x_m (scale parameter) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::ParetoDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "Pareto" "Distribution:"
        is >> s >> s >> s >> s >> t; // "k" "(shape" "parameter)" "=" VALUE
        const double k = std::stod(t);
        is >> s >> s >> s >> s >> t; // "x_m" "(scale" "parameter)" "=" VALUE
        const double xm = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good()) {
            distribution.setK(k);
            distribution.setXm(xm);
        }
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

void ParetoDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                  std::span<double> out) const {
    ensureCache();
    performance::get_double_vec_ops().pareto_batch(
        observations.data(), out.data(), observations.size(), kPlus1_, xm_, logK_ + kLogXm_);
}

std::string ParetoDistribution::to_json() const {
    return json::write_distribution("Pareto", {{"k", k_}, {"xm", xm_}});
}
std::unique_ptr<EmissionDistribution> ParetoDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double k = r.read_double();
    r.read_key();
    const double xm = r.read_double();
    r.consume('}');
    return std::make_unique<ParetoDistribution>(k, xm);
}

} // namespace libhmm
