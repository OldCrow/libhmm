#include "libhmm/distributions/log_normal_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/performance/simd_double_ops.h" // runtime dispatch
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>   // For std::accumulate (not in common.h)
#include <algorithm> // For std::for_each (exists in common.h, included for clarity)

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Log-Normal distribution.
 *
 * For continuous distributions in discrete sampling contexts, we approximate
 * the probability as P(x - ε <= X <= x) = F(x) - F(x - ε) where ε is a small tolerance.
 *
 * This provides a numerically stable approximation of the PDF scaled by the tolerance,
 * which is appropriate for discrete sampling of continuous distributions.
 *
 * @param x The value at which to evaluate the probability
 * @return Approximated probability for discrete sampling
 */
double LogNormalDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x) || x <= math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;
    ensureCache();

    // Use direct PDF calculation for better performance
    // f(x) = 1/(x*σ*√(2π)) * exp(-½*((ln(x)-μ)/σ)²)
    // Optimize by using cached values and avoiding repeated calculations

    const double logX = std::log(x);
    const double delta = logX - mean_;

    // Use cached values: negHalfSigmaSquaredInv_ = -1/(2σ²), logNormalizationConstant_ = ln(σ√(2π))
    // log f(x) = -ln(x) - logNormalizationConstant_ - ((ln(x)-μ)^2)/(2σ²)
    const double logPdf =
        -logX - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * delta * delta;

    const double pdf = std::exp(logPdf);

    // Ensure numerical stability
    if (std::isnan(pdf) || pdf < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    return pdf;
}

double LogNormalDistribution::getLogProbability(double value) const noexcept {
    // Log-normal distribution is only defined for positive values
    if (value <= 0.0 || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    ensureCache();
    const double logX = std::log(value);
    const double delta = logX - mean_;
    return -logX - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * delta * delta;
}

double LogNormalDistribution::getCumulativeProbability(double value) const noexcept {
    // Handle boundary cases
    if (value <= 0.0) {
        return 0.0;
    }
    if (std::isnan(value) || std::isinf(value)) {
        return (std::isinf(value) && value > 0.0) ? 1.0 : 0.0;
    }

    // CDF: F(x) = ½(1 + erf((ln(x)-μ)/(σ√2)))
    double logX = std::log(value);
    double standardized = (logX - mean_) / (standardDeviation_ * std::sqrt(2.0));

    return 0.5 * (1.0 + std::erf(standardized));
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 *
 * For Log-Normal distribution, the MLE estimators are:
 * μ = mean(ln(x_i)) for positive x_i
 * σ = std_dev(ln(x_i)) for positive x_i
 *
 * Only positive values are used since Log-Normal distribution has support (0, ∞).
 *
 * @param values Vector of observed data points
 */
double LogNormalDistribution::sample(std::mt19937_64 &rng) const {
    // std::lognormal_distribution<double>(m, s): m = log-mean, s = log-stddev.
    std::lognormal_distribution<double> dist(mean_, standardDeviation_);
    return dist(rng);
}

void LogNormalDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }
    double mean = 0.0, M2 = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        if (val > 0.0 && std::isfinite(val)) {
            ++count;
            const double logVal = std::log(val);
            const double delta = logVal - mean;
            mean += delta / static_cast<double>(count);
            M2 += delta * (logVal - mean);
        }
    }
    if (count == 0) {
        reset();
        return;
    }
    if (count == 1) {
        mean_ = mean;
        standardDeviation_ = precision::LIMIT_TOLERANCE;
        invalidateCache();
        return;
    }
    // MLE variance: biased N denominator, not N-1, for true log-likelihood maximisation.
    const double stddev = std::sqrt(M2 / static_cast<double>(count));
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= precision::ZERO) {
        reset();
        return;
    }
    mean_ = mean;
    standardDeviation_ = stddev;
    invalidateCache();
}

void LogNormalDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;
    double mean = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            mean += (weights[i] / sumW) * std::log(data[i]);
    double var = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0) {
            const double d = std::log(data[i]) - mean;
            var += weights[i] * d * d;
        }
    var /= sumW;
    const double stddev = std::sqrt(var);
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= precision::ZERO) {
        reset();
        return;
    }
    mean_ = mean;
    standardDeviation_ = stddev;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (μ = 0.0, σ = 1.0).
 * This corresponds to the standard log-normal distribution.
 */
void LogNormalDistribution::reset() noexcept {
    mean_ = 0.0;
    standardDeviation_ = 1.0;
    invalidateCache();
}

std::string LogNormalDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "LogNormal Distribution:\n";
    oss << "      μ (log mean) = " << mean_ << "\n";
    oss << "      σ (log std. deviation) = " << standardDeviation_ << "\n";
    oss << "      Mean = " << getDistributionMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::LogNormalDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   LogNormal Distribution:
//     \u03bc (log mean) = VALUE
//     \u03c3 (log std. deviation) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::LogNormalDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "LogNormal" "Distribution:"
        is >> s >> s >> s >> s >> t; // "\u03bc" "(log" "mean)" "=" VALUE
        const double mean = std::stod(t);
        is >> s >> s >> s >> s >> s >> t; // "\u03c3" "(log" "std." "deviation)" "=" VALUE
        const double sd = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good()) {
            distribution.setMean(mean);
            distribution.setStandardDeviation(sd);
        }
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}
void LogNormalDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                     std::span<double> out) const {
    ensureCache();
    performance::get_double_vec_ops().lognormal_batch(
        observations.data(), out.data(), observations.size(), mean_, negHalfSigmaSquaredInv_,
        logNormalizationConstant_);
}

std::string LogNormalDistribution::to_json() const {
    return json::write_distribution("LogNormal", {{"mu", mean_}, {"sigma", standardDeviation_}});
}
std::unique_ptr<EmissionDistribution> LogNormalDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double mu = r.read_double();
    r.read_key();
    const double sigma = r.read_double();
    r.consume('}');
    return std::make_unique<LogNormalDistribution>(mu, sigma);
}

} // namespace libhmm
