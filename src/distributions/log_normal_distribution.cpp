#include "libhmm/distributions/log_normal_distribution.h"
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
    if (!isCacheValid())
        updateCache();

    // Use direct PDF calculation for better performance
    // f(x) = 1/(x*σ*√(2π)) * exp(-½*((ln(x)-μ)/σ)²)
    // Optimize by using cached values and avoiding repeated calculations

    const double logX = std::log(x);
    const double standardized = (logX - mean_) / standardDeviation_;
    const double standardizedSquared = standardized * standardized;

    // Use cached values: negHalfSigmaSquaredInv_ = -1/(2σ²), logNormalizationConstant_ = ln(σ√(2π))
    // f(x) = exp(-ln(x) - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * standardizedSquared * 2σ²)
    //      = exp(-ln(x) - logNormalizationConstant_ - ½*standardizedSquared)
    const double logPdf =
        -logX - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * standardizedSquared;

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

    if (!isCacheValid())
        updateCache();
    const double logX = std::log(value);
    const double standardized = (logX - mean_) / standardDeviation_;
    return -logX - logNormalizationConstant_ +
           negHalfSigmaSquaredInv_ * standardized * standardized;
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
    const double stddev = std::sqrt(M2 / static_cast<double>(count - 1));
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= precision::ZERO) {
        reset();
        return;
    }
    mean_ = mean;
    standardDeviation_ = stddev;
    invalidateCache();
}

void LogNormalDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    double sumW = 0.0;
    for (const double w : weights)
        sumW += w;
    if (sumW < precision::ZERO || std::isnan(sumW)) {
        reset();
        return;
    }
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

    os << "LogNormal Distribution:" << std::endl;
    os << "    Mean = " << distribution.getMean() << std::endl;
    os << "    Standard Deviation = " << distribution.getStandardDeviation();
    os << std::endl;

    return os;
}

std::istream &operator>>(std::istream &is, libhmm::LogNormalDistribution &distribution) {
    try {
        std::string token, mean_str, stddev_str;
        is >> token; //" Mean"
        is >> token; // "="
        is >> mean_str;
        double mean = std::stod(mean_str);

        is >> token; // "Standard"
        is >> token; // "Deviation"
        is >> token; // " = "
        is >> stddev_str;
        double stdDev = std::stod(stddev_str);

        if (is.good()) {
            distribution.setMean(mean);
            distribution.setStandardDeviation(stdDev);
        }

    } catch (const std::exception &) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

void LogNormalDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                     std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native / /arch:AVX512.
    // Tier 2 upgrade requires vectorised log(x): the inner loop is essentially
    // Gaussian on log(x), so once a vectorised log is available the pattern is
    // identical to GaussianDistribution tier 2 but with an extra log-transform
    // step. Available via Intel SVML, GNU libmvec, or Apple Accelerate vvlog,
    // but not portably without a math-library dependency.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = LogNormalDistribution::getLogProbability(observations[i]);
    }
}

} // namespace libhmm
