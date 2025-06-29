#include "libhmm/distributions/rayleigh_distribution.h"
// Header already includes: <iostream>, <cmath>, <cassert>, <stdexcept>, <sstream>, <iomanip> via common.h
#include <limits>     // For std::numeric_limits (not in common.h)

namespace libhmm {

/**
 * Computes the probability density function for the Rayleigh distribution.
 * 
 * PDF: f(x) = (x/σ²) * exp(-x²/(2σ²)) for x ≥ 0
 * 
 * @param value The value at which to evaluate the PDF
 * @return Probability density
 */
double RayleighDistribution::getProbability(double value) {
    if (value < constants::math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return constants::math::ZERO_DOUBLE;
    }

    if (!cacheValid_) {
        updateCache();
    }

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

    if (!cacheValid_) {
        updateCache();
    }

    return std::log(value) - constants::math::TWO * logSigma_ + negHalfInvSigmaSquared_ * value * value;
}

/**
 * Computes the cumulative distribution function for the Rayleigh distribution.
 * 
 * CDF: F(x) = 1 - exp(-x²/(2σ²)) for x ≥ 0
 * 
 * @param value The value at which to evaluate the CDF
 * @return Cumulative probability
 */
double RayleighDistribution::getCumulativeProbability(double value) const noexcept {
    if (value < constants::math::ZERO_DOUBLE) {
        return constants::math::ZERO_DOUBLE;
    }

    if (!cacheValid_) {
        updateCache();
    }

    return constants::math::ONE - std::exp(negHalfInvSigmaSquared_ * value * value);
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 * This method is efficient as it requires only a single pass through the data
 * to compute the sum of squares.
 * 
 * @param values Vector of observed data
 */
void RayleighDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        reset();
        return;
    }

    double sumSq = constants::math::ZERO_DOUBLE;
    double n = static_cast<double>(values.size());

    for (const auto& value : values) {
        if (value <= constants::math::ZERO_DOUBLE) {
            reset();
            return; // Invalid data for Rayleigh distribution
        }
        sumSq += value * value;
    }

    sigma_ = std::sqrt(sumSq / (constants::math::TWO * n));
    cacheValid_ = false;
}

/**
 * Resets the distribution to default parameters (σ = 1.0).
 */
void RayleighDistribution::reset() noexcept {
    sigma_ = constants::math::ONE;
    cacheValid_ = false;
}

std::string RayleighDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Rayleigh Distribution:\n";
    oss << "      σ (scale parameter) = " << sigma_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    oss << "      Median = " << getMedian() << "\n";
    oss << "      Mode = " << getMode() << "\n";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const RayleighDistribution& distribution) {
    os << distribution.toString();
    return os;
}

std::istream& operator>>(std::istream& is, RayleighDistribution& distribution) {
    std::string token, sigma_str;
    try {
        is >> token >> token >> token; // Read "σ", "(scale", "parameter)"
        is >> sigma_str;
        double sigma = std::stod(sigma_str);
        distribution.setSigma(sigma);
    } catch (const std::exception&) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

} // namespace libhmm

