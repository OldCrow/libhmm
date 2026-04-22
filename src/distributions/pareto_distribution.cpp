#include "libhmm/distributions/pareto_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>     // For std::accumulate (not in common.h)
#include <algorithm>   // For std::min_element (exists in common.h, included for clarity)
#include <cfloat>      // For FLT_* constants (not in common.h)

using namespace libhmm::constants;

namespace libhmm
{

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
    if (std::isnan(x) || std::isinf(x) || x < xm_) return math::ZERO_DOUBLE;
    if (!isCacheValid()) updateCache();
    
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
    
    if (!isCacheValid()) updateCache();
    return logK_ + kLogXm_ - kPlus1_ * std::log(value);
}

double ParetoDistribution::getCumulativeProbability(double value) const noexcept {
    if (std::isnan(value) || value < xm_) return math::ZERO_DOUBLE;
    if (!isCacheValid()) updateCache();
    
    return math::ONE - std::pow(xm_ / value, k_);
}

/**
 * Evaluates the CDF for the Pareto distribution at x.
 * 
 * Formula: F(x) = 1 - (x_m/x)^k for x ≥ x_m
 * 
 * @param x The value at which to evaluate the CDF
 * @return Cumulative probability P(X ≤ x)
 */
double ParetoDistribution::CDF(double x) const noexcept {
    return getCumulativeProbability(x);
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 * 
 * For Pareto distribution, the MLE estimators are:
 * x_m = min(x_i) for all i
 * k = n / Σ(ln(x_i) - ln(x_m)) for i = 1 to n
 * 
 * @param values Vector of observed data
 */                   
void ParetoDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) { reset(); return; }
    double minVal = *std::min_element(data.begin(), data.end());
    if (minVal <= math::ZERO_DOUBLE) { reset(); return; }
    double sumLog = 0.0;
    for (const double val : data)
        if (val > math::ZERO_DOUBLE) sumLog += std::log(val) - std::log(minVal);
    if (sumLog <= math::ZERO_DOUBLE) { reset(); return; }
    xm_ = minVal;
    k_ = static_cast<double>(data.size()) / sumLog;
    if (!std::isfinite(k_) || k_ <= math::ZERO_DOUBLE) { reset(); return; }
    invalidateCache();
}

void ParetoDistribution::fit(std::span<const double> data,
                             std::span<const double> weights) {
    double sumW = 0.0;
    for (const double w : weights) sumW += w;
    if (sumW < precision::ZERO || std::isnan(sumW)) { reset(); return; }
    double minVal = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            minVal = std::min(minVal, data[i]);
    if (minVal <= math::ZERO_DOUBLE || !std::isfinite(minVal)) { reset(); return; }
    double sumWLog = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            sumWLog += weights[i] * (std::log(data[i]) - std::log(minVal));
    if (sumWLog <= math::ZERO_DOUBLE) { reset(); return; }
    xm_ = minVal;
    k_ = sumW / sumWLog;
    if (!std::isfinite(k_) || k_ <= math::ZERO_DOUBLE) { reset(); return; }
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (k = 1.0, x_m = 1.0).
 * This corresponds to a standard Pareto distribution.
 */
void ParetoDistribution::reset() noexcept {
    k_ = math::ONE; xm_ = math::ONE;
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

std::ostream& operator<<( std::ostream& os, 
        const libhmm::ParetoDistribution& distribution ){
    os << distribution.toString();
    return os;
}

std::istream& operator>>( std::istream& is,
        libhmm::ParetoDistribution& distribution ){
    std::string token, k_str, xm_str;
    
    try {
        is >> token; //" k"
        is >> token; // "="
        is >> k_str;
        double k = std::stod(k_str);

        is >> token; // " xm"
        is >> token; // " ="
        is >> xm_str;
        double xm = std::stod(xm_str);
        
        if (is.good()) {
            distribution.setK(k);
            distribution.setXm(xm);
        }
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

void ParetoDistribution::getBatchLogProbabilities(
        std::span<const double> observations,
        std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native / /arch:AVX512.
    // Tier 2 upgrade requires vectorised log(x): inner loop is
    // log(α) + α*log(x_m) - (α+1)*log(x), so a vectorised log is needed.
    // Available via Intel SVML, GNU libmvec, or Apple Accelerate vvlog, but
    // not portably without a math-library dependency.
    if (!isCacheValid()) updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = ParetoDistribution::getLogProbability(observations[i]);
    }
}

}
