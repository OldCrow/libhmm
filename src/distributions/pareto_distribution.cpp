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
double ParetoDistribution::getProbability(double x) {
    // Pareto distribution has support [x_m, ∞)
    if (std::isnan(x) || std::isinf(x) || x < xm_) {
        return math::ZERO_DOUBLE;
    }
    
    // Handle boundary case - PDF is undefined exactly at x_m
    if (x == xm_) {
        return math::ZERO_DOUBLE;
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
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
    
    if (!cacheValid_) {
        updateCache();
    }
    
    // log(f(x)) = log(k) + k*log(x_m) - (k+1)*log(x)
    return logK_ + kLogXm_ - kPlus1_ * std::log(value);
}

/**
 * Computes the cumulative distribution function for the Pareto distribution.
 * 
 * CDF: F(x) = 1 - (x_m/x)^k for x ≥ x_m
 * 
 * @param value The value at which to evaluate the CDF
 * @return Cumulative probability, or 0.0 for values below x_m
 */
double ParetoDistribution::getCumulativeProbability(double value) const noexcept {
    if (std::isnan(value) || value < xm_) {
        return math::ZERO_DOUBLE;
    }
    
    if (!cacheValid_) {
        updateCache();
    }
    
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
void ParetoDistribution::fit(const std::vector<Observation>& values) {
    // Handle edge cases: empty data or single data point
    if (values.empty()) {
        reset();
        return;
    }
    
    if (values.size() == 1) {
        reset(); // MLE is not well-defined for single point
        return;
    }

    // Validate that all values are positive (required for Pareto distribution)
    auto minValue = *std::min_element(values.begin(), values.end());
    if (minValue <= math::ZERO_DOUBLE) {
        reset(); // Fall back to default if data contains non-positive values
        return;
    }
    
    // Set scale parameter to minimum value
    xm_ = minValue;
    
    // Calculate sum of log differences for shape parameter
    double sum = math::ZERO_DOUBLE;
    for (const auto& val : values) {
        if (val > math::ZERO_DOUBLE) {
            sum += std::log(val) - std::log(xm_);
        }
    }
    
    // Validate that sum is positive for numerical stability
    if (sum <= math::ZERO_DOUBLE) {
        reset(); // Fall back to default
        return;
    }
    
    // Calculate MLE estimate for shape parameter
    k_ = static_cast<double>(values.size()) / sum;
    
    // Validate computed shape parameter
    if (std::isnan(k_) || std::isinf(k_) || k_ <= math::ZERO_DOUBLE) {
        reset(); // Fall back to default
        return;
    }
    
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (k = 1.0, x_m = 1.0).
 * This corresponds to a standard Pareto distribution.
 */
void ParetoDistribution::reset() noexcept {
    k_ = math::ONE;
    xm_ = math::ONE;
    cacheValid_ = false; // Invalidate cache since parameters changed
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


}
