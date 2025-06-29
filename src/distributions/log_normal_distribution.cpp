#include "libhmm/distributions/log_normal_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>     // For std::accumulate (not in common.h)
#include <algorithm>   // For std::for_each (exists in common.h, included for clarity)

using namespace libhmm::constants;

namespace libhmm
{

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
double LogNormalDistribution::getProbability(double x) {
    // Log-Normal distribution has support (0, ∞)
    if (std::isnan(x) || std::isinf(x) || x <= math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // Use direct PDF calculation for better performance
    // f(x) = 1/(x*σ*√(2π)) * exp(-½*((ln(x)-μ)/σ)²)
    // Optimize by using cached values and avoiding repeated calculations
    
    const double logX = std::log(x);
    const double standardized = (logX - mean_) / standardDeviation_;
    const double standardizedSquared = standardized * standardized;
    
    // Use cached values: negHalfSigmaSquaredInv_ = -1/(2σ²), logNormalizationConstant_ = ln(σ√(2π))
    // f(x) = exp(-ln(x) - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * standardizedSquared * 2σ²)
    //      = exp(-ln(x) - logNormalizationConstant_ - ½*standardizedSquared)
    const double logPdf = -logX - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * standardizedSquared;
    
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
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // Log PDF: log(f(x)) = -ln(x) - ln(σ√(2π)) - ½((ln(x)-μ)/σ)²
    double logX = std::log(value);
    double standardized = (logX - mean_) / standardDeviation_;
    
    return -logX - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * standardized * standardized;
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
void LogNormalDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        reset();
        return;
    }
    
    // Single-pass Welford's algorithm for log-transformed data
    double mean = math::ZERO_DOUBLE;
    double M2 = math::ZERO_DOUBLE;  // Sum of squared differences from current mean
    std::size_t validCount = 0;
    
    for (const auto& val : values) {
        // Only process positive values (support of Log-Normal distribution)
        if (val > math::ZERO_DOUBLE && std::isfinite(val)) {
            const double logVal = std::log(val);
            ++validCount;
            const double delta = logVal - mean;
            mean += delta / static_cast<double>(validCount);
            const double delta2 = logVal - mean;
            M2 += delta * delta2;
        }
        // Skip zero or negative values as they're not in the support
    }
    
    // Handle edge cases
    if (validCount == 0) {
        reset(); // Fall back to default if no valid data
        return;
    }
    
    if (validCount == 1) {
        // For a single data point, set mean to ln(value) and use a small stddev
        mean_ = mean;
        standardDeviation_ = precision::LIMIT_TOLERANCE; // Use small but non-zero value
        cacheValid_ = false;
        return;
    }
    
    // Calculate sample variance using Bessel's correction (N-1)
    const double variance = M2 / static_cast<double>(validCount - 1);
    const double stddev = std::sqrt(variance);
    
    // Validate computed parameters using constants
    if (std::isnan(mean) || std::isinf(mean) || 
        std::isnan(stddev) || std::isinf(stddev) || stddev <= precision::ZERO) {
        reset(); // Fall back to default
        return;
    }
    
    // Update parameters
    mean_ = mean;
    standardDeviation_ = stddev;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (μ = 0.0, σ = 1.0).
 * This corresponds to the standard log-normal distribution.
 */
void LogNormalDistribution::reset() noexcept {
    mean_ = 0.0;
    standardDeviation_ = 1.0;
    cacheValid_ = false; // Invalidate cache since parameters changed
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

std::ostream& operator<<( std::ostream& os, 
        const libhmm::LogNormalDistribution& distribution ){

    os << "LogNormal Distribution:" << std::endl;
    os << "    Mean = " << distribution.getMean( ) << std::endl;
    os << "    Standard Deviation = " << distribution.getStandardDeviation( );
    os << std::endl;

    return os;
}

std::istream& operator>>( std::istream& is,
        libhmm::LogNormalDistribution& distribution ){
    std::string token, mean_str, stddev_str;
    
    try {
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
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}


}
