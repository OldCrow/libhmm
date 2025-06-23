#include "libhmm/distributions/log_normal_distribution.h"
#include <iostream>
#include <numeric>
#include <algorithm>

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
    if (std::isnan(x) || std::isinf(x) || x <= 0.0) {
        return 0.0;
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    double p = 0.0;
    if (x > LIMIT_TOLERANCE) {
        p = CDF(x) - CDF(x - LIMIT_TOLERANCE);
    } else if (x > 0 && x < LIMIT_TOLERANCE) {
        // For very small positive values, use the PDF scaled by tolerance
        // to avoid numerical issues with CDF differences
        const double logX = std::log(x);
        const double standardized = (logX - mean_) / standardDeviation_;
        p = LIMIT_TOLERANCE * std::exp(negHalfSigmaSquaredInv_ * standardized * standardized) /
            (x * std::exp(logNormalizationConstant_));
    }
    
    // Ensure numerical stability
    if (std::isnan(p) || p < 0.0) {
        p = ZERO;
    }
    
    assert(p <= 1.0);
    return p;
}

double LogNormalDistribution::CDF(double x) noexcept {
    const double y = 0.5 + 0.5 * 
        errorf((std::log(x) - mean_) / (standardDeviation_ * std::sqrt(2.0)));

    assert(y <= 1.0);
    return y;
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
    
    if (values.size() == 1) {
        // For a single data point, set mean to ln(value) and use a small stddev
        // This matches legacy behavior expectations
        const double value = values[0];
        if (value > 0.0) {
            mean_ = std::log(value);
            standardDeviation_ = ZERO; // Use very small value as in legacy behavior
            cacheValid_ = false;
        } else {
            reset(); // Fall back to default if invalid data
        }
        return;
    }

    double sum = 0.0;
    std::size_t validCount = 0;
    
    // Calculate sum of log values for positive values only
    for (const auto& val : values) {
        if (val > 0.0) {
            sum += std::log(val);
            validCount++;
        }
        // Skip zero or negative values as they're not in the support of Log-Normal distribution
    }
    
    if (validCount == 0) {
        reset(); // Fall back to default if no valid data
        return;
    }
    
    if (validCount == 1) {
        reset(); // Need at least 2 points for variance estimation
        return;
    }

    // Calculate mean of log values
    mean_ = sum / static_cast<double>(validCount);
    
    // Validate computed mean
    if (std::isnan(mean_) || std::isinf(mean_)) {
        reset(); // Fall back to default
        return;
    }

    // Calculate standard deviation of log values
    double sumDeviance = 0.0;
    for (const auto& val : values) {
        if (val > 0.0) {
            const double logVal = std::log(val);
            sumDeviance += (logVal - mean_) * (logVal - mean_);
        }
    }
    
    // Use sample standard deviation (N-1 in denominator)
    standardDeviation_ = std::sqrt(sumDeviance / static_cast<double>(validCount - 1));
    
    // Validate computed standard deviation
    if (std::isnan(standardDeviation_) || std::isinf(standardDeviation_) || standardDeviation_ <= 0.0) {
        reset(); // Fall back to default
        return;
    }
    
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
    oss << "      Mean = " << getMean() << "\n";
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
    std::string s, t;
    is >> s; //" Mean"
    is >> s; // "="
    is >> t;
    distribution.setMean(std::stod(t));

    is >> s; // "Standard"
    is >> s; // "Deviation"
    is >> s; // " = "
    is >> t; // ""
    distribution.setStandardDeviation(std::stod(t));

    return is;
}


}
