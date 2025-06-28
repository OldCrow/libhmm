#include "libhmm/distributions/gaussian_distribution.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>

using namespace libhmm::constants;

namespace libhmm
{
/**
 * Returns the probability density function value for the Gaussian distribution.
 * 
 * Formula: PDF(x) = (1/σ√(2π)) * exp(-½((x-μ)/σ)²)
 */            
double GaussianDistribution::getProbability(double x) {
    // Validate input
    if (std::isnan(x) || std::isinf(x)) {
        return 0.0;
    }
    
    if (!cacheValid_) {
        updateCache();
    }
    
    const double exponent = (x - mean_) * (x - mean_) * negHalfSigmaSquaredInv_;
    return normalizationConstant_ * std::exp(exponent);
}

/*
 * Evaluates the CDF for the Normal distribution at x.  The CDF is defined as
 *
 *          1             x - mean
 *   F(x) = -( 1 + erf(---------------) )
 *          2           sigma*sqrt(2)
 */
/**
 * Returns the log probability density function value for numerical stability.
 * Formula: log PDF(x) = -½log(2π) - log(σ) - ½((x-μ)/σ)²
 */
double GaussianDistribution::getLogProbability(double x) const noexcept {
    // Validate input
    if (std::isnan(x) || std::isinf(x)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    if (!cacheValid_) {
        updateCache();
    }
    
    // Use cached values for maximum performance
    const double z = (x - mean_) * invStandardDeviation_;  // Multiply instead of divide
    const double logPdf = -0.5 * math::LN_2PI - logStandardDeviation_ - 0.5 * z * z;
    
    return logPdf;
}

double GaussianDistribution::CDF(double x) noexcept {
    // Handle problematic inputs
    if (std::isnan(x) || std::isnan(mean_) || std::isnan(standardDeviation_)) {
        return 0.0;
    }
    if (std::isinf(x)) {
        return (x > 0) ? 1.0 : 0.0;
    }
    if (standardDeviation_ <= 0.0) {
        return (x >= mean_) ? 1.0 : 0.0;
    }
    
    if (!cacheValid_) {
        updateCache();
    }
    
    // Use cached sigma*sqrt(2) for efficiency
    const double y = 0.5 * (1 + errorf((x - mean_) / sigmaSqrt2_));
    
    // Ensure valid probability range
    if (std::isnan(y) || y < 0.0) {
        return 0.0;
    }
    if (y > 1.0) {
        return 1.0;
    }
    
    return y;
}

/*
 * Fits the distribution parameters using maximum likelihood estimation with optimized algorithm.
 * 
 * Uses single-pass Welford's algorithm for numerically stable variance calculation:
 * - Better cache locality than two-pass algorithm
 * - Numerically stable for extreme values
 * - O(n) time complexity with single data traversal
 */                   
void GaussianDistribution::fit(const std::vector<Observation>& values) {
    // Cannot fit with insufficient data
    if(values.empty() || values.size() == 1) {
        reset();
        return;
    }

    // Use Welford's online algorithm for numerically stable mean and variance
    // This is more cache-friendly and numerically stable than two-pass methods
    const auto n = static_cast<double>(values.size());
    double mean = 0.0;
    double m2 = 0.0;  // Sum of squared differences from current mean
    
    std::size_t count = 0;
    for(const auto& val : values) {
        ++count;
        const double delta = val - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    mean_ = mean;
    
    // Sample variance uses Bessel's correction (N-1)
    const double sampleVariance = m2 / (n - 1.0);
    standardDeviation_ = std::sqrt(sampleVariance);

    // Handle edge cases using consolidated constants
    if (standardDeviation_ <= 0.0 || std::isnan(standardDeviation_) || std::isinf(standardDeviation_)) {
        std::cerr << "Warning: Invalid standard deviation (" << standardDeviation_ 
                  << ") in GaussianDistribution::fit(). Using minimum value." << std::endl;
        standardDeviation_ = precision::MIN_STD_DEV;
    } else if (standardDeviation_ < precision::MIN_STD_DEV) {
        std::cerr << "Warning: Very small standard deviation (" << standardDeviation_ 
                  << ") in GaussianDistribution::fit(). Using minimum value." << std::endl;
        standardDeviation_ = precision::MIN_STD_DEV;
    }
    
    // Invalidate cache since parameters changed
    cacheValid_ = false;
}

/*
 * Resets the the distribution to some default value. 
 */
void GaussianDistribution::reset() noexcept {
    mean_ = 0.0;
    standardDeviation_ = 1.0;
    cacheValid_ = false;
}

std::string GaussianDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Gaussian Distribution:\n";
    oss << "      μ (mean) = " << mean_ << "\n";
    oss << "      σ (std. deviation) = " << standardDeviation_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::GaussianDistribution& distribution ){
    os << "Normal Distribution: " << std::endl;
    os << "    Mean = " << distribution.getMean( ) << std::endl;
    os << "    Standard deviation = " << distribution.getStandardDeviation( );
    os << std::endl;
    
    return os;
}

std::istream& operator>>( std::istream& is,
        libhmm::GaussianDistribution& distribution ){
    std::string token, mean_str, stddev_str;
    
    try {
        is >> token; // "Mean"
        is >> token; // "="
        is >> mean_str;
        double mean = std::stod(mean_str);

        is >> token; // "Standard"
        is >> token; // "Deviation"
        is >> token; // "="
        is >> stddev_str;
        double stdDev = std::stod(stddev_str);
        
        // Use setParameters for validation
        distribution.setParameters(mean, stdDev);
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

bool GaussianDistribution::operator==(const GaussianDistribution& other) const {
    using namespace libhmm::constants;
    return std::abs(mean_ - other.mean_) < precision::LIMIT_TOLERANCE &&
           std::abs(standardDeviation_ - other.standardDeviation_) < precision::LIMIT_TOLERANCE;
}

}
