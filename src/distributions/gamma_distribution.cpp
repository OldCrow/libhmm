#include "libhmm/distributions/gamma_distribution.h"
#include <iostream>
#include <cfloat>
#include <numeric>
#include <algorithm>
#include <limits>
#include <sstream>
#include <iomanip>

using namespace libhmm::constants;

namespace libhmm
{

/**
 * Computes the probability density function for the Gamma distribution.
 * PDF: f(x) = (1/(Γ(k)θ^k)) * x^(k-1) * exp(-x/θ) for x ≥ 0
 * 
 * @param x The value at which to evaluate the probability
 * @return Probability density
 */            
double GammaDistribution::getProbability(double x) {
    // Gamma distribution has support [0, ∞)
    if (std::isnan(x) || std::isinf(x) || x < 0.0) {
        return 0.0;
    }
    
    if (x == 0.0) {
        // Handle x=0 case: PDF is 0 unless k < 1 (then it's infinite)
        return (k_ < 1.0) ? std::numeric_limits<double>::infinity() : 0.0;
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // Use log space for numerical stability then exponentiate
    const double logPdf = getLogProbability(x);
    if (logPdf == -std::numeric_limits<double>::infinity()) {
        return 0.0;
    }
    
    return std::exp(logPdf);
}

/**
 * Evaluates the logarithm of the probability density function for numerical stability.
 * Formula: log PDF(x) = (k-1)*ln(x) - x/θ - k*ln(θ) - ln(Γ(k))
 * 
 * @param x The value at which to evaluate the log PDF
 * @return Log probability density
 */
double GammaDistribution::getLogProbability(double x) noexcept {
    // Gamma distribution has support [0, ∞)
    if (std::isnan(x) || std::isinf(x) || x < 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    
    if (x == 0.0) {
        // For x=0: log(PDF) = -∞ unless k < 1 and we're at the boundary
        return (k_ < 1.0) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // log PDF(x) = (k-1)*ln(x) - x/θ - k*ln(θ) - ln(Γ(k))
    const double logPdf = kMinus1_ * std::log(x) - x / theta_ - kLogTheta_ - logGammaK_;
    
    return logPdf;
}

/*
 * Returns the value of the CDF of the gamma distribution.
 *
 * The CDF is given as 
 *
 *          ligamma( k, x / theta )
 *   F(x) = -----------------------
 *                gamma( k )
 *
 * We have P( a, x ) and loggamma( a ).
 */
double GammaDistribution::CDF(double x) noexcept {
    if (x <= 0) return 0.0;

    double i = gammap(k_, x / theta_);
    if(std::isnan(i) || i < 0.0) i = 0.0;
    
    // Clamp to valid probability range
    if (i > 1.0) i = 1.0;
    
    return i;
}

/*
 * Returns the value of the LOWER INCOMPLETE gamma function given a and x.
 */
double GammaDistribution::ligamma(double a, double x) noexcept {
    return std::exp(std::log(gammap(a, x)) + std::lgamma(a));
}


/*
 * Sets k and theta such that the resulting Gamma distribution fits the data.
 *
 * Wikipedia states that there is no closed form value of k, but we can
 * approximate to 1.5% if we use the value
 *
 *   s = ln( sum( x_i, i = 1..N ) / N ) - sum( ln( x_i ), i = 1..N ) / N
 *
 * where x_i is a data  point to use in setting the PDF and N is the total
 * number of values of x.  k can then be computed as 
 *
 *   k ~= 3 - s + sqrt( (s - 3)^2 + 24s )
 *        ---------------------------------
 *                        12s
 * 
 * Using k, theta is defined as
 *
 *   theta = sum( x_i, i = 1..N )
 *           --------------------
 *                   kN
 * 
 * Note that the presence of a zero in the list of values will tend to screw
 * with things because log( 0 ) is undefined (it approaches -infinity).  I've
 * written an approximation that if there is a zero, to add libhmm::ZERO to the
 * sum and logsum values, but that's something that should really be solved with
 * some more intensive numerical methods.  Perhaps the logsum value should be
 * increased by -REALLY_BIG_NUMBER.                  
 */                   
/**
 * Fits the distribution parameters to the given data using method of moments estimation.
 * 
 * Method of moments uses:
 * sample_mean = k*θ
 * sample_variance = k*θ²
 * 
 * Solving: θ = sample_variance/sample_mean, k = sample_mean²/sample_variance
 * 
 * This is more numerically stable than MLE approximations for the Gamma distribution.
 * 
 * @param values Vector of observed data points
 */
void GammaDistribution::fit(const std::vector<Observation>& values) {
    // Handle edge cases: empty data or single data point
    if (values.empty()) {
        reset();
        return;
    }
    
    if (values.size() == 1) {
        reset(); // Cannot estimate both parameters from single point
        return;
    }

    // Use Welford's algorithm for numerically stable mean and variance calculation
    double mean = 0.0;
    double m2 = 0.0;  // Sum of squared differences from current mean
    std::size_t validCount = 0;

    for (const auto& val : values) {
        // Only positive values are valid for Gamma distribution
        if (val > 0.0 && std::isfinite(val)) {
            ++validCount;
            const double delta = val - mean;
            mean += delta / static_cast<double>(validCount);
            const double delta2 = val - mean;
            m2 += delta * delta2;
        }
    }
    
    if (validCount < 2) {
        reset(); // Need at least 2 valid points
        return;
    }

    // Calculate sample variance (with Bessel's correction)
    const double sampleVariance = m2 / (static_cast<double>(validCount) - 1.0);
    
    // Validate statistics using standardized precision constants
    if (mean <= precision::ZERO || sampleVariance <= precision::ZERO || !std::isfinite(mean) || !std::isfinite(sampleVariance)) {
        reset(); // Fall back to default if invalid statistics
        return;
    }

    // Method of moments estimators
    theta_ = sampleVariance / mean;
    k_ = (mean * mean) / sampleVariance;
    
    // Validate computed parameters using standardized precision constants
    if (!std::isfinite(k_) || !std::isfinite(theta_) || k_ <= precision::ZERO || theta_ <= precision::ZERO) {
        reset(); // Fall back to default if parameters are invalid
        return;
    }
    
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (k = 1.0, θ = 1.0).
 * This corresponds to the standard exponential distribution.
 */
void GammaDistribution::reset() noexcept {
    k_ = 1.0;
    theta_ = 1.0;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

std::string GammaDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Gamma Distribution:\n";
    oss << "      k (shape parameter) = " << k_ << "\n";
    oss << "      θ (scale parameter) = " << theta_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::GammaDistribution& distribution ){
    os << "Gamma Distribution: " << std::endl;
    os << "    k (shape) = " << distribution.getK( ) << std::endl;
    os << "    theta (scale) = " << distribution.getTheta( ) << std::endl;
    os << "    Mean = " << distribution.getMean() << std::endl;
    os << "    Variance = " << distribution.getVariance() << std::endl;

    return os;
}

std::istream& operator>>( std::istream& is,
        libhmm::GammaDistribution& distribution ){
    std::string token, k_str, theta_str;
    
    try {
        is >> token; // "k"
        is >> token; // "(shape)"
        is >> token; // "="
        is >> k_str;
        double k = std::stod(k_str);

        is >> token; // "theta"
        is >> token; // "(scale)"
        is >> token; // "="
        is >> theta_str;
        double theta = std::stod(theta_str);
        
        // Use setParameters for validation
        distribution.setParameters(k, theta);
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

bool GammaDistribution::operator==(const GammaDistribution& other) const {
    using namespace libhmm::constants;
    return std::abs(k_ - other.k_) < precision::LIMIT_TOLERANCE &&
           std::abs(theta_ - other.theta_) < precision::LIMIT_TOLERANCE;
}

}//namespace
