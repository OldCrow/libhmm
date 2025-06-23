#include "libhmm/distributions/gaussian_distribution.h"
#include <iostream>
#include <numeric>
#include <algorithm>

namespace libhmm
{

/**
 * Returns the probability density function value for the Gaussian distribution.
 * 
 * Temporarily using the legacy CDF-based approach while debugging the PDF calculation.
 */            
double GaussianDistribution::getProbability(double x) {
    // Validate input
    if (std::isnan(x) || std::isinf(x)) {
        return 0.0;
    }
    
    const double LIMIT_TOLERANCE = 1e-6;
    const double ZERO = 0.0;
    const double p = CDF(x) - CDF(x - LIMIT_TOLERANCE);
    
    // Handle numerical issues
    if (std::isnan(p) || p < 0.0) {
        return ZERO;
    }
    
    assert(p <= 1.0);
    return p;
}

/*
 * Evaluates the CDF for the Normal distribution at x.  The CDF is defined as
 *
 *          1             x - mean
 *   F(x) = -( 1 + erf(---------------) )
 *          2           sigma*sqrt(2)
 */
double GaussianDistribution::CDF(double x) noexcept {
    const double y = 0.5 * (1 + errorf((x - mean_) / (standardDeviation_ * std::sqrt(2.0))));
    assert(y >= 0);
    return y;
}

/*
 * Sets mean and standard deviation such that the resulting Gaussian 
 * distribution fits the data.
 *
 * The mean is defined as:
 *
 * u = sum( x_i, i=1..N ) / N
 *
 * where N is the number of values.
 *
 * The standard deviation is calculated as
 *
 * sigma = sqrt( sum( (x_i - u)^2, i = 1..N ) / N )
 *
 */                   
void GaussianDistribution::fit(const std::vector<Observation>& values) {
    // You can't fit a normal distribution with a data set of only one 
    // point or zero points.
    if(values.empty() || values.size() == 1) {
        reset();
        return;
    }

    // Calculate mean using modern algorithm
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    mean_ = sum / values.size();

    // Calculate variance using modern algorithm
    double variance = 0.0;
    for(const auto& val : values) {
        variance += std::pow(val - mean_, 2);
    }
    
    standardDeviation_ = std::sqrt(variance / (values.size() - 1));

    // Normal distribution functions are undefined for standard deviation
    // values of 0.  (Variance = (standard deviation)^2 > 0)
    assert(standardDeviation_ > 0);
    
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
