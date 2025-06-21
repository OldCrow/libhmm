#include "libhmm/distributions/log_normal_distribution.h"
#include <iostream>
#include <numeric>
#include <algorithm>

namespace libhmm
{

/*
 * Returns the value of the PDF of the LogNormal distribution.
 *
 * The normal PDF is defined as
 *
 * f(x) = exp( -(ln x-u)^2 / (2 * sigma^2) ) / ( sigma * 2 * pi );
 */            
double LogNormalDistribution::getProbability(double x) {
    // Validate input
    if (std::isnan(x) || std::isinf(x) || x < 0.0) {
        return 0.0;
    }
    
    double p;
    if(x != 0) {
        p = CDF(x) - CDF(x - LIMIT_TOLERANCE);
    } else {
        p = 0;
    }

    if(std::isnan(p)) p = ZERO;
    assert(p <= 1);
    return p;
}

double LogNormalDistribution::CDF(double x) noexcept {
    const double y = 0.5 + 0.5 * 
        errorf((std::log(x) - mean_) / (standardDeviation_ * std::sqrt(2.0)));

    assert(y <= 1.0);
    return y;
}

/*
 * Sets mean and standard deviation such that the resulting LogNormal 
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
void LogNormalDistribution::fit(const std::vector<Observation>& values) {
    if(values.empty()) {
        reset();
        return;
    }

    double sum = 0;
    std::size_t validCount = 0;
    
    // Calculate sum of log values for positive values only
    for(const auto& val : values) {
        if(val > 0) {
            sum += std::log(val);
            validCount++;
        }
    }
    
    if(validCount == 0) {
        reset();
        return;
    }

    mean_ = sum / validCount;
    if(std::isnan(mean_) || mean_ < ZERO) mean_ = ZERO;

    double sumDeviance = 0;
    for(const auto& val : values) {
        if(val > 0) {
            sumDeviance += std::pow(std::log(val) - mean_, 2);
        }
    }
    standardDeviation_ = std::sqrt(sumDeviance / (validCount - 1));
    if(std::isnan(standardDeviation_) || standardDeviation_ < ZERO) standardDeviation_ = ZERO;
}

/*
 * Resets the the distribution to some default value. 
 */
void LogNormalDistribution::reset() noexcept {
    mean_ = 0.0;
    standardDeviation_ = 1.0;
}

std::string LogNormalDistribution::toString() const {
    std::stringstream os;
    os << "LogNormal Distribution:\n      Mean = ";
    os << mean_ << "\n      Standard Deviation = ";
    os << standardDeviation_ << "\n";

    return os.str();
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
