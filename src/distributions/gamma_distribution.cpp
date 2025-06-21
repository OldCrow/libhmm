#include "libhmm/distributions/gamma_distribution.h"
#include <iostream>
#include <cfloat>

namespace libhmm
{

/*
 * Returns P(x) of the Gamma distribution.
 *
 * Approximated.  See GaussianDistribution::getProbability()
 */            
double GammaDistribution::getProbability(double x) {
    // Validate input
    if (std::isnan(x) || std::isinf(x) || x < 0) {
        return 0.0;
    }
    if(x == 0) return 0;

    double p = CDF(x) - CDF(x - LIMIT_TOLERANCE);
    
    if(std::isnan(p)) p = ZERO;
    assert(p <= 1);
    return p;
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
    assert(x >= 0);
    if(x == 0) return 0;

    double i = gammap(k_, x / theta_);
    if(std::isnan(i) || i < ZERO) i = ZERO;
    
    double j = loggamma(k_);
    if(std::isnan(j) || j < ZERO) j = ZERO;
    
    double y = std::exp(std::log(i) - 2 * j);
    if(std::isnan(y) || y < ZERO) y = ZERO; 
    
    assert(y <= 1.0);
    return y;
}

/*
 * Returns the value of the LOWER INCOMPLETE gamma function given a and x.
 */
double GammaDistribution::ligamma(double a, double x) noexcept {
    return std::exp(std::log(gammap(a, x)) + loggamma(a));
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
void GammaDistribution::fit(const std::vector<Observation>& values) {
    const auto N = values.size();

    /* Empty cluster */
    if(N == 0) {
        k_ = ZERO;
        theta_ = ZERO;
        return;
    }

    double sum = 0;
    double logsum = 0;

    for(const auto& val : values) {
        // Handle log( 0 ) = NaN
        if(val != 0) {
            sum += val;
            logsum += std::log(val);
        } else {
            sum += ZERO;
            logsum += DBL_MIN;
        }
    }

    const double s = std::log(sum / N) - logsum / N;

    k_ = (3 - s + std::sqrt(std::pow((s - 3), 2) + 24*s)) / (12 * s);
    if(std::isnan(k_) || k_ < ZERO) k_ = ZERO;

    theta_ = sum / (k_ * N);
    if(std::isnan(theta_) || theta_ < ZERO) theta_ = ZERO;
}

/*
 * Resets the the distribution to some default value.  Sets k and theta to 1.
 */
void GammaDistribution::reset() noexcept {
    k_ = 1.0;
    theta_ = 1.0;
}

std::string GammaDistribution::toString() const {
    std::stringstream os;
    os << "Gamma Distribution:\n      k = ";
    os << k_ << "\n      theta = ";
    os << theta_ << "\n";

    return os.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::GammaDistribution& distribution ){

    os << "Gamma Distribution: " << std::endl;
    os << "    k = " << distribution.getK( ) << std::endl;
    os << "    theta = " << distribution.getTheta( ) << std::endl;

    return os;
}


}//namespace
