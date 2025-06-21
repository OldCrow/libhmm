#include "libhmm/distributions/exponential_distribution.h"
#include <iostream>
#include <numeric>

namespace libhmm
{

/*
 * Returns the value of the P(x) of the Exponential distribution.
 *
 * Note that the Exponential PDF at x is *NOT* the value of P(x).  For continuous
 * distributions, the PDF at x is defined as zero.  We can, however, use
 * P( x - e <= x <= x + e ) and use the CDF.  Wikipedia states that the "point
 * probability" that X is exactly b can be found as
 *
 * P(X=b) = F(b) - lim F(x)
 *                    x->b-
 * 
 * I can't take a limit from the left side, but I can approximate with
 *
 * P(X=b) = F(b) - F(x - e)
 *
 * where e = 1e-6
 *
 * Exponential distributions are 0 for x = 0;
 */            
double ExponentialDistribution::getProbability(double x) {
    if(x == 0) {
        return 0;
    }
    
    double p = 0.0;
    if(x > LIMIT_TOLERANCE) {
        p = CDF(x) - CDF(x - LIMIT_TOLERANCE);
    } else if(x > 0 && x < LIMIT_TOLERANCE) {
        // Number is small enough...
        p = 0;
    }

    if(std::isnan(p)) p = ZERO;
    assert(p <= 1);
    return p;
}

/*
 * Evaluates the CDF for the Normal distribution at x.  The CDF is defined as
 *
 *   F(x) = 1 - exp( -lambda * x )
 */
double ExponentialDistribution::CDF(double x) noexcept {
    const double y = 1 - std::exp(-lambda_ * x);
    assert(y >= 0);
    return y;
}

/*
 * Sets rate parameter such that the resulting Exponential 
 * distribution fits the data.
 *
 * The rate parameter is defined as
 *
 * lambda = 1 / mean
 *
 * The mean is defined as:
 *
 * u = sum( x_i, i=1..N ) / N
 *
 * where N is the number of values.
 *
 */                   
void ExponentialDistribution::fit(const std::vector<Observation>& values) {
    // You can't fit an exponential distribution with a data set of only one 
    // point or zero points.
    if(values.empty() || values.size() == 1) {
        reset();
        return;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double mean = sum / values.size();
    
    lambda_ = 1.0 / mean;

    // I am *really* not sure we can assert this
    //assert(lambda > 0);
}

/*
 * Resets the the distribution to some default value. 
 */
void ExponentialDistribution::reset() noexcept {
    lambda_ = 1.0;
}

std::string ExponentialDistribution::toString() const {
    std::stringstream os;
    os << "Exponential Distribution:\n      Rate parameter = ";
    os << lambda_ << "\n";
    return os.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::ExponentialDistribution& distribution ){
    os << "Exponential Distribution: " << std::endl;
    os << "    Rate parameter = " << distribution.getLambda( ) << std::endl;
    os << std::endl;
    
    return os;
}

std::istream& operator>>( std::istream& is,
        libhmm::ExponentialDistribution& distribution ){
    std::string s, t;
    is >> s; //" Rate"
    is >> s; // "parameter"
    is >> s; // "="
    is >> t;
    distribution.setLambda(std::stod(t));

    return is;
}


}
