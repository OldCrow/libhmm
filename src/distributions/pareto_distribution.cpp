#include "libhmm/distributions/pareto_distribution.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cfloat>

namespace libhmm
{

/*
 * Returns the value of the P(x) of the Pareto distribution.
 *
 * Note that the Pareto PDF at x is *NOT* the value of P(x).  For continuous
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
 * Pareto distributions are 0 for x <= xm;
 */            
double ParetoDistribution::getProbability(double x) {
    double p = 0.0;
    if(x <= xm_) {
        return 0;
    }
    else if(x > xm_ + LIMIT_TOLERANCE) {
        p = CDF(x) - CDF(x - LIMIT_TOLERANCE);
    }
    else if(x > xm_ && x < xm_ + LIMIT_TOLERANCE) {
        // Number is small enough...
        p = 0;
    }

    if(std::isnan(p)) p = ZERO;
    assert(p <= 1);
    return p;
}

/*
 * Evaluates the CDF for the Pareto distribution at x.  The CDF is defined as
 *
 *   F(x) = 1 - (xm/x)^k
 */
double ParetoDistribution::CDF(double x) noexcept {
    const double y = 1 - std::pow(xm_ / x, k_);
    assert(y >= 0);
    return y;
}

/*
 * Sets rate parameter such that the resulting Pareto 
 * distribution fits the data.
 *
 * xm is defined as min( x_i, i = 1..N )
 *
 * The rate parameter is defined as
 *                    n
 * k = ------------------------------
 *     sum( ln(x_i) - ln(xm), i=1..N)
 *
 */                   
void ParetoDistribution::fit(const std::vector<Observation>& values) {
    // You can't fit a Pareto distribution with a data set of only one 
    // point or zero points.
    if(values.empty() || values.size() == 1) {
        reset();
        return;
    }

    // Find minimum value for xm
    const auto minValue = *std::min_element(values.begin(), values.end());
    xm_ = minValue;
    
    // Calculate sum of log differences
    double sum = 0;
    for(const auto& val : values) {
        sum += std::log(val) - std::log(xm_);
    }
    
    k_ = static_cast<double>(values.size()) / sum;

    // I am *really* not sure we can assert this
    //assert(k > 0);
}

/*
 * Resets the the distribution to some default value. 
 */
void ParetoDistribution::reset() noexcept {
    xm_ = 1.0;
    k_ = 1.0;
}

std::string ParetoDistribution::toString() const {
    std::stringstream os;
    os << "Pareto Distribution:" << std::endl;
    os << "   k = " << k_ << std::endl;
    os << "   xm = " << xm_ << std::endl;
    return os.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::ParetoDistribution& distribution ){
    os << "Pareto Distribution: " << std::endl;
    os << "    k = " << distribution.getK( ) << std::endl;
    os << "    xm = " << distribution.getXm( ) << std::endl;
    os << std::endl;
    
    return os;
}

std::istream& operator>>( std::istream& is,
        libhmm::ParetoDistribution& distribution ){
    std::string s, t;
    is >> s; //" k"
    is >> s; // "="
    is >> t;
    distribution.setK(std::stod(t));

    is >> s; // " xm"
    is >> s; // " ="
    is >> t;
    distribution.setXm(std::stod(t));

    return is;
}


}
