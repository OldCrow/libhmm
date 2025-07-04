#ifndef PROBABILITYDISTRIBUTION_H_
#define PROBABILITYDISTRIBUTION_H_

#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <limits>
#include <cmath>
#include "libhmm/common/common.h"
// Removed Boost includes - using C++17 serialization
#include "libhmm/common/string_tokenizer.h"

namespace libhmm
{

/**
 * Base class for any probability distribution attached to an HMM.
 * This class came out of the necessity to have a case for discrete
 * distributions as well as continuous (gamma, Gaussian, etc) 
 * ways of modeling emissions.
 */
class ProbabilityDistribution
{
protected:
    // Removed loggamma - using std::lgamma directly for better performance
    
    /**
     * Returns the value of the error function at x.
     *
     * Source: Numerical Recipes in C
     */
    double errorf(double x) noexcept;

    /**
     * Returns the value of the regularized incomplete Gamma function at x.
     * Needed to compute the error function and the lower incomplete Gamma
     * function.
     *
     * Source: Numerical Recipes in C
     */
    double gammap(double a, double x) noexcept;

    /**
     * Returns the incomplete gamma function Q(a,x) evaluated by its continued
     * fraction representation as gammcf.  Also returns ln gamma(a) as gln.
     *
     * Source: Numerical Recipes in C
     */
    void gcf(double& gammcf, double a, double x, double& gln) noexcept;

    /**
     * Returns the incomplete gamma function P(a,x) evaluated by its series
     * representation as gamser.  Also returns ln( gamma(a) ) as gln.
     *
     * Source: Numerical Recipes in C.
     */
    void gser(double& gamser, double a, double x, double& gln) noexcept;

public:
    ProbabilityDistribution() = default;
    virtual ~ProbabilityDistribution() = default;
    
    // Non-copyable but movable for now
    ProbabilityDistribution(const ProbabilityDistribution&) = default;
    ProbabilityDistribution& operator=(const ProbabilityDistribution&) = default;
    ProbabilityDistribution(ProbabilityDistribution&&) = default;
    ProbabilityDistribution& operator=(ProbabilityDistribution&&) = default;

    /**
     * Returns the probability of an Observation.
     */
    virtual double getProbability(Observation val) = 0;
    
    /**
     * Returns the log probability of an Observation.
     * Default implementation uses log(getProbability(val)) but derived classes
     * can override for better numerical stability and performance.
     */
    virtual double getLogProbability(Observation val) const {
        const double prob = const_cast<ProbabilityDistribution*>(this)->getProbability(val);
        return (prob > 0.0) ? std::log(prob) : -std::numeric_limits<double>::infinity();
    }

    /**
     * Fits the values to match the distribution.
     */
    virtual void fit(const std::vector<Observation>& values) = 0;

    /**
     * Resets the distribution to some default.
     */
    virtual void reset() noexcept = 0;

    /**
     * Creates a string representation of the ProbabilityDistribution.
     */
    virtual std::string toString() const = 0;

    /**
     * Returns the value of the inverse error function at x.
     *
     * Source: http://www.codecogs.com/d-ox/maths/special/errorfn_inv.php
     */
    double errorf_inv(double y) noexcept;
            
}; // class Probability Distribution

}//namespace

#endif
