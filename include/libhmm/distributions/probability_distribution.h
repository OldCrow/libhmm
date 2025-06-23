#ifndef PROBABILITYDISTRIBUTION_H_
#define PROBABILITYDISTRIBUTION_H_

#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include "libhmm/common/common.h"
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include "libhmm/common/string_tokenizer.h"

namespace libhmm
{

/*
 * Base class for any probability distribution attached to an HMM.
 * This class came out of the necessity to have a case for discrete
 * distributions as well as continuous (gamma, Gaussian, etc) 
 * ways of modeling emissions.
 */
class ProbabilityDistribution
{
protected:
    /*
     * Implements the gamma function.
     *
     * This code comes from Numerical Recipes in C, p.214
     */
    double loggamma(double x) const noexcept;
    
    /*
     * Returns the value of the error function at x.
     *
     * Source: Numerical Recipes in C
     */
    double errorf(double x) noexcept;

    /*
     * Returns the value of the regularized incomplete Gamma function at x.
     * Needed to compute the error function and the lower incomplete Gamma
     * function.
     *
     * Source: Numerical Recipes in C
     */
    double gammap(double a, double x) noexcept;

    /*
     * Returns the incomplete gamma function Q(a,x) evaluated by its continued
     * fraction representation as gammcf.  Also returns ln gamma(a) as gln.
     *
     * Source: Numerical Recipes in C
     */
    void gcf(double& gammcf, double a, double x, double& gln) noexcept;

    /*
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

    /*
     * Returns the probability of an Observation.
     */
    virtual double getProbability(Observation val) = 0;

    /*
     * Fits the values to match the distribution.
     */
    virtual void fit(const std::vector<Observation>& values) = 0;

    /*
     * Resets the distribution to some default.
     */
    virtual void reset() noexcept = 0;

    /*
     * Creates a string representation of the ProbabilityDistribution.
     */
    virtual std::string toString() const = 0;

    /*
     * Returns the value of the inverse error function at x.
     *
     * Source: http://www.codecogs.com/d-ox/maths/special/errorfn_inv.php
     */
    double errorf_inv(double x) noexcept;

            
}; // class Probability Distribution

//BOOST_IS_ABSTRACT(ProbabilityDistribution);

/*std::ostream& operator<<( std::ostream& os, 
        const libhmm::ProbabilityDistribution& p );*/

}//namespace

#endif
