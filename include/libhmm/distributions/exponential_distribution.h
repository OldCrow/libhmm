#ifndef EXPONENTIALDISTRIBUTION_H_
#define EXPONENTIALDISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm{

class ExponentialDistribution : public ProbabilityDistribution
{   
private:
    /*
     * Represents the rate parameter Exponential distribution
     * In MATLAB, mu = 1 / lambda
     */
    double lambda_{1.0};

    /* 
     * Evaluates the CDF at x
     */
    double CDF(double x) noexcept;

    friend std::istream& operator>>(std::istream& is,
            libhmm::ExponentialDistribution& distribution);

public:
    ExponentialDistribution(double lambda = 1.0) : lambda_{lambda} {
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
            throw std::invalid_argument("Lambda must be a positive finite number");
        }
    }

    double getProbability(double value) override;

    void fit(const std::vector<Observation>& values) override;

    void reset() noexcept override;

    std::string toString() const override;

    double getLambda() const noexcept { return lambda_; }
    
    void setLambda(double lambda) {
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
            throw std::invalid_argument("Lambda must be a positive finite number");
        }
        lambda_ = lambda;
    }

};

std::ostream& operator<<( std::ostream&, 
        const libhmm::ExponentialDistribution& );
//std::istream& operator>>( std::istream&,
//        const libhmm::ExponentialDistribution& );
} // namespace
#endif
