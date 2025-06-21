#ifndef GAMMADISTRIBUTION_H_
#define GAMMADISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm{

class GammaDistribution : public ProbabilityDistribution
{    
private:
    /*
     * Shape parameter
     */
    double k_{1.0};

    /*
     * Scale parameter
     */
    double theta_{1.0};

    /*
     * Evaluates the LOWER INCOMPLETE gamma function at x
     */
    double ligamma(double a, double x) noexcept;

    /* 
     * Evaluates the CDF at x
     */
    double CDF(double x) noexcept;

public:    
    GammaDistribution(double k = 1.0, double theta = 1.0)
        : k_{k}, theta_{theta} {
        if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
            throw std::invalid_argument("Shape parameter k must be a positive finite number");
        }
        if (std::isnan(theta) || std::isinf(theta) || theta <= 0.0) {
            throw std::invalid_argument("Scale parameter theta must be a positive finite number");
        }
    }

    double getProbability(double value) override;

    void fit(const std::vector<Observation>& values) override;

    void reset() noexcept override;

    std::string toString() const override;

    double getK() const noexcept { return k_; }

    double getTheta() const noexcept { return theta_; }
};

std::ostream& operator<<( std::ostream&, 
        const libhmm::GammaDistribution& );

} // namespace
#endif
