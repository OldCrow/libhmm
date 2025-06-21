#ifndef PARETODISTRIBUTION_H_
#define PARETODISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm{

class ParetoDistribution : public ProbabilityDistribution
{   
private:
    /*
     * Represents the rate parameter 
     */
    double k_{1.0};

    /*
     * Represents x_m
     */
    double xm_{1.0};

    /* 
     * Evaluates the CDF at x
     */
    double CDF(double x) noexcept;

    friend std::istream& operator>>(std::istream& is,
            libhmm::ParetoDistribution& distribution);

public:
    ParetoDistribution(double k = 1.0, double xm = 1.0)
        : k_{k}, xm_{xm} {
        if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
            throw std::invalid_argument("Shape parameter k must be a positive finite number");
        }
        if (std::isnan(xm) || std::isinf(xm) || xm <= 0.0) {
            throw std::invalid_argument("Scale parameter xm must be a positive finite number");
        }
    }

    double getProbability(double value) override;

    void fit(const std::vector<Observation>& values) override;

    void reset() noexcept override;

    std::string toString() const override;

    double getK() const noexcept { return k_; }
    
    void setK(double k) {
        if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
            throw std::invalid_argument("Shape parameter k must be a positive finite number");
        }
        k_ = k;
    }

    double getXm() const noexcept { return xm_; }
    
    void setXm(double xm) {
        if (std::isnan(xm) || std::isinf(xm) || xm <= 0.0) {
            throw std::invalid_argument("Scale parameter xm must be a positive finite number");
        }
        xm_ = xm;
    }

};

std::ostream& operator<<( std::ostream&, 
        const libhmm::ParetoDistribution& );
//std::istream& operator>>( std::istream&,
//        const libhmm::ParetoDistribution& );
} // namespace
#endif
