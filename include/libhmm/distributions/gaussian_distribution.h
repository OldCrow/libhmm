#ifndef GAUSSIANDISTRIBUTION_H_
#define GAUSSIANDISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm{

class GaussianDistribution : public ProbabilityDistribution
{   
private:
    /*
     * Represents the mean of a Gaussian distribution
     */
    double mean_{0.0};

    /*
     * Represents the standard deviation of a Gaussian distribution
     */
    double standardDeviation_{1.0};

    /* 
     * Evaluates the CDF at x
     */
    double CDF(double x) noexcept;

    friend std::istream& operator>>(std::istream& is,
            libhmm::GaussianDistribution& distribution);

public:
    GaussianDistribution(double mean = 0.0, double standardDeviation = 1.0)
        : mean_{mean}, standardDeviation_{standardDeviation} {
        if (std::isnan(mean) || std::isinf(mean)) {
            throw std::invalid_argument("Mean must be a finite number");
        }
        if (std::isnan(standardDeviation) || std::isinf(standardDeviation) || standardDeviation <= 0.0) {
            throw std::invalid_argument("Standard deviation must be a positive finite number");
        }
    }

    double getProbability(double value) override;

    void fit(const std::vector<Observation>& values) override;

    void reset() noexcept override;

    std::string toString() const override;

    double getMean() const noexcept { return mean_; }
    
    void setMean(double mean) {
        if (std::isnan(mean) || std::isinf(mean)) {
            throw std::invalid_argument("Mean must be a finite number");
        }
        mean_ = mean;
    }

    double getStandardDeviation() const noexcept { return standardDeviation_; }
    
    void setStandardDeviation(double stdDev) {
        if (std::isnan(stdDev) || std::isinf(stdDev) || stdDev <= 0.0) {
            throw std::invalid_argument("Standard deviation must be a positive finite number");
        }
        standardDeviation_ = stdDev;
    }

};

std::ostream& operator<<( std::ostream&, 
        const libhmm::GaussianDistribution& );
//std::istream& operator>>( std::istream&,
//        const libhmm::GaussianDistribution& );
} // namespace
#endif
