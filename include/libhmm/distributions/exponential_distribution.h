#ifndef EXPONENTIALDISTRIBUTION_H_
#define EXPONENTIALDISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm{

/**
 * Modern C++17 Exponential distribution for modeling waiting times and decay processes.
 * 
 * The Exponential distribution is a continuous probability distribution that describes
 * the time between events in a Poisson point process. It's commonly used to model
 * lifetimes, waiting times, and decay processes.
 * 
 * PDF: f(x) = λ * exp(-λx) for x ≥ 0, 0 otherwise
 * CDF: F(x) = 1 - exp(-λx) for x ≥ 0, 0 otherwise
 * where λ is the rate parameter (λ > 0)
 * 
 * Properties:
 * - Mean: 1/λ
 * - Variance: 1/λ²
 * - Support: x ∈ [0, ∞)
 * - Memoryless property: P(X > s+t | X > s) = P(X > t)
 */
class ExponentialDistribution : public ProbabilityDistribution
{   
private:
    /**
     * Rate parameter λ - must be positive
     * Higher values indicate faster decay/shorter expected waiting times
     */
    double lambda_{1.0};
    
    /**
     * Cached value of ln(λ) for efficiency in probability calculations
     */
    mutable double logLambda_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates cached values when parameters change
     */
    void updateCache() const noexcept {
        logLambda_ = std::log(lambda_);
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Exponential distribution
     * @param lambda Rate parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double lambda) const {
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
            throw std::invalid_argument("Lambda (rate parameter) must be a positive finite number");
        }
    }

    /**
     * Evaluates the CDF at x using the standard exponential CDF formula
     */
    double CDF(double x) noexcept;

    friend std::istream& operator>>(std::istream& is,
            libhmm::ExponentialDistribution& distribution);

public:
    /**
     * Constructs an Exponential distribution with given rate parameter.
     * 
     * @param lambda Rate parameter λ (must be positive)
     * @throws std::invalid_argument if lambda is invalid
     */
    ExponentialDistribution(double lambda = 1.0)
        : lambda_{lambda}, logLambda_{0.0}, cacheValid_{false} {
        validateParameters(lambda);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    ExponentialDistribution(const ExponentialDistribution& other) 
        : lambda_{other.lambda_}, logLambda_{other.logLambda_}, 
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    ExponentialDistribution& operator=(const ExponentialDistribution& other) {
        if (this != &other) {
            lambda_ = other.lambda_;
            logLambda_ = other.logLambda_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    ExponentialDistribution(ExponentialDistribution&& other) noexcept
        : lambda_{other.lambda_}, logLambda_{other.logLambda_}, 
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    ExponentialDistribution& operator=(ExponentialDistribution&& other) noexcept {
        if (this != &other) {
            lambda_ = other.lambda_;
            logLambda_ = other.logLambda_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Computes the probability density function for the Exponential distribution.
     * 
     * @param value The value at which to evaluate the PDF
     * @return Probability density (or approximated probability for discrete sampling)
     */
    double getProbability(double value) override;

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Exponential distribution, MLE gives λ = 1/sample_mean.
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (λ = 1.0).
     * This corresponds to the standard exponential distribution.
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the rate parameter λ.
     * 
     * @return Current rate parameter value
     */
    double getLambda() const noexcept { return lambda_; }
    
    /**
     * Sets the rate parameter λ.
     * 
     * @param lambda New rate parameter (must be positive)
     * @throws std::invalid_argument if lambda <= 0 or is not finite
     */
    void setLambda(double lambda) {
        validateParameters(lambda);
        lambda_ = lambda;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the distribution.
     * For Exponential distribution, mean = 1/λ
     * 
     * @return Mean value
     */
    double getMean() const noexcept { 
        return 1.0 / lambda_; 
    }
    
    /**
     * Gets the variance of the distribution.
     * For Exponential distribution, variance = 1/λ²
     * 
     * @return Variance value
     */
    double getVariance() const noexcept { 
        return 1.0 / (lambda_ * lambda_); 
    }
    
    /**
     * Gets the standard deviation of the distribution.
     * For Exponential distribution, std_dev = 1/λ
     * 
     * @return Standard deviation value
     */
    double getStandardDeviation() const noexcept { 
        return 1.0 / lambda_; 
    }
    
    /**
     * Gets the scale parameter (reciprocal of rate parameter).
     * This is equivalent to the mean for exponential distributions.
     * 
     * @return Scale parameter (1/λ)
     */
    double getScale() const noexcept {
        return 1.0 / lambda_;
    }

};

std::ostream& operator<<( std::ostream&, 
        const libhmm::ExponentialDistribution& );
//std::istream& operator>>( std::istream&,
//        const libhmm::ExponentialDistribution& );
} // namespace
#endif
