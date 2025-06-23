#ifndef WEIBULLDISTRIBUTION_H_
#define WEIBULLDISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm{

/**
 * Weibull distribution for reliability analysis and survival modeling.
 * 
 * The Weibull distribution is a continuous probability distribution defined 
 * on the interval [0,∞) and parameterized by two positive parameters:
 * k (shape parameter) and λ (scale parameter).
 * 
 * PDF: f(x; k, λ) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)  for x ≥ 0
 * CDF: F(x; k, λ) = 1 - exp(-(x/λ)^k)  for x ≥ 0
 * 
 * Special cases:
 * - k = 1: Exponential distribution with rate λ
 * - k = 2: Rayleigh distribution  
 * - k < 1: Decreasing failure rate (infant mortality)
 * - k = 1: Constant failure rate (random failures)
 * - k > 1: Increasing failure rate (wear-out failures)
 * 
 * Applications:
 * - Reliability engineering and failure analysis
 * - Survival analysis and lifetime modeling
 * - Weather modeling (wind speeds)
 * - Materials science (strength of materials)
 */
class WeibullDistribution : public ProbabilityDistribution
{   
private:
    /**
     * Shape parameter k - must be positive
     * Controls the shape of the distribution and failure rate behavior
     */
    double k_{1.0};

    /**
     * Scale parameter λ (lambda) - must be positive  
     * Controls the scale/spread of the distribution
     */
    double lambda_{1.0};

    /**
     * Cached value of log(k) for efficiency in probability calculations
     */
    mutable double logK_{0.0};
    
    /**
     * Cached value of log(λ) for efficiency in probability calculations
     */
    mutable double logLambda_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates cached values when parameters change
     */
    void updateCache() noexcept {
        logK_ = std::log(k_);
        logLambda_ = std::log(lambda_);
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Weibull distribution
     * @param k Shape parameter (must be positive and finite)
     * @param lambda Scale parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double k, double lambda) const {
        if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
            throw std::invalid_argument("Shape parameter k must be a positive finite number");
        }
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
            throw std::invalid_argument("Scale parameter lambda must be a positive finite number");
        }
    }

    /**
     * Evaluates the CDF at x
     * CDF(x) = 1 - exp(-(x/λ)^k) for x ≥ 0
     */
    double CDF(double x) noexcept;

    friend std::istream& operator>>(std::istream& is,
            libhmm::WeibullDistribution& distribution);

public:
    /**
     * Constructs a Weibull distribution with given parameters.
     * 
     * @param k Shape parameter (must be positive)
     * @param lambda Scale parameter (must be positive)
     * @throws std::invalid_argument if parameters are not positive finite numbers
     */
    WeibullDistribution(double k = 1.0, double lambda = 1.0)
        : k_{k}, lambda_{lambda} {
        validateParameters(k, lambda);
        updateCache();
    }

    /**
     * Copy constructor
     */
    WeibullDistribution(const WeibullDistribution& other) 
        : k_{other.k_}, lambda_{other.lambda_}, 
          logK_{other.logK_}, logLambda_{other.logLambda_}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    WeibullDistribution& operator=(const WeibullDistribution& other) {
        if (this != &other) {
            k_ = other.k_;
            lambda_ = other.lambda_;
            logK_ = other.logK_;
            logLambda_ = other.logLambda_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    WeibullDistribution(WeibullDistribution&& other) noexcept
        : k_{other.k_}, lambda_{other.lambda_}, 
          logK_{other.logK_}, logLambda_{other.logLambda_}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    WeibullDistribution& operator=(WeibullDistribution&& other) noexcept {
        if (this != &other) {
            k_ = other.k_;
            lambda_ = other.lambda_;
            logK_ = other.logK_;
            logLambda_ = other.logLambda_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Computes the probability density function for the Weibull distribution.
     * 
     * @param value The value at which to evaluate the PDF (should be ≥ 0)
     * @return Probability density, or 0.0 if value is negative
     */
    double getProbability(double value) override;

    /**
     * Fits the distribution parameters to the given data using method of moments.
     * 
     * For Weibull distribution, we use an iterative approach since there's no
     * closed-form solution for maximum likelihood estimation. We estimate:
     * 1. Initial k using method of moments approximation
     * 2. λ using the relationship with sample mean and fitted k
     * 
     * @param values Vector of observed data (should be ≥ 0)
     * @throws std::invalid_argument if values contain negative data
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (k = 1.0, λ = 1.0).
     * This corresponds to an exponential distribution with rate 1.
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the shape parameter k.
     * 
     * @return Current k value
     */
    double getK() const noexcept { return k_; }
    
    /**
     * Sets the shape parameter k.
     * 
     * @param k New shape parameter (must be positive)
     * @throws std::invalid_argument if k <= 0 or is not finite
     */
    void setK(double k) {
        validateParameters(k, lambda_);
        k_ = k;
        cacheValid_ = false;
    }

    /**
     * Gets the scale parameter λ (lambda).
     * 
     * @return Current lambda value
     */
    double getLambda() const noexcept { return lambda_; }
    
    /**
     * Sets the scale parameter λ (lambda).
     * 
     * @param lambda New scale parameter (must be positive)
     * @throws std::invalid_argument if lambda <= 0 or is not finite
     */
    void setLambda(double lambda) {
        validateParameters(k_, lambda);
        lambda_ = lambda;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the distribution.
     * For Weibull(k, λ), mean = λ * Γ(1 + 1/k)
     * 
     * @return Mean value
     */
    double getMean() const noexcept { 
        return lambda_ * std::exp(loggamma(1.0 + 1.0/k_));
    }
    
    /**
     * Gets the variance of the distribution.
     * For Weibull(k, λ), variance = λ² * [Γ(1 + 2/k) - (Γ(1 + 1/k))²]
     * 
     * @return Variance value
     */
    double getVariance() const noexcept { 
        double gamma1 = std::exp(loggamma(1.0 + 1.0/k_));
        double gamma2 = std::exp(loggamma(1.0 + 2.0/k_));
        return lambda_ * lambda_ * (gamma2 - gamma1 * gamma1);
    }
    
    /**
     * Gets the standard deviation of the distribution.
     * 
     * @return Standard deviation
     */
    double getStandardDeviation() const noexcept { return std::sqrt(getVariance()); }
    
    /**
     * Gets the scale parameter (alternative name for lambda).
     * This is sometimes called the "characteristic life" in reliability contexts.
     * 
     * @return Scale parameter value
     */
    double getScale() const noexcept { return lambda_; }
    
    /**
     * Gets the shape parameter (alternative name for k).
     * This is sometimes called the "Weibull modulus" in reliability contexts.
     * 
     * @return Shape parameter value
     */
    double getShape() const noexcept { return k_; }
};

/**
 * Stream output operator for Weibull distribution.
 */
std::ostream& operator<<(std::ostream& os, const libhmm::WeibullDistribution& distribution);

} // namespace libhmm

#endif // WEIBULLDISTRIBUTION_H_
