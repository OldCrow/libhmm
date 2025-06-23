#ifndef LOGNORMALDISTRIBUTION_H_
#define LOGNORMALDISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cassert>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace libhmm{

/**
 * Modern C++17 Log-Normal distribution for modeling positive continuous data.
 * 
 * The Log-Normal distribution is a continuous probability distribution of a
 * random variable whose logarithm is normally distributed. It's commonly used
 * to model sizes, lengths, and other positive quantities that arise from
 * multiplicative processes.
 * 
 * Important note about parameterization:
 * This implementation uses the "log-scale" parameterization where:
 * - μ (mean_) is the mean of the underlying normal distribution ln(X)
 * - σ (standardDeviation_) is the standard deviation of ln(X)
 * 
 * PDF: f(x) = (1/(x·σ·√(2π))) * exp(-½((ln(x)-μ)/σ)²) for x > 0
 * where μ is the mean of ln(X) and σ is the std dev of ln(X)
 * 
 * Properties:
 * - Mean: exp(μ + σ²/2)
 * - Variance: (exp(σ²) - 1) * exp(2μ + σ²)
 * - Mode: exp(μ - σ²)
 * - Support: x ∈ (0, ∞)
 */
class LogNormalDistribution : public ProbabilityDistribution
{   
private:
    /**
     * Mean parameter μ of the underlying normal distribution (mean of ln(X))
     * Can be any finite real number
     */
    double mean_{0.0};

    /**
     * Standard deviation parameter σ of the underlying normal distribution (std dev of ln(X))
     * Must be positive
     */
    double standardDeviation_{1.0};
    
    /**
     * Cached value of ln(σ√(2π)) for efficiency in probability calculations
     */
    mutable double logNormalizationConstant_{0.0};
    
    /**
     * Cached value of -1/(2σ²) for efficiency in probability calculations
     */
    mutable double negHalfSigmaSquaredInv_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates cached values when parameters change
     */
    void updateCache() const noexcept {
        double sigma2 = standardDeviation_ * standardDeviation_;
        logNormalizationConstant_ = std::log(standardDeviation_ * std::sqrt(2.0 * M_PI));
        negHalfSigmaSquaredInv_ = -0.5 / sigma2;
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Log-Normal distribution
     * @param mean Mean of ln(X) (any finite value)
     * @param stdDev Standard deviation of ln(X) (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double mean, double stdDev) const {
        if (std::isnan(mean) || std::isinf(mean)) {
            throw std::invalid_argument("Mean parameter must be a finite number");
        }
        if (std::isnan(stdDev) || std::isinf(stdDev) || stdDev <= 0.0) {
            throw std::invalid_argument("Standard deviation must be a positive finite number");
        }
    }

    /**
     * Evaluates the CDF at x using the error function
     */
    double CDF(double x) noexcept;

    friend std::istream& operator>>(std::istream& is,
            libhmm::LogNormalDistribution& distribution);

public:
    /**
     * Constructs a Log-Normal distribution with given parameters.
     * 
     * @param mean Mean of the underlying normal distribution (μ, any finite value)
     * @param standardDeviation Standard deviation of the underlying normal distribution (σ, must be positive)
     * @throws std::invalid_argument if parameters are invalid
     */
    LogNormalDistribution(double mean = 0.0, double standardDeviation = 1.0)
        : mean_{mean}, standardDeviation_{standardDeviation},
          logNormalizationConstant_{0.0}, negHalfSigmaSquaredInv_{0.0}, cacheValid_{false} {
        validateParameters(mean, standardDeviation);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    LogNormalDistribution(const LogNormalDistribution& other) 
        : mean_{other.mean_}, standardDeviation_{other.standardDeviation_}, 
          logNormalizationConstant_{other.logNormalizationConstant_}, 
          negHalfSigmaSquaredInv_{other.negHalfSigmaSquaredInv_}, 
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    LogNormalDistribution& operator=(const LogNormalDistribution& other) {
        if (this != &other) {
            mean_ = other.mean_;
            standardDeviation_ = other.standardDeviation_;
            logNormalizationConstant_ = other.logNormalizationConstant_;
            negHalfSigmaSquaredInv_ = other.negHalfSigmaSquaredInv_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    LogNormalDistribution(LogNormalDistribution&& other) noexcept
        : mean_{other.mean_}, standardDeviation_{other.standardDeviation_}, 
          logNormalizationConstant_{other.logNormalizationConstant_}, 
          negHalfSigmaSquaredInv_{other.negHalfSigmaSquaredInv_}, 
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    LogNormalDistribution& operator=(LogNormalDistribution&& other) noexcept {
        if (this != &other) {
            mean_ = other.mean_;
            standardDeviation_ = other.standardDeviation_;
            logNormalizationConstant_ = other.logNormalizationConstant_;
            negHalfSigmaSquaredInv_ = other.negHalfSigmaSquaredInv_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Computes the probability density function for the Log-Normal distribution.
     * 
     * @param value The value at which to evaluate the PDF
     * @return Probability density (or approximated probability for discrete sampling)
     */
    double getProbability(double value) override;

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Log-Normal distribution, MLE gives μ = mean(ln(x_i)) and σ = std_dev(ln(x_i)).
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (μ = 0.0, σ = 1.0).
     * This corresponds to the standard log-normal distribution.
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the mean parameter μ of the underlying normal distribution.
     * 
     * @return Current mean parameter value
     */
    double getMean() const noexcept { return mean_; }
    
    /**
     * Sets the mean parameter μ of the underlying normal distribution.
     * 
     * @param mean New mean parameter (any finite value)
     * @throws std::invalid_argument if mean is not finite
     */
    void setMean(double mean) {
        validateParameters(mean, standardDeviation_);
        mean_ = mean;
        cacheValid_ = false;
    }

    /**
     * Gets the standard deviation parameter σ of the underlying normal distribution.
     * 
     * @return Current standard deviation parameter value
     */
    double getStandardDeviation() const noexcept { return standardDeviation_; }
    
    /**
     * Sets the standard deviation parameter σ of the underlying normal distribution.
     * 
     * @param stdDev New standard deviation parameter (must be positive)
     * @throws std::invalid_argument if stdDev <= 0 or is not finite
     */
    void setStandardDeviation(double stdDev) {
        validateParameters(mean_, stdDev);
        standardDeviation_ = stdDev;
        cacheValid_ = false;
    }
    
    /**
     * Sets both parameters simultaneously.
     * 
     * @param mean New mean parameter
     * @param stdDev New standard deviation parameter
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParameters(double mean, double stdDev) {
        validateParameters(mean, stdDev);
        mean_ = mean;
        standardDeviation_ = stdDev;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the Log-Normal distribution (not the underlying normal).
     * For Log-Normal distribution, mean = exp(μ + σ²/2)
     * 
     * @return Mean of the Log-Normal distribution
     */
    double getDistributionMean() const noexcept { 
        double sigma2 = standardDeviation_ * standardDeviation_;
        return std::exp(mean_ + sigma2 / 2.0); 
    }
    
    /**
     * Gets the variance of the Log-Normal distribution.
     * For Log-Normal distribution, variance = (exp(σ²) - 1) * exp(2μ + σ²)
     * 
     * @return Variance of the Log-Normal distribution
     */
    double getVariance() const noexcept { 
        double sigma2 = standardDeviation_ * standardDeviation_;
        return (std::exp(sigma2) - 1.0) * std::exp(2.0 * mean_ + sigma2); 
    }
    
    /**
     * Gets the standard deviation of the Log-Normal distribution.
     * 
     * @return Standard deviation of the Log-Normal distribution
     */
    double getDistributionStandardDeviation() const noexcept { 
        return std::sqrt(getVariance()); 
    }
    
    /**
     * Gets the mode of the Log-Normal distribution.
     * For Log-Normal distribution, mode = exp(μ - σ²)
     * 
     * @return Mode of the Log-Normal distribution
     */
    double getMode() const noexcept {
        double sigma2 = standardDeviation_ * standardDeviation_;
        return std::exp(mean_ - sigma2);
    }
    
    /**
     * Gets the median of the Log-Normal distribution.
     * For Log-Normal distribution, median = exp(μ)
     * 
     * @return Median of the Log-Normal distribution
     */
    double getMedian() const noexcept {
        return std::exp(mean_);
    }

};

std::ostream& operator<<( std::ostream&, 
        const libhmm::LogNormalDistribution& );

} // namespace
#endif
