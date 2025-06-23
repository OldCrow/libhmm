#ifndef GAUSSIANDISTRIBUTION_H_
#define GAUSSIANDISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace libhmm{

/**
 * Modern C++17 Gaussian (Normal) distribution for modeling continuous data.
 * 
 * The Gaussian distribution is a continuous probability distribution that is
 * symmetric around the mean, showing that data near the mean are more frequent
 * in occurrence than data far from the mean.
 * 
 * PDF: f(x) = (1/(σ√(2π))) * exp(-½((x-μ)/σ)²)
 * where μ is the mean and σ is the standard deviation
 * 
 * Properties:
 * - Mean: μ
 * - Variance: σ²
 * - Support: x ∈ (-∞, ∞)
 */
class GaussianDistribution : public ProbabilityDistribution
{   
private:
    /**
     * Mean parameter μ - can be any finite real number
     */
    double mean_{0.0};

    /**
     * Standard deviation parameter σ - must be positive
     */
    double standardDeviation_{1.0};

    /**
     * Cached normalization constant for efficiency in probability calculations
     * Stores 1/(σ√(2π))
     */
    mutable double normalizationConstant_{0.0};
    
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
        normalizationConstant_ = 1.0 / (standardDeviation_ * std::sqrt(2.0 * M_PI));
        negHalfSigmaSquaredInv_ = -0.5 / sigma2;
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Gaussian distribution
     * @param mean Mean parameter (any finite value)
     * @param stdDev Standard deviation parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double mean, double stdDev) const {
        if (std::isnan(mean) || std::isinf(mean)) {
            throw std::invalid_argument("Mean must be a finite number");
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
            libhmm::GaussianDistribution& distribution);

public:
    /**
     * Constructs a Gaussian distribution with given parameters.
     * 
     * @param mean Mean parameter μ (any finite value)
     * @param standardDeviation Standard deviation parameter σ (must be positive)
     * @throws std::invalid_argument if parameters are invalid
     */
    GaussianDistribution(double mean = 0.0, double standardDeviation = 1.0)
        : mean_{mean}, standardDeviation_{standardDeviation}, 
          normalizationConstant_{0.0}, negHalfSigmaSquaredInv_{0.0}, cacheValid_{false} {
        validateParameters(mean, standardDeviation);
        updateCache();
    }

    /**
     * Copy constructor
     */
    GaussianDistribution(const GaussianDistribution& other) 
        : mean_{other.mean_}, standardDeviation_{other.standardDeviation_}, 
          normalizationConstant_{other.normalizationConstant_}, 
          negHalfSigmaSquaredInv_{other.negHalfSigmaSquaredInv_}, 
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    GaussianDistribution& operator=(const GaussianDistribution& other) {
        if (this != &other) {
            mean_ = other.mean_;
            standardDeviation_ = other.standardDeviation_;
            normalizationConstant_ = other.normalizationConstant_;
            negHalfSigmaSquaredInv_ = other.negHalfSigmaSquaredInv_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    GaussianDistribution(GaussianDistribution&& other) noexcept
        : mean_{other.mean_}, standardDeviation_{other.standardDeviation_}, 
          normalizationConstant_{other.normalizationConstant_}, 
          negHalfSigmaSquaredInv_{other.negHalfSigmaSquaredInv_}, 
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    GaussianDistribution& operator=(GaussianDistribution&& other) noexcept {
        if (this != &other) {
            mean_ = other.mean_;
            standardDeviation_ = other.standardDeviation_;
            normalizationConstant_ = other.normalizationConstant_;
            negHalfSigmaSquaredInv_ = other.negHalfSigmaSquaredInv_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Computes the probability density function for the Gaussian distribution.
     * 
     * @param value The value at which to evaluate the PDF
     * @return Probability density
     */
    double getProbability(double value) override;

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Gaussian distribution, MLE gives sample mean and sample standard deviation.
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (μ = 0.0, σ = 1.0).
     * This corresponds to the standard normal distribution.
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the mean parameter μ.
     * 
     * @return Current mean value
     */
    double getMean() const noexcept { return mean_; }
    
    /**
     * Sets the mean parameter μ.
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
     * Gets the standard deviation parameter σ.
     * 
     * @return Current standard deviation value
     */
    double getStandardDeviation() const noexcept { return standardDeviation_; }
    
    /**
     * Sets the standard deviation parameter σ.
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
     * Gets the variance of the distribution.
     * For Gaussian distribution, variance = σ²
     * 
     * @return Variance value
     */
    double getVariance() const noexcept { 
        return standardDeviation_ * standardDeviation_; 
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

};

std::ostream& operator<<( std::ostream&, 
        const libhmm::GaussianDistribution& );
//std::istream& operator>>( std::istream&,
//        const libhmm::GaussianDistribution& );
} // namespace
#endif
