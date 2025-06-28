#ifndef NEGATIVE_BINOMIAL_DISTRIBUTION_H_
#define NEGATIVE_BINOMIAL_DISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <vector>
#include <limits>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm {

/**
 * Modern C++17 Negative Binomial distribution for modeling discrete count data.
 * 
 * The Negative Binomial distribution models the number of failures before 
 * the r-th success in a sequence of independent Bernoulli trials, each with 
 * success probability p.
 * 
 * PMF: P(X = k) = C(k+r-1, k) * p^r * (1-p)^k
 * where C(k+r-1, k) is the binomial coefficient
 * 
 * Alternative parameterization (often used in practice):
 * - r: number of successes (positive real number)
 * - p: success probability (in [0,1])
 * 
 * Properties:
 * - Mean: r * (1-p) / p
 * - Variance: r * (1-p) / p²
 * - Support: k ∈ {0, 1, 2, ...}
 */
class NegativeBinomialDistribution : public ProbabilityDistribution
{
private:
    /**
     * Number of successes r - must be positive
     */
    double r_{5.0};
    
    /**
     * Success probability p - must be in (0,1]
     */
    double p_{0.5};
    
    /**
     * Cached values for efficiency in probability calculations
     */
    mutable double logP_{0.0};
    mutable double log1MinusP_{0.0};
    mutable double logGammaR_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates cached values when parameters change
     */
    void updateCache() const noexcept {
        logP_ = std::log(p_);
        log1MinusP_ = std::log(1.0 - p_);
        logGammaR_ = std::lgamma(r_);
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Negative Binomial distribution
     * @param r Number of successes (must be positive)
     * @param p Success probability (must be in (0,1])
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double r, double p) const {
        if (std::isnan(r) || std::isinf(r) || r <= 0.0) {
            throw std::invalid_argument("Number of successes must be positive");
        }
        if (std::isnan(p) || std::isinf(p) || p <= 0.0 || p > 1.0) {
            throw std::invalid_argument("Success probability must be in (0,1]");
        }
    }
    
    /**
     * Computes log of generalized binomial coefficient log(C(k+r-1, k))
     * using the gamma function: C(k+r-1, k) = Γ(k+r) / (Γ(k+1) * Γ(r))
     */
    double logGeneralizedBinomialCoefficient(int k) const {
        if (k < 0) return -std::numeric_limits<double>::infinity();
        
        // Ensure cache is valid
        if (!cacheValid_) {
            updateCache();
        }
        
        // log C(k+r-1, k) = log Γ(k+r) - log Γ(k+1) - log Γ(r)
        //                 = log Γ(k+r) - log k! - log Γ(r)
        const double logGammaKPlusR = std::lgamma(k + r_);
        const double logFactorialK = std::lgamma(k + 1); // log k! = log Γ(k+1)
        
        return logGammaKPlusR - logFactorialK - logGammaR_;
    }

public:
    /**
     * Constructs a Negative Binomial distribution with given parameters.
     * 
     * @param r Number of successes (must be positive)
     * @param p Success probability (must be in (0,1])
     * @throws std::invalid_argument if parameters are invalid
     */
    NegativeBinomialDistribution(double r = 5.0, double p = 0.5)
        : r_{r}, p_{p}, cacheValid_{false} {
        validateParameters(r, p);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    NegativeBinomialDistribution(const NegativeBinomialDistribution& other)
        : r_{other.r_}, p_{other.p_}, 
          logP_{other.logP_}, log1MinusP_{other.log1MinusP_}, logGammaR_{other.logGammaR_},
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    NegativeBinomialDistribution& operator=(const NegativeBinomialDistribution& other) {
        if (this != &other) {
            r_ = other.r_;
            p_ = other.p_;
            logP_ = other.logP_;
            log1MinusP_ = other.log1MinusP_;
            logGammaR_ = other.logGammaR_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    NegativeBinomialDistribution(NegativeBinomialDistribution&& other) noexcept
        : r_{other.r_}, p_{other.p_},
          logP_{other.logP_}, log1MinusP_{other.log1MinusP_}, logGammaR_{other.logGammaR_},
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    NegativeBinomialDistribution& operator=(NegativeBinomialDistribution&& other) noexcept {
        if (this != &other) {
            r_ = other.r_;
            p_ = other.p_;
            logP_ = other.logP_;
            log1MinusP_ = other.log1MinusP_;
            logGammaR_ = other.logGammaR_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Destructor - explicitly defaulted to satisfy Rule of Five
     */
    ~NegativeBinomialDistribution() override = default;
    
    /**
     * Computes the probability mass function for the Negative Binomial distribution.
     * 
     * @param value The value at which to evaluate the PMF (will be rounded to nearest integer)
     * @return Probability mass
     */
    double getProbability(double value) override;
    
    /**
     * Fits the distribution parameters to the given data using method of moments.
     * 
     * For Negative Binomial distribution:
     * - p̂ = sample_mean / (sample_mean + sample_variance - sample_mean)
     * - r̂ = sample_mean² / (sample_variance - sample_mean)
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;
    
    /**
     * Resets the distribution to default parameters (r = 5.0, p = 0.5).
     */
    void reset() noexcept override;
    
    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;
    
    /**
     * Gets the number of successes parameter r.
     * 
     * @return Current number of successes
     */
    double getR() const noexcept { return r_; }
    
    /**
     * Sets the number of successes parameter r.
     * 
     * @param r New number of successes (must be positive)
     * @throws std::invalid_argument if r <= 0
     */
    void setR(double r) {
        validateParameters(r, p_);
        r_ = r;
        cacheValid_ = false;
    }
    
    /**
     * Gets the success probability parameter p.
     * 
     * @return Current success probability
     */
    double getP() const noexcept { return p_; }
    
    /**
     * Sets the success probability parameter p.
     * 
     * @param p New success probability (must be in (0,1])
     * @throws std::invalid_argument if p not in (0,1]
     */
    void setP(double p) {
        validateParameters(r_, p);
        p_ = p;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the distribution.
     * For Negative Binomial distribution, mean = r * (1-p) / p
     * 
     * @return Mean value
     */
    double getMean() const noexcept {
        return r_ * (1.0 - p_) / p_;
    }
    
    /**
     * Gets the variance of the distribution.
     * For Negative Binomial distribution, variance = r * (1-p) / p²
     * 
     * @return Variance value
     */
    double getVariance() const noexcept {
        return r_ * (1.0 - p_) / (p_ * p_);
    }
    
    /**
     * Gets the standard deviation of the distribution.
     * 
     * @return Standard deviation value
     */
    double getStandardDeviation() const noexcept {
        return std::sqrt(getVariance());
    }
    
    /**
     * Sets both parameters simultaneously.
     * 
     * @param r New number of successes
     * @param p New success probability
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParameters(double r, double p) {
        validateParameters(r, p);
        r_ = r;
        p_ = p;
        cacheValid_ = false;
    }
    
    /**
     * Evaluates the logarithm of the probability mass function
     * More numerically stable for small probabilities
     * 
     * @param value The value at which to evaluate the log PMF
     * @return Log probability mass
     */
    [[nodiscard]] double getLogProbability(double value) const noexcept override;
    
    /**
     * Evaluates the CDF at k using cumulative sum approach
     * Formula: CDF(k) = ∑(i=0 to k) P(X = i)
     * 
     * @param value The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ value)
     */
    [[nodiscard]] double CDF(double value) noexcept;
    
    /**
     * Gets the mode of the distribution.
     * For Negative Binomial distribution, mode = floor((r-1)*(1-p)/p) if r > 1, else 0
     * 
     * @return Mode value
     */
    int getMode() const noexcept {
        if (r_ <= 1.0) {
            return 0;
        }
        return static_cast<int>(std::floor((r_ - 1.0) * (1.0 - p_) / p_));
    }
    
    /**
     * Gets the skewness of the distribution.
     * For Negative Binomial distribution, skewness = (2-p)/sqrt(r*(1-p))
     * 
     * @return Skewness value
     */
    double getSkewness() const noexcept {
        return (2.0 - p_) / std::sqrt(r_ * (1.0 - p_));
    }
    
    /**
     * Gets the kurtosis of the distribution.
     * For Negative Binomial distribution, kurtosis = 3 + (6/r) + (p²/(r*(1-p)))
     * 
     * @return Kurtosis value
     */
    double getKurtosis() const noexcept {
        return 3.0 + (6.0 / r_) + (p_ * p_) / (r_ * (1.0 - p_));
    }
    
    /**
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const NegativeBinomialDistribution& other) const;
    
    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const NegativeBinomialDistribution& other) const { return !(*this == other); }
    
private:
    friend std::istream& operator>>(std::istream& is,
            libhmm::NegativeBinomialDistribution& distribution);
};

std::ostream& operator<<(std::ostream&, const libhmm::NegativeBinomialDistribution&);

} // namespace libhmm

#endif
