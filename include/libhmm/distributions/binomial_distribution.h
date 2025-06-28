#ifndef BINOMIAL_DISTRIBUTION_H_
#define BINOMIAL_DISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm {

/**
 * Modern C++17 Binomial distribution for modeling discrete count data.
 * 
 * The Binomial distribution models the number of successes in n independent
 * Bernoulli trials, each with success probability p.
 * 
 * PMF: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
 * where C(n,k) is the binomial coefficient "n choose k"
 * 
 * Properties:
 * - Mean: n * p
 * - Variance: n * p * (1-p)
 * - Support: k ∈ {0, 1, 2, ..., n}
 */
class BinomialDistribution : public ProbabilityDistribution
{
private:
    /**
     * Number of trials n - must be a positive integer
     */
    int n_{10};
    
    /**
     * Success probability p - must be in [0,1]
     */
    double p_{0.5};
    
    /**
     * Cached log factorial values for efficiency
     */
    mutable std::vector<double> logFactorialCache_;
    
    /**
     * Cached values for efficiency in probability calculations
     */
    mutable double logP_{0.0};
    mutable double log1MinusP_{0.0};
    
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
        
        // Precompute log factorials up to n
        logFactorialCache_.resize(n_ + 1);
        logFactorialCache_[0] = 0.0; // log(0!) = log(1) = 0
        for (int i = 1; i <= n_; ++i) {
            logFactorialCache_[i] = logFactorialCache_[i-1] + std::log(i);
        }
        
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Binomial distribution
     * @param n Number of trials (must be positive integer)
     * @param p Success probability (must be in [0,1])
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(int n, double p) const {
        if (n <= 0) {
            throw std::invalid_argument("Number of trials must be positive");
        }
        if (std::isnan(p) || std::isinf(p) || p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Success probability must be in [0,1]");
        }
    }
    
    /**
     * Computes log of binomial coefficient log(C(n,k)) = log(n!/(k!(n-k)!))
     */
    double logBinomialCoefficient(int n, int k) const {
        if (k < 0 || k > n) return -std::numeric_limits<double>::infinity();
        
        // Ensure cache is valid
        if (!cacheValid_) {
            updateCache();
        }
        
        return logFactorialCache_[n] - logFactorialCache_[k] - logFactorialCache_[n-k];
    }

    friend std::istream& operator>>(std::istream& is,
            libhmm::BinomialDistribution& distribution);

public:
    /**
     * Constructs a Binomial distribution with given parameters.
     * 
     * @param n Number of trials (must be positive)
     * @param p Success probability (must be in [0,1])
     * @throws std::invalid_argument if parameters are invalid
     */
    BinomialDistribution(int n = 10, double p = 0.5)
        : n_{n}, p_{p}, cacheValid_{false} {
        validateParameters(n, p);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    BinomialDistribution(const BinomialDistribution& other)
        : n_{other.n_}, p_{other.p_}, 
          logFactorialCache_{other.logFactorialCache_},
          logP_{other.logP_}, log1MinusP_{other.log1MinusP_},
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    BinomialDistribution& operator=(const BinomialDistribution& other) {
        if (this != &other) {
            n_ = other.n_;
            p_ = other.p_;
            logFactorialCache_ = other.logFactorialCache_;
            logP_ = other.logP_;
            log1MinusP_ = other.log1MinusP_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    BinomialDistribution(BinomialDistribution&& other) noexcept
        : n_{other.n_}, p_{other.p_},
          logFactorialCache_{std::move(other.logFactorialCache_)},
          logP_{other.logP_}, log1MinusP_{other.log1MinusP_},
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    BinomialDistribution& operator=(BinomialDistribution&& other) noexcept {
        if (this != &other) {
            n_ = other.n_;
            p_ = other.p_;
            logFactorialCache_ = std::move(other.logFactorialCache_);
            logP_ = other.logP_;
            log1MinusP_ = other.log1MinusP_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Destructor - explicitly defaulted to satisfy Rule of Five
     */
    ~BinomialDistribution() override = default;
    
    /**
     * Computes the probability mass function for the Binomial distribution.
     * 
     * @param value The value at which to evaluate the PMF (will be rounded to nearest integer)
     * @return Probability mass
     */
    double getProbability(double value) override;
    
    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Binomial distribution: p̂ = sample_mean / n (n must be known/fixed)
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;
    
    /**
     * Resets the distribution to default parameters (n = 10, p = 0.5).
     */
    void reset() noexcept override;
    
    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;
    
    /**
     * Gets the number of trials parameter n.
     * 
     * @return Current number of trials
     */
    int getN() const noexcept { return n_; }
    
    /**
     * Sets the number of trials parameter n.
     * 
     * @param n New number of trials (must be positive)
     * @throws std::invalid_argument if n <= 0
     */
    void setN(int n) {
        validateParameters(n, p_);
        n_ = n;
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
     * @param p New success probability (must be in [0,1])
     * @throws std::invalid_argument if p not in [0,1]
     */
    void setP(double p) {
        validateParameters(n_, p);
        p_ = p;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the distribution.
     * For Binomial distribution, mean = n * p
     * 
     * @return Mean value
     */
    double getMean() const noexcept {
        return n_ * p_;
    }
    
    /**
     * Gets the variance of the distribution.
     * For Binomial distribution, variance = n * p * (1-p)
     * 
     * @return Variance value
     */
    double getVariance() const noexcept {
        return n_ * p_ * (1.0 - p_);
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
     * @param n New number of trials
     * @param p New success probability
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParameters(int n, double p) {
        validateParameters(n, p);
        n_ = n;
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
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const BinomialDistribution& other) const;
    
    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const BinomialDistribution& other) const { return !(*this == other); }
};

std::ostream& operator<<(std::ostream&, const libhmm::BinomialDistribution&);

} // namespace libhmm

#endif
