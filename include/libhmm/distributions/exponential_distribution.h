#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

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
class ExponentialDistribution : public DistributionBase
{   
private:
    /**
     * Rate parameter λ - must be positive
     * Higher values indicate faster decay/shorter expected waiting times
     */
    double lambda_{1.0};
    
    /**
     * Cached value of ln(λ) for efficiency in log probability calculations
     */
    mutable double logLambda_{0.0};
    
    /**
     * Cached value of 1/λ (mean and scale parameter) for efficiency
     * This eliminates division in getMean(), getVariance(), getStandardDeviation()
     */
    mutable double invLambda_{1.0};
    
    /**
     * Cached value of -λ for efficiency in PDF and log-PDF calculations
     * This eliminates negation operations in hot paths
     */
    mutable double negLambda_{-1.0};
    
    /**
     * Cached value of 1/λ² for variance calculation efficiency
     * This eliminates the need to square invLambda_ repeatedly
     */
    mutable double invLambdaSquared_{1.0};
    
    void updateCache() const noexcept {
        logLambda_      = std::log(lambda_);
        invLambda_      = 1.0 / lambda_;
        negLambda_      = -lambda_;
        invLambdaSquared_ = invLambda_ * invLambda_;
        markCacheValid();
    }
    
    /**
     * Validates parameters for the Exponential distribution
     * @param lambda Rate parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double lambda) {
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
            throw std::invalid_argument("Lambda (rate parameter) must be a positive finite number");
        }
    }

    friend std::istream& operator>>(std::istream& is,
            libhmm::ExponentialDistribution& distribution);

public:
    /**
     * Constructs an Exponential distribution with given rate parameter.
     * 
     * @param lambda Rate parameter λ (must be positive)
     * @throws std::invalid_argument if lambda is invalid
     */
    explicit ExponentialDistribution(double lambda = 1.0)
        : lambda_{lambda} {
        validateParameters(lambda);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    ExponentialDistribution(const ExponentialDistribution& other)
        : DistributionBase{other}, lambda_{other.lambda_},
          logLambda_{other.logLambda_}, invLambda_{other.invLambda_},
          negLambda_{other.negLambda_}, invLambdaSquared_{other.invLambdaSquared_} {}

    ExponentialDistribution& operator=(const ExponentialDistribution& other) {
        if (this != &other) {
            DistributionBase::operator=(other);
            lambda_           = other.lambda_;
            logLambda_        = other.logLambda_;
            invLambda_        = other.invLambda_;
            negLambda_        = other.negLambda_;
            invLambdaSquared_ = other.invLambdaSquared_;
        }
        return *this;
    }

    ExponentialDistribution(ExponentialDistribution&& other) noexcept
        : DistributionBase{std::move(other)}, lambda_{other.lambda_},
          logLambda_{other.logLambda_}, invLambda_{other.invLambda_},
          negLambda_{other.negLambda_}, invLambdaSquared_{other.invLambdaSquared_} {}

    ExponentialDistribution& operator=(ExponentialDistribution&& other) noexcept {
        if (this != &other) {
            DistributionBase::operator=(std::move(other));
            lambda_           = other.lambda_;
            logLambda_        = other.logLambda_;
            invLambda_        = other.invLambda_;
            negLambda_        = other.negLambda_;
            invLambdaSquared_ = other.invLambdaSquared_;
        }
        return *this;
    }

    ~ExponentialDistribution() = default;

    /**
     * Computes the probability density function for the Exponential distribution.
     * 
     * @param value The value at which to evaluate the PDF
     * @return Probability density (or approximated probability for discrete sampling)
     */
    [[nodiscard]] double getProbability(double value) const override;
    [[nodiscard]] double getLogProbability(double value) const noexcept override;

    /** Fit λ = 1 / sample_mean (unweighted MLE). */
    void fit(std::span<const double> data) override;

    /**
     * Weighted MLE: λ = Σ(weights) / Σ(w_i · x_i) = 1 / weighted_mean.
     * Falls back to reset() if the weighted mean is zero or near-zero.
     */
    void fit(std::span<const double> data, std::span<const double> weights) override;

    /** Returns false — Exponential is a continuous distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

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
        invalidateCache();
    }
    
    /**
     * Gets the mean of the distribution.
     * For Exponential distribution, mean = 1/λ
     * Uses cached value to eliminate division.
     * 
     * @return Mean value
     */
    double getMean()              const noexcept { if (!isCacheValid()) updateCache(); return invLambda_; }
    double getVariance()          const noexcept { if (!isCacheValid()) updateCache(); return invLambdaSquared_; }
    double getStandardDeviation() const noexcept { if (!isCacheValid()) updateCache(); return invLambda_; }
    double getScale()             const noexcept { if (!isCacheValid()) updateCache(); return invLambda_; }
    
    /**
     * Evaluates the CDF at x using the standard exponential CDF formula
     * For exponential distribution: F(x) = 1 - exp(-λx) for x ≥ 0, 0 otherwise
     * 
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    double getCumulativeProbability(double x) const noexcept;
    
    /**
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const ExponentialDistribution& other) const;
    
    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const ExponentialDistribution& other) const { return !(*this == other); }

};

std::ostream& operator<<( std::ostream&, 
        const libhmm::ExponentialDistribution& );
//std::istream& operator>>( std::istream&,
//        const libhmm::ExponentialDistribution& );
} // namespace
