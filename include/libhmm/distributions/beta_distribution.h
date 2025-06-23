#ifndef BETADISTRIBUTION_H_
#define BETADISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm{

/**
 * Beta distribution for modeling probabilities and proportions.
 * 
 * The Beta distribution is a continuous probability distribution defined 
 * on the interval [0,1] and parameterized by two positive shape parameters
 * α (alpha) and β (beta).
 * 
 * PDF: f(x; α, β) = (x^(α-1) * (1-x)^(β-1)) / B(α, β)
 * where B(α, β) is the Beta function: B(α, β) = Γ(α)Γ(β)/Γ(α+β)
 * 
 * Special cases:
 * - α = β = 1: Uniform distribution on [0,1]
 * - α = β: Symmetric around 0.5
 * - α < β: Skewed toward 0
 * - α > β: Skewed toward 1
 */
class BetaDistribution : public ProbabilityDistribution
{   
private:
    /**
     * Shape parameter α (alpha) - must be positive
     */
    double alpha_{1.0};

    /**
     * Shape parameter β (beta) - must be positive  
     */
    double beta_{1.0};

    /**
     * Cached value of log(B(α, β)) for efficiency in probability calculations
     */
    mutable double logBeta_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates cached values when parameters change
     */
    void updateCache() const noexcept {
        // B(α, β) = Γ(α)Γ(β)/Γ(α+β)
        // log(B(α, β)) = log(Γ(α)) + log(Γ(β)) - log(Γ(α+β))
        logBeta_ = loggamma(alpha_) + loggamma(beta_) - loggamma(alpha_ + beta_);
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Beta distribution
     * @param alpha Alpha parameter (must be positive and finite)
     * @param beta Beta parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double alpha, double beta) const {
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0) {
            throw std::invalid_argument("Alpha parameter must be a positive finite number");
        }
        if (std::isnan(beta) || std::isinf(beta) || beta <= 0.0) {
            throw std::invalid_argument("Beta parameter must be a positive finite number");
        }
    }

    /**
     * Evaluates the CDF at x using the incomplete Beta function
     * CDF(x) = I_x(α, β) where I_x is the regularized incomplete Beta function
     */
    double CDF(double x) noexcept;

    friend std::istream& operator>>(std::istream& is,
            libhmm::BetaDistribution& distribution);

public:
    /**
     * Constructs a Beta distribution with given shape parameters.
     * 
     * @param alpha Shape parameter α (must be positive)
     * @param beta Shape parameter β (must be positive)
     * @throws std::invalid_argument if parameters are not positive finite numbers
     */
    BetaDistribution(double alpha = 1.0, double beta = 1.0)
        : alpha_{alpha}, beta_{beta} {
        validateParameters(alpha, beta);
        updateCache();
    }

    /**
     * Copy constructor
     */
    BetaDistribution(const BetaDistribution& other) 
        : alpha_{other.alpha_}, beta_{other.beta_}, 
          logBeta_{other.logBeta_}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    BetaDistribution& operator=(const BetaDistribution& other) {
        if (this != &other) {
            alpha_ = other.alpha_;
            beta_ = other.beta_;
            logBeta_ = other.logBeta_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    BetaDistribution(BetaDistribution&& other) noexcept
        : alpha_{other.alpha_}, beta_{other.beta_}, 
          logBeta_{other.logBeta_}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    BetaDistribution& operator=(BetaDistribution&& other) noexcept {
        if (this != &other) {
            alpha_ = other.alpha_;
            beta_ = other.beta_;
            logBeta_ = other.logBeta_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Computes the probability density function for the Beta distribution.
     * 
     * @param value The value at which to evaluate the PDF (should be in [0,1])
     * @return Probability density, or 0.0 if value is outside [0,1]
     */
    double getProbability(double value) override;

    /**
     * Fits the distribution parameters to the given data using method of moments.
     * 
     * Given sample mean μ and variance σ², the method of moments estimators are:
     * α̂ = μ * (μ(1-μ)/σ² - 1)
     * β̂ = (1-μ) * (μ(1-μ)/σ² - 1)
     * 
     * @param values Vector of observed data (should be in [0,1])
     * @throws std::invalid_argument if values contain data outside [0,1]
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (α = 1.0, β = 1.0).
     * This corresponds to a uniform distribution on [0,1].
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the alpha (α) shape parameter.
     * 
     * @return Current alpha value
     */
    double getAlpha() const noexcept { return alpha_; }
    
    /**
     * Sets the alpha (α) shape parameter.
     * 
     * @param alpha New alpha parameter (must be positive)
     * @throws std::invalid_argument if alpha <= 0 or is not finite
     */
    void setAlpha(double alpha) {
        validateParameters(alpha, beta_);
        alpha_ = alpha;
        cacheValid_ = false;
    }

    /**
     * Gets the beta (β) shape parameter.
     * 
     * @return Current beta value
     */
    double getBeta() const noexcept { return beta_; }
    
    /**
     * Sets the beta (β) shape parameter.
     * 
     * @param beta New beta parameter (must be positive)
     * @throws std::invalid_argument if beta <= 0 or is not finite
     */
    void setBeta(double beta) {
        validateParameters(alpha_, beta);
        beta_ = beta;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the distribution.
     * For Beta(α, β), mean = α/(α+β)
     * 
     * @return Mean value
     */
    double getMean() const noexcept { return alpha_ / (alpha_ + beta_); }
    
    /**
     * Gets the variance of the distribution.
     * For Beta(α, β), variance = αβ/((α+β)²(α+β+1))
     * 
     * @return Variance value
     */
    double getVariance() const noexcept { 
        double sum = alpha_ + beta_;
        return (alpha_ * beta_) / (sum * sum * (sum + 1.0));
    }
    
    /**
     * Gets the standard deviation of the distribution.
     * 
     * @return Standard deviation
     */
    double getStandardDeviation() const noexcept { return std::sqrt(getVariance()); }
};

/**
 * Stream output operator for Beta distribution.
 */
std::ostream& operator<<(std::ostream& os, const libhmm::BetaDistribution& distribution);

} // namespace libhmm

#endif // BETADISTRIBUTION_H_
