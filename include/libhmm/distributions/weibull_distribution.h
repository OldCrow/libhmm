#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

namespace libhmm {

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
class WeibullDistribution : public DistributionBase {
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
     * Cached value of (k-1) for efficiency in probability calculations
     */
    mutable double kMinus1_{0.0};

    /**
     * Cached value of 1/λ for efficiency (multiply instead of divide)
     */
    mutable double invLambda_{1.0};

    /**
     * Cached value of k/λ for PDF normalization
     */
    mutable double kOverLambda_{1.0};

    void updateCache() const noexcept {
        logK_ = std::log(k_);
        logLambda_ = std::log(lambda_);
        kMinus1_ = k_ - 1.0;
        invLambda_ = 1.0 / lambda_;
        kOverLambda_ = k_ * invLambda_;
        markCacheValid();
    }

    /**
     * Validates parameters for the Weibull distribution
     * @param k Shape parameter (must be positive and finite)
     * @param lambda Scale parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double k, double lambda) {
        if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
            throw std::invalid_argument("Shape parameter k must be a positive finite number");
        }
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
            throw std::invalid_argument("Scale parameter lambda must be a positive finite number");
        }
    }

    friend std::istream &operator>>(std::istream &is, libhmm::WeibullDistribution &distribution);

public:
    /**
     * Constructs a Weibull distribution with given parameters.
     *
     * @param k Shape parameter (must be positive)
     * @param lambda Scale parameter (must be positive)
     * @throws std::invalid_argument if parameters are not positive finite numbers
     */
    explicit WeibullDistribution(double k = 1.0, double lambda = 1.0) : k_{k}, lambda_{lambda} {
        validateParameters(k, lambda);
        updateCache();
    }

    /**
     * Copy constructor
     */
    WeibullDistribution(const WeibullDistribution &other) = default;
    WeibullDistribution &operator=(const WeibullDistribution &other) = default;
    WeibullDistribution(WeibullDistribution &&other) noexcept = default;
    WeibullDistribution &operator=(WeibullDistribution &&other) noexcept = default;
    ~WeibullDistribution() override = default;

    /**
     * Computes the probability density function for the Weibull distribution.
     *
     * @param value The value at which to evaluate the PDF (should be ≥ 0)
     * @return Probability density, or 0.0 if value is negative
     */
    [[nodiscard]] double getProbability(double x) const override;
    [[nodiscard]] double sample(std::mt19937_64 &rng) const override;
    [[nodiscard]] double getLogProbability(double value) const noexcept override;

    /// Concrete non-virtual batch log-PDF. Eliminates per-element virtual dispatch.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;

    /// MLE fit: Newton–Raphson on the Weibull profile score for k,
    /// then λ = (Σx_i^k/n)^(1/k); seeded from MoM. Requires all x > 0.
    void fit(std::span<const double> data) override;
    /// Weighted MLE: same Newton–Raphson system with weighted sums;
    /// near-zero-weight observations are skipped; near-zero total weight retains current params.
    void fit(std::span<const double> data, std::span<const double> weights) override;

public:
    /** Returns false — Weibull is a continuous distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }
    [[nodiscard]] std::size_t getNumParameters() const noexcept override { return 2; }

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
    [[nodiscard]] std::string to_json() const override;
    /// @internal JSON factory — called by the distribution registry in src/io/hmm_json.cpp.
    static std::unique_ptr<EmissionDistribution> from_json(json::Reader &r);

    /**
     * Computes the cumulative distribution function
     *
     * @param x The value at which to evaluate the CDF (should be ≥ 0)
     * @return Cumulative probability P(X ≤ x), or 0.0 if x is negative
     */
    [[nodiscard]] double CDF(double x) const noexcept;

    /**
     * Equality comparison operator with tolerance for floating-point comparison.
     *
     * @param other Distribution to compare with
     * @return true if distributions have the same parameters within tolerance
     */
    bool operator==(const WeibullDistribution &other) const noexcept;

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
        invalidateCache();
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
        invalidateCache();
    }

    /**
     * Gets the mean of the distribution.
     * For Weibull(k, λ), mean = λ * Γ(1 + 1/k)
     *
     * @return Mean value
     */
    double getMean() const noexcept { return lambda_ * std::exp(std::lgamma(1.0 + 1.0 / k_)); }

    /**
     * Gets the variance of the distribution.
     * For Weibull(k, λ), variance = λ² * [Γ(1 + 2/k) - (Γ(1 + 1/k))²]
     *
     * @return Variance value
     */
    double getVariance() const noexcept {
        double gamma1 = std::exp(std::lgamma(1.0 + 1.0 / k_));
        double gamma2 = std::exp(std::lgamma(1.0 + 2.0 / k_));
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
std::ostream &operator<<(std::ostream &os, const libhmm::WeibullDistribution &distribution);

/**
 * Stream input operator for Weibull distribution.
 */
std::istream &operator>>(std::istream &is, libhmm::WeibullDistribution &distribution);

} // namespace libhmm
