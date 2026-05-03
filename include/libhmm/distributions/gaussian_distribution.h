#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

namespace libhmm {

/**
 * Gaussian (Normal) distribution for modeling continuous emission data.
 *
 * PDF: f(x) = (1/(σ√(2π))) * exp(-½((x-μ)/σ)²)
 *
 * Ported to DistributionBase in Phase 2:
 *   - getProbability() and getLogProbability() are now const
 *   - Cache uses std::atomic<bool> (thread-safe for calculator thread pool)
 *   - fit() accepts std::span<const double> (no copies, no Observation alias)
 *   - Weighted fit() added for Baum-Welch M-step
 */
class GaussianDistribution : public DistributionBase {
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
     * Cached log(σ) for efficiency in log probability calculations
     */
    mutable double logStandardDeviation_{0.0};

    /**
     * Cached σ√2 for efficiency in CDF calculations
     */
    mutable double sigmaSqrt2_{0.0};

    /**
     * Cached 1/σ for efficiency in log probability calculations
     */
    mutable double invStandardDeviation_{0.0};

    /**
     * Updates all cached values. Called on first use after parameters change.
     * Ends by calling markCacheValid() from DistributionBase.
     */
    void updateCache() const noexcept {
        const double sigma2 = standardDeviation_ * standardDeviation_;
        invStandardDeviation_ = 1.0 / standardDeviation_;
        normalizationConstant_ = invStandardDeviation_ / constants::math::SQRT_2PI;
        negHalfSigmaSquaredInv_ = -0.5 / sigma2;
        logStandardDeviation_ = std::log(standardDeviation_);
        sigmaSqrt2_ = standardDeviation_ * constants::math::SQRT_2;
        markCacheValid();
    }

    /**
     * Validates parameters for the Gaussian distribution
     * @param mean Mean parameter (any finite value)
     * @param stdDev Standard deviation parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double mean, double stdDev) {
        if (std::isnan(mean) || std::isinf(mean)) {
            throw std::invalid_argument("Mean must be a finite number");
        }
        if (std::isnan(stdDev) || std::isinf(stdDev) || stdDev <= 0.0) {
            throw std::invalid_argument("Standard deviation must be a positive finite number");
        }
    }

    friend std::istream &operator>>(std::istream &is, libhmm::GaussianDistribution &distribution);

public:
    explicit GaussianDistribution(double mean = 0.0, double standardDeviation = 1.0)
        : mean_{mean}, standardDeviation_{standardDeviation} {
        validateParameters(mean, standardDeviation);
        updateCache();
    }

    // Rule of Five: atomic<bool> in base requires explicit copy/move.
    // Parameter data is trivially copyable; base handles the atomic.
    GaussianDistribution(const GaussianDistribution &other)
        : DistributionBase{other}, mean_{other.mean_}, standardDeviation_{other.standardDeviation_},
          normalizationConstant_{other.normalizationConstant_},
          negHalfSigmaSquaredInv_{other.negHalfSigmaSquaredInv_},
          logStandardDeviation_{other.logStandardDeviation_}, sigmaSqrt2_{other.sigmaSqrt2_},
          invStandardDeviation_{other.invStandardDeviation_} {}

    GaussianDistribution &operator=(const GaussianDistribution &other) {
        if (this != &other) {
            DistributionBase::operator=(other);
            mean_ = other.mean_;
            standardDeviation_ = other.standardDeviation_;
            normalizationConstant_ = other.normalizationConstant_;
            negHalfSigmaSquaredInv_ = other.negHalfSigmaSquaredInv_;
            logStandardDeviation_ = other.logStandardDeviation_;
            sigmaSqrt2_ = other.sigmaSqrt2_;
            invStandardDeviation_ = other.invStandardDeviation_;
        }
        return *this;
    }

    GaussianDistribution(GaussianDistribution &&other) noexcept
        : DistributionBase{std::move(other)}, mean_{other.mean_},
          standardDeviation_{other.standardDeviation_},
          normalizationConstant_{other.normalizationConstant_},
          negHalfSigmaSquaredInv_{other.negHalfSigmaSquaredInv_},
          logStandardDeviation_{other.logStandardDeviation_}, sigmaSqrt2_{other.sigmaSqrt2_},
          invStandardDeviation_{other.invStandardDeviation_} {}

    GaussianDistribution &operator=(GaussianDistribution &&other) noexcept {
        if (this != &other) {
            DistributionBase::operator=(std::move(other));
            mean_ = other.mean_;
            standardDeviation_ = other.standardDeviation_;
            normalizationConstant_ = other.normalizationConstant_;
            negHalfSigmaSquaredInv_ = other.negHalfSigmaSquaredInv_;
            logStandardDeviation_ = other.logStandardDeviation_;
            sigmaSqrt2_ = other.sigmaSqrt2_;
            invStandardDeviation_ = other.invStandardDeviation_;
        }
        return *this;
    }

    ~GaussianDistribution() override = default;

    /**
     * Computes the probability density function for the Gaussian distribution.
     * Formula: PDF(x) = (1/(σ√(2π))) * exp(-½((x-μ)/σ)²)
     *
     * @param x The value at which to evaluate the PDF
     * @return Probability density
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * Evaluates the logarithm of the probability density function
     * Formula: log PDF(x) = -½log(2π) - log(σ) - ½((x-μ)/σ)²
     * More numerically stable for small probabilities
     *
     * @param x The value at which to evaluate the log PDF
     * @return Log probability density
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /// Vectorised batch log-PDF.
    /// Uses AVX-512 (8-wide), AVX2 (4-wide), SSE2 (2-wide), or NEON (2-wide) SIMD
    /// when available; falls back to a scalar tail. NaN inputs yield -Inf output.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;

    /**
     * Evaluates the CDF at x using the error function
     * Formula: CDF(x) = (1/2) * (1 + erf((x-μ)/(σ√2)))
     *
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    [[nodiscard]] double getCumulativeProbability(double x) const noexcept;

    /** Fit parameters to unweighted data using Welford's online algorithm. */
    void fit(std::span<const double> data) override;

    /**
     * Fit parameters to weighted data (Baum-Welch M-step).
     * Weights are unnormalized γ values; normalization by sum(weights) is done internally.
     * Uses weighted Welford's algorithm for numerical stability.
     * Falls back to reset() if sum(weights) is near zero.
     */
    void fit(std::span<const double> data, std::span<const double> weights) override;

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
        invalidateCache();
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
        invalidateCache();
    }

    /**
     * Gets the variance of the distribution.
     * For Gaussian distribution, variance = σ²
     *
     * @return Variance value
     */
    double getVariance() const noexcept { return standardDeviation_ * standardDeviation_; }

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
        invalidateCache();
    }

    /** Returns false — Gaussian is a continuous distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /**
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const GaussianDistribution &other) const;

    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const GaussianDistribution &other) const { return !(*this == other); }
};

std::ostream &operator<<(std::ostream &, const libhmm::GaussianDistribution &);
//std::istream& operator>>( std::istream&,
//        const libhmm::GaussianDistribution& );
} // namespace libhmm
