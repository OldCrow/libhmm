#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

namespace libhmm {

/**
 * Modern C++20 Rayleigh distribution for modeling magnitudes and speeds.
 *
 * The Rayleigh distribution is a continuous probability distribution that arises
 * when modeling the magnitude of a 2D random vector whose components are independent,
 * identically distributed, zero-mean Gaussian random variables.
 *
 * This is a special case of the Weibull distribution with shape parameter k = 2,
 * but implemented as a standalone class for maximum efficiency.
 *
 * PDF: f(x) = (x/σ²) * exp(-x²/(2σ²)) for x ≥ 0, 0 otherwise
 * CDF: F(x) = 1 - exp(-x²/(2σ²)) for x ≥ 0, 0 otherwise
 * where σ is the scale parameter (σ > 0)
 *
 * Properties:
 * - Mean: σ * √(π/2) ≈ 1.253 * σ
 * - Variance: σ² * (4-π)/2 ≈ 0.429 * σ²
 * - Mode: σ
 * - Support: x ∈ [0, ∞)
 *
 * Applications:
 * - Wind speed modeling
 * - Wave height analysis
 * - Signal processing (magnitude of complex Gaussian noise)
 * - Materials science (fiber strength)
 * - Communications (fading channel modeling)
 */
class RayleighDistribution : public DistributionBase {
private:
    /**
     * Scale parameter σ (sigma) - must be positive
     * Controls the spread and scale of the distribution
     */
    double sigma_{1.0};

    /**
     * Cached value of ln(σ) for efficiency in log probability calculations
     */
    mutable double logSigma_{0.0};

    /**
     * Cached value of 1/σ for efficiency (multiply instead of divide)
     */
    mutable double invSigma_{1.0};

    /**
     * Cached value of 1/σ² for efficiency in PDF and CDF calculations
     */
    mutable double invSigmaSquared_{1.0};

    /**
     * Cached value of -1/(2σ²) for CDF and log-PDF calculations
     * This eliminates the need for division and negation in hot paths
     */
    mutable double negHalfInvSigmaSquared_{-0.5};

    /**
     * Cached value of σ² for variance and other calculations
     */
    mutable double sigmaSquared_{1.0};

    /**
     * Cached value of σ * √(π/2) for mean calculation
     * Mean = σ * √(π/2) ≈ 1.2533141373 * σ
     */
    mutable double mean_{constants::math::SQRT_PI_OVER_TWO};

    /**
     * Cached value of σ² * (4-π)/2 for variance calculation
     * Variance = σ² * (4-π)/2 ≈ 0.4292036732 * σ²
     */
    mutable double variance_{constants::math::FOUR_MINUS_PI_OVER_TWO};

    void updateCache() const noexcept {
        logSigma_ = std::log(sigma_);
        invSigma_ = constants::math::ONE / sigma_;
        sigmaSquared_ = sigma_ * sigma_;
        invSigmaSquared_ = invSigma_ * invSigma_;
        negHalfInvSigmaSquared_ = -constants::math::HALF * invSigmaSquared_;
        mean_ = sigma_ * constants::math::SQRT_PI_OVER_TWO;
        variance_ = sigmaSquared_ * constants::math::FOUR_MINUS_PI_OVER_TWO;
        markCacheValid();
    }

    /**
     * Validates parameters for the Rayleigh distribution
     * @param sigma Scale parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double sigma) {
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) {
            throw std::invalid_argument("Sigma (scale parameter) must be a positive finite number");
        }
    }

    friend std::istream &operator>>(std::istream &is, libhmm::RayleighDistribution &distribution);

public:
    /**
     * Constructs a Rayleigh distribution with given scale parameter.
     *
     * @param sigma Scale parameter σ (must be positive)
     * @throws std::invalid_argument if sigma is invalid
     */
    explicit RayleighDistribution(double sigma = 1.0) : sigma_{sigma} {
        validateParameters(sigma);
        updateCache();
    }

    RayleighDistribution(const RayleighDistribution &other) = default;
    RayleighDistribution &operator=(const RayleighDistribution &other) = default;
    RayleighDistribution(RayleighDistribution &&other) noexcept = default;
    RayleighDistribution &operator=(RayleighDistribution &&other) noexcept = default;
    ~RayleighDistribution() override = default;

    [[nodiscard]] double getProbability(double value) const override;
    [[nodiscard]] double sample(std::mt19937_64 &rng) const override;
    [[nodiscard]] double getLogProbability(double value) const noexcept override;

    /// Concrete non-virtual batch log-PDF. Eliminates per-element virtual dispatch.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;
    [[nodiscard]] double getCumulativeProbability(double value) const noexcept;

    /** MLE: σ̂ = √(Σx² / (2n)). */
    void fit(std::span<const double> data) override;
    /** Weighted MLE: σ̂ = √(Σ(w_i * x_i²) / (2 * Σw_i)). */
    void fit(std::span<const double> data, std::span<const double> weights) override;

    /** Returns false — Rayleigh is a continuous distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }
    [[nodiscard]] std::size_t getNumParameters() const noexcept override { return 1; }

    /**
     * Resets the distribution to default parameters (σ = 1.0).
     * This corresponds to the standard Rayleigh distribution.
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
     * Gets the scale parameter σ.
     *
     * @return Current scale parameter value
     */
    double getSigma() const noexcept { return sigma_; }

    /**
     * Sets the scale parameter σ.
     *
     * @param sigma New scale parameter (must be positive)
     * @throws std::invalid_argument if sigma is invalid
     */
    void setSigma(double sigma) {
        validateParameters(sigma);
        sigma_ = sigma;
        invalidateCache();
    }

    /**
     * Gets the mean of the distribution.
     * Mean = σ * √(π/2)
     *
     * @return Mean value
     */
    double getMean() const noexcept {
        if (!isCacheValid())
            updateCache();
        return mean_;
    }
    double getVariance() const noexcept {
        if (!isCacheValid())
            updateCache();
        return variance_;
    }

    /**
     * Gets the standard deviation of the distribution.
     *
     * @return Standard deviation (square root of variance)
     */
    double getStandardDeviation() const noexcept { return std::sqrt(getVariance()); }

    /**
     * Gets the mode of the distribution.
     * Mode = σ
     *
     * @return Mode value
     */
    double getMode() const noexcept { return sigma_; }

    /**
     * Gets the median of the distribution.
     * Median = σ * √(2 * ln(2)) ≈ 1.177 * σ
     *
     * @return Median value
     */
    double getMedian() const noexcept { return sigma_ * constants::math::SQRT_TWO_LN_TWO; }

    /**
     * Equality operator
     */
    bool operator==(const RayleighDistribution &other) const noexcept {
        return std::abs(sigma_ - other.sigma_) <
               constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE;
    }

    /**
     * Inequality operator
     */
    bool operator!=(const RayleighDistribution &other) const noexcept { return !(*this == other); }
};

} // namespace libhmm
