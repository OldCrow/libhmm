#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

namespace libhmm {

/**
 * Von Mises distribution for circular / directional data.
 *
 * The von Mises distribution is the circular analogue of the Gaussian. It is
 * the canonical emission distribution for angular observations in Hidden Markov
 * Models of animal movement (turning angles), speech, robotics, and any domain
 * where observations are directions on the unit circle.
 *
 * PDF: f(x | μ, κ) = exp(κ cos(x − μ)) / (2π I₀(κ))
 * where μ ∈ (−π, π] is the mean direction, κ ≥ 0 is the concentration
 * parameter, and I₀ is the modified Bessel function of the first kind.
 *
 * Properties:
 * - Mean direction: μ
 * - Circular variance: 1 − I₁(κ)/I₀(κ)  (ranges from 1 at κ=0 to 0 as κ→∞)
 * - Support: x ∈ (−π, π]
 * - κ = 0: uniform distribution on the circle
 * - κ → ∞: point mass at μ (distribution collapses to a single direction)
 *
 * Bessel implementation:
 * Uses std::cyl_bessel_i (C++17) where LIBHMM_HAS_CXX17_BESSEL is defined,
 * otherwise falls back to Abramowitz & Stegun §9.8.1–9.8.4 polynomial
 * approximations. Both paths yield < 1.6×10⁻⁷ absolute error.
 * See include/libhmm/math/bessel.h.
 *
 * Applications:
 * - Animal movement ecology (turning angles, see moveHMM R package)
 * - Wind direction analysis
 * - Robotics heading estimation
 * - Speech / audio phase modelling
 */
class VonMisesDistribution : public DistributionBase<VonMisesDistribution> {
    friend class DistributionBase<VonMisesDistribution>;

private:
    /**
     * Mean direction μ — maintained in (−π, π].
     * The distribution is 2π-periodic; μ is wrapped to this range on
     * construction and after fit().
     */
    double mu_{0.0};

    /**
     * Concentration parameter κ ≥ 0.
     * κ = 0  → uniform; κ → ∞ → point mass at μ.
     */
    double kappa_{1.0};

    /**
     * Cached log-normaliser: log(2π) + log I₀(κ).
     * The log-PDF is: κ cos(x − μ) − logNormaliser_
     */
    mutable double logNormaliser_{0.0};

    /**
     * Cached circular variance: 1 − I₁(κ)/I₀(κ).
     * Ranges from 1 (κ=0, uniform) to 0 (κ→∞, point mass).
     */
    mutable double circularVariance_{0.0};

    void updateCache() const noexcept;

    /**
     * Wraps angle to (−π, π].
     */
    [[nodiscard]] static double wrap_angle(double x) noexcept;

    /**
     * Validates parameters.
     * @throws std::invalid_argument if parameters are out of range or non-finite.
     */
    static void validateParameters(double mu, double kappa);

    friend std::istream &operator>>(std::istream &is, libhmm::VonMisesDistribution &distribution);

public:
    /**
     * Constructs a von Mises distribution.
     *
     * @param mu    Mean direction in radians; wrapped to (−π, π].
     * @param kappa Concentration parameter κ ≥ 0.
     * @throws std::invalid_argument if parameters are invalid.
     */
    explicit VonMisesDistribution(double mu = 0.0, double kappa = 1.0);

    VonMisesDistribution(const VonMisesDistribution &) = default;
    VonMisesDistribution &operator=(const VonMisesDistribution &) = default;
    VonMisesDistribution(VonMisesDistribution &&) noexcept = default;
    VonMisesDistribution &operator=(VonMisesDistribution &&) noexcept = default;
    ~VonMisesDistribution() override = default;

    [[nodiscard]] double getProbability(double value) const override;
    [[nodiscard]] double sample(std::mt19937_64 &rng) const override;
    [[nodiscard]] double getLogProbability(double value) const noexcept override;

    /// Concrete non-virtual batch log-PDF. Eliminates per-element virtual dispatch.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;

    /**
     * CDF evaluated via numerical integration (trapezoidal rule, 512 steps).
     * The von Mises CDF has no closed form; this implementation is suitable
     * for general use but not for inner-loop computation.
     *
     * @param value Angle in radians; wrapped to (−π, π] before evaluation.
     * @return Cumulative probability P(X ≤ value).
     */
    [[nodiscard]] double getCumulativeProbability(double value) const noexcept;

    /**
     * MLE: estimates μ and κ from unweighted angle observations.
     * Uses the circular mean for μ and the Mardia-Jupp approximation for κ.
     *
     * @param data Span of angle observations in radians.
     */
    void fit(std::span<const double> data) override;

    /**
     * Weighted MLE for the Baum-Welch M-step.
     * μ is estimated via weighted circular mean (atan2).
     * κ is estimated via the Mardia-Jupp approximation applied to the
     * weighted mean resultant length R̄.
     *
     * @param data    Span of angle observations in radians.
     * @param weights Span of non-negative responsibility weights.
     */
    void fit(std::span<const double> data, std::span<const double> weights) override;

    /** Returns false — von Mises is a continuous distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }
    [[nodiscard]] std::size_t getNumParameters() const noexcept override { return 2; }

    /** Resets to default parameters: μ = 0, κ = 1 (moderate concentration). */
    void reset() noexcept override;

    [[nodiscard]] std::string toString() const override;
    [[nodiscard]] std::string to_json() const override;
    /// @internal JSON factory — called by the distribution registry in src/io/hmm_json.cpp.
    static std::unique_ptr<EmissionDistribution> from_json(json::Reader &r);

    // ---- Accessors -----------------------------------------------------------

    /** Returns the mean direction μ in (−π, π]. */
    [[nodiscard]] double getMu() const noexcept { return mu_; }
    /** Returns the concentration parameter κ. */
    [[nodiscard]] double getKappa() const noexcept { return kappa_; }

    /** Sets mean direction; wraps to (−π, π]. */
    void setMu(double mu);
    /** Sets concentration parameter; must be ≥ 0. */
    void setKappa(double kappa);

    /**
     * Returns the circular variance 1 − I₁(κ)/I₀(κ).
     * 0 = perfectly concentrated; 1 = uniform.
     */
    [[nodiscard]] double getCircularVariance() const noexcept {
        ensureCache();
        return circularVariance_;
    }

    /** Mean direction (same as getMu()). */
    [[nodiscard]] double getMean() const noexcept { return mu_; }

    /** Circular variance. */
    [[nodiscard]] double getVariance() const noexcept { return getCircularVariance(); }

    bool operator==(const VonMisesDistribution &o) const noexcept;
    bool operator!=(const VonMisesDistribution &o) const noexcept { return !(*this == o); }
};

} // namespace libhmm
