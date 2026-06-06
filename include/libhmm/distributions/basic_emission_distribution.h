#pragma once

#include <memory>
#include <span>
#include <string>
#include <type_traits>

namespace libhmm {

/**
 * @brief Abstract HMM emission distribution interface, parameterised on
 * observation type.
 *
 * @tparam Obs  Observation type.  Default is `double` (scalar), which
 *              preserves the v3 API exactly through the `EmissionDistribution`
 *              type alias in emission_distribution.h.  For multivariate use
 *              a view type such as `std::span<const double>`.
 *
 * The parameter-passing convention for single observations adapts to `Obs`:
 *   - Scalar types (integral or floating-point): passed by value.
 *   - All other types: passed by `const Obs&`.
 *
 * This ensures `BasicEmissionDistribution<double>` carries identical virtual
 * signatures to v3's `EmissionDistribution`, so all existing derived classes
 * compile unchanged after the alias is introduced.
 */
template<typename Obs = double>
class BasicEmissionDistribution {
public:
    /// Observation parameter type: by value for scalars, const-ref otherwise.
    using obs_param_t = std::conditional_t<std::is_scalar_v<Obs>, Obs, const Obs&>;

    virtual ~BasicEmissionDistribution() = default;

    BasicEmissionDistribution() = default;
    BasicEmissionDistribution(const BasicEmissionDistribution&) = default;
    BasicEmissionDistribution& operator=(const BasicEmissionDistribution&) = default;
    BasicEmissionDistribution(BasicEmissionDistribution&&) = default;
    BasicEmissionDistribution& operator=(BasicEmissionDistribution&&) = default;

    // =========================================================================
    // Scalar evaluation
    // =========================================================================

    /**
     * @brief Probability density/mass at x.
     * Used by non-SIMD calculators and for small models.
     */
    [[nodiscard]] virtual double getProbability(obs_param_t x) const = 0;

    /**
     * @brief Log probability density/mass at x.
     * Numerically preferred in log-space calculator variants.
     */
    [[nodiscard]] virtual double getLogProbability(obs_param_t x) const = 0;

    // =========================================================================
    // Batch evaluation (in-place, no allocation)
    // =========================================================================

    /**
     * @brief Compute log probabilities for a sequence of observations.
     *
     * Writes log p(observations[i]) into out[i] for all i.
     * The default implementation is a scalar loop calling getLogProbability().
     * Override for SIMD vectorization.
     *
     * Precondition: observations.size() == out.size()
     */
    virtual void getBatchLogProbabilities(std::span<const Obs> observations,
                                          std::span<double> out) const = 0;

    // =========================================================================
    // Parameter estimation
    // =========================================================================

    /**
     * @brief Fit parameters to unweighted data (MLE).
     * Used for initialization and Viterbi training.
     */
    virtual void fit(std::span<const Obs> data) = 0;

    /**
     * @brief Fit parameters to weighted data (weighted MLE).
     * Used by the Baum-Welch M-step.
     *
     * Weights are unnormalized gamma values from the E-step. Each distribution
     * normalizes by sum(weights) internally. If sum(weights) is near zero
     * (state rarely visited), the implementation should call reset().
     *
     * Precondition: data.size() == weights.size()
     */
    virtual void fit(std::span<const Obs> data, std::span<const double> weights) = 0;

    /**
     * @brief Reset parameters to distribution-specific defaults.
     */
    virtual void reset() noexcept = 0;

    // =========================================================================
    // Cloning
    // =========================================================================

    /**
     * @brief Polymorphic copy.
     *
     * Returns a heap-allocated copy of this distribution. Implemented
     * automatically by DistributionBase<Derived, Obs> via CRTP; requires
     * Derived to be copy-constructible.
     */
    [[nodiscard]] virtual std::unique_ptr<BasicEmissionDistribution<Obs>>
    clone() const = 0;

    // =========================================================================
    // Metadata
    // =========================================================================

    /** @brief Human-readable string representation.  Delegates to to_json(). */
    [[nodiscard]] virtual std::string toString() const { return to_json(); }

    /**
     * @brief Serialise to a compact JSON object string.
     *
     * Must produce output that round-trips exactly through the matching
     * static from_json() factory registered in src/io/hmm_json.cpp.
     * Use json::write_distribution() from libhmm/io/json_utils.h.
     */
    [[nodiscard]] virtual std::string to_json() const = 0;

    /** @brief Returns true for discrete (PMF) distributions, false for continuous (PDF). */
    [[nodiscard]] virtual bool isDiscrete() const noexcept = 0;

    /**
     * @brief Number of free parameters in this distribution.
     *
     * Used by model-selection utilities (AIC, BIC, AICc) to compute the
     * total parameter count for an HMM.
     */
    [[nodiscard]] virtual std::size_t getNumParameters() const noexcept = 0;
};

} // namespace libhmm
