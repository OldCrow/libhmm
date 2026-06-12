#pragma once

#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "libhmm/common/common.h"
#include "libhmm/distributions/distribution_base.h"
#include "libhmm/distributions/emission_distribution.h"
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {

/**
 * @brief Multivariate emission distribution modelled as D independent
 *        scalar components.
 *
 * Each dimension d is handled by an independent `EmissionDistribution`
 * (a scalar `BasicEmissionDistribution<double>`).  The joint log-probability
 * is the sum of individual component log-probabilities:
 *
 *   log p(x) = Σ_{d=0}^{D-1} log p_d(x[d])
 *
 * This is the weakest multivariate extension — it cannot capture cross-
 * dimensional correlations — but it is the most flexible: any mix of
 * distribution families (Gaussian, Exponential, Poisson, …) is supported.
 *
 * Obs = ObservationVectorView = std::span<const double>.
 */
class IndependentComponentsDistribution
    : public DistributionBase<IndependentComponentsDistribution, ObservationVectorView> {
private:
    std::size_t dim_{0};
    std::vector<std::unique_ptr<EmissionDistribution>> components_;

    void validateDim(std::size_t d, std::string_view func) const {
        if (d >= dim_) {
            throw std::out_of_range(std::string(func) + ": dimension index out of range");
        }
    }

public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Create D independent GaussianDistribution(0, 1) components.
     * @param dim  Number of dimensions D (must be > 0).
     */
    explicit IndependentComponentsDistribution(std::size_t dim);

    /**
     * @brief Take ownership of pre-built component distributions.
     * @param components  Exactly D non-null emission distributions.
     * @throws std::invalid_argument if components is empty or contains null.
     */
    explicit IndependentComponentsDistribution(
        std::vector<std::unique_ptr<EmissionDistribution>> components);

    /// Deep-copy constructor required by CRTP clone().
    IndependentComponentsDistribution(const IndependentComponentsDistribution &other);
    IndependentComponentsDistribution &
    operator=(const IndependentComponentsDistribution &other) = delete;
    IndependentComponentsDistribution(IndependentComponentsDistribution &&) = default;
    IndependentComponentsDistribution &operator=(IndependentComponentsDistribution &&) = default;
    ~IndependentComponentsDistribution() override = default;

    // =========================================================================
    // EmissionDistribution interface
    // =========================================================================

    /** @brief exp(getLogProbability(x)). Prefer the log version. */
    [[nodiscard]] double getProbability(const ObservationVectorView &x) const override;

    /**
     * @brief log p(x) = Σ_d component[d].getLogProbability(x[d])
     * Precondition: x.size() == getDimension()
     */
    [[nodiscard]] double getLogProbability(const ObservationVectorView &x) const noexcept override;

    /** @brief Fit each component independently from the corresponding dimension. */
    void fit(std::span<const ObservationVectorView> data) override;

    /** @brief Weighted fit: extract per-dimension weighted data and fit each component. */
    void fit(std::span<const ObservationVectorView> data, std::span<const double> weights) override;

    /** @brief Reset all components to their defaults. */
    void reset() noexcept override;

    /**
     * @brief Scalar sample() is not meaningful for multivariate distributions.
     * Use sample_mv() instead.
     * @throws std::logic_error always.
     */
    [[nodiscard]] double sample(std::mt19937_64 &rng) const override;

    /** @brief Draw a D-dimensional sample: result[d] ~ component[d]. */
    [[nodiscard]] std::vector<double> sample_mv(std::mt19937_64 &rng) const;

    [[nodiscard]] std::string to_json() const override;
    [[nodiscard]] std::string toString() const override;

    /** @brief true only if ALL components are discrete. */
    [[nodiscard]] bool isDiscrete() const noexcept override;

    /** @brief Sum of component parameter counts. */
    [[nodiscard]] std::size_t getNumParameters() const noexcept override;

    // =========================================================================
    // Accessors
    // =========================================================================

    [[nodiscard]] std::size_t getDimension() const noexcept override { return dim_; }

    /** @brief Read-only access to component d. */
    [[nodiscard]] const EmissionDistribution &getComponent(std::size_t d) const {
        validateDim(d, "getComponent");
        return *components_[d];
    }

    /** @brief Mutable access to component d (e.g. for manual parameter setting). */
    [[nodiscard]] EmissionDistribution &getComponent(std::size_t d) {
        validateDim(d, "getComponent");
        return *components_[d];
    }

    /**
     * @brief Replace component d with a new distribution.
     * @throws std::invalid_argument if d >= getDimension() or component is null.
     */
    void setComponent(std::size_t d, std::unique_ptr<EmissionDistribution> component);
};

} // namespace libhmm
