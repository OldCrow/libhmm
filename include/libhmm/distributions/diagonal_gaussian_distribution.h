#pragma once

#include <cmath>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

#include "libhmm/common/common.h"
#include "libhmm/distributions/distribution_base.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {

/**
 * @brief Multivariate Gaussian with diagonal covariance.
 *
 * Each dimension d has its own mean μ_d and variance σ²_d.  Cross-dimensional
 * correlations are assumed zero.  Log-probability:
 *
 *   log p(x; μ, σ²) = -D/2 · log(2π) - ½ Σ_d [log(σ²_d) + (x_d - μ_d)²/σ²_d]
 *
 * This is simpler and faster than `FullCovarianceGaussianDistribution` and should
 * be preferred when correlations between features are expected to be weak.
 *
 * Weighted MLE (Baum-Welch M-step):
 *   μ̂_d = Σ_i w_i x_{i,d} / Σ_i w_i
 *   σ̂²_d = Σ_i w_i (x_{i,d} - μ̂_d)² / Σ_i w_i  +  regularisation
 *
 * Free parameters: 2·D (D means + D variances).
 *
 * Obs = ObservationVectorView = std::span<const double>.
 */
class DiagonalGaussianDistribution
    : public DistributionBase<DiagonalGaussianDistribution, ObservationVectorView> {
private:
    std::size_t dim_{0};
    std::vector<double> mean_; ///< μ_d  for d in [0, D)
    std::vector<double> var_;  ///< σ²_d for d in [0, D)

    // Cached values (updated when parameters change)
    mutable std::vector<double> log_var_; ///< log(σ²_d)
    mutable std::vector<double> inv_var_; ///< 1 / σ²_d
    mutable double log_normalizer_{0.0};  ///< D/2 · log(2π)

    /// Minimum allowed variance (prevents numerical collapse).
    static constexpr double kMinVar = 1e-6;

    /// Guards concurrent writes to log_var_, inv_var_, and log_normalizer_.
    /// Not copied/moved: each object has its own independent mutex.
    mutable std::mutex cache_mutex_;

    void updateCache() const noexcept {
        // Double-checked locking: re-check the flag under the mutex so that
        // two threads racing on a stale cache only compute once.
        std::lock_guard lock(cache_mutex_);
        if (isCacheValid())
            return;
        // log_var_ and inv_var_ are pre-sized to dim_ in the constructor and
        // never change size thereafter; no resize needed here.
        log_normalizer_ = 0.5 * static_cast<double>(dim_) * constants::math::LN_2PI;
        for (std::size_t d = 0; d < dim_; ++d) {
            log_var_[d] = std::log(var_[d]);
            inv_var_[d] = 1.0 / var_[d];
        }
        markCacheValid();
    }

public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Construct with D dimensions and optional initial parameters.
     * @param dim   Number of dimensions D (must be > 0).
     * @param mean  Initial mean for all dimensions (default 0.0).
     * @param var   Initial variance for all dimensions (default 1.0, must be > 0).
     */
    explicit DiagonalGaussianDistribution(std::size_t dim, double mean = 0.0, double var = 1.0);

    // Explicit copy/move: std::mutex is non-copyable/non-movable; each object
    // gets its own independent mutex; cached data is copied/moved explicitly.
    DiagonalGaussianDistribution(const DiagonalGaussianDistribution &);
    DiagonalGaussianDistribution &operator=(const DiagonalGaussianDistribution &);
    DiagonalGaussianDistribution(DiagonalGaussianDistribution &&) noexcept;
    DiagonalGaussianDistribution &operator=(DiagonalGaussianDistribution &&) noexcept;
    ~DiagonalGaussianDistribution() override = default;

    // =========================================================================
    // EmissionDistribution interface
    // =========================================================================

    /** @brief exp(getLogProbability(x)). Prefer the log version. */
    [[nodiscard]] double getProbability(const ObservationVectorView &x) const override;

    /**
     * @brief log p(x) = -D/2·log(2π) - ½·Σ_d [log(σ²_d) + (x_d-μ_d)²/σ²_d]
     * Precondition: x.size() == getDimension()
     */
    [[nodiscard]] double getLogProbability(const ObservationVectorView &x) const noexcept override;

    /** @brief Unweighted MLE: per-dimension sample mean and variance. */
    void fit(std::span<const ObservationVectorView> data) override;

    /** @brief Weighted MLE (Baum-Welch M-step): per-dimension weighted mean and variance. */
    void fit(std::span<const ObservationVectorView> data, std::span<const double> weights) override;

    /** @brief Reset to zero mean and unit variance for all dimensions. */
    void reset() noexcept override;

    /**
     * @brief Scalar sample() is not meaningful for multivariate distributions.
     * Use sample_mv() instead.
     * @throws std::logic_error always.
     */
    [[nodiscard]] double sample(std::mt19937_64 &rng) const override;

    /** @brief Draw a D-dimensional sample: result[d] ~ N(μ_d, σ²_d). */
    [[nodiscard]] std::vector<double> sample_mv(std::mt19937_64 &rng) const;

    [[nodiscard]] std::string to_json() const override;
    [[nodiscard]] std::string toString() const override;
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief 2·D (D means + D variances). */
    [[nodiscard]] std::size_t getNumParameters() const noexcept override { return 2 * dim_; }

    // =========================================================================
    // Accessors
    // =========================================================================

    [[nodiscard]] std::size_t getDimension() const noexcept override { return dim_; }
    [[nodiscard]] const std::vector<double> &getMean() const noexcept { return mean_; }
    [[nodiscard]] const std::vector<double> &getVariance() const noexcept { return var_; }

    /**
     * @brief Set per-dimension means and variances simultaneously.
     * @throws std::invalid_argument if sizes differ from getDimension() or any variance <= 0.
     */
    void setParameters(std::vector<double> means, const std::vector<double> &variances) {
        if (means.size() != dim_ || variances.size() != dim_)
            throw std::invalid_argument(
                "DiagonalGaussianDistribution::setParameters: size mismatch");
        for (double v : variances)
            if (v <= 0.0)
                throw std::invalid_argument(
                    "DiagonalGaussianDistribution::setParameters: variances must be > 0");
        mean_ = std::move(means);
        for (std::size_t d = 0; d < dim_; ++d)
            var_[d] = std::max(variances[d], kMinVar);
        invalidateCache();
    }

    /**
     * @brief Set per-dimension means only.
     * @throws std::invalid_argument if size differs from getDimension().
     */
    void setMeans(std::vector<double> means) {
        if (means.size() != dim_)
            throw std::invalid_argument("DiagonalGaussianDistribution::setMeans: size mismatch");
        mean_ = std::move(means);
        invalidateCache();
    }

    /**
     * @brief Set per-dimension variances only.
     * @throws std::invalid_argument if size differs from getDimension() or any value <= 0.
     */
    void setVariances(const std::vector<double> &variances) {
        if (variances.size() != dim_)
            throw std::invalid_argument(
                "DiagonalGaussianDistribution::setVariances: size mismatch");
        for (double v : variances)
            if (v <= 0.0)
                throw std::invalid_argument(
                    "DiagonalGaussianDistribution::setVariances: variances must be > 0");
        for (std::size_t d = 0; d < dim_; ++d)
            var_[d] = std::max(variances[d], kMinVar);
        invalidateCache();
    }

    /**
     * @brief Deserialize from JSON.  Reader must be positioned immediately
     * after the opening '{' and the "type" value have been consumed.
     * Reads remaining fields and the closing '}'.
     */
    [[nodiscard]] static std::unique_ptr<BasicEmissionDistribution<ObservationVectorView>>
    from_json(json::Reader &r);
};

} // namespace libhmm
