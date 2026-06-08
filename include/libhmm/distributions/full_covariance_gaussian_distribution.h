#pragma once

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "libhmm/common/common.h"
#include "libhmm/distributions/distribution_base.h"
#include "libhmm/linalg/cholesky.h"
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {

/**
 * @brief Multivariate Gaussian with full covariance matrix.
 *
 * Models correlated features via a D×D positive-definite covariance Σ.
 * The Cholesky factor L = chol(Σ) is cached and used for numerically
 * stable log-probability and sampling:
 *
 *   log p(x; μ, Σ) = -D/2·log(2π) - ½·log|det Σ| - ½·(x-μ)ᵀ Σ⁻¹ (x-μ)
 *                  = -D/2·log(2π) - log_det(L)/2  - inv_quad_form(L, x-μ)/2
 *
 * Weighted MLE (Baum-Welch M-step):
 *   μ̂   = Σ_i w_i x_i / Σ_i w_i
 *   Σ̂   = Σ_i w_i (x_i-μ̂)(x_i-μ̂)ᵀ / Σ_i w_i  +  reg·I
 *
 * The regularisation term (default 1e-5·I) ensures positive-definiteness
 * when sample count is low relative to dimensionality.
 *
 * Free parameters: D + D·(D+1)/2  (D means + upper triangle of Σ).
 *
 * Obs = ObservationVectorView = std::span<const double>.
 */
class FullCovarianceGaussianDistribution
    : public DistributionBase<FullCovarianceGaussianDistribution,
                               ObservationVectorView> {
private:
    std::size_t dim_{0};
    std::vector<double>   mean_;    ///< D-dimensional mean vector
    BasicMatrix<double>   cov_;     ///< D×D covariance matrix

    // Cached Cholesky factor and log-determinant
    mutable BasicMatrix<double> chol_L_{};   ///< lower-triangular L: Σ = L·Lᵀ
    mutable double log_det_{0.0};            ///< log(det(Σ))

    /// Regularisation added to the diagonal before factorization.
    double reg_{1e-5};

    /// D/2 · log(2π) — precomputed at construction.
    double half_d_log2pi_{0.0};

    void updateCache() const noexcept {
        auto res = chol::factorize(cov_);
        if (res.success) {
            chol_L_  = std::move(res.L);
            log_det_ = chol::log_det(chol_L_);
        }
        // If factorization fails the cached values retain their previous state.
        markCacheValid();
    }

    static void validateDim(std::size_t dim, std::string_view tag) {
        if (dim == 0) {
            throw std::invalid_argument(std::string(tag) + ": dimension must be > 0");
        }
    }

public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Construct a D-dimensional Gaussian with μ = 0, Σ = I.
     * @param dim          Number of dimensions (must be > 0).
     * @param regularise   Regularisation added to Σ diagonal before
     *                     factorization (default 1e-5).
     */
    explicit FullCovarianceGaussianDistribution(std::size_t dim,
                                                 double regularise = 1e-5);

    FullCovarianceGaussianDistribution(
        const FullCovarianceGaussianDistribution&) = default;
    FullCovarianceGaussianDistribution& operator=(
        const FullCovarianceGaussianDistribution&) = default;
    FullCovarianceGaussianDistribution(
        FullCovarianceGaussianDistribution&&) = default;
    FullCovarianceGaussianDistribution& operator=(
        FullCovarianceGaussianDistribution&&) = default;
    ~FullCovarianceGaussianDistribution() override = default;

    // =========================================================================
    // EmissionDistribution interface
    // =========================================================================

    /** @brief exp(getLogProbability(x)). Prefer the log version. */
    [[nodiscard]] double getProbability(const ObservationVectorView& x) const override;

    /**
     * @brief log p(x) = -D/2·log(2π) - log_det(L)/2 - inv_quad_form(L, x-μ)/2
     * Precondition: x.size() == getDimension()
     */
    [[nodiscard]] double getLogProbability(
        const ObservationVectorView& x) const noexcept override;

    /** @brief Unweighted MLE: sample mean and covariance. */
    void fit(std::span<const ObservationVectorView> data) override;

    /** @brief Weighted MLE (Baum-Welch M-step): weighted mean and covariance. */
    void fit(std::span<const ObservationVectorView> data,
             std::span<const double> weights) override;

    /** @brief Reset to μ = 0, Σ = I (and refactorize). */
    void reset() noexcept override;

    /**
     * @brief Scalar sample() is not meaningful for multivariate distributions.
     * Use sample_mv() instead.
     * @throws std::logic_error always.
     */
    [[nodiscard]] double sample(std::mt19937_64& rng) const override;

    /**
     * @brief Draw a D-dimensional sample: x = μ + L·z, z ~ N(0, I).
     * Precondition: the Cholesky factor must be valid (i.e., fit() or
     * construction must have succeeded).
     */
    [[nodiscard]] std::vector<double> sample_mv(std::mt19937_64& rng) const;

    [[nodiscard]] std::string to_json() const override;
    [[nodiscard]] std::string toString() const override;
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief D + D·(D+1)/2 (mean vector + upper triangle of Σ). */
    [[nodiscard]] std::size_t getNumParameters() const noexcept override {
        return dim_ + dim_ * (dim_ + 1) / 2;
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    [[nodiscard]] std::size_t getDimension() const noexcept { return dim_; }
    [[nodiscard]] const std::vector<double>&  getMean()       const noexcept { return mean_; }
    [[nodiscard]] const BasicMatrix<double>&  getCovariance() const noexcept { return cov_;  }

    /** @brief Cached Cholesky factor L; valid after a successful fit/construction. */
    [[nodiscard]] const BasicMatrix<double>& getCholeskyFactor() const noexcept {
        if (!isCacheValid()) updateCache();
        return chol_L_;
    }

    /** @brief log(det(Σ)); valid after a successful fit/construction. */
    [[nodiscard]] double getLogDet() const noexcept {
        if (!isCacheValid()) updateCache();
        return log_det_;
    }
};

} // namespace libhmm
