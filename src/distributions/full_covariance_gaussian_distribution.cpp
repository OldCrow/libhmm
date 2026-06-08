#include "libhmm/distributions/full_covariance_gaussian_distribution.h"

#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace libhmm {

// =============================================================================
// Construction
// =============================================================================

FullCovarianceGaussianDistribution::FullCovarianceGaussianDistribution(
    std::size_t dim, double regularise)
    : dim_{dim}
    , mean_(dim, 0.0)
    , cov_(dim, dim, 0.0)
    , reg_{regularise}
    , half_d_log2pi_{0.5 * static_cast<double>(dim) * constants::math::LN_2PI}
{
    validateDim(dim, "FullCovarianceGaussianDistribution");
    // Initialise to identity covariance + regularisation
    for (std::size_t d = 0; d < dim; ++d) {
        cov_(d, d) = 1.0;
    }
    // Factorize immediately so the cache is valid
    auto res = chol::factorize(cov_);
    if (res.success) {
        chol_L_  = std::move(res.L);
        log_det_ = chol::log_det(chol_L_);
    }
    markCacheValid();
}

// =============================================================================
// Evaluation
// =============================================================================

double FullCovarianceGaussianDistribution::getLogProbability(
    const ObservationVectorView& x) const noexcept
{
    if (x.size() != dim_) return -std::numeric_limits<double>::infinity();
    if (!isCacheValid()) updateCache();

    // Compute residual r = x - μ
    std::vector<double> r(dim_);
    for (std::size_t d = 0; d < dim_; ++d) { r[d] = x[d] - mean_[d]; }

    const double q = chol::inv_quad_form(chol_L_, r);
    return -half_d_log2pi_ - 0.5 * log_det_ - 0.5 * q;
}

double FullCovarianceGaussianDistribution::getProbability(
    const ObservationVectorView& x) const
{
    return std::exp(getLogProbability(x));
}

// =============================================================================
// Fitting helpers
// =============================================================================

namespace {

/// Apply regularisation εI and attempt Cholesky factorization.
/// Returns the factorization result; success==false if still not SPD.
chol::CholeskyResult regularise_and_factorize(BasicMatrix<double>& cov,
                                               std::size_t dim,
                                               double reg) noexcept
{
    for (std::size_t d = 0; d < dim; ++d) {
        cov(d, d) += reg;
    }
    return chol::factorize(cov);
}

} // anonymous namespace

// =============================================================================
// Fitting — unweighted MLE
// =============================================================================

void FullCovarianceGaussianDistribution::fit(
    std::span<const ObservationVectorView> data)
{
    if (data.size() < 2) { reset(); return; }
    const double inv_n = 1.0 / static_cast<double>(data.size());

    // Weighted mean (equal weights)
    std::fill(mean_.begin(), mean_.end(), 0.0);
    for (const auto& x : data) {
        for (std::size_t d = 0; d < dim_; ++d) { mean_[d] += x[d]; }
    }
    for (auto& m : mean_) { m *= inv_n; }

    // Covariance
    BasicMatrix<double> new_cov(dim_, dim_, 0.0);
    for (const auto& x : data) {
        for (std::size_t i = 0; i < dim_; ++i) {
            const double ri = x[i] - mean_[i];
            for (std::size_t j = i; j < dim_; ++j) {
                new_cov(i, j) += ri * (x[j] - mean_[j]);
            }
        }
    }
    for (std::size_t i = 0; i < dim_; ++i) {
        for (std::size_t j = i; j < dim_; ++j) {
            new_cov(i, j) *= inv_n;
            new_cov(j, i)  = new_cov(i, j);  // symmetry
        }
    }

    auto res = regularise_and_factorize(new_cov, dim_, reg_);
    if (res.success) {
        cov_     = new_cov;
        chol_L_  = std::move(res.L);
        log_det_ = chol::log_det(chol_L_);
        markCacheValid();
    }
    // On failure: keep current parameters
}

// =============================================================================
// Fitting — weighted MLE (Baum-Welch M-step)
// =============================================================================

void FullCovarianceGaussianDistribution::fit(
    std::span<const ObservationVectorView> data,
    std::span<const double> weights)
{
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW <= 0.0 || data.empty()) return;
    const double inv_sumW = 1.0 / sumW;
    const std::size_t n   = data.size();

    // Weighted mean
    std::fill(mean_.begin(), mean_.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < dim_; ++d) {
            mean_[d] += weights[i] * data[i][d];
        }
    }
    for (auto& m : mean_) { m *= inv_sumW; }

    // Weighted covariance
    BasicMatrix<double> new_cov(dim_, dim_, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < dim_; ++r) {
            const double wr = weights[i] * (data[i][r] - mean_[r]);
            for (std::size_t c = r; c < dim_; ++c) {
                new_cov(r, c) += wr * (data[i][c] - mean_[c]);
            }
        }
    }
    for (std::size_t r = 0; r < dim_; ++r) {
        for (std::size_t c = r; c < dim_; ++c) {
            new_cov(r, c) *= inv_sumW;
            new_cov(c, r)  = new_cov(r, c);
        }
    }

    auto res = regularise_and_factorize(new_cov, dim_, reg_);
    if (res.success) {
        cov_     = new_cov;
        chol_L_  = std::move(res.L);
        log_det_ = chol::log_det(chol_L_);
        markCacheValid();
    }
}

void FullCovarianceGaussianDistribution::reset() noexcept
{
    std::fill(mean_.begin(), mean_.end(), 0.0);
    // Reset covariance to identity
    for (std::size_t i = 0; i < dim_; ++i) {
        for (std::size_t j = 0; j < dim_; ++j) {
            cov_(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }
    invalidateCache();
}

// =============================================================================
// Sampling
// =============================================================================

double FullCovarianceGaussianDistribution::sample(std::mt19937_64&) const
{
    throw std::logic_error(
        "FullCovarianceGaussianDistribution::sample() returns double which is "
        "not meaningful for D-dimensional distributions. Use sample_mv().");
}

std::vector<double> FullCovarianceGaussianDistribution::sample_mv(
    std::mt19937_64& rng) const
{
    if (!isCacheValid()) updateCache();

    // x = μ + L·z,  z ~ N(0, I)
    std::normal_distribution<double> std_normal(0.0, 1.0);
    std::vector<double> z(dim_);
    for (std::size_t d = 0; d < dim_; ++d) { z[d] = std_normal(rng); }

    std::vector<double> result(mean_);   // copy mean
    for (std::size_t i = 0; i < dim_; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {   // L is lower triangular
            result[i] += chol_L_(i, j) * z[j];
        }
    }
    return result;
}

// =============================================================================
// Metadata
// =============================================================================

std::string FullCovarianceGaussianDistribution::to_json() const
{
    std::ostringstream oss;
    oss << "{\"type\":\"FullCovarianceGaussian\",\"dim\":" << dim_ << "}";
    return oss.str();
}

std::string FullCovarianceGaussianDistribution::toString() const
{
    std::ostringstream oss;
    oss << std::fixed;
    oss << "FullCovarianceGaussian Distribution (D=" << dim_ << "):\n";
    oss << "  mean: [";
    for (std::size_t d = 0; d < dim_; ++d) {
        oss << mean_[d];
        if (d + 1 < dim_) oss << ", ";
    }
    oss << "]\n  log_det(Σ): " << (isCacheValid() ? log_det_ : 0.0) << "\n";
    return oss.str();
}

} // namespace libhmm
