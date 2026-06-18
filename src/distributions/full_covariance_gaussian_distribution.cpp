#include "libhmm/distributions/full_covariance_gaussian_distribution.h"

#include <cassert>
#include <cmath>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "libhmm/io/json_utils.h"

namespace libhmm {

// =============================================================================
// Construction — including explicit copy/move for the mutable mutex member
// =============================================================================

// Copy constructor: each object gets its own independent mutex.
// DistributionBase(other) above reads only the atomic cacheValid_ flag, which
// is self-synchronizing. All non-atomic fields (including the mutable chol_L_
// and log_det_ that updateCache() writes) are copied in the constructor body
// under other.cache_mutex_ to prevent a data race with a concurrent
// getLogProbability() call that triggers updateCache() on the source object.
FullCovarianceGaussianDistribution::FullCovarianceGaussianDistribution(
    const FullCovarianceGaussianDistribution &other)
    : DistributionBase(other) {
    std::lock_guard lock(other.cache_mutex_);
    dim_ = other.dim_;
    mean_ = other.mean_;
    cov_ = other.cov_;
    chol_L_ = other.chol_L_;
    log_det_ = other.log_det_;
    reg_ = other.reg_;
    half_d_log2pi_ = other.half_d_log2pi_;
    // cache_mutex_ is default-constructed; each copy owns its own mutex.
}

FullCovarianceGaussianDistribution &
FullCovarianceGaussianDistribution::operator=(const FullCovarianceGaussianDistribution &other) {
    if (this != &other) {
        // Lock other's mutex to safely read its cached fields.
        std::lock_guard lock(other.cache_mutex_);
        DistributionBase::operator=(other);
        dim_ = other.dim_;
        mean_ = other.mean_;
        cov_ = other.cov_;
        chol_L_ = other.chol_L_;
        log_det_ = other.log_det_;
        reg_ = other.reg_;
        half_d_log2pi_ = other.half_d_log2pi_;
        // this->cache_mutex_ is unchanged; it is independent per object.
    }
    return *this;
}

// Move constructor: cached data is moved; each object retains its own mutex.
FullCovarianceGaussianDistribution::FullCovarianceGaussianDistribution(
    FullCovarianceGaussianDistribution &&other) noexcept
    : DistributionBase(std::move(other)), dim_(other.dim_), mean_(std::move(other.mean_)),
      cov_(std::move(other.cov_)), chol_L_(std::move(other.chol_L_)), log_det_(other.log_det_),
      reg_(other.reg_), half_d_log2pi_(other.half_d_log2pi_) {
    // cache_mutex_ is default-constructed; other.cache_mutex_ remains with other.
}

FullCovarianceGaussianDistribution &
FullCovarianceGaussianDistribution::operator=(FullCovarianceGaussianDistribution &&other) noexcept {
    if (this != &other) {
        DistributionBase::operator=(std::move(other));
        dim_ = other.dim_;
        mean_ = std::move(other.mean_);
        cov_ = std::move(other.cov_);
        chol_L_ = std::move(other.chol_L_);
        log_det_ = other.log_det_;
        reg_ = other.reg_;
        half_d_log2pi_ = other.half_d_log2pi_;
        // cache_mutex_ is independent per object; not transferred on move.
    }
    return *this;
}

FullCovarianceGaussianDistribution::FullCovarianceGaussianDistribution(std::size_t dim,
                                                                       double regularise)
    : dim_{dim}, mean_(dim, 0.0), cov_(dim, dim, 0.0), reg_{regularise},
      half_d_log2pi_{0.5 * static_cast<double>(dim) * constants::math::LN_2PI} {
    validateDim(dim, "FullCovarianceGaussianDistribution");
    // cov_ = I (unregularised); cache is built from (I + reg*I).
    for (std::size_t d = 0; d < dim; ++d)
        cov_(d, d) = 1.0;
    auto tmp = cov_;
    for (std::size_t d = 0; d < dim; ++d)
        tmp(d, d) += reg_;
    auto res = chol::factorize(tmp);
    if (res.success) {
        chol_L_ = std::move(res.L);
        log_det_ = chol::log_det(chol_L_);
    }
    markCacheValid();
}

// =============================================================================
// Evaluation
// =============================================================================

double FullCovarianceGaussianDistribution::getLogProbability(
    const ObservationVectorView &x) const noexcept {
    if (x.size() != dim_)
        return -std::numeric_limits<double>::infinity();
    if (!isCacheValid())
        updateCache();
    // inv_quad_form_mv uses a thread_local scratch buffer: zero heap allocation
    // in steady state, keeping noexcept sound on this hot path.
    const double q = chol::inv_quad_form_mv(chol_L_, mean_, x);
    return -half_d_log2pi_ - 0.5 * log_det_ - 0.5 * q;
}

double FullCovarianceGaussianDistribution::getProbability(const ObservationVectorView &x) const {
    return std::exp(getLogProbability(x));
}

// =============================================================================
// Fitting helpers
// =============================================================================

namespace {

/// Apply regularisation εI and attempt Cholesky factorization.
/// Returns the factorization result; success==false if still not SPD.
chol::CholeskyResult regularise_and_factorize(BasicMatrix<double> &cov, std::size_t dim,
                                              double reg) noexcept {
    for (std::size_t d = 0; d < dim; ++d)
        cov(d, d) += reg;
    return chol::factorize(cov);
}

/// Unweighted sample mean from @p data rows.
std::vector<double> compute_mean(std::span<const ObservationVectorView> data, std::size_t dim) {
    const double inv_n = 1.0 / static_cast<double>(data.size());
    std::vector<double> mean(dim, 0.0);
    for (const auto &x : data)
        for (std::size_t d = 0; d < dim; ++d)
            mean[d] += x[d];
    for (auto &m : mean)
        m *= inv_n;
    return mean;
}

/// Unweighted sample covariance (upper triangle filled and symmetrized).
BasicMatrix<double> compute_cov(std::span<const ObservationVectorView> data,
                                const std::vector<double> &mean, std::size_t dim) {
    const double inv_n = 1.0 / static_cast<double>(data.size());
    BasicMatrix<double> cov(dim, dim, 0.0);
    for (const auto &x : data) {
        for (std::size_t i = 0; i < dim; ++i) {
            const double ri = x[i] - mean[i];
            for (std::size_t j = i; j < dim; ++j)
                cov(i, j) += ri * (x[j] - mean[j]);
        }
    }
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i; j < dim; ++j) {
            cov(i, j) *= inv_n;
            cov(j, i) = cov(i, j);
        }
    }
    return cov;
}

/// Weighted mean from @p data rows with @p weights summing to @p sumW.
std::vector<double> compute_weighted_mean(std::span<const ObservationVectorView> data,
                                          std::span<const double> weights, double sumW,
                                          std::size_t dim) {
    const double inv_sumW = 1.0 / sumW;
    std::vector<double> mean(dim, 0.0);
    for (std::size_t i = 0; i < data.size(); ++i)
        for (std::size_t d = 0; d < dim; ++d)
            mean[d] += weights[i] * data[i][d];
    for (auto &m : mean)
        m *= inv_sumW;
    return mean;
}

/// Weighted covariance (upper triangle and symmetric lower) about @p mean.
BasicMatrix<double> compute_weighted_cov(std::span<const ObservationVectorView> data,
                                         std::span<const double> weights,
                                         const std::vector<double> &mean, double inv_sumW,
                                         std::size_t dim) {
    BasicMatrix<double> cov(dim, dim, 0.0);
    for (std::size_t i = 0; i < data.size(); ++i) {
        for (std::size_t r = 0; r < dim; ++r) {
            const double wr = weights[i] * (data[i][r] - mean[r]);
            for (std::size_t c = r; c < dim; ++c)
                cov(r, c) += wr * (data[i][c] - mean[c]);
        }
    }
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = r; c < dim; ++c) {
            cov(r, c) *= inv_sumW;
            cov(c, r) = cov(r, c);
        }
    }
    return cov;
}

} // anonymous namespace

// =============================================================================
// Fitting — unweighted MLE
// =============================================================================

void FullCovarianceGaussianDistribution::fit(std::span<const ObservationVectorView> data) {
    if (data.size() < 2) {
        reset();
        return;
    }
    for (const auto &x : data)
        if (x.size() != dim_)
            throw std::invalid_argument(
                "FullCovarianceGaussianDistribution::fit: observation dimension mismatch");
    // Invalidate before computing: if factorization fails, getLogProbability()
    // will safely recompute from the unchanged pre-fit parameters rather than
    // returning a log-probability with a mismatched mean_ and old chol_L_.
    invalidateCache();
    auto new_mean = compute_mean(data, dim_);
    auto new_cov = compute_cov(data, new_mean, dim_);
    auto tmp = new_cov; // scratch copy for regularisation; new_cov stays unregularised
    auto res = regularise_and_factorize(tmp, dim_, reg_);
    if (res.success) {
        mean_ = std::move(new_mean);
        cov_ = std::move(new_cov); // store unregularised empirical covariance
        chol_L_ = std::move(res.L);
        log_det_ = chol::log_det(chol_L_);
        markCacheValid();
    }
    // On failure: mean_/cov_/chol_L_ are unchanged; cache stays invalid so the
    // next getLogProbability() recomputes from the pre-fit parameters.
}

// =============================================================================
// Fitting — weighted MLE (Baum-Welch M-step)
// =============================================================================

void FullCovarianceGaussianDistribution::fit(std::span<const ObservationVectorView> data,
                                             std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW <= 0.0 || data.empty())
        return;
    for (const auto &x : data)
        if (x.size() != dim_)
            throw std::invalid_argument(
                "FullCovarianceGaussianDistribution::fit: observation dimension mismatch");
    invalidateCache();
    auto new_mean = compute_weighted_mean(data, weights, sumW, dim_);
    auto new_cov = compute_weighted_cov(data, weights, new_mean, 1.0 / sumW, dim_);
    auto tmp = new_cov; // scratch copy for regularisation; new_cov stays unregularised
    auto res = regularise_and_factorize(tmp, dim_, reg_);
    if (res.success) {
        mean_ = std::move(new_mean);
        cov_ = std::move(new_cov); // store unregularised empirical covariance
        chol_L_ = std::move(res.L);
        log_det_ = chol::log_det(chol_L_);
        markCacheValid();
    }
}

void FullCovarianceGaussianDistribution::setCovariance(BasicMatrix<double> cov) {
    if (cov.size1() != dim_ || cov.size2() != dim_)
        throw std::invalid_argument(
            "FullCovarianceGaussianDistribution::setCovariance: dimension mismatch");
    auto tmp = cov; // scratch copy for regularisation; cov remains unregularised
    auto res = regularise_and_factorize(tmp, dim_, reg_);
    if (!res.success)
        throw std::invalid_argument("FullCovarianceGaussianDistribution::setCovariance: "
                                    "matrix is not positive-definite");
    cov_ = std::move(cov); // store unregularised user-provided matrix
    chol_L_ = std::move(res.L);
    log_det_ = chol::log_det(chol_L_);
    markCacheValid();
}

void FullCovarianceGaussianDistribution::setParameters(std::vector<double> mean,
                                                       BasicMatrix<double> cov) {
    if (mean.size() != dim_)
        throw std::invalid_argument(
            "FullCovarianceGaussianDistribution::setParameters: mean size mismatch");
    if (cov.size1() != dim_ || cov.size2() != dim_)
        throw std::invalid_argument(
            "FullCovarianceGaussianDistribution::setParameters: cov dimension mismatch");
    auto tmp = cov; // scratch copy for regularisation; cov remains unregularised
    auto res = regularise_and_factorize(tmp, dim_, reg_);
    if (!res.success)
        throw std::invalid_argument("FullCovarianceGaussianDistribution::setParameters: "
                                    "matrix is not positive-definite");
    mean_ = std::move(mean);
    cov_ = std::move(cov); // store unregularised user-provided matrix
    chol_L_ = std::move(res.L);
    log_det_ = chol::log_det(chol_L_);
    markCacheValid();
}

void FullCovarianceGaussianDistribution::reset() noexcept {
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

double FullCovarianceGaussianDistribution::sample(std::mt19937_64 &) const {
    throw std::logic_error("FullCovarianceGaussianDistribution::sample() returns double which is "
                           "not meaningful for D-dimensional distributions. Use sample_mv().");
}

std::vector<double> FullCovarianceGaussianDistribution::sample_mv(std::mt19937_64 &rng) const {
    if (!isCacheValid())
        updateCache();

    // x = μ + L·z,  z ~ N(0, I)
    std::normal_distribution<double> std_normal(0.0, 1.0);
    std::vector<double> z(dim_);
    for (std::size_t d = 0; d < dim_; ++d) {
        z[d] = std_normal(rng);
    }

    std::vector<double> result(mean_); // copy mean
    for (std::size_t i = 0; i < dim_; ++i) {
        for (std::size_t j = 0; j <= i; ++j) { // L is lower triangular
            result[i] += chol_L_(i, j) * z[j];
        }
    }
    return result;
}

// =============================================================================
// Metadata
// =============================================================================

std::string FullCovarianceGaussianDistribution::to_json() const {
    std::string s;
    s.reserve(64 + dim_ * 20 + dim_ * dim_ * 20);
    s += "{\"type\":\"FullCovarianceGaussian\"";
    s += ",\"dim\":";
    s += json::write_double(static_cast<double>(dim_));
    s += ",\"reg\":";
    s += json::write_double(reg_);
    s += ",\"mean\":";
    s += json::write_array(std::span<const double>(mean_.data(), dim_));
    s += ",\"cov\":";
    s += json::write_matrix(dim_, dim_, std::span<const double>(cov_.data(), dim_ * dim_));
    s += '}';
    return s;
}

std::unique_ptr<BasicEmissionDistribution<ObservationVectorView>>
FullCovarianceGaussianDistribution::from_json(json::Reader &r) {
    // Reader is positioned after '{' and "type":"FullCovarianceGaussian" consumed.
    r.read_key(); // "dim"
    const auto dim_raw = r.read_double();
    // Upper bound matches kMaxMvDimensions in hmm_json.cpp (1024).
    if (!std::isfinite(dim_raw) || dim_raw < 1.0 || dim_raw > 1024.0)
        throw std::runtime_error("FullCovarianceGaussian JSON: dim out of range [1, 1024]");
    const std::size_t D = static_cast<std::size_t>(dim_raw);

    r.read_key(); // "reg"
    const double reg = r.read_double();

    r.read_key(); // "mean"
    const auto mean_data = r.read_double_array(D);
    if (mean_data.size() != D)
        throw std::runtime_error("FullCovarianceGaussian JSON: mean size mismatch");

    r.read_key(); // "cov"
    const auto cov_rows = r.read_double_matrix(D, D);
    if (cov_rows.size() != D)
        throw std::runtime_error("FullCovarianceGaussian JSON: cov row count mismatch");
    for (std::size_t i = 0; i < D; ++i)
        if (cov_rows[i].size() != D)
            throw std::runtime_error("FullCovarianceGaussian JSON: cov row length mismatch");

    r.consume('}');

    auto dist = std::make_unique<FullCovarianceGaussianDistribution>(D, reg);
    for (std::size_t d = 0; d < D; ++d)
        dist->mean_[d] = mean_data[d];
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j)
            dist->cov_(i, j) = cov_rows[i][j];
    dist->invalidateCache();
    return dist;
}

std::string FullCovarianceGaussianDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed;
    oss << "FullCovarianceGaussian Distribution (D=" << dim_ << "):\n";
    oss << "  mean: [";
    for (std::size_t d = 0; d < dim_; ++d) {
        oss << mean_[d];
        if (d + 1 < dim_)
            oss << ", ";
    }
    oss << "]\n  log_det(Σ): " << (isCacheValid() ? log_det_ : 0.0) << "\n";
    return oss.str();
}

} // namespace libhmm
