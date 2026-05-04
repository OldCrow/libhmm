#include "libhmm/distributions/exponential_distribution.h"
#include "libhmm/math/weighted_stats.h"
#include "libhmm/platform/simd_platform.h"
#include <limits>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Exponential distribution.
 *
 * For continuous distributions in discrete sampling contexts, we approximate
 * the probability as P(x - ε <= X <= x) = F(x) - F(x - ε) where ε is a small tolerance.
 *
 * This provides a numerically stable approximation of the PDF scaled by the tolerance,
 * which is appropriate for discrete sampling of continuous distributions.
 *
 * @param x The value at which to evaluate the probability
 * @return Approximated probability for discrete sampling
 */
double ExponentialDistribution::getProbability(double value) const {
    // Exponential distribution has support [0, ∞)
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    // For continuous distributions, we return the actual PDF value
    // This is more mathematically correct than the old discrete approximation
    if (value == math::ZERO_DOUBLE) {
        // At x=0, PDF equals λ (the rate parameter)
        return lambda_;
    }

    if (!isCacheValid())
        updateCache();
    return lambda_ * std::exp(negLambda_ * value);
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 *
 * For exponential distribution: log(f(x)) = log(λ) - λx for x ≥ 0
 *
 * @param x The value at which to evaluate the log-PDF
 * @return Natural logarithm of the probability density, or -∞ for invalid values
 */
double ExponentialDistribution::getLogProbability(double value) const noexcept {
    // Exponential distribution has support [0, ∞)
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    if (!isCacheValid())
        updateCache();
    return logLambda_ - lambda_ * value;
}

/**
 * Evaluates the CDF for the Exponential distribution at x.
 *
 * Formula: F(x) = 1 - exp(-λx) for x ≥ 0
 *
 * @param x The value at which to evaluate the CDF
 * @return Cumulative probability P(X ≤ x)
 */
double ExponentialDistribution::getCumulativeProbability(double x) const noexcept {
    const double y = math::ONE - std::exp(-lambda_ * x);
    assert(y >= math::ZERO_DOUBLE);
    return y;
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 *
 * For the Exponential distribution, the MLE of the rate parameter is:
 * λ = 1 / sample_mean
 *
 * The sample mean is calculated as:
 * mean = Σ(x_i) / N for i = 1 to N
 *
 * @param values Vector of observed data points
 */
void ExponentialDistribution::fit(std::span<const double> data) {
    if (data.size() <= 1) {
        reset();
        return;
    }

    double mean = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        if (val < 0.0 || std::isnan(val) || std::isinf(val)) {
            reset();
            return;
        }
        ++count;
        mean += (val - mean) / static_cast<double>(count);
    }

    if (mean <= 0.0 || !std::isfinite(mean)) {
        reset();
        return;
    }
    const double lam = 1.0 / mean;
    if (!std::isfinite(lam) || lam <= 0.0) {
        reset();
        return;
    }
    lambda_ = lam;
    invalidateCache();
}

void ExponentialDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    // Weighted MLE: λ = 1 / weighted_mean
    const auto mean = detail::compute_weighted_mean(data, weights);
    if (!mean || *mean <= 0.0 || !std::isfinite(*mean)) {
        reset();
        return;
    }
    const double lam = 1.0 / *mean;
    if (!std::isfinite(lam) || lam <= 0.0) {
        reset();
        return;
    }
    lambda_ = lam;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (λ = 1.0).
 * This corresponds to the standard exponential distribution.
 */
void ExponentialDistribution::reset() noexcept {
    lambda_ = math::ONE;
    invalidateCache();
}

std::string ExponentialDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Exponential Distribution:\n";
    oss << "      λ (rate parameter) = " << lambda_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::ExponentialDistribution &distribution) {
    os << "Exponential Distribution: " << std::endl;
    os << "    Rate parameter = " << distribution.getLambda() << std::endl;
    os << std::endl;

    return os;
}

std::istream &operator>>(std::istream &is, libhmm::ExponentialDistribution &distribution) {
    try {
        std::string token, lambda_str;
        is >> token; // "Rate"
        is >> token; // "parameter"
        is >> token; // "="
        is >> lambda_str;
        double lambda = std::stod(lambda_str);

        // Use setLambda for validation
        distribution.setLambda(lambda);

    } catch (const std::exception &) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

bool ExponentialDistribution::operator==(const ExponentialDistribution &other) const {
    using namespace libhmm::constants;
    return std::abs(lambda_ - other.lambda_) < precision::LIMIT_TOLERANCE;
}

// =============================================================================
// Batch log-PDF — explicit SIMD intrinsics (tier 2)
//
// Formula (valid inputs, x >= 0): log(λ) + (−λ)·x  — a single FMA per element.
// x = +Inf naturally yields −Inf via the formula; only x < 0 and NaN need masking.
//
// Free function — no class access; extractable to a separate TU for future
// runtime dispatch without interface changes (see design decision 6 in WARP.md).
// =============================================================================
namespace detail {

void exponential_logpdf_batch(const double *obs, double *out, std::size_t n, double log_lambda,
                              double neg_lambda) noexcept {
    std::size_t i = 0;
    const double neg_inf = -std::numeric_limits<double>::infinity();

#if defined(LIBHMM_HAS_AVX512)
    {
        const __m512d loglam_v = _mm512_set1_pd(log_lambda);
        const __m512d neglam_v = _mm512_set1_pd(neg_lambda);
        const __m512d zero_v = _mm512_setzero_pd();
        const __m512d neg_inf_v = _mm512_set1_pd(neg_inf);
        for (; i + 8 <= n; i += 8) {
            __m512d x = _mm512_loadu_pd(obs + i);
            __m512d res = _mm512_add_pd(loglam_v, _mm512_mul_pd(neglam_v, x));
            __mmask8 invalid = _mm512_cmp_pd_mask(x, zero_v, _CMP_LT_OS) // x < 0
                               | _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q); // NaN
            res = _mm512_mask_blend_pd(invalid, res, neg_inf_v);
            _mm512_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        const __m256d loglam_v = _mm256_set1_pd(log_lambda);
        const __m256d neglam_v = _mm256_set1_pd(neg_lambda);
        const __m256d zero_v = _mm256_setzero_pd();
        const __m256d neg_inf_v = _mm256_set1_pd(neg_inf);
        for (; i + 4 <= n; i += 4) {
            __m256d x = _mm256_loadu_pd(obs + i);
            __m256d res = _mm256_add_pd(loglam_v, _mm256_mul_pd(neglam_v, x));
            __m256d invalid = _mm256_or_pd(_mm256_cmp_pd(x, zero_v, _CMP_LT_OS), // x < 0
                                           _mm256_cmp_pd(x, x, _CMP_UNORD_Q));   // NaN
            res = _mm256_blendv_pd(res, neg_inf_v, invalid);
            _mm256_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        const __m128d loglam_v = _mm_set1_pd(log_lambda);
        const __m128d neglam_v = _mm_set1_pd(neg_lambda);
        const __m128d zero_v = _mm_setzero_pd();
        const __m128d neg_inf_v = _mm_set1_pd(neg_inf);
        for (; i + 2 <= n; i += 2) {
            __m128d x = _mm_loadu_pd(obs + i);
            __m128d res = _mm_add_pd(loglam_v, _mm_mul_pd(neglam_v, x));
            __m128d invalid = _mm_or_pd(_mm_cmplt_pd(x, zero_v), // x < 0
                                        _mm_cmpunord_pd(x, x));  // NaN
            res = _mm_or_pd(_mm_andnot_pd(invalid, res), _mm_and_pd(invalid, neg_inf_v));
            _mm_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        const float64x2_t loglam_v = vdupq_n_f64(log_lambda);
        const float64x2_t neglam_v = vdupq_n_f64(neg_lambda);
        const float64x2_t zero_v = vdupq_n_f64(0.0);
        const float64x2_t neg_inf_v = vdupq_n_f64(neg_inf);
        for (; i + 2 <= n; i += 2) {
            float64x2_t x = vld1q_f64(obs + i);
            float64x2_t res = vaddq_f64(loglam_v, vmulq_f64(neglam_v, x));
            // valid = (x >= 0) & (x == x)  — the latter is false for NaN
            uint64x2_t valid = vandq_u64(vcgeq_f64(x, zero_v), vceqq_f64(x, x));
            res = vbslq_f64(valid, res, neg_inf_v);
            vst1q_f64(out + i, res);
        }
    }
#endif

    // Scalar tail (also covers platforms without any SIMD path above).
    // +Inf naturally yields -Inf via the formula; only x < 0 and NaN need masking.
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < 0.0 || std::isnan(x)) ? neg_inf : log_lambda + neg_lambda * x;
    }
}

} // namespace detail

void ExponentialDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                       std::span<double> out) const {
    if (!isCacheValid())
        updateCache();
    detail::exponential_logpdf_batch(observations.data(), out.data(), observations.size(),
                                     logLambda_, negLambda_);
}

} // namespace libhmm
