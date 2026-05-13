#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/platform/simd_platform.h" // compile-time SIMD macros + intrinsics
#include <algorithm>
#include <limits>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {
/**
 * Returns the probability density function value for the Gaussian distribution.
 *
 * Formula: PDF(x) = (1/σ√(2π)) * exp(-½((x-μ)/σ)²)
 */
double GaussianDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x)) {
        return 0.0;
    }
    if (!isCacheValid()) {
        updateCache();
    }

    const double exponent = (x - mean_) * (x - mean_) * negHalfSigmaSquaredInv_;
    return normalizationConstant_ * std::exp(exponent);
}

/**
 * Returns the log probability density function value for numerical stability.
 * Formula: log PDF(x) = -½log(2π) - log(σ) - ½((x-μ)/σ)²
 */
double GaussianDistribution::getLogProbability(double x) const noexcept {
    // Validate input
    if (std::isnan(x) || std::isinf(x)) {
        return -std::numeric_limits<double>::infinity();
    }

    if (!isCacheValid()) {
        updateCache();
    }
    // Use cached values for maximum performance
    const double z = (x - mean_) * invStandardDeviation_;
    const double logPdf = -0.5 * math::LN_2PI - logStandardDeviation_ - 0.5 * z * z;

    return logPdf;
}

/**
 * Evaluates the CDF for the Normal distribution at x.  The CDF is defined as
 *
 *          1             x - mean
 *   F(x) = -( 1 + erf(---------------) )
 *          2           sigma*sqrt(2)
 */
double GaussianDistribution::getCumulativeProbability(double x) const noexcept {
    // Handle problematic inputs
    if (std::isnan(x) || std::isnan(mean_) || std::isnan(standardDeviation_)) {
        return 0.0;
    }
    if (std::isinf(x)) {
        return (x > 0) ? 1.0 : 0.0;
    }
    if (standardDeviation_ <= 0.0) {
        return (x >= mean_) ? 1.0 : 0.0;
    }

    if (!isCacheValid()) {
        updateCache();
    }
    // Use cached sigma*sqrt(2) for efficiency
    const double y = 0.5 * (1 + std::erf((x - mean_) / sigmaSqrt2_));

    // Ensure valid probability range
    if (std::isnan(y) || y < 0.0) {
        return 0.0;
    }
    if (y > 1.0) {
        return 1.0;
    }

    return y;
}

/*
 * Fits the distribution parameters using maximum likelihood estimation with optimized algorithm.
 *
 * Uses single-pass Welford's algorithm for numerically stable variance calculation:
 * - Better cache locality than two-pass algorithm
 * - Numerically stable for extreme values
 * - O(n) time complexity with single data traversal
 */
void GaussianDistribution::fit(std::span<const double> data) {
    if (data.size() <= 1) {
        reset();
        return;
    }

    // Welford's online algorithm: single-pass, numerically stable
    double mean = 0.0;
    double m2 = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        ++count;
        const double delta = val - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = val - mean;
        m2 += delta * delta2;
    }

    mean_ = mean;
    // MLE variance: biased N denominator, not N-1, for true log-likelihood maximisation.
    double sd = std::sqrt(m2 / static_cast<double>(count));
    if (sd <= 0.0 || std::isnan(sd) || std::isinf(sd) || sd < precision::MIN_STD_DEV) {
        sd = precision::MIN_STD_DEV;
    }
    standardDeviation_ = sd;
    invalidateCache();
}

void GaussianDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    // Weighted Gaussian MLE for Baum-Welch M-step.
    // Weights are unnormalized γ values; we normalize by their sum.
    const double sumW = [&] {
        double s = 0.0;
        for (const double w : weights)
            s += w;
        return s;
    }();

    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;

    // Weighted Welford for numerical stability
    double mean = 0.0;
    double m2 = 0.0;
    double cumW = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        cumW += weights[i];
        const double delta = data[i] - mean;
        mean += (weights[i] / cumW) * delta;
        const double delta2 = data[i] - mean;
        m2 += weights[i] * delta * delta2;
    }

    mean_ = mean;
    // Population-weighted variance (no Bessel correction for EM)
    double sd = std::sqrt(m2 / sumW);
    if (sd <= 0.0 || std::isnan(sd) || std::isinf(sd) || sd < precision::MIN_STD_DEV) {
        sd = precision::MIN_STD_DEV;
    }
    standardDeviation_ = sd;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (μ = 0.0, σ = 1.0).
 * This corresponds to the standard normal distribution.
 */
void GaussianDistribution::reset() noexcept {
    mean_ = 0.0;
    standardDeviation_ = 1.0;
    invalidateCache();
}

std::string GaussianDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Gaussian Distribution:\n";
    oss << "      μ (mean) = " << mean_ << "\n";
    oss << "      σ (std. deviation) = " << standardDeviation_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::GaussianDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   Gaussian Distribution:
//     \u03bc (mean) = VALUE
//     \u03c3 (std. deviation) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::GaussianDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;           // "Gaussian" "Distribution:"
        is >> s >> s >> s >> t; // "\u03bc" "(mean)" "=" VALUE
        const double mean = std::stod(t);
        is >> s >> s >> s >> s >> t; // "\u03c3" "(std." "deviation)" "=" VALUE
        const double sd = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good())
            distribution.setParameters(mean, sd);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

bool GaussianDistribution::operator==(const GaussianDistribution &other) const {
    using namespace libhmm::constants;
    return std::abs(mean_ - other.mean_) < precision::LIMIT_TOLERANCE &&
           std::abs(standardDeviation_ - other.standardDeviation_) < precision::LIMIT_TOLERANCE;
}

// =============================================================================
// Batch log-PDF — explicit SIMD intrinsics (tier 2)
//
// Free function: takes only plain data — no class access.
// Pattern is deliberately extractable to a separate TU for future runtime
// dispatch without any class or interface changes.
//
// ISA chain (highest available wins; lower paths handle tail elements):
//   AVX-512  8-wide __m512d   (Ryzen 7000, Intel Skylake-X+)
//   AVX/AVX2 4-wide __m256d   (Haswell+, Ivy Bridge with AVX)
//   SSE2     2-wide __m128d   (x86-64 baseline)
//   NEON     2-wide float64x2 (AArch64 — always present)
//   scalar                    (tail + fallback)
//
// NaN inputs yield -Inf output, matching GaussianDistribution::getLogProbability.
// =============================================================================
namespace detail {

void gaussian_logpdf_batch(const double *obs, double *out, std::size_t n, double mean,
                           double neg_half_inv_sigma_sq, double log_norm) noexcept {
    std::size_t i = 0;

#if defined(LIBHMM_HAS_AVX512)
    {
        const __m512d mean_v = _mm512_set1_pd(mean);
        const __m512d lognorm_v = _mm512_set1_pd(log_norm);
        const __m512d scale_v = _mm512_set1_pd(neg_half_inv_sigma_sq);
        const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
        for (; i + 8 <= n; i += 8) {
            __m512d x = _mm512_loadu_pd(obs + i);
            __m512d diff = _mm512_sub_pd(x, mean_v);
            __m512d sq = _mm512_mul_pd(diff, diff);
            __m512d res = _mm512_add_pd(lognorm_v, _mm512_mul_pd(scale_v, sq));
            __mmask8 is_nan = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q); // 1 where NaN
            res = _mm512_mask_blend_pd(is_nan, res, neg_inf_v);       // neg_inf where NaN
            _mm512_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        const __m256d mean_v = _mm256_set1_pd(mean);
        const __m256d lognorm_v = _mm256_set1_pd(log_norm);
        const __m256d scale_v = _mm256_set1_pd(neg_half_inv_sigma_sq);
        const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
        for (; i + 4 <= n; i += 4) {
            __m256d x = _mm256_loadu_pd(obs + i);
            __m256d diff = _mm256_sub_pd(x, mean_v);
            __m256d sq = _mm256_mul_pd(diff, diff);
            __m256d res = _mm256_add_pd(lognorm_v, _mm256_mul_pd(scale_v, sq));
            __m256d is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q); // all-1s where NaN
            res = _mm256_blendv_pd(res, neg_inf_v, is_nan);     // neg_inf where NaN
            _mm256_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        const __m128d mean_v = _mm_set1_pd(mean);
        const __m128d lognorm_v = _mm_set1_pd(log_norm);
        const __m128d scale_v = _mm_set1_pd(neg_half_inv_sigma_sq);
        const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());
        for (; i + 2 <= n; i += 2) {
            __m128d x = _mm_loadu_pd(obs + i);
            __m128d diff = _mm_sub_pd(x, mean_v);
            __m128d sq = _mm_mul_pd(diff, diff);
            __m128d res = _mm_add_pd(lognorm_v, _mm_mul_pd(scale_v, sq));
            // SSE2 has no blendv: use andnot/or to select neg_inf where NaN
            __m128d is_nan = _mm_cmpunord_pd(x, x); // all-1s where NaN
            res = _mm_or_pd(_mm_andnot_pd(is_nan, res), _mm_and_pd(is_nan, neg_inf_v));
            _mm_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        const float64x2_t mean_v = vdupq_n_f64(mean);
        const float64x2_t lognorm_v = vdupq_n_f64(log_norm);
        const float64x2_t scale_v = vdupq_n_f64(neg_half_inv_sigma_sq);
        const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());
        for (; i + 2 <= n; i += 2) {
            float64x2_t x = vld1q_f64(obs + i);
            float64x2_t diff = vsubq_f64(x, mean_v);
            float64x2_t sq = vmulq_f64(diff, diff);
            float64x2_t res = vaddq_f64(lognorm_v, vmulq_f64(scale_v, sq));
            // vceqq_f64(x,x) = all-1s where NOT NaN (NaN != NaN by IEEE)
            // vbslq_f64(mask, a, b): lane = a where mask=1, b where mask=0
            uint64x2_t not_nan = vceqq_f64(x, x);
            res = vbslq_f64(not_nan, res, neg_inf_v); // res if valid, neg_inf if NaN
            vst1q_f64(out + i, res);
        }
    }
#endif

    // Scalar tail: handles remaining elements and platforms without SIMD
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (std::isnan(x) || std::isinf(x))
                     ? neg_inf
                     : log_norm + neg_half_inv_sigma_sq * (x - mean) * (x - mean);
    }
}

} // namespace detail

void GaussianDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                    std::span<double> out) const {
    if (!isCacheValid())
        updateCache();
    const double log_norm = -0.5 * math::LN_2PI - logStandardDeviation_;
    detail::gaussian_logpdf_batch(observations.data(), out.data(), observations.size(), mean_,
                                  negHalfSigmaSquaredInv_, log_norm);
}

std::string GaussianDistribution::to_json() const {
    return json::write_distribution("Gaussian", {{"mu", mean_}, {"sigma", standardDeviation_}});
}
std::unique_ptr<EmissionDistribution> GaussianDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double mu = r.read_double();
    r.read_key();
    const double sigma = r.read_double();
    r.consume('}');
    return std::make_unique<GaussianDistribution>(mu, sigma);
}

} // namespace libhmm
