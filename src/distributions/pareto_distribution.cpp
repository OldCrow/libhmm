#include "libhmm/distributions/pareto_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/performance/simd_kernels_internal.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>   // For std::accumulate (not in common.h)
#include <algorithm> // For std::min_element (exists in common.h, included for clarity)
#include <cfloat>    // For FLT_* constants (not in common.h)

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Pareto distribution.
 *
 * For Pareto distribution: f(x) = (k * x_m^k) / x^(k+1) for x ≥ x_m
 *
 * Uses direct PDF calculation for optimal performance, avoiding expensive CDF differences.
 *
 * @param x The value at which to evaluate the probability density
 * @return Probability density for the given value
 */
double ParetoDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x) || x < xm_)
        return math::ZERO_DOUBLE;
    if (!isCacheValid())
        updateCache();

    // Direct PDF calculation: f(x) = (k * x_m^k) / x^(k+1)
    // Using cached kXmPowK_ = k * x_m^k and kPlus1_ = k + 1 for efficiency
    const double pdf = kXmPowK_ / std::pow(x, kPlus1_);

    // Ensure numerical stability
    if (std::isnan(pdf) || pdf < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    return pdf;
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 *
 * For Pareto distribution: log(f(x)) = log(k) + k*log(x_m) - (k+1)*log(x) for x ≥ x_m
 *
 * @param value The value at which to evaluate the log-PDF
 * @return Natural logarithm of the probability density, or -∞ for invalid values
 */
double ParetoDistribution::getLogProbability(double value) const noexcept {
    if (std::isnan(value) || std::isinf(value) || value < xm_) {
        return -std::numeric_limits<double>::infinity();
    }

    if (!isCacheValid())
        updateCache();
    return logK_ + kLogXm_ - kPlus1_ * std::log(value);
}

double ParetoDistribution::getCumulativeProbability(double value) const noexcept {
    if (std::isnan(value) || value < xm_)
        return math::ZERO_DOUBLE;
    if (!isCacheValid())
        updateCache();

    return math::ONE - std::pow(xm_ / value, k_);
}

/**
 * Evaluates the CDF for the Pareto distribution at x.
 *
 * Formula: F(x) = 1 - (x_m/x)^k for x ≥ x_m
 *
 * @param x The value at which to evaluate the CDF
 * @return Cumulative probability P(X ≤ x)
 */
double ParetoDistribution::CDF(double x) const noexcept {
    return getCumulativeProbability(x);
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 *
 * For Pareto distribution, the MLE estimators are:
 * x_m = min(x_i) for all i
 * k = n / Σ(ln(x_i) - ln(x_m)) for i = 1 to n
 *
 * @param values Vector of observed data
 */
void ParetoDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }
    double minVal = *std::min_element(data.begin(), data.end());
    if (minVal <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    double sumLog = 0.0;
    for (const double val : data)
        if (val > math::ZERO_DOUBLE)
            sumLog += std::log(val) - std::log(minVal);
    if (sumLog <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    xm_ = minVal;
    k_ = static_cast<double>(data.size()) / sumLog;
    if (!std::isfinite(k_) || k_ <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    invalidateCache();
}

void ParetoDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    double sumW = 0.0;
    for (const double w : weights)
        sumW += w;
    if (sumW < precision::ZERO || std::isnan(sumW)) {
        reset();
        return;
    }
    double minVal = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            minVal = std::min(minVal, data[i]);
    if (minVal <= math::ZERO_DOUBLE || !std::isfinite(minVal)) {
        reset();
        return;
    }
    double sumWLog = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            sumWLog += weights[i] * (std::log(data[i]) - std::log(minVal));
    if (sumWLog <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    xm_ = minVal;
    k_ = sumW / sumWLog;
    if (!std::isfinite(k_) || k_ <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (k = 1.0, x_m = 1.0).
 * This corresponds to a standard Pareto distribution.
 */
void ParetoDistribution::reset() noexcept {
    k_ = math::ONE;
    xm_ = math::ONE;
    invalidateCache();
}

std::string ParetoDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Pareto Distribution:\n";
    oss << "      k (shape parameter) = " << k_ << "\n";
    oss << "      x_m (scale parameter) = " << xm_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::ParetoDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   Pareto Distribution:
//     k (shape parameter) = VALUE
//     x_m (scale parameter) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::ParetoDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "Pareto" "Distribution:"
        is >> s >> s >> s >> s >> t; // "k" "(shape" "parameter)" "=" VALUE
        const double k = std::stod(t);
        is >> s >> s >> s >> s >> t; // "x_m" "(scale" "parameter)" "=" VALUE
        const double xm = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good()) {
            distribution.setK(k);
            distribution.setXm(xm);
        }
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

// =============================================================================
// Batch log-PDF — explicit SIMD intrinsics (tier 2)
//
// Formula: log f(x) = logK + kLogXm - kPlus1 * log(x)  for x >= xm
//                   = -inf                               for x < xm
// =============================================================================
namespace detail {

void pareto_logpdf_batch(const double *obs, double *out, std::size_t n, double xm,
                         double logK_plus_kLogXm, double kPlus1) noexcept {
    using namespace performance::detail::kernels;
    std::size_t i = 0;
    const double neg_inf = -std::numeric_limits<double>::infinity();

#if defined(LIBHMM_HAS_AVX512)
    {
        const __m512d vxm = _mm512_set1_pd(xm);
        const __m512d vconst = _mm512_set1_pd(logK_plus_kLogXm);
        const __m512d vkp1 = _mm512_set1_pd(kPlus1);
        const __m512d vneg_inf = _mm512_set1_pd(neg_inf);
        for (; i + 8 <= n; i += 8) {
            __m512d x = _mm512_loadu_pd(obs + i);
            // x < xm: -inf
            __mmask8 invalid = _mm512_cmp_pd_mask(x, vxm, _CMP_LT_OS);
            __m512d lx = k_log_pd_avx512(x);
            __m512d res = _mm512_fnmadd_pd(vkp1, lx, vconst); // const - kp1*log(x)
            res = _mm512_mask_blend_pd(invalid, res, vneg_inf);
            _mm512_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        const __m256d vxm = _mm256_set1_pd(xm);
        const __m256d vconst = _mm256_set1_pd(logK_plus_kLogXm);
        const __m256d vkp1 = _mm256_set1_pd(kPlus1);
        const __m256d vneg_inf = _mm256_set1_pd(neg_inf);
        for (; i + 4 <= n; i += 4) {
            __m256d x = _mm256_loadu_pd(obs + i);
            __m256d inv = _mm256_cmp_pd(x, vxm, _CMP_LT_OS); // all-1s where x < xm
            __m256d lx = k_log_pd_avx(x);
            __m256d res = _mm256_sub_pd(vconst, _mm256_mul_pd(vkp1, lx));
            res = _mm256_blendv_pd(res, vneg_inf, inv);
            _mm256_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        const __m128d vxm = _mm_set1_pd(xm);
        const __m128d vconst = _mm_set1_pd(logK_plus_kLogXm);
        const __m128d vkp1 = _mm_set1_pd(kPlus1);
        const __m128d vneg_inf = _mm_set1_pd(neg_inf);
        for (; i + 2 <= n; i += 2) {
            __m128d x = _mm_loadu_pd(obs + i);
            __m128d inv = _mm_cmplt_pd(x, vxm);
            __m128d lx = k_log_pd_sse2(x);
            __m128d res = _mm_sub_pd(vconst, _mm_mul_pd(vkp1, lx));
            res = _mm_or_pd(_mm_andnot_pd(inv, res), _mm_and_pd(inv, vneg_inf));
            _mm_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        const float64x2_t vxm = vdupq_n_f64(xm);
        const float64x2_t vconst = vdupq_n_f64(logK_plus_kLogXm);
        const float64x2_t vkp1 = vdupq_n_f64(kPlus1);
        const float64x2_t vneg_inf = vdupq_n_f64(neg_inf);
        for (; i + 2 <= n; i += 2) {
            float64x2_t x = vld1q_f64(obs + i);
            uint64x2_t inv = vcltq_f64(x, vxm); // x < xm
            float64x2_t lx = k_log_pd_neon(x);
            float64x2_t res = vsubq_f64(vconst, vmulq_f64(vkp1, lx));
            res = vbslq_f64(inv, vneg_inf, res);
            vst1q_f64(out + i, res);
        }
    }
#endif

    // Scalar tail.
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (std::isnan(x) || std::isinf(x) || x < xm)
                     ? neg_inf
                     : logK_plus_kLogXm - kPlus1 * std::log(x);
    }
}

} // namespace detail

void ParetoDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                  std::span<double> out) const {
    // Tier 2 — explicit SIMD via simd_kernels_internal.h
    if (!isCacheValid())
        updateCache();
    // logK_ + kLogXm_ is a single scalar constant — compute once.
    detail::pareto_logpdf_batch(observations.data(), out.data(), observations.size(), xm_,
                                logK_ + kLogXm_, kPlus1_);
}

std::string ParetoDistribution::to_json() const {
    return json::write_distribution("Pareto", {{"k", k_}, {"xm", xm_}});
}
std::unique_ptr<EmissionDistribution> ParetoDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double k = r.read_double();
    r.read_key();
    const double xm = r.read_double();
    r.consume('}');
    return std::make_unique<ParetoDistribution>(k, xm);
}

} // namespace libhmm
