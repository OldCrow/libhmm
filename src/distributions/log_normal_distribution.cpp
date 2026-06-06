#include "libhmm/distributions/log_normal_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/performance/simd_kernels_internal.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>   // For std::accumulate (not in common.h)
#include <algorithm> // For std::for_each (exists in common.h, included for clarity)

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Log-Normal distribution.
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
double LogNormalDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x) || x <= math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;
    if (!isCacheValid())
        updateCache();

    // Use direct PDF calculation for better performance
    // f(x) = 1/(x*σ*√(2π)) * exp(-½*((ln(x)-μ)/σ)²)
    // Optimize by using cached values and avoiding repeated calculations

    const double logX = std::log(x);
    const double delta = logX - mean_;

    // Use cached values: negHalfSigmaSquaredInv_ = -1/(2σ²), logNormalizationConstant_ = ln(σ√(2π))
    // log f(x) = -ln(x) - logNormalizationConstant_ - ((ln(x)-μ)^2)/(2σ²)
    const double logPdf =
        -logX - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * delta * delta;

    const double pdf = std::exp(logPdf);

    // Ensure numerical stability
    if (std::isnan(pdf) || pdf < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    return pdf;
}

double LogNormalDistribution::getLogProbability(double value) const noexcept {
    // Log-normal distribution is only defined for positive values
    if (value <= 0.0 || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    if (!isCacheValid())
        updateCache();
    const double logX = std::log(value);
    const double delta = logX - mean_;
    return -logX - logNormalizationConstant_ + negHalfSigmaSquaredInv_ * delta * delta;
}

double LogNormalDistribution::getCumulativeProbability(double value) const noexcept {
    // Handle boundary cases
    if (value <= 0.0) {
        return 0.0;
    }
    if (std::isnan(value) || std::isinf(value)) {
        return (std::isinf(value) && value > 0.0) ? 1.0 : 0.0;
    }

    // CDF: F(x) = ½(1 + erf((ln(x)-μ)/(σ√2)))
    double logX = std::log(value);
    double standardized = (logX - mean_) / (standardDeviation_ * std::sqrt(2.0));

    return 0.5 * (1.0 + std::erf(standardized));
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 *
 * For Log-Normal distribution, the MLE estimators are:
 * μ = mean(ln(x_i)) for positive x_i
 * σ = std_dev(ln(x_i)) for positive x_i
 *
 * Only positive values are used since Log-Normal distribution has support (0, ∞).
 *
 * @param values Vector of observed data points
 */
double LogNormalDistribution::sample(std::mt19937_64& rng) const {
    // std::lognormal_distribution<double>(m, s): m = log-mean, s = log-stddev.
    std::lognormal_distribution<double> dist(mean_, standardDeviation_);
    return dist(rng);
}

void LogNormalDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }
    double mean = 0.0, M2 = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        if (val > 0.0 && std::isfinite(val)) {
            ++count;
            const double logVal = std::log(val);
            const double delta = logVal - mean;
            mean += delta / static_cast<double>(count);
            M2 += delta * (logVal - mean);
        }
    }
    if (count == 0) {
        reset();
        return;
    }
    if (count == 1) {
        mean_ = mean;
        standardDeviation_ = precision::LIMIT_TOLERANCE;
        invalidateCache();
        return;
    }
    // MLE variance: biased N denominator, not N-1, for true log-likelihood maximisation.
    const double stddev = std::sqrt(M2 / static_cast<double>(count));
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= precision::ZERO) {
        reset();
        return;
    }
    mean_ = mean;
    standardDeviation_ = stddev;
    invalidateCache();
}

void LogNormalDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;
    double mean = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            mean += (weights[i] / sumW) * std::log(data[i]);
    double var = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0) {
            const double d = std::log(data[i]) - mean;
            var += weights[i] * d * d;
        }
    var /= sumW;
    const double stddev = std::sqrt(var);
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= precision::ZERO) {
        reset();
        return;
    }
    mean_ = mean;
    standardDeviation_ = stddev;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (μ = 0.0, σ = 1.0).
 * This corresponds to the standard log-normal distribution.
 */
void LogNormalDistribution::reset() noexcept {
    mean_ = 0.0;
    standardDeviation_ = 1.0;
    invalidateCache();
}

std::string LogNormalDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "LogNormal Distribution:\n";
    oss << "      μ (log mean) = " << mean_ << "\n";
    oss << "      σ (log std. deviation) = " << standardDeviation_ << "\n";
    oss << "      Mean = " << getDistributionMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::LogNormalDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   LogNormal Distribution:
//     \u03bc (log mean) = VALUE
//     \u03c3 (log std. deviation) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::LogNormalDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "LogNormal" "Distribution:"
        is >> s >> s >> s >> s >> t; // "\u03bc" "(log" "mean)" "=" VALUE
        const double mean = std::stod(t);
        is >> s >> s >> s >> s >> s >> t; // "\u03c3" "(log" "std." "deviation)" "=" VALUE
        const double sd = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good()) {
            distribution.setMean(mean);
            distribution.setStandardDeviation(sd);
        }
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}
// =============================================================================
// Batch log-PDF — explicit SIMD intrinsics (tier 2)
//
// Formula: log f(x) = -log(x) - logNormConst + negHalfInvSigma2*(log(x)-mu)^2
// Per element: log_x = log(x); then result = -log_x - C + S*(log_x - mu)^2
// where C = logNormalizationConstant_, S = negHalfSigmaSquaredInv_.
//
// x <= 0 lanes: log(x) is -inf; guard produces -inf output.
// Pattern mirrors gaussian_logpdf_batch (gaussian_distribution.cpp).
// =============================================================================
namespace detail {

void lognormal_logpdf_batch(const double *obs, double *out, std::size_t n, double mu, double S,
                            double C) noexcept {
    using namespace performance::detail::kernels;
    std::size_t i = 0;
    const double neg_inf = -std::numeric_limits<double>::infinity();

#if defined(LIBHMM_HAS_AVX512)
    {
        const __m512d vmu = _mm512_set1_pd(mu);
        const __m512d vS = _mm512_set1_pd(S);
        const __m512d vC = _mm512_set1_pd(C);
        for (; i + 8 <= n; i += 8) {
            __m512d x = _mm512_loadu_pd(obs + i);
            __m512d lx = k_log_pd_avx512(x);    // -inf where x<=0
            __m512d d = _mm512_sub_pd(lx, vmu); // log(x) - mu
            __m512d res = _mm512_fmadd_pd(
                d, _mm512_mul_pd(d, vS),
                _mm512_sub_pd(_mm512_setzero_pd(), _mm512_add_pd(lx, vC))); // -lx - C + S*d^2
            _mm512_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        const __m256d vmu = _mm256_set1_pd(mu);
        const __m256d vS = _mm256_set1_pd(S);
        const __m256d vC = _mm256_set1_pd(C);
        for (; i + 4 <= n; i += 4) {
            __m256d x = _mm256_loadu_pd(obs + i);
            __m256d lx = k_log_pd_avx(x);
            __m256d d = _mm256_sub_pd(lx, vmu);
            __m256d res = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(d, d), vS),
                                        _mm256_sub_pd(_mm256_setzero_pd(), _mm256_add_pd(lx, vC)));
            _mm256_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        const __m128d vmu = _mm_set1_pd(mu);
        const __m128d vS = _mm_set1_pd(S);
        const __m128d vC = _mm_set1_pd(C);
        for (; i + 2 <= n; i += 2) {
            __m128d x = _mm_loadu_pd(obs + i);
            __m128d lx = k_log_pd_sse2(x);
            __m128d d = _mm_sub_pd(lx, vmu);
            __m128d res = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(d, d), vS),
                                     _mm_sub_pd(_mm_setzero_pd(), _mm_add_pd(lx, vC)));
            _mm_storeu_pd(out + i, res);
        }
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        const float64x2_t vmu = vdupq_n_f64(mu);
        const float64x2_t vS = vdupq_n_f64(S);
        const float64x2_t vC = vdupq_n_f64(C);
        for (; i + 2 <= n; i += 2) {
            float64x2_t x = vld1q_f64(obs + i);
            float64x2_t lx = k_log_pd_neon(x);
            float64x2_t d = vsubq_f64(lx, vmu);
            // res = S*d^2 + (-lx - C) = S*d^2 - lx - C
            float64x2_t res = vfmaq_f64(vsubq_f64(vnegq_f64(lx), vC), vmulq_f64(d, d), vS);
            vst1q_f64(out + i, res);
        }
    }
#endif

    // Scalar tail.
    for (; i < n; ++i) {
        const double x = obs[i];
        if (x <= 0.0 || std::isnan(x) || std::isinf(x)) {
            out[i] = neg_inf;
        } else {
            const double lx = std::log(x);
            const double d = lx - mu;
            out[i] = -lx - C + S * d * d;
        }
    }
}

} // namespace detail

void LogNormalDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                     std::span<double> out) const {
    // Tier 2 — explicit SIMD via simd_kernels_internal.h
    if (!isCacheValid())
        updateCache();
    detail::lognormal_logpdf_batch(observations.data(), out.data(), observations.size(), mean_,
                                   negHalfSigmaSquaredInv_, logNormalizationConstant_);
}

std::string LogNormalDistribution::to_json() const {
    return json::write_distribution("LogNormal", {{"mu", mean_}, {"sigma", standardDeviation_}});
}
std::unique_ptr<EmissionDistribution> LogNormalDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double mu = r.read_double();
    r.read_key();
    const double sigma = r.read_double();
    r.consume('}');
    return std::make_unique<LogNormalDistribution>(mu, sigma);
}

} // namespace libhmm
