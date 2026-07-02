// simd_double_ops_avx2.cpp — AVX/AVX2 + FMA (4-wide) kernels.
// Compiled with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC).
// Only included in the build on x86/x86-64 platforms.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h> // AVX/AVX2/FMA
#include <limits>

#include "libhmm/detail/simd_math_helpers.h"

namespace libhmm::performance::detail {
using namespace libhmm::detail::simd; // log_pd, exp_pd, cos_pd, log1p_pd

// gaussian: log_norm + neg_half_inv_sq * (x - mean)^2, NaN/Inf → -Inf.
void gaussian_batch_avx2(const double *obs, double *out, std::size_t n, double mean,
                         double neg_half_inv_sq, double log_norm) noexcept {
    const __m256d mean_v = _mm256_set1_pd(mean);
    const __m256d lognorm_v = _mm256_set1_pd(log_norm);
    const __m256d scale_v = _mm256_set1_pd(neg_half_inv_sq);
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d diff = _mm256_sub_pd(x, mean_v);
        __m256d sq = _mm256_mul_pd(diff, diff);
        __m256d res = _mm256_add_pd(lognorm_v, _mm256_mul_pd(scale_v, sq));
        // NaN mask: _CMP_UNORD_Q gives all-1s where either operand is NaN.
        __m256d is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
        res = _mm256_blendv_pd(res, neg_inf_v, is_nan);
        _mm256_storeu_pd(out + i, res);
    }
    // scalar tail
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (std::isnan(x) || std::isinf(x)) {
            out[i] = neg_inf;
        } else {
            const double d = x - mean;
            out[i] = log_norm + neg_half_inv_sq * d * d;
        }
    }
}

// exponential: log_lambda + neg_lambda * x, x < 0 or NaN → -Inf.
void exponential_batch_avx2(const double *obs, double *out, std::size_t n, double log_lambda,
                            double neg_lambda) noexcept {
    const __m256d loglam_v = _mm256_set1_pd(log_lambda);
    const __m256d neglam_v = _mm256_set1_pd(neg_lambda);
    const __m256d zero_v = _mm256_setzero_pd();
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d res = _mm256_add_pd(loglam_v, _mm256_mul_pd(neglam_v, x));
        __m256d invalid = _mm256_or_pd(_mm256_cmp_pd(x, zero_v, _CMP_LT_OS), // x < 0
                                       _mm256_cmp_pd(x, x, _CMP_UNORD_Q));   // NaN
        res = _mm256_blendv_pd(res, neg_inf_v, invalid);
        _mm256_storeu_pd(out + i, res);
    }
    // scalar tail
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < 0.0 || std::isnan(x)) ? neg_inf : log_lambda + neg_lambda * x;
    }
}

// ============================================================================
// File-private SIMD inline helpers — now sourced from simd_math_helpers.h
// (log_pd, exp_pd, cos_pd, log1p_pd via using namespace libhmm::detail::simd)
// ============================================================================

// Sentinel — avoids empty section warning. Remove when all call sites below
// are confirmed to use the shared overloads (log_pd / exp_pd / cos_pd).
static inline __m256d avx2_log_4pd(__m256d x) noexcept {
    return log_pd(x);
}

// SLEEF-inspired exp, < 1 ULP. Clamped to [−708, 709.8].
static inline __m256d avx2_exp_4pd(__m256d x) noexcept {
    return exp_pd(x);
}

// 7-term Horner cosine with FMA and two-step range reduction.
// Max error ≈ 1×10⁻¹⁰ for |y| ≤ π/2. All finite x.
static inline __m256d avx2_cos_4pd(__m256d x) noexcept {
    return cos_pd(x);
}

// ============================================================================
// Generic math primitives
// ============================================================================

void log_batch_avx2(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, avx2_log_4pd(_mm256_loadu_pd(in + i)));
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double xv = in[i];
        out[i] = (xv <= 0.0) ? neg_inf : std::log(xv);
    }
}

void exp_batch_avx2(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, avx2_exp_4pd(_mm256_loadu_pd(in + i)));
    for (; i < n; ++i)
        out[i] = std::exp(in[i]);
}

void cos_batch_avx2(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, avx2_cos_4pd(_mm256_loadu_pd(in + i)));
    for (; i < n; ++i)
        out[i] = std::cos(in[i]);
}

void log1p_batch_avx2(const double *in, double *out, std::size_t n) noexcept {
    const __m256d one = _mm256_set1_pd(1.0);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, avx2_log_4pd(_mm256_add_pd(_mm256_loadu_pd(in + i), one)));
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double xv = in[i];
        out[i] = (xv <= -1.0) ? neg_inf : std::log1p(xv);
    }
}

// ============================================================================
// Tier-1 distribution kernels
// ============================================================================

// lognormal: −log_norm_const + neg_half_inv_sq·(log(x)−mean_log)²; x≤0 → −∞.
void lognormal_batch_avx2(const double *obs, double *out, std::size_t n, double mean_log,
                          double neg_half_inv_sq, double log_norm_const) noexcept {
    const __m256d mean_v = _mm256_set1_pd(mean_log);
    const __m256d scale_v = _mm256_set1_pd(neg_half_inv_sq);
    const __m256d neg_norm = _mm256_set1_pd(-log_norm_const);
    const __m256d zero_v = _mm256_setzero_pd();
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d invalid = _mm256_cmp_pd(x, zero_v, _CMP_LE_OQ);
        __m256d lx = avx2_log_4pd(x);
        __m256d diff = _mm256_sub_pd(lx, mean_v);
        // res = -lx - log_norm_const + neg_half_inv_sq * d^2
        __m256d res =
            _mm256_sub_pd(_mm256_fmadd_pd(_mm256_mul_pd(diff, diff), scale_v, neg_norm), lx);
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (x <= 0.0) {
            out[i] = neg_inf;
        } else {
            const double lx = std::log(x);
            const double d = lx - mean_log;
            out[i] = -lx - log_norm_const + neg_half_inv_sq * d * d;
        }
    }
}

// gamma: const_term + k_minus_1·log(x) − x·inv_theta; x≤0 → −∞.
void gamma_batch_avx2(const double *obs, double *out, std::size_t n, double k_minus_1,
                      double inv_theta, double const_term) noexcept {
    const __m256d km1_v = _mm256_set1_pd(k_minus_1);
    const __m256d invth_v = _mm256_set1_pd(inv_theta);
    const __m256d cterm_v = _mm256_set1_pd(const_term);
    const __m256d zero_v = _mm256_setzero_pd();
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d invalid = _mm256_cmp_pd(x, zero_v, _CMP_LE_OQ);
        __m256d lx = avx2_log_4pd(x);
        __m256d res = _mm256_fmadd_pd(km1_v, lx, cterm_v);
        res = _mm256_fnmadd_pd(invth_v, x, res);
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : const_term + k_minus_1 * std::log(x) - x * inv_theta;
    }
}

// chisq: const_term + half_k_minus_1·log(x) − x·0.5; x≤0 → −∞.
void chisq_batch_avx2(const double *obs, double *out, std::size_t n, double half_k_minus_1,
                      double const_term) noexcept {
    const __m256d hkm1_v = _mm256_set1_pd(half_k_minus_1);
    const __m256d half_v = _mm256_set1_pd(0.5);
    const __m256d cterm_v = _mm256_set1_pd(const_term);
    const __m256d zero_v = _mm256_setzero_pd();
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d invalid = _mm256_cmp_pd(x, zero_v, _CMP_LE_OQ);
        __m256d lx = avx2_log_4pd(x);
        __m256d res = _mm256_fmadd_pd(hkm1_v, lx, cterm_v);
        res = _mm256_fnmadd_pd(half_v, x, res);
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : const_term + half_k_minus_1 * std::log(x) - x * 0.5;
    }
}

// rayleigh: log_norm + log(x) − x²·inv2sigma_sq; x≤0 → −∞.
void rayleigh_batch_avx2(const double *obs, double *out, std::size_t n, double inv2sigma_sq,
                         double log_norm) noexcept {
    const __m256d inv2sig_v = _mm256_set1_pd(inv2sigma_sq);
    const __m256d lognorm_v = _mm256_set1_pd(log_norm);
    const __m256d zero_v = _mm256_setzero_pd();
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d invalid = _mm256_cmp_pd(x, zero_v, _CMP_LE_OQ);
        __m256d lx = avx2_log_4pd(x);
        __m256d res = _mm256_add_pd(lognorm_v, lx);
        res = _mm256_fnmadd_pd(inv2sig_v, _mm256_mul_pd(x, x), res);
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : log_norm + std::log(x) - x * x * inv2sigma_sq;
    }
}

// pareto: log_norm − k_plus_1·log(x); x<xm → −∞.
void pareto_batch_avx2(const double *obs, double *out, std::size_t n, double k_plus_1, double xm,
                       double log_norm) noexcept {
    const __m256d kp1_v = _mm256_set1_pd(k_plus_1);
    const __m256d xm_v = _mm256_set1_pd(xm);
    const __m256d lognorm_v = _mm256_set1_pd(log_norm);
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d invalid = _mm256_cmp_pd(x, xm_v, _CMP_LT_OQ);
        __m256d res = _mm256_fnmadd_pd(kp1_v, avx2_log_4pd(x), lognorm_v);
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < xm) ? neg_inf : log_norm - k_plus_1 * std::log(x);
    }
}

// weibull: log_norm + (k−1)·log(x) − exp(k·log(x) + neg_k_log_lambda); x≤0 → −∞.
void weibull_batch_avx2(const double *obs, double *out, std::size_t n, double k_minus_1, double k,
                        double log_norm, double neg_k_log_lambda) noexcept {
    const __m256d km1_v = _mm256_set1_pd(k_minus_1);
    const __m256d k_v = _mm256_set1_pd(k);
    const __m256d lognorm_v = _mm256_set1_pd(log_norm);
    const __m256d nkll_v = _mm256_set1_pd(neg_k_log_lambda);
    const __m256d zero_v = _mm256_setzero_pd();
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d invalid = _mm256_cmp_pd(x, zero_v, _CMP_LE_OQ);
        __m256d lx = avx2_log_4pd(x);
        __m256d pow_term = avx2_exp_4pd(_mm256_fmadd_pd(k_v, lx, nkll_v));
        __m256d res = _mm256_sub_pd(_mm256_fmadd_pd(km1_v, lx, lognorm_v), pow_term);
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (x <= 0.0) {
            out[i] = neg_inf;
        } else {
            const double lx = std::log(x);
            out[i] = log_norm + k_minus_1 * lx - std::exp(k * lx + neg_k_log_lambda);
        }
    }
}

// beta: neg_log_beta + (α−1)·log(x) + (β−1)·log(1−x); x∉(0,1) → −∞.
void beta_batch_avx2(const double *obs, double *out, std::size_t n, double alpha_minus_1,
                     double beta_minus_1, double neg_log_beta) noexcept {
    const __m256d am1_v = _mm256_set1_pd(alpha_minus_1);
    const __m256d bm1_v = _mm256_set1_pd(beta_minus_1);
    const __m256d nlb_v = _mm256_set1_pd(neg_log_beta);
    const __m256d zero_v = _mm256_setzero_pd();
    const __m256d one_v = _mm256_set1_pd(1.0);
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d invalid = _mm256_or_pd(
            _mm256_or_pd(_mm256_cmp_pd(x, zero_v, _CMP_LE_OQ), _mm256_cmp_pd(x, one_v, _CMP_GE_OQ)),
            _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
        __m256d lx = avx2_log_4pd(x);
        __m256d l1mx = avx2_log_4pd(_mm256_sub_pd(one_v, x));
        __m256d res = _mm256_fmadd_pd(bm1_v, l1mx, _mm256_fmadd_pd(am1_v, lx, nlb_v));
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (x <= 0.0 || x >= 1.0) {
            out[i] = neg_inf;
        } else {
            out[i] = neg_log_beta + alpha_minus_1 * std::log(x) + beta_minus_1 * std::log1p(-x);
        }
    }
}

// student_t: log_norm − half_nu_plus_1·log1p(((x−loc)·inv_scale)²·inv_nu); NaN/Inf → −∞.
void student_t_batch_avx2(const double *obs, double *out, std::size_t n, double location,
                          double inv_scale, double half_nu_plus_1, double log_norm,
                          double inv_nu) noexcept {
    const __m256d loc_v = _mm256_set1_pd(location);
    const __m256d iscale_v = _mm256_set1_pd(inv_scale);
    const __m256d hnp1_v = _mm256_set1_pd(half_nu_plus_1);
    const __m256d lognorm_v = _mm256_set1_pd(log_norm);
    const __m256d invnu_v = _mm256_set1_pd(inv_nu);
    const __m256d one_v = _mm256_set1_pd(1.0);
    const __m256d pos_inf_v = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d abs_x = _mm256_andnot_pd(sign_mask, x);
        __m256d invalid = _mm256_or_pd(_mm256_cmp_pd(x, x, _CMP_UNORD_Q),
                                       _mm256_cmp_pd(abs_x, pos_inf_v, _CMP_EQ_OQ));
        __m256d t = _mm256_mul_pd(_mm256_sub_pd(x, loc_v), iscale_v);
        __m256d t2nu = _mm256_mul_pd(_mm256_mul_pd(t, t), invnu_v);
        __m256d res = _mm256_fnmadd_pd(hnp1_v, avx2_log_4pd(_mm256_add_pd(one_v, t2nu)), lognorm_v);
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (std::isnan(x) || std::isinf(x)) {
            out[i] = neg_inf;
        } else {
            const double tv = (x - location) * inv_scale;
            out[i] = log_norm - half_nu_plus_1 * std::log1p(tv * tv * inv_nu);
        }
    }
}

// vonmises: κ·cos(x−μ) − log_normaliser; NaN/Inf → −∞.
void vonmises_batch_avx2(const double *obs, double *out, std::size_t n, double mu, double kappa,
                         double log_normaliser) noexcept {
    const __m256d mu_v = _mm256_set1_pd(mu);
    const __m256d kappa_v = _mm256_set1_pd(kappa);
    const __m256d neg_ln_v = _mm256_set1_pd(-log_normaliser);
    const __m256d pos_inf_v = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    const __m256d neg_inf_v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(obs + i);
        __m256d abs_x = _mm256_andnot_pd(sign_mask, x);
        __m256d invalid = _mm256_or_pd(_mm256_cmp_pd(x, x, _CMP_UNORD_Q),
                                       _mm256_cmp_pd(abs_x, pos_inf_v, _CMP_EQ_OQ));
        __m256d res = _mm256_fmadd_pd(kappa_v, avx2_cos_4pd(_mm256_sub_pd(x, mu_v)), neg_ln_v);
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(res, neg_inf_v, invalid));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (std::isnan(x) || std::isinf(x)) {
            out[i] = neg_inf;
        } else {
            out[i] = kappa * std::cos(x - mu) - log_normaliser;
        }
    }
}

} // namespace libhmm::performance::detail

#endif // x86 guard
