// simd_double_ops_avx512.cpp — AVX-512F/DQ (8-wide) kernels.
// Compiled with -mavx512f -mavx512dq (GCC/Clang) or /arch:AVX512 (MSVC).
// Only included in the build on x86/x86-64 platforms with AVX-512 support.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cmath>
#include <cstddef>
#include <immintrin.h> // AVX-512
#include <limits>

#include "libhmm/detail/simd_math_helpers.h"

namespace libhmm::performance::detail {
using namespace libhmm::detail::simd; // log_pd, exp_pd, cos_pd, log1p_pd

// gaussian: log_norm + neg_half_inv_sq * (x - mean)^2, NaN/Inf → -Inf.
void gaussian_batch_avx512(const double *obs, double *out, std::size_t n, double mean,
                           double neg_half_inv_sq, double log_norm) noexcept {
    const __m512d mean_v = _mm512_set1_pd(mean);
    const __m512d lognorm_v = _mm512_set1_pd(log_norm);
    const __m512d scale_v = _mm512_set1_pd(neg_half_inv_sq);
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __m512d diff = _mm512_sub_pd(x, mean_v);
        __m512d sq = _mm512_mul_pd(diff, diff);
        __m512d res = _mm512_add_pd(lognorm_v, _mm512_mul_pd(scale_v, sq));
        // _CMP_UNORD_Q: 1 where NaN; mask_blend replaces those lanes with neg_inf.
        __mmask8 is_nan = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
        res = _mm512_mask_blend_pd(is_nan, res, neg_inf_v);
        _mm512_storeu_pd(out + i, res);
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
void exponential_batch_avx512(const double *obs, double *out, std::size_t n, double log_lambda,
                              double neg_lambda) noexcept {
    const __m512d loglam_v = _mm512_set1_pd(log_lambda);
    const __m512d neglam_v = _mm512_set1_pd(neg_lambda);
    const __m512d zero_v = _mm512_setzero_pd();
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __m512d res = _mm512_add_pd(loglam_v, _mm512_mul_pd(neglam_v, x));
        __mmask8 invalid = _mm512_cmp_pd_mask(x, zero_v, _CMP_LT_OS) // x < 0
                           | _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q); // NaN
        res = _mm512_mask_blend_pd(invalid, res, neg_inf_v);
        _mm512_storeu_pd(out + i, res);
    }
    // scalar tail
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < 0.0 || std::isnan(x)) ? neg_inf : log_lambda + neg_lambda * x;
    }
}

// ============================================================================
// File-private SIMD inline helpers (not exported)
// ============================================================================

// SLEEF xlog_u1 core, < 1 ULP. x≤0 → −∞, NaN → NaN, +∞ → +∞.
// Uses _mm512_cvtepi64_pd (AVX-512DQ) for clean int64→double conversion.
static inline __m512d avx512_log_8pd(__m512d x) noexcept {
    return log_pd(x);
}

// SLEEF-inspired exp, < 1 ULP. Clamped to [−708, 709.8].
// Uses _mm512_roundscale_pd and _mm512_cvtpd_epi32 + _mm512_cvtepi32_epi64 (AVX-512F).
static inline __m512d avx512_exp_8pd(__m512d x) noexcept {
    return exp_pd(x);
}

// 7-term Horner cosine with FMA and two-step range reduction.
// Max error ≈ 1×10⁻¹⁰ for |y| ≤ π/2. All finite x.
static inline __m512d avx512_cos_8pd(__m512d x) noexcept {
    return cos_pd(x);
}

// ============================================================================
// Generic math primitives
// ============================================================================

void log_batch_avx512(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(out + i, avx512_log_8pd(_mm512_loadu_pd(in + i)));
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double xv = in[i];
        out[i] = (xv <= 0.0) ? neg_inf : std::log(xv);
    }
}

void exp_batch_avx512(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(out + i, avx512_exp_8pd(_mm512_loadu_pd(in + i)));
    for (; i < n; ++i)
        out[i] = std::exp(in[i]);
}

void cos_batch_avx512(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(out + i, avx512_cos_8pd(_mm512_loadu_pd(in + i)));
    for (; i < n; ++i)
        out[i] = std::cos(in[i]);
}

void log1p_batch_avx512(const double *in, double *out, std::size_t n) noexcept {
    const __m512d one = _mm512_set1_pd(1.0);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(out + i, avx512_log_8pd(_mm512_add_pd(_mm512_loadu_pd(in + i), one)));
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
void lognormal_batch_avx512(const double *obs, double *out, std::size_t n, double mean_log,
                            double neg_half_inv_sq, double log_norm_const) noexcept {
    const __m512d mean_v = _mm512_set1_pd(mean_log);
    const __m512d scale_v = _mm512_set1_pd(neg_half_inv_sq);
    const __m512d neg_norm = _mm512_set1_pd(-log_norm_const);
    const __m512d zero_v = _mm512_setzero_pd();
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __mmask8 invalid = _mm512_cmp_pd_mask(x, zero_v, _CMP_LE_OQ);
        __m512d lx = avx512_log_8pd(x);
        __m512d diff = _mm512_sub_pd(lx, mean_v);
        // res = -lx - log_norm_const + neg_half_inv_sq * d^2
        __m512d res =
            _mm512_sub_pd(_mm512_fmadd_pd(_mm512_mul_pd(diff, diff), scale_v, neg_norm), lx);
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
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
void gamma_batch_avx512(const double *obs, double *out, std::size_t n, double k_minus_1,
                        double inv_theta, double const_term) noexcept {
    const __m512d km1_v = _mm512_set1_pd(k_minus_1);
    const __m512d invth_v = _mm512_set1_pd(inv_theta);
    const __m512d cterm_v = _mm512_set1_pd(const_term);
    const __m512d zero_v = _mm512_setzero_pd();
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __mmask8 invalid = _mm512_cmp_pd_mask(x, zero_v, _CMP_LE_OQ);
        __m512d lx = avx512_log_8pd(x);
        __m512d res = _mm512_fmadd_pd(km1_v, lx, cterm_v);
        res = _mm512_fnmadd_pd(invth_v, x, res);
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : const_term + k_minus_1 * std::log(x) - x * inv_theta;
    }
}

// chisq: const_term + half_k_minus_1·log(x) − x·0.5; x≤0 → −∞.
void chisq_batch_avx512(const double *obs, double *out, std::size_t n, double half_k_minus_1,
                        double const_term) noexcept {
    const __m512d hkm1_v = _mm512_set1_pd(half_k_minus_1);
    const __m512d half_v = _mm512_set1_pd(0.5);
    const __m512d cterm_v = _mm512_set1_pd(const_term);
    const __m512d zero_v = _mm512_setzero_pd();
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __mmask8 invalid = _mm512_cmp_pd_mask(x, zero_v, _CMP_LE_OQ);
        __m512d lx = avx512_log_8pd(x);
        __m512d res = _mm512_fmadd_pd(hkm1_v, lx, cterm_v);
        res = _mm512_fnmadd_pd(half_v, x, res);
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : const_term + half_k_minus_1 * std::log(x) - x * 0.5;
    }
}

// rayleigh: log_norm + log(x) − x²·inv2sigma_sq; x≤0 → −∞.
void rayleigh_batch_avx512(const double *obs, double *out, std::size_t n, double inv2sigma_sq,
                           double log_norm) noexcept {
    const __m512d inv2sig_v = _mm512_set1_pd(inv2sigma_sq);
    const __m512d lognorm_v = _mm512_set1_pd(log_norm);
    const __m512d zero_v = _mm512_setzero_pd();
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __mmask8 invalid = _mm512_cmp_pd_mask(x, zero_v, _CMP_LE_OQ);
        __m512d lx = avx512_log_8pd(x);
        __m512d res = _mm512_add_pd(lognorm_v, lx);
        res = _mm512_fnmadd_pd(inv2sig_v, _mm512_mul_pd(x, x), res);
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : log_norm + std::log(x) - x * x * inv2sigma_sq;
    }
}

// pareto: log_norm − k_plus_1·log(x); x<xm → −∞.
void pareto_batch_avx512(const double *obs, double *out, std::size_t n, double k_plus_1, double xm,
                         double log_norm) noexcept {
    const __m512d kp1_v = _mm512_set1_pd(k_plus_1);
    const __m512d xm_v = _mm512_set1_pd(xm);
    const __m512d lognorm_v = _mm512_set1_pd(log_norm);
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __mmask8 invalid = _mm512_cmp_pd_mask(x, xm_v, _CMP_LT_OQ);
        __m512d res = _mm512_fnmadd_pd(kp1_v, avx512_log_8pd(x), lognorm_v);
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < xm) ? neg_inf : log_norm - k_plus_1 * std::log(x);
    }
}

// weibull: log_norm + (k−1)·log(x) − exp(k·log(x) + neg_k_log_lambda); x≤0 → −∞.
void weibull_batch_avx512(const double *obs, double *out, std::size_t n, double k_minus_1, double k,
                          double log_norm, double neg_k_log_lambda) noexcept {
    const __m512d km1_v = _mm512_set1_pd(k_minus_1);
    const __m512d k_v = _mm512_set1_pd(k);
    const __m512d lognorm_v = _mm512_set1_pd(log_norm);
    const __m512d nkll_v = _mm512_set1_pd(neg_k_log_lambda);
    const __m512d zero_v = _mm512_setzero_pd();
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __mmask8 invalid = _mm512_cmp_pd_mask(x, zero_v, _CMP_LE_OQ);
        __m512d lx = avx512_log_8pd(x);
        __m512d pow_term = avx512_exp_8pd(_mm512_fmadd_pd(k_v, lx, nkll_v));
        __m512d res = _mm512_sub_pd(_mm512_fmadd_pd(km1_v, lx, lognorm_v), pow_term);
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
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
void beta_batch_avx512(const double *obs, double *out, std::size_t n, double alpha_minus_1,
                       double beta_minus_1, double neg_log_beta) noexcept {
    const __m512d am1_v = _mm512_set1_pd(alpha_minus_1);
    const __m512d bm1_v = _mm512_set1_pd(beta_minus_1);
    const __m512d nlb_v = _mm512_set1_pd(neg_log_beta);
    const __m512d zero_v = _mm512_setzero_pd();
    const __m512d one_v = _mm512_set1_pd(1.0);
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __mmask8 invalid = _mm512_cmp_pd_mask(x, zero_v, _CMP_LE_OQ) |
                           _mm512_cmp_pd_mask(x, one_v, _CMP_GE_OQ) |
                           _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
        __m512d lx = avx512_log_8pd(x);
        __m512d l1mx = avx512_log_8pd(_mm512_sub_pd(one_v, x));
        __m512d res = _mm512_fmadd_pd(bm1_v, l1mx, _mm512_fmadd_pd(am1_v, lx, nlb_v));
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
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

// student_t: log_norm − half_nu_plus_1·log(1+((x−loc)·inv_scale)²·inv_nu); NaN/Inf → −∞.
void student_t_batch_avx512(const double *obs, double *out, std::size_t n, double location,
                            double inv_scale, double half_nu_plus_1, double log_norm,
                            double inv_nu) noexcept {
    const __m512d loc_v = _mm512_set1_pd(location);
    const __m512d iscale_v = _mm512_set1_pd(inv_scale);
    const __m512d hnp1_v = _mm512_set1_pd(half_nu_plus_1);
    const __m512d lognorm_v = _mm512_set1_pd(log_norm);
    const __m512d invnu_v = _mm512_set1_pd(inv_nu);
    const __m512d one_v = _mm512_set1_pd(1.0);
    const __m512d pos_inf_v = _mm512_set1_pd(std::numeric_limits<double>::infinity());
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    const __m512d sign_mask = _mm512_set1_pd(-0.0); // for |x|

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __m512d abs_x = _mm512_andnot_pd(sign_mask, x); // AVX-512DQ
        __mmask8 invalid = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q) |
                           _mm512_cmp_pd_mask(abs_x, pos_inf_v, _CMP_EQ_OQ);
        __m512d t = _mm512_mul_pd(_mm512_sub_pd(x, loc_v), iscale_v);
        __m512d t2nu = _mm512_mul_pd(_mm512_mul_pd(t, t), invnu_v);
        __m512d res =
            _mm512_fnmadd_pd(hnp1_v, avx512_log_8pd(_mm512_add_pd(one_v, t2nu)), lognorm_v);
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
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
void vonmises_batch_avx512(const double *obs, double *out, std::size_t n, double mu, double kappa,
                           double log_normaliser) noexcept {
    const __m512d mu_v = _mm512_set1_pd(mu);
    const __m512d kappa_v = _mm512_set1_pd(kappa);
    const __m512d neg_ln_v = _mm512_set1_pd(-log_normaliser);
    const __m512d pos_inf_v = _mm512_set1_pd(std::numeric_limits<double>::infinity());
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    const __m512d sign_mask = _mm512_set1_pd(-0.0); // for |x|

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(obs + i);
        __m512d abs_x = _mm512_andnot_pd(sign_mask, x); // AVX-512DQ
        __mmask8 invalid = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q) |
                           _mm512_cmp_pd_mask(abs_x, pos_inf_v, _CMP_EQ_OQ);
        __m512d res = _mm512_fmadd_pd(kappa_v, avx512_cos_8pd(_mm512_sub_pd(x, mu_v)), neg_ln_v);
        _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(invalid, res, neg_inf_v));
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
