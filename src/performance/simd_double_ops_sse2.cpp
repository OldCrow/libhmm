// simd_double_ops_sse2.cpp — SSE2 (2-wide) kernels.
// Compiled with -msse2 (GCC/Clang) or no extra flag (MSVC x64 baseline).
// Only included in the build on x86/x86-64 platforms.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cmath>
#include <cstddef>
#include <emmintrin.h> // SSE2
#include <limits>

#include "libhmm/detail/simd_math_helpers.h" // sse2_blend + log_pd/exp_pd/cos_pd

namespace libhmm::performance::detail {
using namespace libhmm::detail::simd; // log_pd, exp_pd, cos_pd, log1p_pd

static inline __m128d sse2_abs_2pd(__m128d x) noexcept {
    const __m128d sign_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    return _mm_and_pd(x, sign_mask);
}

static inline __m128d sse2_finite_mask_2pd(__m128d x) noexcept {
    const __m128d max_finite = _mm_set1_pd(std::numeric_limits<double>::max());
    return _mm_cmple_pd(sse2_abs_2pd(x), max_finite);
}

static inline __m128d sse2_log_2pd(__m128d x) noexcept {
    return log_pd(x);
}

static inline __m128d sse2_exp_2pd(__m128d x) noexcept {
    return exp_pd(x);
}

static inline __m128d sse2_cos_2pd(__m128d x) noexcept {
    return cos_pd(x);
}

// gaussian: log_norm + neg_half_inv_sq * (x - mean)^2, NaN/Inf → -Inf.
void gaussian_batch_sse2(const double *obs, double *out, std::size_t n, double mean,
                         double neg_half_inv_sq, double log_norm) noexcept {
    const __m128d mean_v = _mm_set1_pd(mean);
    const __m128d lognorm_v = _mm_set1_pd(log_norm);
    const __m128d scale_v = _mm_set1_pd(neg_half_inv_sq);
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d x = _mm_loadu_pd(obs + i);
        __m128d diff = _mm_sub_pd(x, mean_v);
        __m128d sq = _mm_mul_pd(diff, diff);
        __m128d res = _mm_add_pd(lognorm_v, _mm_mul_pd(scale_v, sq));
        // NaN check: NaN != NaN. SSE2 has no blendv; use andnot/or.
        __m128d is_nan = _mm_cmpunord_pd(
            x,
            x); // all-1s where NaN or Inf (cmpunord also catches Inf pairs, but we need a separate Inf check)
        // Actually cmpunord catches NaN only. For Inf, the formula gives -Inf naturally
        // for finite results, but Inf input gives Inf-mean = Inf, Inf*scale = -Inf or Inf.
        // We need explicit NaN masking; Inf naturally yields -Inf via the formula for
        // neg_half_inv_sq < 0: Inf * neg = -Inf. So just mask NaN.
        res = _mm_or_pd(_mm_andnot_pd(is_nan, res), _mm_and_pd(is_nan, neg_inf_v));
        _mm_storeu_pd(out + i, res);
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
void exponential_batch_sse2(const double *obs, double *out, std::size_t n, double log_lambda,
                            double neg_lambda) noexcept {
    const __m128d loglam_v = _mm_set1_pd(log_lambda);
    const __m128d neglam_v = _mm_set1_pd(neg_lambda);
    const __m128d zero_v = _mm_setzero_pd();
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d x = _mm_loadu_pd(obs + i);
        __m128d res = _mm_add_pd(loglam_v, _mm_mul_pd(neglam_v, x));
        __m128d invalid = _mm_or_pd(_mm_cmplt_pd(x, zero_v), // x < 0
                                    _mm_cmpunord_pd(x, x));  // NaN
        res = _mm_or_pd(_mm_andnot_pd(invalid, res), _mm_and_pd(invalid, neg_inf_v));
        _mm_storeu_pd(out + i, res);
    }
    // scalar tail
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < 0.0 || std::isnan(x)) ? neg_inf : log_lambda + neg_lambda * x;
    }
}

// ---- generic math primitives ----

// log: log(x); x ≤ 0 → -Inf, NaN → NaN.
void log_batch_sse2(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        _mm_storeu_pd(out + i, sse2_log_2pd(_mm_loadu_pd(in + i)));
    for (; i < n; ++i) {
        const double x = in[i];
        out[i] = (x <= 0.0) ? -std::numeric_limits<double>::infinity() : std::log(x);
    }
}

// exp: exp(x), clamped to [-708, 709.8].
void exp_batch_sse2(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        _mm_storeu_pd(out + i, sse2_exp_2pd(_mm_loadu_pd(in + i)));
    for (; i < n; ++i)
        out[i] = std::exp(in[i]);
}

// cos: cos(x) for all finite x; NaN/Inf → NaN via arithmetic propagation.
void cos_batch_sse2(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        _mm_storeu_pd(out + i, sse2_cos_2pd(_mm_loadu_pd(in + i)));
    for (; i < n; ++i)
        out[i] = std::cos(in[i]);
}

// log1p: log(1 + x); x ≤ -1 → -Inf.
void log1p_batch_sse2(const double *in, double *out, std::size_t n) noexcept {
    const __m128d one_v = _mm_set1_pd(1.0);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        _mm_storeu_pd(out + i, sse2_log_2pd(_mm_add_pd(_mm_loadu_pd(in + i), one_v)));
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = in[i];
        out[i] = (x < -1.0) ? neg_inf : std::log1p(x);
    }
}

// ---- tier-1 distribution kernels ----

// lognormal: -log_norm_const + neg_half_inv_sq * (log(x) - mean_log)^2; x <= 0 -> -Inf.
void lognormal_batch_sse2(const double *obs, double *out, std::size_t n, double mean_log,
                          double neg_half_inv_sq, double log_norm_const) noexcept {
    const __m128d mean_log_v = _mm_set1_pd(mean_log);
    const __m128d nhi_sq_v = _mm_set1_pd(neg_half_inv_sq);
    const __m128d neg_lnc_v = _mm_set1_pd(-log_norm_const);
    const __m128d zero_v = _mm_setzero_pd();
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        const __m128d valid = _mm_and_pd(_mm_cmpgt_pd(x, zero_v), sse2_finite_mask_2pd(x));
        const __m128d lx = sse2_log_2pd(x);
        const __m128d d = _mm_sub_pd(lx, mean_log_v);
        // res = -lx - log_norm_const + neg_half_inv_sq * d^2
        const __m128d res =
            _mm_sub_pd(_mm_add_pd(neg_lnc_v, _mm_mul_pd(nhi_sq_v, _mm_mul_pd(d, d))), lx);
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (!std::isfinite(x) || x <= 0.0) {
            out[i] = neg_inf;
        } else {
            const double lx = std::log(x);
            const double d = lx - mean_log;
            out[i] = -lx - log_norm_const + neg_half_inv_sq * d * d;
        }
    }
}

// gamma: const_term + k_minus_1 * log(x) - x * inv_theta; x <= 0 -> -Inf.
void gamma_batch_sse2(const double *obs, double *out, std::size_t n, double k_minus_1,
                      double inv_theta, double const_term) noexcept {
    const __m128d km1_v = _mm_set1_pd(k_minus_1);
    const __m128d invth_v = _mm_set1_pd(inv_theta);
    const __m128d ct_v = _mm_set1_pd(const_term);
    const __m128d zero_v = _mm_setzero_pd();
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        const __m128d valid = _mm_and_pd(_mm_cmpgt_pd(x, zero_v), sse2_finite_mask_2pd(x));
        const __m128d lx = sse2_log_2pd(x);
        const __m128d res =
            _mm_sub_pd(_mm_add_pd(ct_v, _mm_mul_pd(km1_v, lx)), _mm_mul_pd(x, invth_v));
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x) || x <= 0.0)
                     ? neg_inf
                     : const_term + k_minus_1 * std::log(x) - x * inv_theta;
    }
}

// chisq: const_term + half_k_minus_1 * log(x) - x * 0.5; x <= 0 -> -Inf.
void chisq_batch_sse2(const double *obs, double *out, std::size_t n, double half_k_minus_1,
                      double const_term) noexcept {
    const __m128d hkm1_v = _mm_set1_pd(half_k_minus_1);
    const __m128d ct_v = _mm_set1_pd(const_term);
    const __m128d half_v = _mm_set1_pd(0.5);
    const __m128d zero_v = _mm_setzero_pd();
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        const __m128d valid = _mm_and_pd(_mm_cmpgt_pd(x, zero_v), sse2_finite_mask_2pd(x));
        const __m128d lx = sse2_log_2pd(x);
        const __m128d res =
            _mm_sub_pd(_mm_add_pd(ct_v, _mm_mul_pd(hkm1_v, lx)), _mm_mul_pd(x, half_v));
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x) || x <= 0.0)
                     ? neg_inf
                     : const_term + half_k_minus_1 * std::log(x) - x * 0.5;
    }
}

// rayleigh: log_norm + log(x) - x^2 * inv2sigma_sq; x <= 0 -> -Inf.
void rayleigh_batch_sse2(const double *obs, double *out, std::size_t n, double inv2sigma_sq,
                         double log_norm) noexcept {
    const __m128d inv2s2_v = _mm_set1_pd(inv2sigma_sq);
    const __m128d lnorm_v = _mm_set1_pd(log_norm);
    const __m128d zero_v = _mm_setzero_pd();
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        const __m128d valid = _mm_and_pd(_mm_cmpgt_pd(x, zero_v), sse2_finite_mask_2pd(x));
        const __m128d lx = sse2_log_2pd(x);
        const __m128d x2 = _mm_mul_pd(x, x);
        const __m128d res = _mm_sub_pd(_mm_add_pd(lnorm_v, lx), _mm_mul_pd(x2, inv2s2_v));
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x) || x <= 0.0) ? neg_inf
                                                 : log_norm + std::log(x) - x * x * inv2sigma_sq;
    }
}

// pareto: log_norm - k_plus_1 * log(x); x < xm -> -Inf.
void pareto_batch_sse2(const double *obs, double *out, std::size_t n, double k_plus_1, double xm,
                       double log_norm) noexcept {
    const __m128d kp1_v = _mm_set1_pd(k_plus_1);
    const __m128d xm_v = _mm_set1_pd(xm);
    const __m128d lnorm_v = _mm_set1_pd(log_norm);
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        // x >= xm (excludes NaN since NaN >= xm is false) and finite (excludes +Inf)
        const __m128d valid = _mm_and_pd(_mm_cmpge_pd(x, xm_v), sse2_finite_mask_2pd(x));
        const __m128d lx = sse2_log_2pd(x);
        const __m128d res = _mm_sub_pd(lnorm_v, _mm_mul_pd(kp1_v, lx));
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x) || x < xm) ? neg_inf : log_norm - k_plus_1 * std::log(x);
    }
}

// weibull: log_norm + k_minus_1*log(x) - exp(k*log(x)+neg_k_log_lambda); x <= 0 -> -Inf.
void weibull_batch_sse2(const double *obs, double *out, std::size_t n, double k_minus_1, double k,
                        double log_norm, double neg_k_log_lambda) noexcept {
    const __m128d km1_v = _mm_set1_pd(k_minus_1);
    const __m128d k_v = _mm_set1_pd(k);
    const __m128d lnorm_v = _mm_set1_pd(log_norm);
    const __m128d nkll_v = _mm_set1_pd(neg_k_log_lambda);
    const __m128d zero_v = _mm_setzero_pd();
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        const __m128d valid = _mm_and_pd(_mm_cmpgt_pd(x, zero_v), sse2_finite_mask_2pd(x));
        const __m128d lx = sse2_log_2pd(x);
        // exp_arg = k * lx + neg_k_log_lambda (no FMA: mul then add)
        const __m128d exp_arg = _mm_add_pd(_mm_mul_pd(k_v, lx), nkll_v);
        const __m128d res =
            _mm_sub_pd(_mm_add_pd(lnorm_v, _mm_mul_pd(km1_v, lx)), sse2_exp_2pd(exp_arg));
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (!std::isfinite(x) || x <= 0.0) {
            out[i] = neg_inf;
        } else {
            const double lx = std::log(x);
            out[i] = log_norm + k_minus_1 * lx - std::exp(k * lx + neg_k_log_lambda);
        }
    }
}

// beta: neg_log_beta + alpha_minus_1*log(x) + beta_minus_1*log(1-x); x not in (0,1) -> -Inf.
void beta_batch_sse2(const double *obs, double *out, std::size_t n, double alpha_minus_1,
                     double beta_minus_1, double neg_log_beta) noexcept {
    const __m128d am1_v = _mm_set1_pd(alpha_minus_1);
    const __m128d bm1_v = _mm_set1_pd(beta_minus_1);
    const __m128d nlb_v = _mm_set1_pd(neg_log_beta);
    const __m128d zero_v = _mm_setzero_pd();
    const __m128d one_v = _mm_set1_pd(1.0);
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        // NaN comparisons return false, so NaN lanes are invalid
        const __m128d valid = _mm_and_pd(_mm_cmpgt_pd(x, zero_v), _mm_cmplt_pd(x, one_v));
        const __m128d lx = sse2_log_2pd(x);
        // log(1 - x) via log helper; for x in (0,1): 1-x in (0,1), valid
        const __m128d l1mx = sse2_log_2pd(_mm_sub_pd(one_v, x));
        const __m128d res =
            _mm_add_pd(nlb_v, _mm_add_pd(_mm_mul_pd(am1_v, lx), _mm_mul_pd(bm1_v, l1mx)));
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (!std::isfinite(x) || x <= 0.0 || x >= 1.0) {
            out[i] = neg_inf;
        } else {
            out[i] = neg_log_beta + alpha_minus_1 * std::log(x) + beta_minus_1 * std::log1p(-x);
        }
    }
}

// student_t: log_norm - half_nu_plus_1*log1p(((x-location)*inv_scale)^2*inv_nu); NaN/Inf -> -Inf.
void student_t_batch_sse2(const double *obs, double *out, std::size_t n, double location,
                          double inv_scale, double half_nu_plus_1, double log_norm,
                          double inv_nu) noexcept {
    const __m128d loc_v = _mm_set1_pd(location);
    const __m128d isc_v = _mm_set1_pd(inv_scale);
    const __m128d hnp1_v = _mm_set1_pd(half_nu_plus_1);
    const __m128d lnorm_v = _mm_set1_pd(log_norm);
    const __m128d inv_nu_v = _mm_set1_pd(inv_nu);
    const __m128d one_v = _mm_set1_pd(1.0);
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        const __m128d valid = sse2_finite_mask_2pd(x);
        const __m128d z = _mm_mul_pd(_mm_sub_pd(x, loc_v), isc_v);
        const __m128d z2 = _mm_mul_pd(z, z);
        // arg = 1 + z^2 * inv_nu; for finite x and real nu, arg >= 1 > 0
        const __m128d arg = _mm_add_pd(one_v, _mm_mul_pd(z2, inv_nu_v));
        const __m128d res = _mm_sub_pd(lnorm_v, _mm_mul_pd(hnp1_v, sse2_log_2pd(arg)));
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        if (!std::isfinite(x)) {
            out[i] = neg_inf;
        } else {
            const double z = (x - location) * inv_scale;
            out[i] = log_norm - half_nu_plus_1 * std::log1p(z * z * inv_nu);
        }
    }
}

// vonmises: kappa * cos(x - mu) - log_normaliser; NaN/Inf -> -Inf.
void vonmises_batch_sse2(const double *obs, double *out, std::size_t n, double mu, double kappa,
                         double log_normaliser) noexcept {
    const __m128d mu_v = _mm_set1_pd(mu);
    const __m128d kappa_v = _mm_set1_pd(kappa);
    const __m128d lnorm_v = _mm_set1_pd(log_normaliser);
    const __m128d neg_inf_v = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d x = _mm_loadu_pd(obs + i);
        const __m128d valid = sse2_finite_mask_2pd(x);
        const __m128d diff = _mm_sub_pd(x, mu_v);
        const __m128d cos_diff = sse2_cos_2pd(diff);
        const __m128d res = _mm_sub_pd(_mm_mul_pd(kappa_v, cos_diff), lnorm_v);
        _mm_storeu_pd(out + i, sse2_blend(valid, res, neg_inf_v));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x)) ? neg_inf : kappa * std::cos(x - mu) - log_normaliser;
    }
}

} // namespace libhmm::performance::detail

#endif // x86 guard
