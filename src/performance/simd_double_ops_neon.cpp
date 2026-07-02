// simd_double_ops_neon.cpp — ARM NEON (2-wide float64x2) kernels.
// Compiled without extra flags on AArch64 (NEON is the mandatory baseline ISA).
// Only included in the build on AArch64 platforms.

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <limits>

#include "libhmm/detail/simd_math_helpers.h"

namespace libhmm::performance::detail {
using namespace libhmm::detail::simd; // log_pd, exp_pd, cos_pd, log1p_pd

// gaussian: log_norm + neg_half_inv_sq * (x - mean)^2, NaN/Inf → -Inf.
void gaussian_batch_neon(const double *obs, double *out, std::size_t n, double mean,
                         double neg_half_inv_sq, double log_norm) noexcept {
    const float64x2_t mean_v = vdupq_n_f64(mean);
    const float64x2_t lognorm_v = vdupq_n_f64(log_norm);
    const float64x2_t scale_v = vdupq_n_f64(neg_half_inv_sq);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        float64x2_t diff = vsubq_f64(x, mean_v);
        float64x2_t sq = vmulq_f64(diff, diff);
        float64x2_t res = vaddq_f64(lognorm_v, vmulq_f64(scale_v, sq));
        // vceqq_f64(x, x): all-1s where NOT NaN (NaN != NaN by IEEE 754).
        // vbslq_f64(mask, a, b): a where mask=1, b where mask=0.
        uint64x2_t not_nan = vceqq_f64(x, x);
        res = vbslq_f64(not_nan, res, neg_inf_v);
        vst1q_f64(out + i, res);
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
void exponential_batch_neon(const double *obs, double *out, std::size_t n, double log_lambda,
                            double neg_lambda) noexcept {
    const float64x2_t loglam_v = vdupq_n_f64(log_lambda);
    const float64x2_t neglam_v = vdupq_n_f64(neg_lambda);
    const float64x2_t zero_v = vdupq_n_f64(0.0);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        float64x2_t res = vaddq_f64(loglam_v, vmulq_f64(neglam_v, x));
        // valid = (x >= 0) AND (x == x)  — the latter is false for NaN.
        uint64x2_t valid = vandq_u64(vcgeq_f64(x, zero_v), vceqq_f64(x, x));
        res = vbslq_f64(valid, res, neg_inf_v);
        vst1q_f64(out + i, res);
    }
    // scalar tail
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < 0.0 || std::isnan(x)) ? neg_inf : log_lambda + neg_lambda * x;
    }
}

// ============================================================================
// File-private SIMD inline helpers (not exported; AArch64 only)
// ============================================================================

// SLEEF xlog_u1 core on float64x2_t, < 1 ULP.
// x ≤ 0 → −∞; x = NaN → NaN (original value); x = +∞ → +∞.
// AArch64: vcvtq_f64_s64 converts int64 → double directly (no store/reload).
static inline float64x2_t neon_log_2pd(float64x2_t x) noexcept {
    return log_pd(x);
}

// SLEEF-inspired FMA exp(x), < 1 ULP.  Clamps input to [−708, 709.78].
// Range reduction: x = n·ln2 + r; reconstructs 2^n via IEEE 754 exponent bit-shift.
// AArch64: vcvtq_s64_f64 + vshlq_n_s64 replace the 32-bit round-trip needed on x86.
static inline float64x2_t neon_exp_2pd(float64x2_t x) noexcept {
    return exp_pd(x);
}

// 7-term Horner cosine with two-step range reduction.  Max error ≈ 2×10⁻¹⁰ for finite x.
// NaN/Inf input may produce unspecified output; callers must guard when required.
static inline float64x2_t neon_cos_2pd(float64x2_t x) noexcept {
    return cos_pd(x);
}

// ============================================================================
// Generic math primitives
// ============================================================================

void log_batch_neon(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        vst1q_f64(out + i, neon_log_2pd(vld1q_f64(in + i)));
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double xv = in[i];
        out[i] = (xv <= 0.0) ? neg_inf : std::log(xv);
    }
}

void exp_batch_neon(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        vst1q_f64(out + i, neon_exp_2pd(vld1q_f64(in + i)));
    for (; i < n; ++i)
        out[i] = std::exp(in[i]);
}

void cos_batch_neon(const double *in, double *out, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        vst1q_f64(out + i, neon_cos_2pd(vld1q_f64(in + i)));
    for (; i < n; ++i)
        out[i] = std::cos(in[i]);
}

// log1p: log(1+x); x≤−1 → −∞.  SIMD: add 1.0 then delegate to inline log helper.
void log1p_batch_neon(const double *in, double *out, std::size_t n) noexcept {
    const float64x2_t one_v = vdupq_n_f64(1.0);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2)
        vst1q_f64(out + i, neon_log_2pd(vaddq_f64(vld1q_f64(in + i), one_v)));
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
void lognormal_batch_neon(const double *obs, double *out, std::size_t n, double mean_log,
                          double neg_half_inv_sq, double log_norm_const) noexcept {
    const float64x2_t mean_v = vdupq_n_f64(mean_log);
    const float64x2_t scale_v = vdupq_n_f64(neg_half_inv_sq);
    const float64x2_t neg_norm = vdupq_n_f64(-log_norm_const);
    const float64x2_t zero_v = vdupq_n_f64(0.0);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        uint64x2_t invalid = vcleq_f64(x, zero_v);
        float64x2_t lx = neon_log_2pd(x);
        float64x2_t diff = vsubq_f64(lx, mean_v);
        // res = -lx - log_norm_const + neg_half_inv_sq * d^2
        float64x2_t res = vsubq_f64(vfmaq_f64(neg_norm, vmulq_f64(diff, diff), scale_v), lx);
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
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
void gamma_batch_neon(const double *obs, double *out, std::size_t n, double k_minus_1,
                      double inv_theta, double const_term) noexcept {
    const float64x2_t km1_v = vdupq_n_f64(k_minus_1);
    const float64x2_t invth_v = vdupq_n_f64(inv_theta);
    const float64x2_t cterm_v = vdupq_n_f64(const_term);
    const float64x2_t zero_v = vdupq_n_f64(0.0);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        uint64x2_t invalid = vcleq_f64(x, zero_v);
        float64x2_t lx = neon_log_2pd(x);
        float64x2_t res = vfmaq_f64(cterm_v, km1_v, lx); // const_term + km1·log(x)
        res = vfmsq_f64(res, invth_v, x);                // res − inv_theta·x
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : const_term + k_minus_1 * std::log(x) - x * inv_theta;
    }
}

// chisq: const_term + half_k_minus_1·log(x) − x·0.5; x≤0 → −∞.
void chisq_batch_neon(const double *obs, double *out, std::size_t n, double half_k_minus_1,
                      double const_term) noexcept {
    const float64x2_t hkm1_v = vdupq_n_f64(half_k_minus_1);
    const float64x2_t half_v = vdupq_n_f64(0.5);
    const float64x2_t cterm_v = vdupq_n_f64(const_term);
    const float64x2_t zero_v = vdupq_n_f64(0.0);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        uint64x2_t invalid = vcleq_f64(x, zero_v);
        float64x2_t lx = neon_log_2pd(x);
        float64x2_t res = vfmaq_f64(cterm_v, hkm1_v, lx); // const_term + hkm1·log(x)
        res = vfmsq_f64(res, half_v, x);                  // res − 0.5·x
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : const_term + half_k_minus_1 * std::log(x) - x * 0.5;
    }
}

// rayleigh: log_norm + log(x) − x²·inv2sigma_sq; x≤0 → −∞.
void rayleigh_batch_neon(const double *obs, double *out, std::size_t n, double inv2sigma_sq,
                         double log_norm) noexcept {
    const float64x2_t inv2sig_v = vdupq_n_f64(inv2sigma_sq);
    const float64x2_t lognorm_v = vdupq_n_f64(log_norm);
    const float64x2_t zero_v = vdupq_n_f64(0.0);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        uint64x2_t invalid = vcleq_f64(x, zero_v);
        float64x2_t lx = neon_log_2pd(x);
        float64x2_t res = vaddq_f64(lognorm_v, lx);       // log_norm + log(x)
        res = vfmsq_f64(res, inv2sig_v, vmulq_f64(x, x)); // res − inv2σ²·x²
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : log_norm + std::log(x) - x * x * inv2sigma_sq;
    }
}

// pareto: log_norm − k_plus_1·log(x); x<xm → −∞.
void pareto_batch_neon(const double *obs, double *out, std::size_t n, double k_plus_1, double xm,
                       double log_norm) noexcept {
    const float64x2_t kp1_v = vdupq_n_f64(k_plus_1);
    const float64x2_t xm_v = vdupq_n_f64(xm);
    const float64x2_t lognorm_v = vdupq_n_f64(log_norm);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        uint64x2_t invalid = vcltq_f64(x, xm_v);                        // x < xm
        float64x2_t res = vfmsq_f64(lognorm_v, kp1_v, neon_log_2pd(x)); // log_norm−kp1·log(x)
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < xm) ? neg_inf : log_norm - k_plus_1 * std::log(x);
    }
}

// weibull: log_norm + (k−1)·log(x) − exp(k·log(x) + neg_k_log_lambda); x≤0 → −∞.
void weibull_batch_neon(const double *obs, double *out, std::size_t n, double k_minus_1, double k,
                        double log_norm, double neg_k_log_lambda) noexcept {
    const float64x2_t km1_v = vdupq_n_f64(k_minus_1);
    const float64x2_t k_v = vdupq_n_f64(k);
    const float64x2_t lognorm_v = vdupq_n_f64(log_norm);
    const float64x2_t nkll_v = vdupq_n_f64(neg_k_log_lambda);
    const float64x2_t zero_v = vdupq_n_f64(0.0);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        uint64x2_t invalid = vcleq_f64(x, zero_v);
        float64x2_t lx = neon_log_2pd(x);
        float64x2_t pow_term = neon_exp_2pd(vfmaq_f64(nkll_v, k_v, lx)); // exp(k·log(x)+nkll)
        float64x2_t res = vsubq_f64(vfmaq_f64(lognorm_v, km1_v, lx), pow_term);
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
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

// beta: neg_log_beta + (α−1)·log(x) + (β−1)·log(1−x); x∉(0,1) or NaN → −∞.
void beta_batch_neon(const double *obs, double *out, std::size_t n, double alpha_minus_1,
                     double beta_minus_1, double neg_log_beta) noexcept {
    const float64x2_t am1_v = vdupq_n_f64(alpha_minus_1);
    const float64x2_t bm1_v = vdupq_n_f64(beta_minus_1);
    const float64x2_t nlb_v = vdupq_n_f64(neg_log_beta);
    const float64x2_t zero_v = vdupq_n_f64(0.0);
    const float64x2_t one_v = vdupq_n_f64(1.0);
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());
    const uint64x2_t all_ones = vdupq_n_u64(~0ULL);

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        // invalid = x≤0 OR x≥1 OR NaN
        uint64x2_t is_nan = veorq_u64(vceqq_f64(x, x), all_ones);
        uint64x2_t invalid =
            vorrq_u64(vorrq_u64(vcleq_f64(x, zero_v), vcgeq_f64(x, one_v)), is_nan);
        float64x2_t lx = neon_log_2pd(x);
        float64x2_t l1mx = neon_log_2pd(vsubq_f64(one_v, x)); // log(1−x)
        float64x2_t res = vfmaq_f64(vfmaq_f64(nlb_v, am1_v, lx), bm1_v, l1mx);
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
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
void student_t_batch_neon(const double *obs, double *out, std::size_t n, double location,
                          double inv_scale, double half_nu_plus_1, double log_norm,
                          double inv_nu) noexcept {
    const float64x2_t loc_v = vdupq_n_f64(location);
    const float64x2_t iscale_v = vdupq_n_f64(inv_scale);
    const float64x2_t hnp1_v = vdupq_n_f64(half_nu_plus_1);
    const float64x2_t lognorm_v = vdupq_n_f64(log_norm);
    const float64x2_t invnu_v = vdupq_n_f64(inv_nu);
    const float64x2_t one_v = vdupq_n_f64(1.0);
    const float64x2_t pos_inf_v = vdupq_n_f64(std::numeric_limits<double>::infinity());
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());
    const uint64x2_t all_ones = vdupq_n_u64(~0ULL);

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        // invalid = NaN or ±Inf
        uint64x2_t is_nan = veorq_u64(vceqq_f64(x, x), all_ones);
        uint64x2_t is_inf = vceqq_f64(vabsq_f64(x), pos_inf_v);
        uint64x2_t invalid = vorrq_u64(is_nan, is_inf);
        float64x2_t t = vmulq_f64(vsubq_f64(x, loc_v), iscale_v);
        float64x2_t t2nu = vmulq_f64(vmulq_f64(t, t), invnu_v);
        // log1p(t²·inv_nu) = log(1 + t²·inv_nu); argument ≥ 1 for finite x
        float64x2_t res = vfmsq_f64(lognorm_v, hnp1_v, neon_log_2pd(vaddq_f64(one_v, t2nu)));
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
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
void vonmises_batch_neon(const double *obs, double *out, std::size_t n, double mu, double kappa,
                         double log_normaliser) noexcept {
    const float64x2_t mu_v = vdupq_n_f64(mu);
    const float64x2_t kappa_v = vdupq_n_f64(kappa);
    const float64x2_t neg_ln_v = vdupq_n_f64(-log_normaliser);
    const float64x2_t pos_inf_v = vdupq_n_f64(std::numeric_limits<double>::infinity());
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());
    const uint64x2_t all_ones = vdupq_n_u64(~0ULL);

    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(obs + i);
        uint64x2_t is_nan = veorq_u64(vceqq_f64(x, x), all_ones);
        uint64x2_t is_inf = vceqq_f64(vabsq_f64(x), pos_inf_v);
        uint64x2_t invalid = vorrq_u64(is_nan, is_inf);
        float64x2_t res = vfmaq_f64(neg_ln_v, kappa_v, neon_cos_2pd(vsubq_f64(x, mu_v)));
        vst1q_f64(out + i, vbslq_f64(invalid, neg_inf_v, res));
    }
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x)) ? neg_inf : kappa * std::cos(x - mu) - log_normaliser;
    }
}

} // namespace libhmm::performance::detail

#else

// Non-AArch64 scalar fallbacks — compiled when this TU is processed on non-ARM
// hosts (e.g. cross-compilation checks).  Never called at runtime.
#include <cmath>
#include <cstddef>
#include <limits>

namespace libhmm::performance::detail {

void log_batch_neon(const double *in, double *out, std::size_t n) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = in[i];
        out[i] = (x <= 0.0) ? neg_inf : std::log(x);
    }
}
void exp_batch_neon(const double *in, double *out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = std::exp(in[i]);
}
void cos_batch_neon(const double *in, double *out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = std::cos(in[i]);
}
void log1p_batch_neon(const double *in, double *out, std::size_t n) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = in[i];
        out[i] = (x <= -1.0) ? neg_inf : std::log1p(x);
    }
}
void lognormal_batch_neon(const double *obs, double *out, std::size_t n, double mean_log,
                          double neg_half_inv_sq, double log_norm_const) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
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
void gamma_batch_neon(const double *obs, double *out, std::size_t n, double k_minus_1,
                      double inv_theta, double const_term) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : const_term + k_minus_1 * std::log(x) - x * inv_theta;
    }
}
void chisq_batch_neon(const double *obs, double *out, std::size_t n, double half_k_minus_1,
                      double const_term) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : const_term + half_k_minus_1 * std::log(x) - x * 0.5;
    }
}
void rayleigh_batch_neon(const double *obs, double *out, std::size_t n, double inv2sigma_sq,
                         double log_norm) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x <= 0.0) ? neg_inf : log_norm + std::log(x) - x * x * inv2sigma_sq;
    }
}
void pareto_batch_neon(const double *obs, double *out, std::size_t n, double k_plus_1, double xm,
                       double log_norm) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < xm) ? neg_inf : log_norm - k_plus_1 * std::log(x);
    }
}
void weibull_batch_neon(const double *obs, double *out, std::size_t n, double k_minus_1, double k,
                        double log_norm, double neg_k_log_lambda) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        if (x <= 0.0) {
            out[i] = neg_inf;
        } else {
            const double lx = std::log(x);
            out[i] = log_norm + k_minus_1 * lx - std::exp(k * lx + neg_k_log_lambda);
        }
    }
}
void beta_batch_neon(const double *obs, double *out, std::size_t n, double alpha_minus_1,
                     double beta_minus_1, double neg_log_beta) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        if (x <= 0.0 || x >= 1.0) {
            out[i] = neg_inf;
        } else {
            out[i] = neg_log_beta + alpha_minus_1 * std::log(x) + beta_minus_1 * std::log1p(-x);
        }
    }
}
void student_t_batch_neon(const double *obs, double *out, std::size_t n, double location,
                          double inv_scale, double half_nu_plus_1, double log_norm,
                          double inv_nu) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        if (std::isnan(x) || std::isinf(x)) {
            out[i] = neg_inf;
        } else {
            const double tv = (x - location) * inv_scale;
            out[i] = log_norm - half_nu_plus_1 * std::log1p(tv * tv * inv_nu);
        }
    }
}
void vonmises_batch_neon(const double *obs, double *out, std::size_t n, double mu, double kappa,
                         double log_normaliser) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x)) ? neg_inf : kappa * std::cos(x - mu) - log_normaliser;
    }
}

} // namespace libhmm::performance::detail

#endif // AArch64 guard
