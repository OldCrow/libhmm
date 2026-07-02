// simd_double_ops_neon.cpp — ARM NEON (2-wide float64x2) kernels.
// Compiled without extra flags on AArch64 (NEON is the mandatory baseline ISA).
// Only included in the build on AArch64 platforms.

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <limits>

namespace libhmm::performance::detail {

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

} // namespace libhmm::performance::detail

#endif // AArch64 guard
