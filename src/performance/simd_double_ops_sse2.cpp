// simd_double_ops_sse2.cpp — SSE2 (2-wide) kernels.
// Compiled with -msse2 (GCC/Clang) or no extra flag (MSVC x64 baseline).
// Only included in the build on x86/x86-64 platforms.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cmath>
#include <cstddef>
#include <emmintrin.h> // SSE2
#include <limits>

namespace libhmm::performance::detail {

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

} // namespace libhmm::performance::detail

#endif // x86 guard
