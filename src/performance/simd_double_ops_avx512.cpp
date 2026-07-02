// simd_double_ops_avx512.cpp — AVX-512F/DQ (8-wide) kernels.
// Compiled with -mavx512f -mavx512dq (GCC/Clang) or /arch:AVX512 (MSVC).
// Only included in the build on x86/x86-64 platforms with AVX-512 support.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cmath>
#include <cstddef>
#include <immintrin.h> // AVX-512
#include <limits>

namespace libhmm::performance::detail {

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

} // namespace libhmm::performance::detail

#endif // x86 guard
