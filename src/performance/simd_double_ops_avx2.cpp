// simd_double_ops_avx2.cpp — AVX/AVX2 + FMA (4-wide) kernels.
// Compiled with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC).
// Only included in the build on x86/x86-64 platforms.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cmath>
#include <cstddef>
#include <immintrin.h> // AVX/AVX2/FMA
#include <limits>

namespace libhmm::performance::detail {

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

} // namespace libhmm::performance::detail

#endif // x86 guard
