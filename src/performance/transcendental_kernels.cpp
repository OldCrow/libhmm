// src/performance/transcendental_kernels.cpp
//
// SIMD implementations of TranscendentalKernels methods.
//
// Compiled with LIBHMM_BEST_SIMD_FLAGS, activating the ISA cascade:
//   AVX-512  8-wide __m512d
//   AVX/AVX2 4-wide __m256d   (AVX-1 compatible; compiler fuses FMA under AVX2)
//   SSE2     2-wide __m128d
//   NEON     2-wide float64x2_t
//   scalar   tail and portable fallback
//
// Vector exp helpers (k_exp_pd_*) and log helpers (k_log_pd_*) are defined
// in simd_kernels_internal.h -- the single source of truth shared with
// Tier-2 distribution TUs (log_normal_distribution.cpp, pareto_distribution.cpp).

#include "libhmm/performance/transcendental_kernels.h"
#include "libhmm/performance/simd_kernels_internal.h"
#include "libhmm/math/constants.h"
#include "libhmm/platform/simd_platform.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace libhmm {
namespace performance {
namespace detail {

namespace {

// ---------------------------------------------------------------------------
// Horizontal reduction helpers
// ---------------------------------------------------------------------------

// SSE2: horizontal max of 2-lane vector.
#if defined(LIBHMM_HAS_SSE2)
static inline double hmax_pd_sse2(__m128d v) noexcept {
    __m128d shuf = _mm_shuffle_pd(v, v, 1);
    return _mm_cvtsd_f64(_mm_max_pd(v, shuf));
}
static inline double hadd_pd_sse2(__m128d v) noexcept {
    __m128d shuf = _mm_shuffle_pd(v, v, 1);
    return _mm_cvtsd_f64(_mm_add_pd(v, shuf));
}
#endif

// AVX: horizontal max/sum of 4-lane vector.
#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
static inline double hmax_pd_avx(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d m = _mm_max_pd(lo, hi);
    return hmax_pd_sse2(m);
}
static inline double hadd_pd_avx(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s = _mm_add_pd(lo, hi);
    return hadd_pd_sse2(s);
}
#endif

} // anonymous namespace

// =============================================================================
// TranscendentalKernels method implementations
// =============================================================================

// -----------------------------------------------------------------------------
// reduce_max_sum2: max of (a[i] + b[i])
// -----------------------------------------------------------------------------
double TranscendentalKernels::reduce_max_sum2(const double *a, const double *b,
                                              std::size_t size) noexcept {
    std::size_t i = 0;
    const double neg_inf = -std::numeric_limits<double>::infinity();
    // maxVal accumulates across ISA blocks; each block seeds its vector
    // accumulator from it so the cascade is correct for any size.
    double maxVal = neg_inf;

#if defined(LIBHMM_HAS_AVX512)
    {
        __m512d vmax = _mm512_set1_pd(neg_inf);
        for (; i + 8 <= size; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            vmax = _mm512_max_pd(vmax, _mm512_add_pd(va, vb));
        }
        maxVal = _mm512_reduce_max_pd(vmax); // cppcheck-suppress redundantInitialization
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        __m256d vmax = _mm256_set1_pd(maxVal);
        for (; i + 4 <= size; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            vmax = _mm256_max_pd(vmax, _mm256_add_pd(va, vb));
        }
        maxVal = hmax_pd_avx(vmax);
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        __m128d vmax = _mm_set1_pd(maxVal);
        for (; i + 2 <= size; i += 2) {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            vmax = _mm_max_pd(vmax, _mm_add_pd(va, vb));
        }
        maxVal = hmax_pd_sse2(vmax);
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        float64x2_t vmax = vdupq_n_f64(maxVal);
        for (; i + 2 <= size; i += 2) {
            float64x2_t va = vld1q_f64(a + i);
            float64x2_t vb = vld1q_f64(b + i);
            vmax = vmaxq_f64(vmax, vaddq_f64(va, vb));
        }
        maxVal = vmaxvq_f64(vmax);
    }
#endif

    // Scalar tail.
    for (; i < size; ++i) {
        const double t = a[i] + b[i];
        if (t > maxVal)
            maxVal = t;
    }
    return maxVal;
}

// -----------------------------------------------------------------------------
// sum_exp_sum2_minus_max
// -----------------------------------------------------------------------------
double TranscendentalKernels::sum_exp_sum2_minus_max(const double *a, const double *b,
                                                     std::size_t size, double maxVal) noexcept {
    if (!std::isfinite(maxVal))
        return 0.0;
    std::size_t i = 0;
    double sum = 0.0;

#if defined(LIBHMM_HAS_AVX512)
    {
        const __m512d vmaxv = _mm512_set1_pd(maxVal);
        __m512d vsum = _mm512_setzero_pd();
        for (; i + 8 <= size; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d term = _mm512_sub_pd(_mm512_add_pd(va, vb), vmaxv);
            vsum = _mm512_add_pd(vsum, kernels::k_exp_pd_avx512(term));
        }
        sum += _mm512_reduce_add_pd(vsum);
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        const __m256d vmaxv = _mm256_set1_pd(maxVal);
        __m256d vsum = _mm256_setzero_pd();
        for (; i + 4 <= size; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d term = _mm256_sub_pd(_mm256_add_pd(va, vb), vmaxv);
            vsum = _mm256_add_pd(vsum, kernels::k_exp_pd_avx(term));
        }
        sum += hadd_pd_avx(vsum);
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        const __m128d vmaxv = _mm_set1_pd(maxVal);
        __m128d vsum = _mm_setzero_pd();
        for (; i + 2 <= size; i += 2) {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d term = _mm_sub_pd(_mm_add_pd(va, vb), vmaxv);
            vsum = _mm_add_pd(vsum, kernels::k_exp_pd_sse2(term));
        }
        sum += hadd_pd_sse2(vsum);
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        const float64x2_t vmaxv = vdupq_n_f64(maxVal);
        float64x2_t vsum = vdupq_n_f64(0.0);
        for (; i + 2 <= size; i += 2) {
            float64x2_t va = vld1q_f64(a + i);
            float64x2_t vb = vld1q_f64(b + i);
            float64x2_t term = vsubq_f64(vaddq_f64(va, vb), vmaxv);
            vsum = vaddq_f64(vsum, kernels::k_exp_pd_neon(term));
        }
        sum += vaddvq_f64(vsum);
    }
#endif

    // Scalar tail.
    for (; i < size; ++i) {
        const double t = a[i] + b[i];
        if (std::isfinite(t))
            sum += std::exp(t - maxVal);
    }
    return sum;
}

// -----------------------------------------------------------------------------
// reduce_max_sum3: max of (a[i] + b[i] + c[i])
// -----------------------------------------------------------------------------
double TranscendentalKernels::reduce_max_sum3(const double *a, const double *b, const double *c,
                                              std::size_t size) noexcept {
    std::size_t i = 0;
    const double neg_inf = -std::numeric_limits<double>::infinity();
    double maxVal = neg_inf;

#if defined(LIBHMM_HAS_AVX512)
    {
        __m512d vmax = _mm512_set1_pd(neg_inf);
        for (; i + 8 <= size; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d vc = _mm512_loadu_pd(c + i);
            vmax = _mm512_max_pd(vmax, _mm512_add_pd(_mm512_add_pd(va, vb), vc));
        }
        maxVal = _mm512_reduce_max_pd(vmax); // cppcheck-suppress redundantInitialization
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        __m256d vmax = _mm256_set1_pd(maxVal);
        for (; i + 4 <= size; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d vc = _mm256_loadu_pd(c + i);
            vmax = _mm256_max_pd(vmax, _mm256_add_pd(_mm256_add_pd(va, vb), vc));
        }
        maxVal = hmax_pd_avx(vmax);
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        __m128d vmax = _mm_set1_pd(maxVal);
        for (; i + 2 <= size; i += 2) {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d vc = _mm_loadu_pd(c + i);
            vmax = _mm_max_pd(vmax, _mm_add_pd(_mm_add_pd(va, vb), vc));
        }
        maxVal = hmax_pd_sse2(vmax);
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        float64x2_t vmax = vdupq_n_f64(maxVal);
        for (; i + 2 <= size; i += 2) {
            float64x2_t va = vld1q_f64(a + i);
            float64x2_t vb = vld1q_f64(b + i);
            float64x2_t vc = vld1q_f64(c + i);
            vmax = vmaxq_f64(vmax, vaddq_f64(vaddq_f64(va, vb), vc));
        }
        maxVal = vmaxvq_f64(vmax);
    }
#endif

    // Scalar tail.
    for (; i < size; ++i) {
        const double t = a[i] + b[i] + c[i];
        if (t > maxVal)
            maxVal = t;
    }
    return maxVal;
}

// -----------------------------------------------------------------------------
// sum_exp_sum3_minus_max: sum of exp(a[i]+b[i]+c[i] - maxVal)
// -----------------------------------------------------------------------------
double TranscendentalKernels::sum_exp_sum3_minus_max(const double *a, const double *b,
                                                     const double *c, std::size_t size,
                                                     double maxVal) noexcept {
    if (!std::isfinite(maxVal))
        return 0.0;
    std::size_t i = 0;
    double sum = 0.0;

#if defined(LIBHMM_HAS_AVX512)
    {
        const __m512d vmaxv = _mm512_set1_pd(maxVal);
        __m512d vsum = _mm512_setzero_pd();
        for (; i + 8 <= size; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d vc = _mm512_loadu_pd(c + i);
            __m512d term = _mm512_sub_pd(_mm512_add_pd(_mm512_add_pd(va, vb), vc), vmaxv);
            vsum = _mm512_add_pd(vsum, kernels::k_exp_pd_avx512(term));
        }
        sum += _mm512_reduce_add_pd(vsum);
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        const __m256d vmaxv = _mm256_set1_pd(maxVal);
        __m256d vsum = _mm256_setzero_pd();
        for (; i + 4 <= size; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d vc = _mm256_loadu_pd(c + i);
            __m256d term = _mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(va, vb), vc), vmaxv);
            vsum = _mm256_add_pd(vsum, kernels::k_exp_pd_avx(term));
        }
        sum += hadd_pd_avx(vsum);
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        const __m128d vmaxv = _mm_set1_pd(maxVal);
        __m128d vsum = _mm_setzero_pd();
        for (; i + 2 <= size; i += 2) {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d vc = _mm_loadu_pd(c + i);
            __m128d term = _mm_sub_pd(_mm_add_pd(_mm_add_pd(va, vb), vc), vmaxv);
            vsum = _mm_add_pd(vsum, kernels::k_exp_pd_sse2(term));
        }
        sum += hadd_pd_sse2(vsum);
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        const float64x2_t vmaxv = vdupq_n_f64(maxVal);
        float64x2_t vsum = vdupq_n_f64(0.0);
        for (; i + 2 <= size; i += 2) {
            float64x2_t va = vld1q_f64(a + i);
            float64x2_t vb = vld1q_f64(b + i);
            float64x2_t vc = vld1q_f64(c + i);
            float64x2_t term = vsubq_f64(vaddq_f64(vaddq_f64(va, vb), vc), vmaxv);
            vsum = vaddq_f64(vsum, kernels::k_exp_pd_neon(term));
        }
        sum += vaddvq_f64(vsum);
    }
#endif

    // Scalar tail.
    for (; i < size; ++i) {
        const double t = a[i] + b[i] + c[i];
        if (std::isfinite(t))
            sum += std::exp(t - maxVal);
    }
    return sum;
}

// -----------------------------------------------------------------------------
// accumulate_exp_sum2_bias: dst[i] += exp(a[i] + b[i] + bias)
// -----------------------------------------------------------------------------
void TranscendentalKernels::accumulate_exp_sum2_bias(double *dst, const double *a, const double *b,
                                                     std::size_t size, double bias) noexcept {
    std::size_t i = 0;

#if defined(LIBHMM_HAS_AVX512)
    {
        const __m512d vbias = _mm512_set1_pd(bias);
        for (; i + 8 <= size; i += 8) {
            __m512d vd = _mm512_loadu_pd(dst + i);
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d arg = _mm512_add_pd(_mm512_add_pd(va, vb), vbias);
            vd = _mm512_add_pd(vd, kernels::k_exp_pd_avx512(arg));
            _mm512_storeu_pd(dst + i, vd);
        }
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        const __m256d vbias = _mm256_set1_pd(bias);
        for (; i + 4 <= size; i += 4) {
            __m256d vd = _mm256_loadu_pd(dst + i);
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d arg = _mm256_add_pd(_mm256_add_pd(va, vb), vbias);
            vd = _mm256_add_pd(vd, kernels::k_exp_pd_avx(arg));
            _mm256_storeu_pd(dst + i, vd);
        }
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        const __m128d vbias = _mm_set1_pd(bias);
        for (; i + 2 <= size; i += 2) {
            __m128d vd = _mm_loadu_pd(dst + i);
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d arg = _mm_add_pd(_mm_add_pd(va, vb), vbias);
            vd = _mm_add_pd(vd, kernels::k_exp_pd_sse2(arg));
            _mm_storeu_pd(dst + i, vd);
        }
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        const float64x2_t vbias = vdupq_n_f64(bias);
        for (; i + 2 <= size; i += 2) {
            float64x2_t vd = vld1q_f64(dst + i);
            float64x2_t va = vld1q_f64(a + i);
            float64x2_t vb = vld1q_f64(b + i);
            float64x2_t arg = vaddq_f64(vaddq_f64(va, vb), vbias);
            vd = vaddq_f64(vd, kernels::k_exp_pd_neon(arg));
            vst1q_f64(dst + i, vd);
        }
    }
#endif

    // Scalar tail.
    for (; i < size; ++i) {
        dst[i] += std::exp(a[i] + b[i] + bias);
    }
}

} // namespace detail
} // namespace performance
} // namespace libhmm
