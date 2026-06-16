#pragma once
// include/libhmm/detail/simd_kernels_internal.h
//
// Internal header — NOT part of the public API.
// Not installed to CMAKE_INSTALL_PREFIX/include (detail/ is excluded).
//
// Single source of truth for vector exp/log helpers shared between
// transcendental_kernels.cpp and Tier-2 distribution TUs
// (log_normal_distribution.cpp, pareto_distribution.cpp).
//
// Include only from .cpp files compiled with LIBHMM_BEST_SIMD_FLAGS.

#include "libhmm/platform/simd_platform.h"
#include "libhmm/math/constants.h"

#include <cmath>
#include <limits>

namespace libhmm {
namespace performance {
namespace detail {
namespace kernels {

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------
static constexpr double K_LN2_HI = 6.93147180369123816490e-1;
static constexpr double K_LN2_LO = 1.90821492927058770002e-10;
static constexpr double K_LOG2E = 1.44269504088896338700;
static constexpr double K_SQRT2 = 1.41421356237309504880168872420969807;
static constexpr double K_EXP_UNDERFLOW = constants::probability::MIN_LOG_PROBABILITY; // -700.0
static constexpr double K_EXPONENT_BIAS = 1023.0;

// log polynomial: 2y*(c0 + c1*y^2 + ... + c6*y^12), c_k = 1/(2k+1)
static constexpr double K_LOG_C0 = 1.0;
static constexpr double K_LOG_C1 = 3.3333333333333333e-1;
static constexpr double K_LOG_C2 = 2.0000000000000000e-1;
static constexpr double K_LOG_C3 = 1.4285714285714285e-1;
static constexpr double K_LOG_C4 = 1.1111111111111111e-1;
static constexpr double K_LOG_C5 = 9.0909090909090909e-2;
static constexpr double K_LOG_C6 = 7.6923076923076923e-2;

// exp polynomial: sum(r^k/k!), k=0..12
static constexpr double K_EXP_C0 = 1.0;
static constexpr double K_EXP_C1 = 1.0;
static constexpr double K_EXP_C2 = 0.5;
static constexpr double K_EXP_C3 = 1.6666666666666666e-1;
static constexpr double K_EXP_C4 = 4.1666666666666664e-2;
static constexpr double K_EXP_C5 = 8.3333333333333332e-3;
static constexpr double K_EXP_C6 = 1.3888888888888889e-3;
static constexpr double K_EXP_C7 = 1.9841269841269841e-4;
static constexpr double K_EXP_C8 = 2.4801587301587302e-5;
static constexpr double K_EXP_C9 = 2.7557319223985888e-6;
static constexpr double K_EXP_C10 = 2.7557319223985888e-7;
static constexpr double K_EXP_C11 = 2.5052108385441720e-8;
static constexpr double K_EXP_C12 = 2.0876756987868099e-9;

// ---------------------------------------------------------------------------
// AVX-512 helpers
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_AVX512)

[[nodiscard]] static inline __m512d k_log_pd_avx512(__m512d x) noexcept {
    const __m512d neg_inf_v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    const __m512d sqrt2_v = _mm512_set1_pd(K_SQRT2);
    const __m512d one_v = _mm512_set1_pd(1.0);
    const __m512d half_v = _mm512_set1_pd(0.5);
    const __m512d two_v = _mm512_set1_pd(2.0);
    const __m512d ln2hi_v = _mm512_set1_pd(K_LN2_HI);
    const __m512d ln2lo_v = _mm512_set1_pd(K_LN2_LO);

    const __mmask8 invalid = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_LE_OS);

    __m512i bits = _mm512_castpd_si512(x);
    __m512i e_biased = _mm512_srli_epi64(bits, 52);
    const __m512i mant_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFLL);
    const __m512i exp_one = _mm512_set1_epi64(0x3FF0000000000000LL);
    __m512i mbits = _mm512_or_si512(_mm512_and_si512(bits, mant_mask), exp_one);
    __m512d m = _mm512_castsi512_pd(mbits);

    // Convert int64 exponent to double via scalar (no AVX-512 DQ needed).
    __m512i e_ub = _mm512_sub_epi64(e_biased, _mm512_set1_epi64(1023LL));
    alignas(64) long long e_arr[8];
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(e_arr), e_ub);
    __m512d e = _mm512_set_pd(static_cast<double>(e_arr[7]), static_cast<double>(e_arr[6]),
                              static_cast<double>(e_arr[5]), static_cast<double>(e_arr[4]),
                              static_cast<double>(e_arr[3]), static_cast<double>(e_arr[2]),
                              static_cast<double>(e_arr[1]), static_cast<double>(e_arr[0]));

    __mmask8 adj = _mm512_cmp_pd_mask(m, sqrt2_v, _CMP_GT_OS);
    e = _mm512_mask_add_pd(e, adj, e, one_v);
    m = _mm512_mask_mul_pd(m, adj, m, half_v);

    __m512d y = _mm512_div_pd(_mm512_sub_pd(m, one_v), _mm512_add_pd(m, one_v));
    __m512d y2 = _mm512_mul_pd(y, y);

    __m512d p = _mm512_set1_pd(K_LOG_C6);
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(K_LOG_C5));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(K_LOG_C4));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(K_LOG_C3));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(K_LOG_C2));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(K_LOG_C1));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(K_LOG_C0));
    __m512d log_m = _mm512_mul_pd(_mm512_mul_pd(two_v, y), p);

    __m512d result = _mm512_fmadd_pd(e, ln2hi_v, _mm512_fmadd_pd(e, ln2lo_v, log_m));
    result = _mm512_mask_blend_pd(invalid, result, neg_inf_v);
    return result;
}

[[nodiscard]] static inline __m512d k_exp_pd_avx512(__m512d x) noexcept {
    const __m512d uflow_v = _mm512_set1_pd(K_EXP_UNDERFLOW);
    const __m512d log2e_v = _mm512_set1_pd(K_LOG2E);
    const __m512d half_v = _mm512_set1_pd(0.5);
    const __m512d ln2hi_v = _mm512_set1_pd(K_LN2_HI);
    const __m512d ln2lo_v = _mm512_set1_pd(K_LN2_LO);
    const __m512d zero_v = _mm512_setzero_pd();
    const __mmask8 uflow = _mm512_cmp_pd_mask(x, uflow_v, _CMP_LE_OS);
    x = _mm512_max_pd(x, uflow_v);
    __m512d n = _mm512_floor_pd(_mm512_fmadd_pd(x, log2e_v, half_v));
    __m512d r = _mm512_fnmadd_pd(n, ln2hi_v, x);
    r = _mm512_fnmadd_pd(n, ln2lo_v, r);
    __m512d p = _mm512_set1_pd(K_EXP_C12);
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C11));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C10));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C9));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C8));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C7));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C6));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C5));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C4));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C3));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C2));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C1));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(K_EXP_C0));
    __m256i ni = _mm512_cvtpd_epi32(n);
    __m512i ni64 = _mm512_cvtepi32_epi64(ni);
    ni64 = _mm512_add_epi64(ni64, _mm512_set1_epi64(static_cast<long long>(K_EXPONENT_BIAS)));
    ni64 = _mm512_slli_epi64(ni64, 52);
    __m512d result = _mm512_mul_pd(p, _mm512_castsi512_pd(ni64));
    result = _mm512_mask_blend_pd(uflow, result, zero_v);
    return result;
}

#endif // LIBHMM_HAS_AVX512

// ---------------------------------------------------------------------------
// AVX helpers (AVX-1 compatible)
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)

/// FMA emulation for AVX-1 targets without native _mm256_fmadd_pd.
/// On AVX2+FMA3 hosts the compiler will typically fold this into vfmadd anyway.
/// Returns a * b + c.
[[nodiscard]] static inline __m256d k_fmadd_pd_avx(__m256d a, __m256d b, __m256d c) noexcept {
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
}

[[nodiscard]] static inline __m256d k_log_pd_avx(__m256d x) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    const __m256d neg_inf_v = _mm256_set1_pd(neg_inf);
    const __m256d sqrt2_v = _mm256_set1_pd(K_SQRT2);
    const __m256d one_v = _mm256_set1_pd(1.0);
    const __m256d half_v = _mm256_set1_pd(0.5);
    const __m256d two_v = _mm256_set1_pd(2.0);
    const __m256d ln2hi_v = _mm256_set1_pd(K_LN2_HI);
    const __m256d ln2lo_v = _mm256_set1_pd(K_LN2_LO);
    const __m256d invalid_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LE_OS);

    auto extract_em = [](__m128d xh, __m128d &mh, __m128d &eh) {
        __m128i bits = _mm_castpd_si128(xh);
        __m128i eb = _mm_srli_epi64(bits, 52);
        __m128i mm = _mm_set1_epi64x(0x000FFFFFFFFFFFFFLL);
        __m128i eo = _mm_set1_epi64x(0x3FF0000000000000LL);
        mh = _mm_castsi128_pd(_mm_or_si128(_mm_and_si128(bits, mm), eo));
        __m128i eu = _mm_sub_epi64(eb, _mm_set1_epi64x(1023LL));
        long long e0, e1;
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&e0), eu);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&e1), _mm_unpackhi_epi64(eu, eu));
        eh = _mm_set_pd(static_cast<double>(e1), static_cast<double>(e0));
    };

    __m128d m_lo, e_lo, m_hi, e_hi;
    extract_em(_mm256_castpd256_pd128(x), m_lo, e_lo);
    extract_em(_mm256_extractf128_pd(x, 1), m_hi, e_hi);
    __m256d m = _mm256_set_m128d(m_hi, m_lo);
    __m256d e = _mm256_set_m128d(e_hi, e_lo);

    __m256d adj = _mm256_cmp_pd(m, sqrt2_v, _CMP_GT_OS);
    e = _mm256_add_pd(e, _mm256_and_pd(adj, one_v));
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half_v), adj);

    __m256d y = _mm256_div_pd(_mm256_sub_pd(m, one_v), _mm256_add_pd(m, one_v));
    __m256d y2 = _mm256_mul_pd(y, y);

    __m256d p = _mm256_set1_pd(K_LOG_C6);
    p = k_fmadd_pd_avx(p, y2, _mm256_set1_pd(K_LOG_C5));
    p = k_fmadd_pd_avx(p, y2, _mm256_set1_pd(K_LOG_C4));
    p = k_fmadd_pd_avx(p, y2, _mm256_set1_pd(K_LOG_C3));
    p = k_fmadd_pd_avx(p, y2, _mm256_set1_pd(K_LOG_C2));
    p = k_fmadd_pd_avx(p, y2, _mm256_set1_pd(K_LOG_C1));
    p = k_fmadd_pd_avx(p, y2, _mm256_set1_pd(K_LOG_C0));
    __m256d log_m = _mm256_mul_pd(_mm256_mul_pd(two_v, y), p);
    __m256d result =
        _mm256_add_pd(_mm256_mul_pd(e, ln2hi_v), _mm256_add_pd(_mm256_mul_pd(e, ln2lo_v), log_m));
    result = _mm256_blendv_pd(result, neg_inf_v, invalid_mask);
    return result;
}

[[nodiscard]] static inline __m256d k_exp_pd_avx(__m256d x) noexcept {
    const __m256d uflow_v = _mm256_set1_pd(K_EXP_UNDERFLOW);
    const __m256d log2e_v = _mm256_set1_pd(K_LOG2E);
    const __m256d half_v = _mm256_set1_pd(0.5);
    const __m256d ln2hi_v = _mm256_set1_pd(K_LN2_HI);
    const __m256d ln2lo_v = _mm256_set1_pd(K_LN2_LO);
    const __m256d zero_v = _mm256_setzero_pd();
    const __m256d ufl_mask = _mm256_cmp_pd(x, uflow_v, _CMP_LE_OS);
    x = _mm256_max_pd(x, uflow_v);
    __m256d n = _mm256_floor_pd(_mm256_add_pd(_mm256_mul_pd(x, log2e_v), half_v));
    __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(n, ln2hi_v));
    r = _mm256_sub_pd(r, _mm256_mul_pd(n, ln2lo_v));

    __m256d p = _mm256_set1_pd(K_EXP_C12);
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C11));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C10));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C9));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C8));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C7));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C6));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C5));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C4));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C3));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C2));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C1));
    p = k_fmadd_pd_avx(p, r, _mm256_set1_pd(K_EXP_C0));

    __m128d n_lo = _mm256_castpd256_pd128(n), n_hi = _mm256_extractf128_pd(n, 1);
    auto bp2 = [](__m128d nd) {
        __m128i ni32 =
            _mm_add_epi32(_mm_cvttpd_epi32(nd), _mm_set1_epi32(static_cast<int>(K_EXPONENT_BIAS)));
        __m128i i64 = _mm_slli_epi64(_mm_unpacklo_epi32(ni32, _mm_setzero_si128()), 52);
        return _mm_castsi128_pd(i64);
    };
    __m256d result = _mm256_mul_pd(p, _mm256_set_m128d(bp2(n_hi), bp2(n_lo)));
    result = _mm256_blendv_pd(result, zero_v, ufl_mask);
    return result;
}

#endif // LIBHMM_HAS_AVX || LIBHMM_HAS_AVX2

// ---------------------------------------------------------------------------
// SSE2 helpers
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_SSE2)

/// FMA emulation for SSE2 targets without native _mm_fmadd_pd.
/// Returns a * b + c.
[[nodiscard]] static inline __m128d k_fmadd_pd_sse2(__m128d a, __m128d b, __m128d c) noexcept {
    return _mm_add_pd(_mm_mul_pd(a, b), c);
}

[[nodiscard]] static inline __m128d k_log_pd_sse2(__m128d x) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    const __m128d neg_inf_v = _mm_set1_pd(neg_inf);
    const __m128d sqrt2_v = _mm_set1_pd(K_SQRT2);
    const __m128d one_v = _mm_set1_pd(1.0);
    const __m128d half_v = _mm_set1_pd(0.5);
    const __m128d two_v = _mm_set1_pd(2.0);
    const __m128d ln2hi_v = _mm_set1_pd(K_LN2_HI);
    const __m128d ln2lo_v = _mm_set1_pd(K_LN2_LO);
    const __m128d invalid = _mm_cmple_pd(x, _mm_setzero_pd());
    __m128i bits = _mm_castpd_si128(x);
    __m128i eb = _mm_srli_epi64(bits, 52);
    __m128i mbits = _mm_or_si128(_mm_and_si128(bits, _mm_set1_epi64x(0x000FFFFFFFFFFFFFLL)),
                                 _mm_set1_epi64x(0x3FF0000000000000LL));
    __m128d m = _mm_castsi128_pd(mbits);
    __m128i eu = _mm_sub_epi64(eb, _mm_set1_epi64x(1023LL));
    long long e0, e1;
    _mm_storel_epi64(reinterpret_cast<__m128i *>(&e0), eu);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(&e1), _mm_unpackhi_epi64(eu, eu));
    __m128d e = _mm_set_pd(static_cast<double>(e1), static_cast<double>(e0));
    __m128d adj = _mm_cmpgt_pd(m, sqrt2_v);
    e = _mm_add_pd(e, _mm_and_pd(adj, one_v));
    m = _mm_or_pd(_mm_andnot_pd(adj, m), _mm_and_pd(adj, _mm_mul_pd(m, half_v)));
    __m128d y = _mm_div_pd(_mm_sub_pd(m, one_v), _mm_add_pd(m, one_v));
    __m128d y2 = _mm_mul_pd(y, y);
    __m128d p = _mm_set1_pd(K_LOG_C6);
    p = k_fmadd_pd_sse2(p, y2, _mm_set1_pd(K_LOG_C5));
    p = k_fmadd_pd_sse2(p, y2, _mm_set1_pd(K_LOG_C4));
    p = k_fmadd_pd_sse2(p, y2, _mm_set1_pd(K_LOG_C3));
    p = k_fmadd_pd_sse2(p, y2, _mm_set1_pd(K_LOG_C2));
    p = k_fmadd_pd_sse2(p, y2, _mm_set1_pd(K_LOG_C1));
    p = k_fmadd_pd_sse2(p, y2, _mm_set1_pd(K_LOG_C0));
    __m128d log_m = _mm_mul_pd(_mm_mul_pd(two_v, y), p);
    __m128d result = _mm_add_pd(_mm_mul_pd(e, ln2hi_v), _mm_add_pd(_mm_mul_pd(e, ln2lo_v), log_m));
    result = _mm_or_pd(_mm_andnot_pd(invalid, result), _mm_and_pd(invalid, neg_inf_v));
    return result;
}

[[nodiscard]] static inline __m128d k_exp_pd_sse2(__m128d x) noexcept {
    const __m128d uflow_v = _mm_set1_pd(K_EXP_UNDERFLOW);
    const __m128d log2e_v = _mm_set1_pd(K_LOG2E);
    const __m128d half_v = _mm_set1_pd(0.5);
    const __m128d ln2hi_v = _mm_set1_pd(K_LN2_HI);
    const __m128d ln2lo_v = _mm_set1_pd(K_LN2_LO);
    const __m128d zero_v = _mm_setzero_pd();
    const __m128d ufl = _mm_cmple_pd(x, uflow_v);
    x = _mm_max_pd(x, uflow_v);
    __m128d t = _mm_add_pd(_mm_mul_pd(x, log2e_v), half_v);
    __m128i ni = _mm_cvttpd_epi32(t);
    __m128d n = _mm_cvtepi32_pd(ni);
    n = _mm_sub_pd(n, _mm_and_pd(_mm_cmpgt_pd(n, t), _mm_set1_pd(1.0)));
    __m128d r = _mm_sub_pd(x, _mm_mul_pd(n, ln2hi_v));
    r = _mm_sub_pd(r, _mm_mul_pd(n, ln2lo_v));
    __m128d p = _mm_set1_pd(K_EXP_C12);
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C11));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C10));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C9));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C8));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C7));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C6));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C5));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C4));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C3));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C2));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C1));
    p = k_fmadd_pd_sse2(p, r, _mm_set1_pd(K_EXP_C0));
    __m128i ni32b =
        _mm_add_epi32(_mm_cvttpd_epi32(n), _mm_set1_epi32(static_cast<int>(K_EXPONENT_BIAS)));
    __m128i i64 = _mm_slli_epi64(_mm_unpacklo_epi32(ni32b, _mm_setzero_si128()), 52);
    __m128d result = _mm_mul_pd(p, _mm_castsi128_pd(i64));
    result = _mm_or_pd(_mm_andnot_pd(ufl, result), _mm_and_pd(ufl, zero_v));
    return result;
}

#endif // LIBHMM_HAS_SSE2

// ---------------------------------------------------------------------------
// NEON helpers
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_NEON)

[[nodiscard]] static inline float64x2_t k_log_pd_neon(float64x2_t x) noexcept {
    const float64x2_t neg_inf_v = vdupq_n_f64(-std::numeric_limits<double>::infinity());
    const float64x2_t sqrt2_v = vdupq_n_f64(K_SQRT2);
    const float64x2_t one_v = vdupq_n_f64(1.0);
    const float64x2_t half_v = vdupq_n_f64(0.5);
    const float64x2_t two_v = vdupq_n_f64(2.0);
    const float64x2_t ln2hi_v = vdupq_n_f64(K_LN2_HI);
    const float64x2_t ln2lo_v = vdupq_n_f64(K_LN2_LO);
    const uint64x2_t invalid = vcleq_f64(x, vdupq_n_f64(0.0));
    uint64x2_t bits = vreinterpretq_u64_f64(x);
    uint64x2_t eb = vshrq_n_u64(bits, 52);
    uint64x2_t mbits = vorrq_u64(vandq_u64(bits, vdupq_n_u64(0x000FFFFFFFFFFFFFULL)),
                                 vdupq_n_u64(0x3FF0000000000000ULL));
    float64x2_t m = vreinterpretq_f64_u64(mbits);
    float64x2_t e = vcvtq_f64_s64(vsubq_s64(vreinterpretq_s64_u64(eb), vdupq_n_s64(1023LL)));
    uint64x2_t adj = vcgtq_f64(m, sqrt2_v);
    e = vbslq_f64(adj, vaddq_f64(e, one_v), e);
    m = vbslq_f64(adj, vmulq_f64(m, half_v), m);
    float64x2_t y = vdivq_f64(vsubq_f64(m, one_v), vaddq_f64(m, one_v));
    float64x2_t y2 = vmulq_f64(y, y);
    float64x2_t p = vdupq_n_f64(K_LOG_C6);
    p = vfmaq_f64(vdupq_n_f64(K_LOG_C5), p, y2);
    p = vfmaq_f64(vdupq_n_f64(K_LOG_C4), p, y2);
    p = vfmaq_f64(vdupq_n_f64(K_LOG_C3), p, y2);
    p = vfmaq_f64(vdupq_n_f64(K_LOG_C2), p, y2);
    p = vfmaq_f64(vdupq_n_f64(K_LOG_C1), p, y2);
    p = vfmaq_f64(vdupq_n_f64(K_LOG_C0), p, y2);
    float64x2_t log_m = vmulq_f64(vmulq_f64(two_v, y), p);
    float64x2_t result = vfmaq_f64(vfmaq_f64(log_m, e, ln2lo_v), e, ln2hi_v);
    result = vbslq_f64(invalid, neg_inf_v, result);
    return result;
}

[[nodiscard]] static inline float64x2_t k_exp_pd_neon(float64x2_t x) noexcept {
    const float64x2_t uflow_v = vdupq_n_f64(K_EXP_UNDERFLOW);
    const float64x2_t log2e_v = vdupq_n_f64(K_LOG2E);
    const float64x2_t half_v = vdupq_n_f64(0.5);
    const float64x2_t ln2hi_v = vdupq_n_f64(K_LN2_HI);
    const float64x2_t ln2lo_v = vdupq_n_f64(K_LN2_LO);
    const float64x2_t zero_v = vdupq_n_f64(0.0);
    const uint64x2_t valid = vcgtq_f64(x, uflow_v);
    x = vmaxq_f64(x, uflow_v);
    float64x2_t n = vrndmq_f64(vfmaq_f64(half_v, x, log2e_v));
    float64x2_t r = vfmsq_f64(x, n, ln2hi_v);
    r = vfmsq_f64(r, n, ln2lo_v);
    float64x2_t p = vdupq_n_f64(K_EXP_C12);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C11), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C10), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C9), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C8), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C7), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C6), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C5), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C4), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C3), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C2), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C1), p, r);
    p = vfmaq_f64(vdupq_n_f64(K_EXP_C0), p, r);
    int64x2_t ni64 =
        vaddq_s64(vcvtq_s64_f64(n), vdupq_n_s64(static_cast<int64_t>(K_EXPONENT_BIAS)));
    float64x2_t result = vmulq_f64(p, vreinterpretq_f64_s64(vshlq_n_s64(ni64, 52)));
    result = vbslq_f64(valid, result, zero_v);
    return result;
}

#endif // LIBHMM_HAS_NEON

} // namespace kernels
} // namespace detail
} // namespace performance
} // namespace libhmm
