#pragma once
// include/libhmm/detail/simd_math_helpers.h
//
// Internal header — NOT part of the public API.
// Not installed to CMAKE_INSTALL_PREFIX/include (detail/ is excluded).
//
// Single source of truth for SIMD math primitives (log, exp, cos, log1p)
// shared between:
//   - src/performance/simd_double_ops_*.cpp  (distribution batch kernels)
//   - src/performance/transcendental_kernels.cpp  (FB recurrence kernels)
//
// Replaces simd_kernels_internal.h, which used older polynomial approximations.
// All implementations here are SLEEF-based (log/exp < 1 ULP; cos ~2e-10).
//
// Include only from .cpp files compiled with the appropriate SIMD flags.
// The ISA-specific sections are guarded by LIBHMM_HAS_* macros from simd_platform.h.
//
// Overloaded on SIMD register type: log_pd(__m512d), log_pd(__m256d),
// log_pd(__m128d), log_pd(float64x2_t) — callers use the same name regardless of
// the active ISA tier; the compiler selects the right overload.

#include "libhmm/platform/simd_platform.h"

#include <cmath>
#include <limits>

// ============================================================================
// Platform-specific intrinsic includes (file scope, before namespace)
// ============================================================================

#if defined(LIBHMM_HAS_SSE2) || defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2) ||             \
    defined(LIBHMM_HAS_AVX512)
#include <immintrin.h>
#endif

#if defined(LIBHMM_HAS_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace libhmm::detail::simd {

#if defined(LIBHMM_HAS_SSE2)
// SSE2 blend helper — no _mm_blendv_pd before SSE4.1.
// Selects trueValue where mask = all-ones, falseValue where mask = all-zeros.
[[nodiscard]] static inline __m128d sse2_blend(__m128d mask, __m128d trueValue,
                                               __m128d falseValue) noexcept {
    return _mm_or_pd(_mm_and_pd(mask, trueValue), _mm_andnot_pd(mask, falseValue));
}
#endif

// ============================================================================
// AVX-512 — 8-wide __m512d  (requires LIBHMM_HAS_AVX512)
// ============================================================================
#if defined(LIBHMM_HAS_AVX512)

// SLEEF xlog_u1 core, < 1 ULP. Uses _mm512_cvtepi64_pd (AVX-512DQ).
[[nodiscard]] static inline __m512d log_pd(__m512d x) noexcept {
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d ln2_hi = _mm512_set1_pd(0.693147180559945286226764);
    const __m512d ln2_lo = _mm512_set1_pd(2.319046813846299558417771e-17);
    const __m512d sqrt2 = _mm512_set1_pd(1.4142135623730950488016887242097);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d two = _mm512_set1_pd(2.0);
    const __m512d neg_inf = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    const __m512d pos_inf = _mm512_set1_pd(std::numeric_limits<double>::infinity());
    const __m512d zero = _mm512_setzero_pd();
    const __m512d c1 = _mm512_set1_pd(0.6666666666667333541e+0);
    const __m512d c2 = _mm512_set1_pd(0.3999999999635251990e+0);
    const __m512d c3 = _mm512_set1_pd(0.2857142932794299317e+0);
    const __m512d c4 = _mm512_set1_pd(0.2222214519839380009e+0);
    const __m512d c5 = _mm512_set1_pd(0.1818605932937785996e+0);
    const __m512d c6 = _mm512_set1_pd(0.1525629051003428716e+0);
    const __m512d c7 = _mm512_set1_pd(0.1532076988502701353e+0);

    __mmask8 is_le_zero = _mm512_cmp_pd_mask(x, zero, _CMP_LE_OQ);
    __mmask8 is_inf = _mm512_cmp_pd_mask(x, pos_inf, _CMP_EQ_OQ);
    __mmask8 is_nan = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
    const __m512d min_normal = _mm512_set1_pd(2.2250738585072014e-308);
    const __m512d scale_up = _mm512_set1_pd(18014398509481984.0); // 2^54
    __mmask8 is_denormal = _mm512_cmp_pd_mask(x, min_normal, _CMP_LT_OQ);
    __m512d sx = _mm512_mask_blend_pd(is_denormal, x, _mm512_mul_pd(x, scale_up));

    __m512i xi = _mm512_castpd_si512(sx);
    __m512i exp_i =
        _mm512_sub_epi64(_mm512_and_si512(_mm512_srli_epi64(xi, 52), _mm512_set1_epi64(0x7FF)),
                         _mm512_set1_epi64(1023));
    __m512d e = _mm512_cvtepi64_pd(exp_i); // AVX-512DQ
    e = _mm512_mask_blend_pd(is_denormal, e, _mm512_sub_pd(e, _mm512_set1_pd(54.0)));

    __m512i mant_i = _mm512_or_si512(_mm512_and_si512(xi, _mm512_set1_epi64(0x000FFFFFFFFFFFFFLL)),
                                     _mm512_set1_epi64(0x3FF0000000000000LL));
    __m512d m = _mm512_castsi512_pd(mant_i);

    __mmask8 needs_adj = _mm512_cmp_pd_mask(m, sqrt2, _CMP_GT_OQ);
    m = _mm512_mask_blend_pd(needs_adj, m, _mm512_mul_pd(m, half));
    e = _mm512_mask_blend_pd(needs_adj, e, _mm512_add_pd(e, one));

    __m512d xr = _mm512_div_pd(_mm512_sub_pd(m, one), _mm512_add_pd(m, one));
    __m512d xr2 = _mm512_mul_pd(xr, xr);
    __m512d t = c7;
    t = _mm512_fmadd_pd(t, xr2, c6);
    t = _mm512_fmadd_pd(t, xr2, c5);
    t = _mm512_fmadd_pd(t, xr2, c4);
    t = _mm512_fmadd_pd(t, xr2, c3);
    t = _mm512_fmadd_pd(t, xr2, c2);
    t = _mm512_fmadd_pd(t, xr2, c1);

    __m512d xr3 = _mm512_mul_pd(xr, xr2);
    __m512d log_m = _mm512_fmadd_pd(xr3, t, _mm512_mul_pd(xr, two));
    __m512d result = _mm512_fmadd_pd(e, ln2_hi, log_m);
    result = _mm512_fmadd_pd(e, ln2_lo, result);

    result = _mm512_mask_blend_pd(is_le_zero, result, neg_inf);
    result = _mm512_mask_blend_pd(is_inf, result, pos_inf);
    result = _mm512_mask_blend_pd(is_nan, result, x);
    return result;
}

// SLEEF-inspired exp, < 1 ULP.
[[nodiscard]] static inline __m512d exp_pd(__m512d x) noexcept {
    const __m512d ln2_inv = _mm512_set1_pd(1.4426950408889634073599246810019);
    const __m512d ln2_hi = _mm512_set1_pd(0.693147180369123816490e+00);
    const __m512d ln2_lo = _mm512_set1_pd(1.90821492927058770002e-10);
    const __m512d exp_max = _mm512_set1_pd(709.782712893383996732223);
    const __m512d exp_min = _mm512_set1_pd(-708.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d c1 = _mm512_set1_pd(0.1666666666666669072e+0);
    const __m512d c2 = _mm512_set1_pd(0.4166666666666602598e-1);
    const __m512d c3 = _mm512_set1_pd(0.8333333333314938210e-2);
    const __m512d c4 = _mm512_set1_pd(0.1388888888914497797e-2);
    const __m512d c5 = _mm512_set1_pd(0.1984126989855865850e-3);
    const __m512d c6 = _mm512_set1_pd(0.2480158687479686264e-4);
    const __m512d c7 = _mm512_set1_pd(0.2755723402025388239e-5);
    const __m512d c8 = _mm512_set1_pd(0.2755762628169491192e-6);
    const __m512d c9 = _mm512_set1_pd(0.2511210703042288022e-7);
    const __m512d c10 = _mm512_set1_pd(0.2081276378237164457e-8);

    x = _mm512_min_pd(x, exp_max);
    x = _mm512_max_pd(x, exp_min);
    __m512d n_float = _mm512_roundscale_pd(_mm512_mul_pd(x, ln2_inv), _MM_FROUND_TO_NEAREST_INT);
    __m512d r = _mm512_fnmadd_pd(n_float, ln2_hi, x);
    r = _mm512_fnmadd_pd(n_float, ln2_lo, r);
    __m512d r2 = _mm512_mul_pd(r, r);
    __m512d poly = c10;
    poly = _mm512_fmadd_pd(poly, r, c9);
    poly = _mm512_fmadd_pd(poly, r, c8);
    poly = _mm512_fmadd_pd(poly, r, c7);
    poly = _mm512_fmadd_pd(poly, r, c6);
    poly = _mm512_fmadd_pd(poly, r, c5);
    poly = _mm512_fmadd_pd(poly, r, c4);
    poly = _mm512_fmadd_pd(poly, r, c3);
    poly = _mm512_fmadd_pd(poly, r, c2);
    poly = _mm512_fmadd_pd(poly, r, c1);
    poly = _mm512_fmadd_pd(poly, r, half);
    poly = _mm512_fmadd_pd(poly, r2, r);
    poly = _mm512_add_pd(poly, one);
    __m256i n_i32 = _mm512_cvtpd_epi32(n_float);
    __m512i n_i64 = _mm512_cvtepi32_epi64(n_i32);
    __m512i ebits = _mm512_slli_epi64(_mm512_add_epi64(n_i64, _mm512_set1_epi64(1023)), 52);
    return _mm512_mul_pd(poly, _mm512_castsi512_pd(ebits));
}

// 7-term Horner cosine, max error ≈ 1×10⁻¹⁰.
[[nodiscard]] static inline __m512d cos_pd(__m512d x) noexcept {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kHalfPi = 1.5707963267948966192313216916397514421;
    const __m512d inv2pi = _mm512_set1_pd(1.0 / (2.0 * kPi));
    const __m512d two_pi = _mm512_set1_pd(2.0 * kPi);
    const __m512d pi = _mm512_set1_pd(kPi);
    const __m512d half_pi = _mm512_set1_pd(kHalfPi);
    const __m512d neg_pi = _mm512_set1_pd(-kPi);
    const __m512d nhalf_pi = _mm512_set1_pd(-kHalfPi);
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d neg_one = _mm512_set1_pd(-1.0);
    const __m512d c1 = _mm512_set1_pd(-0.5);
    const __m512d c2 = _mm512_set1_pd(4.166666666666667e-2);
    const __m512d c3 = _mm512_set1_pd(-1.388888888888889e-3);
    const __m512d c4 = _mm512_set1_pd(2.480158730158730e-5);
    const __m512d c5 = _mm512_set1_pd(-2.755731922398589e-7);
    const __m512d c6 = _mm512_set1_pd(2.087675698786810e-9);
    const __m512d c7 = _mm512_set1_pd(-1.147074559772973e-11);

    __m512d q = _mm512_roundscale_pd(_mm512_mul_pd(x, inv2pi), _MM_FROUND_TO_NEAREST_INT);
    __m512d y = _mm512_sub_pd(x, _mm512_mul_pd(q, two_pi));
    __m512d sign = one;
    __mmask8 gt = _mm512_cmp_pd_mask(y, half_pi, _CMP_GT_OQ);
    __mmask8 lt = _mm512_cmp_pd_mask(y, nhalf_pi, _CMP_LT_OQ);
    y = _mm512_mask_blend_pd(gt, y, _mm512_sub_pd(pi, y));
    sign = _mm512_mask_blend_pd(gt, sign, neg_one);
    y = _mm512_mask_blend_pd(lt, y, _mm512_sub_pd(neg_pi, y));
    sign = _mm512_mask_blend_pd(lt, sign, neg_one);
    __m512d y2 = _mm512_mul_pd(y, y);
    __m512d poly = c7;
    poly = _mm512_fmadd_pd(y2, poly, c6);
    poly = _mm512_fmadd_pd(y2, poly, c5);
    poly = _mm512_fmadd_pd(y2, poly, c4);
    poly = _mm512_fmadd_pd(y2, poly, c3);
    poly = _mm512_fmadd_pd(y2, poly, c2);
    poly = _mm512_fmadd_pd(y2, poly, c1);
    poly = _mm512_fmadd_pd(y2, poly, one);
    return _mm512_mul_pd(poly, sign);
}

// log1p: log(1+x). Uses 8-term polynomial for |x|<1e-4 to avoid catastrophic
// cancellation in 1+x for small x (where log(1+x)=0 due to rounding).
[[nodiscard]] static inline __m512d log1p_pd(__m512d x) noexcept {
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d thr = _mm512_set1_pd(1.0e-4);
    const __m512d nthr = _mm512_set1_pd(-1.0e-4);
    // Horner: x*(1 - x/2 + x²/3 - ... + x⁷/8)
    __m512d p = _mm512_set1_pd(-0.125);
    p = _mm512_fmadd_pd(p, x, _mm512_set1_pd(1.0 / 7.0));
    p = _mm512_fmadd_pd(p, x, _mm512_set1_pd(-1.0 / 6.0));
    p = _mm512_fmadd_pd(p, x, _mm512_set1_pd(0.2));
    p = _mm512_fmadd_pd(p, x, _mm512_set1_pd(-0.25));
    p = _mm512_fmadd_pd(p, x, _mm512_set1_pd(1.0 / 3.0));
    p = _mm512_fmadd_pd(p, x, _mm512_set1_pd(-0.5));
    p = _mm512_fmadd_pd(p, x, one);
    const __m512d small = _mm512_mul_pd(x, p);
    const __m512d general = log_pd(_mm512_add_pd(one, x));
    const __mmask8 sm = _mm512_kand(_mm512_cmp_pd_mask(x, thr, _CMP_LT_OS),
                                    _mm512_cmp_pd_mask(x, nthr, _CMP_GT_OS));
    return _mm512_mask_blend_pd(sm, general, small);
}

#endif // LIBHMM_HAS_AVX512

// ============================================================================
// AVX/AVX2 — 4-wide __m256d  (requires LIBHMM_HAS_AVX or LIBHMM_HAS_AVX2)
// ============================================================================
#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)

// SLEEF xlog_u1 core, < 1 ULP. int64→double via store-and-reload (no AVX-512DQ).
[[nodiscard]] static inline __m256d log_pd(__m256d x) noexcept {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d ln2_hi = _mm256_set1_pd(0.693147180559945286226764);
    const __m256d ln2_lo = _mm256_set1_pd(2.319046813846299558417771e-17);
    const __m256d sqrt2 = _mm256_set1_pd(1.4142135623730950488016887242097);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    const __m256d zero = _mm256_setzero_pd();
    const __m256d c1 = _mm256_set1_pd(0.6666666666667333541e+0);
    const __m256d c2 = _mm256_set1_pd(0.3999999999635251990e+0);
    const __m256d c3 = _mm256_set1_pd(0.2857142932794299317e+0);
    const __m256d c4 = _mm256_set1_pd(0.2222214519839380009e+0);
    const __m256d c5 = _mm256_set1_pd(0.1818605932937785996e+0);
    const __m256d c6 = _mm256_set1_pd(0.1525629051003428716e+0);
    const __m256d c7 = _mm256_set1_pd(0.1532076988502701353e+0);

    __m256d is_le_zero = _mm256_cmp_pd(x, zero, _CMP_LE_OQ);
    __m256d is_inf = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);
    __m256d is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    const __m256d min_normal = _mm256_set1_pd(2.2250738585072014e-308);
    const __m256d scale_up = _mm256_set1_pd(18014398509481984.0);
    __m256d is_denormal = _mm256_cmp_pd(x, min_normal, _CMP_LT_OQ);
    __m256d sx = _mm256_blendv_pd(x, _mm256_mul_pd(x, scale_up), is_denormal);

    __m256i xi = _mm256_castpd_si256(sx);
    __m128i xi_lo = _mm256_castsi256_si128(xi);
    __m128i xi_hi = _mm256_extractf128_si256(xi, 1);
    __m128i emask = _mm_set1_epi64x(0x7FF);
    __m128i ibias = _mm_set1_epi64x(1023);
    __m128i exp_lo = _mm_sub_epi64(_mm_and_si128(_mm_srli_epi64(xi_lo, 52), emask), ibias);
    __m128i exp_hi = _mm_sub_epi64(_mm_and_si128(_mm_srli_epi64(xi_hi, 52), emask), ibias);
    alignas(16) long long elo[2], ehi_arr[2];
    _mm_store_si128(reinterpret_cast<__m128i *>(elo), exp_lo);
    _mm_store_si128(reinterpret_cast<__m128i *>(ehi_arr), exp_hi);
    __m128d elo_d = _mm_set_pd(static_cast<double>(elo[1]), static_cast<double>(elo[0]));
    __m128d ehi_d = _mm_set_pd(static_cast<double>(ehi_arr[1]), static_cast<double>(ehi_arr[0]));
    __m256d e = _mm256_set_m128d(ehi_d, elo_d);
    e = _mm256_blendv_pd(e, _mm256_sub_pd(e, _mm256_set1_pd(54.0)), is_denormal);

    __m128i mmask = _mm_set1_epi64x(0x000FFFFFFFFFFFFF);
    __m128i exp_bias = _mm_set1_epi64x(0x3FF0000000000000);
    __m128i m_lo = _mm_or_si128(_mm_and_si128(xi_lo, mmask), exp_bias);
    __m128i m_hi = _mm_or_si128(_mm_and_si128(xi_hi, mmask), exp_bias);
    __m256d m = _mm256_set_m128d(_mm_castsi128_pd(m_hi), _mm_castsi128_pd(m_lo));

    __m256d needs_adj = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), needs_adj);
    e = _mm256_blendv_pd(e, _mm256_add_pd(e, one), needs_adj);

    __m256d xr = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
    __m256d xr2 = _mm256_mul_pd(xr, xr);
    __m256d t = c7;
    t = _mm256_fmadd_pd(t, xr2, c6);
    t = _mm256_fmadd_pd(t, xr2, c5);
    t = _mm256_fmadd_pd(t, xr2, c4);
    t = _mm256_fmadd_pd(t, xr2, c3);
    t = _mm256_fmadd_pd(t, xr2, c2);
    t = _mm256_fmadd_pd(t, xr2, c1);

    __m256d xr3 = _mm256_mul_pd(xr, xr2);
    __m256d log_m = _mm256_fmadd_pd(xr3, t, _mm256_mul_pd(xr, two));
    __m256d result = _mm256_fmadd_pd(e, ln2_hi, log_m);
    result = _mm256_fmadd_pd(e, ln2_lo, result);

    result = _mm256_blendv_pd(result, neg_inf, is_le_zero);
    result = _mm256_blendv_pd(result, pos_inf, is_inf);
    result = _mm256_blendv_pd(result, x, is_nan);
    return result;
}

// SLEEF-inspired exp, < 1 ULP. FMA range reduction; 2^n via 32-bit round-trip.
[[nodiscard]] static inline __m256d exp_pd(__m256d x) noexcept {
    const __m256d ln2_inv = _mm256_set1_pd(1.4426950408889634073599246810019);
    const __m256d ln2_hi = _mm256_set1_pd(0.693147180369123816490e+00);
    const __m256d ln2_lo = _mm256_set1_pd(1.90821492927058770002e-10);
    const __m256d exp_max = _mm256_set1_pd(709.782712893383996732223);
    const __m256d exp_min = _mm256_set1_pd(-708.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d c1 = _mm256_set1_pd(0.1666666666666669072e+0);
    const __m256d c2 = _mm256_set1_pd(0.4166666666666602598e-1);
    const __m256d c3 = _mm256_set1_pd(0.8333333333314938210e-2);
    const __m256d c4 = _mm256_set1_pd(0.1388888888914497797e-2);
    const __m256d c5 = _mm256_set1_pd(0.1984126989855865850e-3);
    const __m256d c6 = _mm256_set1_pd(0.2480158687479686264e-4);
    const __m256d c7 = _mm256_set1_pd(0.2755723402025388239e-5);
    const __m256d c8 = _mm256_set1_pd(0.2755762628169491192e-6);
    const __m256d c9 = _mm256_set1_pd(0.2511210703042288022e-7);
    const __m256d c10 = _mm256_set1_pd(0.2081276378237164457e-8);

    x = _mm256_min_pd(x, exp_max);
    x = _mm256_max_pd(x, exp_min);
    __m256d n_float =
        _mm256_round_pd(_mm256_mul_pd(x, ln2_inv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(n_float, ln2_hi, x);
    r = _mm256_fnmadd_pd(n_float, ln2_lo, r);
    __m256d r2 = _mm256_mul_pd(r, r);
    __m256d poly = c10;
    poly = _mm256_fmadd_pd(poly, r, c9);
    poly = _mm256_fmadd_pd(poly, r, c8);
    poly = _mm256_fmadd_pd(poly, r, c7);
    poly = _mm256_fmadd_pd(poly, r, c6);
    poly = _mm256_fmadd_pd(poly, r, c5);
    poly = _mm256_fmadd_pd(poly, r, c4);
    poly = _mm256_fmadd_pd(poly, r, c3);
    poly = _mm256_fmadd_pd(poly, r, c2);
    poly = _mm256_fmadd_pd(poly, r, c1);
    poly = _mm256_fmadd_pd(poly, r, half);
    poly = _mm256_fmadd_pd(poly, r2, r);
    poly = _mm256_add_pd(poly, one);
    __m128i n_int = _mm256_cvtpd_epi32(n_float);
    __m128i ebits = _mm_add_epi32(n_int, _mm_set1_epi32(1023));
    __m128i elo = _mm_slli_epi64(_mm_cvtepi32_epi64(ebits), 52);
    __m128i ehi = _mm_slli_epi64(_mm_cvtepi32_epi64(_mm_shuffle_epi32(ebits, 0x0E)), 52);
    __m256d scale = _mm256_set_m128d(_mm_castsi128_pd(ehi), _mm_castsi128_pd(elo));
    return _mm256_mul_pd(poly, scale);
}

// 7-term Horner cosine with FMA, max error ≈ 1×10⁻¹⁰.
[[nodiscard]] static inline __m256d cos_pd(__m256d x) noexcept {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kHalfPi = 1.5707963267948966192313216916397514421;
    const __m256d inv2pi = _mm256_set1_pd(1.0 / (2.0 * kPi));
    const __m256d two_pi = _mm256_set1_pd(2.0 * kPi);
    const __m256d pi = _mm256_set1_pd(kPi);
    const __m256d half_pi = _mm256_set1_pd(kHalfPi);
    const __m256d neg_pi = _mm256_set1_pd(-kPi);
    const __m256d nhalf_pi = _mm256_set1_pd(-kHalfPi);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);
    const __m256d c1 = _mm256_set1_pd(-0.5);
    const __m256d c2 = _mm256_set1_pd(4.166666666666667e-2);
    const __m256d c3 = _mm256_set1_pd(-1.388888888888889e-3);
    const __m256d c4 = _mm256_set1_pd(2.480158730158730e-5);
    const __m256d c5 = _mm256_set1_pd(-2.755731922398589e-7);
    const __m256d c6 = _mm256_set1_pd(2.087675698786810e-9);
    const __m256d c7 = _mm256_set1_pd(-1.147074559772973e-11);

    __m256d q =
        _mm256_round_pd(_mm256_mul_pd(x, inv2pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d y = _mm256_sub_pd(x, _mm256_mul_pd(q, two_pi));
    __m256d sign = one;
    __m256d gt = _mm256_cmp_pd(y, half_pi, _CMP_GT_OQ);
    __m256d lt = _mm256_cmp_pd(y, nhalf_pi, _CMP_LT_OQ);
    y = _mm256_blendv_pd(y, _mm256_sub_pd(pi, y), gt);
    sign = _mm256_blendv_pd(sign, neg_one, gt);
    y = _mm256_blendv_pd(y, _mm256_sub_pd(neg_pi, y), lt);
    sign = _mm256_blendv_pd(sign, neg_one, lt);
    __m256d y2 = _mm256_mul_pd(y, y);
    __m256d poly = c7;
    poly = _mm256_fmadd_pd(y2, poly, c6);
    poly = _mm256_fmadd_pd(y2, poly, c5);
    poly = _mm256_fmadd_pd(y2, poly, c4);
    poly = _mm256_fmadd_pd(y2, poly, c3);
    poly = _mm256_fmadd_pd(y2, poly, c2);
    poly = _mm256_fmadd_pd(y2, poly, c1);
    poly = _mm256_fmadd_pd(y2, poly, one);
    return _mm256_mul_pd(poly, sign);
}

// log1p: 8-term polynomial for |x|<1e-4 to avoid catastrophic cancellation.
[[nodiscard]] static inline __m256d log1p_pd(__m256d x) noexcept {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d thr = _mm256_set1_pd(1.0e-4);
    const __m256d nthr = _mm256_set1_pd(-1.0e-4);
    __m256d p = _mm256_set1_pd(-0.125);
    p = _mm256_fmadd_pd(p, x, _mm256_set1_pd(1.0 / 7.0));
    p = _mm256_fmadd_pd(p, x, _mm256_set1_pd(-1.0 / 6.0));
    p = _mm256_fmadd_pd(p, x, _mm256_set1_pd(0.2));
    p = _mm256_fmadd_pd(p, x, _mm256_set1_pd(-0.25));
    p = _mm256_fmadd_pd(p, x, _mm256_set1_pd(1.0 / 3.0));
    p = _mm256_fmadd_pd(p, x, _mm256_set1_pd(-0.5));
    p = _mm256_fmadd_pd(p, x, one);
    const __m256d small = _mm256_mul_pd(x, p);
    const __m256d general = log_pd(_mm256_add_pd(one, x));
    const __m256d sm =
        _mm256_and_pd(_mm256_cmp_pd(x, thr, _CMP_LT_OS), _mm256_cmp_pd(x, nthr, _CMP_GT_OS));
    return _mm256_blendv_pd(general, small, sm);
}

#endif // LIBHMM_HAS_AVX || LIBHMM_HAS_AVX2

// ============================================================================
// SSE2 — 2-wide __m128d  (requires LIBHMM_HAS_SSE2)
// ============================================================================
#if defined(LIBHMM_HAS_SSE2)

// SLEEF xlog_u1 core, < 1 ULP. Magic-number int64→double (no SSE4.1).
[[nodiscard]] static inline __m128d log_pd(__m128d x) noexcept {
    const __m128d one = _mm_set1_pd(1.0);
    const __m128d half = _mm_set1_pd(0.5);
    const __m128d two = _mm_set1_pd(2.0);
    const __m128d ln2_hi = _mm_set1_pd(0.693147180559945286226764);
    const __m128d ln2_lo = _mm_set1_pd(2.319046813846299558417771e-17);
    const __m128d sqrt2 = _mm_set1_pd(1.4142135623730950488016887242097);
    const __m128d neg_inf = _mm_set1_pd(-std::numeric_limits<double>::infinity());
    const __m128d pos_inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
    const __m128d zero = _mm_setzero_pd();
    const __m128d c1 = _mm_set1_pd(0.6666666666667333541e+0);
    const __m128d c2 = _mm_set1_pd(0.3999999999635251990e+0);
    const __m128d c3 = _mm_set1_pd(0.2857142932794299317e+0);
    const __m128d c4 = _mm_set1_pd(0.2222214519839380009e+0);
    const __m128d c5 = _mm_set1_pd(0.1818605932937785996e+0);
    const __m128d c6 = _mm_set1_pd(0.1525629051003428716e+0);
    const __m128d c7 = _mm_set1_pd(0.1532076988502701353e+0);

    const __m128d is_zero = _mm_cmpeq_pd(x, zero);
    const __m128d is_non_positive = _mm_cmple_pd(x, zero);
    const __m128d is_inf = _mm_cmpeq_pd(x, pos_inf);
    const __m128d is_nan = _mm_cmpunord_pd(x, x);

    const __m128i xi = _mm_castpd_si128(x);
    __m128i exp_i = _mm_srli_epi64(xi, 52);
    exp_i = _mm_and_si128(exp_i, _mm_set1_epi64x(0x7FFLL));
    const __m128i exp_i32 = _mm_shuffle_epi32(exp_i, _MM_SHUFFLE(0, 0, 2, 0));
    __m128d e = _mm_cvtepi32_pd(exp_i32);
    e = _mm_sub_pd(e, _mm_set1_pd(1023.0));

    __m128d m =
        _mm_castsi128_pd(_mm_or_si128(_mm_and_si128(xi, _mm_set1_epi64x(0x000FFFFFFFFFFFFFLL)),
                                      _mm_set1_epi64x(0x3FF0000000000000LL)));

    const __m128d need_adj = _mm_cmpgt_pd(m, sqrt2);
    m = sse2_blend(need_adj, _mm_mul_pd(m, half), m);
    e = sse2_blend(need_adj, _mm_add_pd(e, one), e);

    const __m128d xr = _mm_div_pd(_mm_sub_pd(m, one), _mm_add_pd(m, one));
    const __m128d xr2 = _mm_mul_pd(xr, xr);
    __m128d t = c7;
    t = _mm_add_pd(_mm_mul_pd(t, xr2), c6);
    t = _mm_add_pd(_mm_mul_pd(t, xr2), c5);
    t = _mm_add_pd(_mm_mul_pd(t, xr2), c4);
    t = _mm_add_pd(_mm_mul_pd(t, xr2), c3);
    t = _mm_add_pd(_mm_mul_pd(t, xr2), c2);
    t = _mm_add_pd(_mm_mul_pd(t, xr2), c1);

    const __m128d xr3 = _mm_mul_pd(xr, xr2);
    const __m128d log_m = _mm_add_pd(_mm_mul_pd(xr, two), _mm_mul_pd(xr3, t));
    __m128d result = _mm_add_pd(_mm_mul_pd(e, ln2_hi), log_m);
    result = _mm_add_pd(result, _mm_mul_pd(e, ln2_lo));

    result = sse2_blend(is_zero, neg_inf, result);
    result = sse2_blend(is_inf, pos_inf, result);
    result = sse2_blend(is_non_positive, neg_inf, result);
    result = sse2_blend(is_nan, x, result);
    return result;
}

// SLEEF-inspired exp, < 1 ULP. Magic-number rounding (no _mm_round_pd).
[[nodiscard]] static inline __m128d exp_pd(__m128d x) noexcept {
    const __m128d original = x;
    const __m128d nan_mask = _mm_cmpunord_pd(x, x);
    const __m128d ln2_inv = _mm_set1_pd(1.4426950408889634073599246810019);
    const __m128d ln2_hi = _mm_set1_pd(0.693147180369123816490e+00);
    const __m128d ln2_lo = _mm_set1_pd(1.90821492927058770002e-10);
    const __m128d exp_max = _mm_set1_pd(709.782712893383996732223);
    const __m128d exp_min = _mm_set1_pd(-708.0);
    const __m128d half = _mm_set1_pd(0.5);
    const __m128d one = _mm_set1_pd(1.0);
    const __m128d magic = _mm_set1_pd(6755399441055744.0);
    const __m128d c1 = _mm_set1_pd(0.1666666666666669072e+0);
    const __m128d c2 = _mm_set1_pd(0.4166666666666602598e-1);
    const __m128d c3 = _mm_set1_pd(0.8333333333314938210e-2);
    const __m128d c4 = _mm_set1_pd(0.1388888888914497797e-2);
    const __m128d c5 = _mm_set1_pd(0.1984126989855865850e-3);
    const __m128d c6 = _mm_set1_pd(0.2480158687479686264e-4);
    const __m128d c7 = _mm_set1_pd(0.2755723402025388239e-5);
    const __m128d c8 = _mm_set1_pd(0.2755762628169491192e-6);
    const __m128d c9 = _mm_set1_pd(0.2511210703042288022e-7);
    const __m128d c10 = _mm_set1_pd(0.2081276378237164457e-8);

    x = _mm_min_pd(x, exp_max);
    x = _mm_max_pd(x, exp_min);
    __m128d n_float = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(x, ln2_inv), magic), magic);
    __m128d r = _mm_sub_pd(x, _mm_mul_pd(n_float, ln2_hi));
    r = _mm_sub_pd(r, _mm_mul_pd(n_float, ln2_lo));
    const __m128d r2 = _mm_mul_pd(r, r);
    __m128d poly = c10;
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c9);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c8);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c7);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c6);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c5);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c4);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c3);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c2);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), c1);
    poly = _mm_add_pd(_mm_mul_pd(poly, r), half);
    poly = _mm_add_pd(_mm_mul_pd(poly, r2), r);
    poly = _mm_add_pd(poly, one);
    const __m128i n_i32 = _mm_cvttpd_epi32(n_float);
    const __m128i n_i64 = _mm_unpacklo_epi32(n_i32, _mm_setzero_si128());
    __m128i ebits = _mm_add_epi64(n_i64, _mm_set1_epi64x(1023LL));
    ebits = _mm_slli_epi64(ebits, 52);
    const __m128d result = _mm_mul_pd(poly, _mm_castsi128_pd(ebits));
    return sse2_blend(nan_mask, original, result);
}

// 7-term Horner cosine, magic-number range reduction.
[[nodiscard]] static inline __m128d cos_pd(__m128d x) noexcept {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kHalfPi = 1.5707963267948966192313216916397514421;
    const __m128d inv2pi = _mm_set1_pd(1.0 / (2.0 * kPi));
    const __m128d two_pi = _mm_set1_pd(2.0 * kPi);
    const __m128d pi = _mm_set1_pd(kPi);
    const __m128d half_pi = _mm_set1_pd(kHalfPi);
    const __m128d neg_pi = _mm_set1_pd(-kPi);
    const __m128d nhalf_pi = _mm_set1_pd(-kHalfPi);
    const __m128d one = _mm_set1_pd(1.0);
    const __m128d neg_one = _mm_set1_pd(-1.0);
    const __m128d magic = _mm_set1_pd(6755399441055744.0);
    const __m128d c1 = _mm_set1_pd(-0.5);
    const __m128d c2 = _mm_set1_pd(4.166666666666667e-2);
    const __m128d c3 = _mm_set1_pd(-1.388888888888889e-3);
    const __m128d c4 = _mm_set1_pd(2.480158730158730e-5);
    const __m128d c5 = _mm_set1_pd(-2.755731922398589e-7);
    const __m128d c6 = _mm_set1_pd(2.087675698786810e-9);
    const __m128d c7 = _mm_set1_pd(-1.147074559772973e-11);

    const __m128d q = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(x, inv2pi), magic), magic);
    __m128d y = _mm_sub_pd(x, _mm_mul_pd(q, two_pi));
    __m128d sign = one;
    const __m128d gt = _mm_cmpgt_pd(y, half_pi);
    const __m128d lt = _mm_cmplt_pd(y, nhalf_pi);
    y = sse2_blend(gt, _mm_sub_pd(pi, y), y);
    sign = sse2_blend(gt, neg_one, sign);
    y = sse2_blend(lt, _mm_sub_pd(neg_pi, y), y);
    sign = sse2_blend(lt, neg_one, sign);
    const __m128d y2 = _mm_mul_pd(y, y);
    __m128d poly = c7;
    poly = _mm_add_pd(c6, _mm_mul_pd(y2, poly));
    poly = _mm_add_pd(c5, _mm_mul_pd(y2, poly));
    poly = _mm_add_pd(c4, _mm_mul_pd(y2, poly));
    poly = _mm_add_pd(c3, _mm_mul_pd(y2, poly));
    poly = _mm_add_pd(c2, _mm_mul_pd(y2, poly));
    poly = _mm_add_pd(c1, _mm_mul_pd(y2, poly));
    poly = _mm_add_pd(one, _mm_mul_pd(y2, poly));
    return _mm_mul_pd(poly, sign);
}

// log1p: 8-term polynomial for |x|<1e-4. No FMA on SSE2 — uses mul+add.
[[nodiscard]] static inline __m128d log1p_pd(__m128d x) noexcept {
    const __m128d one = _mm_set1_pd(1.0);
    const __m128d thr = _mm_set1_pd(1.0e-4);
    const __m128d nthr = _mm_set1_pd(-1.0e-4);
    __m128d p = _mm_set1_pd(-0.125);
    p = _mm_add_pd(_mm_mul_pd(p, x), _mm_set1_pd(1.0 / 7.0));
    p = _mm_add_pd(_mm_mul_pd(p, x), _mm_set1_pd(-1.0 / 6.0));
    p = _mm_add_pd(_mm_mul_pd(p, x), _mm_set1_pd(0.2));
    p = _mm_add_pd(_mm_mul_pd(p, x), _mm_set1_pd(-0.25));
    p = _mm_add_pd(_mm_mul_pd(p, x), _mm_set1_pd(1.0 / 3.0));
    p = _mm_add_pd(_mm_mul_pd(p, x), _mm_set1_pd(-0.5));
    p = _mm_add_pd(_mm_mul_pd(p, x), one);
    const __m128d small = _mm_mul_pd(x, p);
    const __m128d general = log_pd(_mm_add_pd(one, x));
    const __m128d sm = _mm_and_pd(_mm_cmplt_pd(x, thr), _mm_cmpgt_pd(x, nthr));
    return sse2_blend(sm, small, general);
}

#endif // LIBHMM_HAS_SSE2

// ============================================================================
// NEON — 2-wide float64x2_t  (requires AArch64)
// ============================================================================
#if defined(LIBHMM_HAS_NEON) && defined(__aarch64__)

// SLEEF xlog_u1 core, < 1 ULP. vcvtq_f64_s64 native on AArch64.
[[nodiscard]] static inline float64x2_t log_pd(float64x2_t x) noexcept {
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t ln2_hi = vdupq_n_f64(0.693147180559945286226764);
    const float64x2_t ln2_lo = vdupq_n_f64(2.319046813846299558417771e-17);
    const float64x2_t sqrt2 = vdupq_n_f64(1.4142135623730950488016887242097);
    const float64x2_t half = vdupq_n_f64(0.5);
    const float64x2_t two = vdupq_n_f64(2.0);
    const float64x2_t neg_inf = vdupq_n_f64(-std::numeric_limits<double>::infinity());
    const float64x2_t pos_inf = vdupq_n_f64(std::numeric_limits<double>::infinity());
    const float64x2_t zero = vdupq_n_f64(0.0);
    const float64x2_t c1 = vdupq_n_f64(0.6666666666667333541e+0);
    const float64x2_t c2 = vdupq_n_f64(0.3999999999635251990e+0);
    const float64x2_t c3 = vdupq_n_f64(0.2857142932794299317e+0);
    const float64x2_t c4 = vdupq_n_f64(0.2222214519839380009e+0);
    const float64x2_t c5 = vdupq_n_f64(0.1818605932937785996e+0);
    const float64x2_t c6 = vdupq_n_f64(0.1525629051003428716e+0);
    const float64x2_t c7 = vdupq_n_f64(0.1532076988502701353e+0);

    uint64x2_t is_le_zero = vcleq_f64(x, zero);
    uint64x2_t is_inf = vceqq_f64(x, pos_inf);
    uint64x2_t is_not_nan = vreinterpretq_u64_f64(vceqq_f64(x, x));
    uint64x2_t is_nan = veorq_u64(is_not_nan, vdupq_n_u64(~0ULL));

    const float64x2_t min_normal = vdupq_n_f64(2.2250738585072014e-308);
    const float64x2_t scale_up = vdupq_n_f64(18014398509481984.0);
    uint64x2_t is_denormal = vcltq_f64(x, min_normal);
    float64x2_t sx = vbslq_f64(is_denormal, vmulq_f64(x, scale_up), x);

    uint64x2_t xi = vreinterpretq_u64_f64(sx);
    int64x2_t e_int =
        vsubq_s64(vreinterpretq_s64_u64(vandq_u64(vshrq_n_u64(xi, 52), vdupq_n_u64(0x7FFULL))),
                  vdupq_n_s64(1023));
    float64x2_t e = vcvtq_f64_s64(e_int);
    e = vbslq_f64(is_denormal, vsubq_f64(e, vdupq_n_f64(54.0)), e);

    uint64x2_t m_bits = vorrq_u64(vandq_u64(xi, vdupq_n_u64(0x000FFFFFFFFFFFFFULL)),
                                  vdupq_n_u64(0x3FF0000000000000ULL));
    float64x2_t m = vreinterpretq_f64_u64(m_bits);

    uint64x2_t needs_adj = vcgtq_f64(m, sqrt2);
    m = vbslq_f64(needs_adj, vmulq_f64(m, half), m);
    e = vbslq_f64(needs_adj, vaddq_f64(e, one), e);

    float64x2_t xr = vdivq_f64(vsubq_f64(m, one), vaddq_f64(m, one));
    float64x2_t xr2 = vmulq_f64(xr, xr);
    float64x2_t t = c7;
    t = vfmaq_f64(c6, t, xr2);
    t = vfmaq_f64(c5, t, xr2);
    t = vfmaq_f64(c4, t, xr2);
    t = vfmaq_f64(c3, t, xr2);
    t = vfmaq_f64(c2, t, xr2);
    t = vfmaq_f64(c1, t, xr2);

    float64x2_t xr3 = vmulq_f64(xr, xr2);
    float64x2_t log_m = vfmaq_f64(vmulq_f64(xr, two), xr3, t);
    float64x2_t res = vfmaq_f64(log_m, e, ln2_hi);
    res = vfmaq_f64(res, e, ln2_lo);

    res = vbslq_f64(is_le_zero, neg_inf, res);
    res = vbslq_f64(is_inf, pos_inf, res);
    res = vbslq_f64(is_nan, x, res);
    return res;
}

// SLEEF-inspired exp, < 1 ULP. vcvtq_s64_f64 + vshlq_n_s64 on AArch64.
[[nodiscard]] static inline float64x2_t exp_pd(float64x2_t x) noexcept {
    const float64x2_t ln2_inv = vdupq_n_f64(1.4426950408889634073599246810019);
    const float64x2_t ln2_hi = vdupq_n_f64(0.693147180369123816490e+00);
    const float64x2_t ln2_lo = vdupq_n_f64(1.90821492927058770002e-10);
    const float64x2_t exp_max = vdupq_n_f64(709.782712893383996732223);
    const float64x2_t exp_min = vdupq_n_f64(-708.0);
    const float64x2_t half = vdupq_n_f64(0.5);
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t c1 = vdupq_n_f64(0.1666666666666669072e+0);
    const float64x2_t c2 = vdupq_n_f64(0.4166666666666602598e-1);
    const float64x2_t c3 = vdupq_n_f64(0.8333333333314938210e-2);
    const float64x2_t c4 = vdupq_n_f64(0.1388888888914497797e-2);
    const float64x2_t c5 = vdupq_n_f64(0.1984126989855865850e-3);
    const float64x2_t c6 = vdupq_n_f64(0.2480158687479686264e-4);
    const float64x2_t c7 = vdupq_n_f64(0.2755723402025388239e-5);
    const float64x2_t c8 = vdupq_n_f64(0.2755762628169491192e-6);
    const float64x2_t c9 = vdupq_n_f64(0.2511210703042288022e-7);
    const float64x2_t c10 = vdupq_n_f64(0.2081276378237164457e-8);

    x = vminq_f64(x, exp_max);
    x = vmaxq_f64(x, exp_min);
    float64x2_t n_float = vrndnq_f64(vmulq_f64(x, ln2_inv));
    float64x2_t r = vfmsq_f64(x, n_float, ln2_hi);
    r = vfmsq_f64(r, n_float, ln2_lo);
    float64x2_t r2 = vmulq_f64(r, r);
    float64x2_t poly = c10;
    poly = vfmaq_f64(c9, poly, r);
    poly = vfmaq_f64(c8, poly, r);
    poly = vfmaq_f64(c7, poly, r);
    poly = vfmaq_f64(c6, poly, r);
    poly = vfmaq_f64(c5, poly, r);
    poly = vfmaq_f64(c4, poly, r);
    poly = vfmaq_f64(c3, poly, r);
    poly = vfmaq_f64(c2, poly, r);
    poly = vfmaq_f64(c1, poly, r);
    poly = vfmaq_f64(half, poly, r);
    poly = vfmaq_f64(r, poly, r2);
    poly = vaddq_f64(poly, one);
    int64x2_t n_int = vcvtq_s64_f64(n_float);
    int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(n_int, vdupq_n_s64(1023)), 52);
    return vmulq_f64(poly, vreinterpretq_f64_s64(exp_bits));
}

// 7-term Horner cosine, max error ≈ 2×10⁻¹⁰.
[[nodiscard]] static inline float64x2_t cos_pd(float64x2_t x) noexcept {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kHalfPi = 1.5707963267948966192313216916397514421;
    const float64x2_t inv2pi = vdupq_n_f64(1.0 / (2.0 * kPi));
    const float64x2_t two_pi = vdupq_n_f64(2.0 * kPi);
    const float64x2_t pi = vdupq_n_f64(kPi);
    const float64x2_t half_pi = vdupq_n_f64(kHalfPi);
    const float64x2_t neg_pi = vdupq_n_f64(-kPi);
    const float64x2_t nhalf_pi = vdupq_n_f64(-kHalfPi);
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t neg_one = vdupq_n_f64(-1.0);
    const float64x2_t c1 = vdupq_n_f64(-0.5);
    const float64x2_t c2 = vdupq_n_f64(4.166666666666667e-2);
    const float64x2_t c3 = vdupq_n_f64(-1.388888888888889e-3);
    const float64x2_t c4 = vdupq_n_f64(2.480158730158730e-5);
    const float64x2_t c5 = vdupq_n_f64(-2.755731922398589e-7);
    const float64x2_t c6 = vdupq_n_f64(2.087675698786810e-9);
    const float64x2_t c7 = vdupq_n_f64(-1.147074559772973e-11);

    float64x2_t q = vrndnq_f64(vmulq_f64(x, inv2pi));
    float64x2_t y = vsubq_f64(x, vmulq_f64(q, two_pi));
    float64x2_t sign = one;
    uint64x2_t gt = vcgtq_f64(y, half_pi);
    uint64x2_t lt = vcltq_f64(y, nhalf_pi);
    y = vbslq_f64(gt, vsubq_f64(pi, y), y);
    sign = vbslq_f64(gt, neg_one, sign);
    y = vbslq_f64(lt, vsubq_f64(neg_pi, y), y);
    sign = vbslq_f64(lt, neg_one, sign);
    float64x2_t y2 = vmulq_f64(y, y);
    float64x2_t poly = c7;
    poly = vfmaq_f64(c6, y2, poly);
    poly = vfmaq_f64(c5, y2, poly);
    poly = vfmaq_f64(c4, y2, poly);
    poly = vfmaq_f64(c3, y2, poly);
    poly = vfmaq_f64(c2, y2, poly);
    poly = vfmaq_f64(c1, y2, poly);
    poly = vfmaq_f64(one, y2, poly);
    return vmulq_f64(poly, sign);
}

// log1p: 8-term polynomial for |x|<1e-4 with FMA. AArch64 NEON.
[[nodiscard]] static inline float64x2_t log1p_pd(float64x2_t x) noexcept {
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t thr = vdupq_n_f64(1.0e-4);
    const float64x2_t nthr = vdupq_n_f64(-1.0e-4);
    float64x2_t p = vdupq_n_f64(-0.125);
    p = vfmaq_f64(vdupq_n_f64(1.0 / 7.0), p, x);
    p = vfmaq_f64(vdupq_n_f64(-1.0 / 6.0), p, x);
    p = vfmaq_f64(vdupq_n_f64(0.2), p, x);
    p = vfmaq_f64(vdupq_n_f64(-0.25), p, x);
    p = vfmaq_f64(vdupq_n_f64(1.0 / 3.0), p, x);
    p = vfmaq_f64(vdupq_n_f64(-0.5), p, x);
    p = vfmaq_f64(one, p, x);
    const float64x2_t small = vmulq_f64(x, p);
    const float64x2_t general = log_pd(vaddq_f64(one, x));
    uint64x2_t sm = vandq_u64(vcltq_f64(x, thr), vcgtq_f64(x, nthr));
    return vbslq_f64(sm, small, general);
}

#endif // LIBHMM_HAS_NEON && __aarch64__

} // namespace libhmm::detail::simd
