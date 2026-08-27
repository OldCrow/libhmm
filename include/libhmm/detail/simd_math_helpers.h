#pragma once
// include/libhmm/detail/simd_math_helpers.h
//
// Internal header — NOT part of the public API.
// Not installed to CMAKE_INSTALL_PREFIX/include: CMakeLists.txt excludes this file
// and trig_cleanroom_data.inc from install(DIRECTORY). Its __m256d section needs
// -mfma and its __m512d section needs -mavx512dq, which LIBHMM_HAS_AVX/AVX512 do not
// imply, so it is only safe inside the per-ISA TUs that pass the right flag pair.
//
// Single source of truth for SIMD math primitives (log, exp, cos, sin, log1p)
// shared between:
//   - src/performance/simd_double_ops_*.cpp  (distribution batch kernels and, since
//     #58, the FB/BW recurrence kernels behind the TranscendentalKernels facade)
//
// Replaces simd_kernels_internal.h, which used older polynomial approximations.
// log/exp are SLEEF-based (< 1 ULP). cos/sin use a clean-room quadrant-reduction
// kernel (issue #74; ported from libstats, same owner, MIT, no third-party
// source) for |x| <= kTrigDMax (2^23); accuracy sub-ULP on FMA tiers, slightly
// worse on plain-arithmetic SSE2 (see per-overload comments below). cos_pd/
// sin_pd are register-level primitives only — see the domain-contract comment
// at each overload; the oversized-input scalar fixup lives in the cos_batch/
// sin_batch callers in src/performance/simd_double_ops_*.cpp.
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

// Clean-room quadrant-reduction cos/sin constants (issue #74), shared by every
// ISA section below — the values are plain doubles with no vector-register
// dependency, so one include here covers all four tiers without duplication
// beyond the ordinary per-TU inclusion of this header. Ported from libstats'
// scripts/gen_neon_trig_cleanroom_table.py (same owner, MIT, clean-room
// derived; no third-party source) — see that project's
// docs/NEON_TRIG_DERIVATION.md for the mathematics. Regenerate via
// scripts/gen_trig_cleanroom_table.py; never hand-edit the .inc.
#include "libhmm/detail/trig_cleanroom_data.inc"

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

// Clean-room quadrant-reduction cos/sin (issue #74): x = n*(pi/2) + r,
// n = round(x*2/pi), compensated reduction into (r, rlo) via the 4-part
// exact-product pi/2 split (kTrigPio2), degree-6 parity cores
// sin(r) = r + r*(u*P(u)), cos(r) = 1 + u*Q(u) with the leading 1-u/2 kept as
// an exact head+tail pair, u = r*r. See docs at kTrigDMax's definition site
// (trig_cleanroom_data.inc) and libstats' docs/NEON_TRIG_DERIVATION.md.
// _mm512_cvtepi32_epi64 requires AVX-512F; _mm512_xor_pd requires AVX-512DQ
// (this TU is compiled with -mavx512f -mavx512dq / /arch:AVX512).

// Reduction shared by cos_pd/sin_pd: n32 = round-to-nearest-even(x*2/pi) via
// the cvt round-trip (exact-product lemma holds for |n| <= 5,340,354, i.e.
// |x| <= kTrigDMax = 2^23); r/rlo carry the reduced argument compensated.
static inline void trig_reduce_8pd(__m512d x, __m512d &r, __m512d &rlo, __m512i &n64) noexcept {
    const __m256i n32 = _mm512_cvtpd_epi32(_mm512_mul_pd(x, _mm512_set1_pd(kTrigTwoOverPi)));
    const __m512d nf = _mm512_cvtepi32_pd(n32); // exact
    n64 = _mm512_cvtepi32_epi64(n32);

    r = _mm512_fnmadd_pd(nf, _mm512_set1_pd(kTrigPio2[0]), x); // exact (step 1)
    rlo = _mm512_setzero_pd();
    for (int k = 1; k < 4; ++k) {
        const __m512d pk = _mm512_set1_pd(kTrigPio2[k]);
        const __m512d rk = _mm512_fnmadd_pd(nf, pk, r);
        const __m512d e = _mm512_fnmadd_pd(nf, pk, _mm512_sub_pd(r, rk));
        rlo = _mm512_add_pd(rlo, e);
        r = rk;
    }
}

// Degree-6 minimax parity cores on u = r*r; cos's 1 - u/2 head is split into
// an exact (h, hl) pair (kTrigCosC[0] == -0.5 exactly, generator-asserted).
static inline void trig_cores_8pd(__m512d r, __m512d rlo, __m512d &s_core,
                                  __m512d &c_core) noexcept {
    const __m512d u = _mm512_mul_pd(r, r);

    __m512d ps = _mm512_set1_pd(kTrigSinC[6]);
    for (int i = 5; i >= 0; --i)
        ps = _mm512_fmadd_pd(ps, u, _mm512_set1_pd(kTrigSinC[i]));
    s_core = _mm512_add_pd(r, _mm512_fmadd_pd(_mm512_mul_pd(r, u), ps, rlo));

    __m512d pc = _mm512_set1_pd(kTrigCosC[6]);
    for (int i = 5; i >= 1; --i)
        pc = _mm512_fmadd_pd(pc, u, _mm512_set1_pd(kTrigCosC[i]));
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d h = _mm512_fnmadd_pd(u, half, one);                    // 1 - u/2, exact
    const __m512d hl = _mm512_fnmadd_pd(u, half, _mm512_sub_pd(one, h)); // (1-h) - u/2, exact
    __m512d mc = _mm512_fmadd_pd(_mm512_mul_pd(u, u), pc, hl);
    mc = _mm512_fnmadd_pd(r, rlo, mc); // first-order effect of compensated reduction
    c_core = _mm512_add_pd(h, mc);
}

// cos(x) for |x| <= kTrigDMax (2^23) only — NOT valid for larger |x| or Inf
// (the batch wrapper's scalar fixup handles those); NaN self-propagates.
// Quadrant table: q=0:+c 1:-s 2:-c 3:+s -> swap core on bit0, sign on
// bit1 XOR bit0 (both taken from the low bits of n's two's-complement form).
[[nodiscard]] static inline __m512d cos_pd(__m512d x) noexcept {
    __m512d r, rlo;
    __m512i n64;
    trig_reduce_8pd(x, r, rlo, n64);
    __m512d s_core, c_core;
    trig_cores_8pd(r, rlo, s_core, c_core);

    const __m512i one_i = _mm512_set1_epi64(1);
    const __mmask8 swap = _mm512_test_epi64_mask(n64, one_i);
    const __m512i bit0 = _mm512_and_si512(n64, one_i);
    const __m512i bit1 = _mm512_and_si512(_mm512_srli_epi64(n64, 1), one_i);
    const __m512i sign_bit = _mm512_xor_si512(bit1, bit0);
    const __m512d cv = _mm512_mask_blend_pd(swap, c_core, s_core);
    const __m512d sign_v = _mm512_castsi512_pd(_mm512_slli_epi64(sign_bit, 63));
    return _mm512_xor_pd(cv, sign_v);
}

// sin(x) for |x| <= kTrigDMax (2^23) only — see cos_pd's domain-contract
// comment above; identical caveats apply. Quadrant table: q=0:+s 1:+c 2:-s
// 3:-c -> swap core on bit0 (opposite selection order from cos_pd), sign on
// bit1 alone. Computed from the quadrant table directly, NOT cos(x - pi/2)
// (that composition loses accuracy through the extra subtraction).
[[nodiscard]] static inline __m512d sin_pd(__m512d x) noexcept {
    __m512d r, rlo;
    __m512i n64;
    trig_reduce_8pd(x, r, rlo, n64);
    __m512d s_core, c_core;
    trig_cores_8pd(r, rlo, s_core, c_core);

    const __m512i one_i = _mm512_set1_epi64(1);
    const __mmask8 swap = _mm512_test_epi64_mask(n64, one_i);
    const __m512i bit1 = _mm512_and_si512(_mm512_srli_epi64(n64, 1), one_i);
    const __m512d sv = _mm512_mask_blend_pd(swap, s_core, c_core);
    const __m512d sign_v = _mm512_castsi512_pd(_mm512_slli_epi64(bit1, 63));
    const __m512d result = _mm512_xor_pd(sv, sign_v);
    // IEEE sign-of-zero (issue #81): for x = -0 the core computes
    // (-0) + (+0) = +0, dropping the sign; sin(+/-0) must be +/-0 exactly.
    // x == 0 matches both zeros and no other double, so blend x itself
    // back in.
    const __mmask8 zmask = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ);
    return _mm512_mask_blend_pd(zmask, result, x);
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
    const __mmask8 sm = _kand_mask8(_mm512_cmp_pd_mask(x, thr, _CMP_LT_OS),
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

// Clean-room quadrant-reduction cos/sin (issue #74) — see the AVX-512 section
// above for the full derivation comment; identical algorithm, 4-wide.

static inline void trig_reduce_4pd(__m256d x, __m256d &r, __m256d &rlo, __m256i &n64) noexcept {
    const __m128i n32 = _mm256_cvtpd_epi32(_mm256_mul_pd(x, _mm256_set1_pd(kTrigTwoOverPi)));
    const __m256d nf = _mm256_cvtepi32_pd(n32); // exact
    n64 = _mm256_cvtepi32_epi64(n32);

    r = _mm256_fnmadd_pd(nf, _mm256_set1_pd(kTrigPio2[0]), x); // exact (step 1)
    rlo = _mm256_setzero_pd();
    for (int k = 1; k < 4; ++k) {
        const __m256d pk = _mm256_set1_pd(kTrigPio2[k]);
        const __m256d rk = _mm256_fnmadd_pd(nf, pk, r);
        const __m256d e = _mm256_fnmadd_pd(nf, pk, _mm256_sub_pd(r, rk));
        rlo = _mm256_add_pd(rlo, e);
        r = rk;
    }
}

static inline void trig_cores_4pd(__m256d r, __m256d rlo, __m256d &s_core,
                                  __m256d &c_core) noexcept {
    const __m256d u = _mm256_mul_pd(r, r);

    __m256d ps = _mm256_set1_pd(kTrigSinC[6]);
    for (int i = 5; i >= 0; --i)
        ps = _mm256_fmadd_pd(ps, u, _mm256_set1_pd(kTrigSinC[i]));
    s_core = _mm256_add_pd(r, _mm256_fmadd_pd(_mm256_mul_pd(r, u), ps, rlo));

    __m256d pc = _mm256_set1_pd(kTrigCosC[6]);
    for (int i = 5; i >= 1; --i)
        pc = _mm256_fmadd_pd(pc, u, _mm256_set1_pd(kTrigCosC[i]));
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d h = _mm256_fnmadd_pd(u, half, one);
    const __m256d hl = _mm256_fnmadd_pd(u, half, _mm256_sub_pd(one, h));
    __m256d mc = _mm256_fmadd_pd(_mm256_mul_pd(u, u), pc, hl);
    mc = _mm256_fnmadd_pd(r, rlo, mc);
    c_core = _mm256_add_pd(h, mc);
}

// cos(x) for |x| <= kTrigDMax (2^23) only — see the AVX-512 cos_pd comment
// for the domain contract and quadrant table; identical here, 4-wide.
[[nodiscard]] static inline __m256d cos_pd(__m256d x) noexcept {
    __m256d r, rlo;
    __m256i n64;
    trig_reduce_4pd(x, r, rlo, n64);
    __m256d s_core, c_core;
    trig_cores_4pd(r, rlo, s_core, c_core);

    const __m256i one_i = _mm256_set1_epi64x(1);
    const __m256i bit0 = _mm256_and_si256(n64, one_i);
    const __m256i bit1 = _mm256_and_si256(_mm256_srli_epi64(n64, 1), one_i);
    // all-ones iff bit0 set (0 - 1 = -1 = all-ones; 0 - 0 = 0), for blendv_pd's MSB test.
    const __m256d swap_mask = _mm256_castsi256_pd(_mm256_sub_epi64(_mm256_setzero_si256(), bit0));
    const __m256d cv = _mm256_blendv_pd(c_core, s_core, swap_mask);
    const __m256i sign_bit = _mm256_xor_si256(bit1, bit0);
    const __m256d sign_v = _mm256_castsi256_pd(_mm256_slli_epi64(sign_bit, 63));
    return _mm256_xor_pd(cv, sign_v);
}

// sin(x) for |x| <= kTrigDMax (2^23) only — see the AVX-512 sin_pd comment
// for the domain contract and quadrant table; identical here, 4-wide.
[[nodiscard]] static inline __m256d sin_pd(__m256d x) noexcept {
    __m256d r, rlo;
    __m256i n64;
    trig_reduce_4pd(x, r, rlo, n64);
    __m256d s_core, c_core;
    trig_cores_4pd(r, rlo, s_core, c_core);

    const __m256i one_i = _mm256_set1_epi64x(1);
    const __m256i bit0 = _mm256_and_si256(n64, one_i);
    const __m256i bit1 = _mm256_and_si256(_mm256_srli_epi64(n64, 1), one_i);
    const __m256d swap_mask = _mm256_castsi256_pd(_mm256_sub_epi64(_mm256_setzero_si256(), bit0));
    const __m256d sv = _mm256_blendv_pd(s_core, c_core, swap_mask);
    const __m256d sign_v = _mm256_castsi256_pd(_mm256_slli_epi64(bit1, 63));
    const __m256d result = _mm256_xor_pd(sv, sign_v);
    // IEEE sign-of-zero (issue #81): for x = -0 the core computes
    // (-0) + (+0) = +0, dropping the sign; sin(+/-0) must be +/-0 exactly.
    // x == 0 matches both zeros and no other double, so blend x itself
    // back in.
    return _mm256_blendv_pd(result, x, _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ));
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

    // Subnormal prescale (issue #85): without this, the exponent-extraction
    // path below treats a subnormal's biased exponent (0) as if it belonged
    // to a normal number, compressing the whole subnormal range. Ported from
    // the AVX2/AVX-512/NEON log_pd overloads in this same file: scale by 2^54
    // (exact — power-of-two) and subtract 54 back out of the exponent.
    const __m128d min_normal = _mm_set1_pd(2.2250738585072014e-308);
    const __m128d scale_up = _mm_set1_pd(18014398509481984.0); // 2^54
    const __m128d is_denormal = _mm_cmplt_pd(x, min_normal);
    const __m128d sx = sse2_blend(is_denormal, _mm_mul_pd(x, scale_up), x);

    const __m128i xi = _mm_castpd_si128(sx);
    __m128i exp_i = _mm_srli_epi64(xi, 52);
    exp_i = _mm_and_si128(exp_i, _mm_set1_epi64x(0x7FFLL));
    const __m128i exp_i32 = _mm_shuffle_epi32(exp_i, _MM_SHUFFLE(0, 0, 2, 0));
    __m128d e = _mm_cvtepi32_pd(exp_i32);
    e = _mm_sub_pd(e, _mm_set1_pd(1023.0));
    e = sse2_blend(is_denormal, _mm_sub_pd(e, _mm_set1_pd(54.0)), e);

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

// Clean-room quadrant-reduction cos/sin (issue #74) — see the AVX-512 section
// above for the full derivation comment. SSE2 has no FMA, so the reduction
// and both cores use plain mul+add/mul+sub throughout; this is EXACTLY as
// accurate as the FMA form here because every nf*p_k product in the
// reduction, and the u*0.5 scaling in cos's head/tail split, is exact by the
// 30-bit-split construction — a plain rounded add/sub after an exact
// multiply commits the identical single rounding an FMA would. The only
// accuracy cost is the ordinary Horner mul+add in the polynomial cores
// (slightly worse rounding per step than FMA; expected ~1.5 ULP landing vs
// ~0.8 for the FMA tiers, measured separately by the ULP gate task).
// n32 has no SSE4.1 cvtepi32_epi64, so bit0/bit1 are read off a duplicated
// 32-bit shuffle instead of a true 64-bit sign-extension (sufficient: only
// the low 2 bits of each lane are ever inspected).

static inline void trig_reduce_2pd(__m128d x, __m128d &r, __m128d &rlo, __m128i &n64) noexcept {
    const __m128i n32 = _mm_cvtpd_epi32(_mm_mul_pd(x, _mm_set1_pd(kTrigTwoOverPi)));
    const __m128d nf = _mm_cvtepi32_pd(n32); // exact
    n64 = _mm_shuffle_epi32(n32, _MM_SHUFFLE(1, 1, 0, 0));

    r = _mm_sub_pd(x, _mm_mul_pd(nf, _mm_set1_pd(kTrigPio2[0]))); // exact (step 1)
    rlo = _mm_setzero_pd();
    for (int k = 1; k < 4; ++k) {
        const __m128d pk = _mm_set1_pd(kTrigPio2[k]);
        const __m128d rk = _mm_sub_pd(r, _mm_mul_pd(nf, pk));
        const __m128d e = _mm_sub_pd(_mm_sub_pd(r, rk), _mm_mul_pd(nf, pk));
        rlo = _mm_add_pd(rlo, e);
        r = rk;
    }
}

static inline void trig_cores_2pd(__m128d r, __m128d rlo, __m128d &s_core,
                                  __m128d &c_core) noexcept {
    const __m128d u = _mm_mul_pd(r, r);

    __m128d ps = _mm_set1_pd(kTrigSinC[6]);
    for (int i = 5; i >= 0; --i)
        ps = _mm_add_pd(_mm_set1_pd(kTrigSinC[i]), _mm_mul_pd(ps, u));
    s_core = _mm_add_pd(r, _mm_add_pd(rlo, _mm_mul_pd(_mm_mul_pd(r, u), ps)));

    __m128d pc = _mm_set1_pd(kTrigCosC[6]);
    for (int i = 5; i >= 1; --i)
        pc = _mm_add_pd(_mm_set1_pd(kTrigCosC[i]), _mm_mul_pd(pc, u));
    const __m128d one = _mm_set1_pd(1.0);
    const __m128d half = _mm_set1_pd(0.5);
    const __m128d h = _mm_sub_pd(one, _mm_mul_pd(u, half));                 // 1 - u/2, exact
    const __m128d hl = _mm_sub_pd(_mm_sub_pd(one, h), _mm_mul_pd(u, half)); // (1-h) - u/2, exact
    __m128d mc = _mm_add_pd(hl, _mm_mul_pd(_mm_mul_pd(u, u), pc));
    mc = _mm_sub_pd(mc, _mm_mul_pd(r, rlo));
    c_core = _mm_add_pd(h, mc);
}

// cos(x) for |x| <= kTrigDMax (2^23) only — see the AVX-512 cos_pd comment
// for the domain contract and quadrant table; identical here, 2-wide.
[[nodiscard]] static inline __m128d cos_pd(__m128d x) noexcept {
    __m128d r, rlo;
    __m128i n64;
    trig_reduce_2pd(x, r, rlo, n64);
    __m128d s_core, c_core;
    trig_cores_2pd(r, rlo, s_core, c_core);

    const __m128i one_i = _mm_set1_epi64x(1);
    const __m128i bit0 = _mm_and_si128(n64, one_i);
    const __m128i bit1 = _mm_and_si128(_mm_srli_epi64(n64, 1), one_i);
    // all-ones iff bit0 set (0 - 1 = -1 = all-ones; 0 - 0 = 0), for sse2_blend.
    const __m128d swap_mask = _mm_castsi128_pd(_mm_sub_epi64(_mm_setzero_si128(), bit0));
    const __m128d cv = sse2_blend(swap_mask, s_core, c_core);
    const __m128i sign_bit = _mm_xor_si128(bit1, bit0);
    const __m128d sign_v = _mm_castsi128_pd(_mm_slli_epi64(sign_bit, 63));
    return _mm_xor_pd(cv, sign_v);
}

// sin(x) for |x| <= kTrigDMax (2^23) only — see the AVX-512 sin_pd comment
// for the domain contract and quadrant table; identical here, 2-wide.
[[nodiscard]] static inline __m128d sin_pd(__m128d x) noexcept {
    __m128d r, rlo;
    __m128i n64;
    trig_reduce_2pd(x, r, rlo, n64);
    __m128d s_core, c_core;
    trig_cores_2pd(r, rlo, s_core, c_core);

    const __m128i one_i = _mm_set1_epi64x(1);
    const __m128i bit0 = _mm_and_si128(n64, one_i);
    const __m128i bit1 = _mm_and_si128(_mm_srli_epi64(n64, 1), one_i);
    const __m128d swap_mask = _mm_castsi128_pd(_mm_sub_epi64(_mm_setzero_si128(), bit0));
    const __m128d sv = sse2_blend(swap_mask, c_core, s_core);
    const __m128d sign_v = _mm_castsi128_pd(_mm_slli_epi64(bit1, 63));
    const __m128d result = _mm_xor_pd(sv, sign_v);
    // IEEE sign-of-zero (issue #81): for x = -0 the core computes
    // (-0) + (+0) = +0, dropping the sign; sin(+/-0) must be +/-0 exactly.
    // x == 0 matches both zeros and no other double, so blend x itself
    // back in.
    return sse2_blend(_mm_cmpeq_pd(x, _mm_setzero_pd()), x, result);
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
    uint64x2_t is_not_nan = vceqq_f64(x, x);
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

// Clean-room quadrant-reduction cos/sin (issue #74) — see the AVX-512 section
// above for the full derivation comment; this NEON port keeps the libstats
// vector_cos_neon expressions verbatim (same owner, MIT, no third-party
// source; see libstats' docs/NEON_TRIG_DERIVATION.md), factored into a
// shared reduce/cores pair plus each of cos_pd/sin_pd doing its own
// quadrant recombination so sin comes from the quadrant table directly
// (never cos(x - pi/2), which would lose accuracy through the extra
// subtraction).

static inline void trig_reduce_2pd(float64x2_t x, float64x2_t &r, float64x2_t &rlo,
                                   int64x2_t &n) noexcept {
    // reduction: n = round(x * 2/pi); r = x - n*pi/2 via exact split parts.
    // Step 1 is always exact (exact product + Sterbenz); steps 2..4 are
    // compensated: when a step rounds, cancellation was small, so
    // (r_prev - r_new) is exact and e recovers the rounding error exactly.
    const float64x2_t nf = vrndnq_f64(vmulq_f64(x, vdupq_n_f64(kTrigTwoOverPi)));
    n = vcvtq_s64_f64(nf); // nf is integral; conversion exact
    r = vfmsq_f64(x, nf, vdupq_n_f64(kTrigPio2[0]));
    rlo = vdupq_n_f64(0.0);
    for (int k = 1; k < 4; ++k) {
        const float64x2_t pk = vdupq_n_f64(kTrigPio2[k]);
        const float64x2_t rk = vfmsq_f64(r, nf, pk);
        const float64x2_t e = vfmsq_f64(vsubq_f64(r, rk), nf, pk);
        rlo = vaddq_f64(rlo, e);
        r = rk;
    }
}

static inline void trig_cores_2pd(float64x2_t r, float64x2_t rlo, float64x2_t &s_core,
                                  float64x2_t &c_core) noexcept {
    const float64x2_t u = vmulq_f64(r, r);

    // sin core: s = r + (r*u*P(u) + rlo)
    float64x2_t ps = vdupq_n_f64(kTrigSinC[6]);
    for (int i = 5; i >= 0; --i)
        ps = vfmaq_f64(vdupq_n_f64(kTrigSinC[i]), ps, u);
    s_core = vaddq_f64(r, vfmaq_f64(rlo, vmulq_f64(r, u), ps));

    // cos core: split the leading 1 - u/2 into an exact head+tail pair
    // (h = fl(1 - u/2); hl = (1 - h) - u/2, both steps exact by Sterbenz/
    // cancellation), accumulate every correction at ~2^-54 magnitude, and
    // pay only the final add's rounding. Q[0] == -1/2 exactly (generator-
    // asserted), so no c0 remainder term is needed. The -r*rlo term is the
    // first-order effect of the compensated reduction on cos.
    float64x2_t pc = vdupq_n_f64(kTrigCosC[6]);
    for (int i = 5; i >= 1; --i)
        pc = vfmaq_f64(vdupq_n_f64(kTrigCosC[i]), pc, u);
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t half = vdupq_n_f64(0.5);
    const float64x2_t h = vfmsq_f64(one, u, half);
    const float64x2_t hl = vfmsq_f64(vsubq_f64(one, h), u, half);
    float64x2_t mc = vfmaq_f64(hl, vmulq_f64(u, u), pc);
    mc = vfmsq_f64(mc, r, rlo);
    c_core = vaddq_f64(h, mc);
}

// cos(x) for |x| <= kTrigDMax (2^23) only — NOT valid for larger |x| or Inf
// (the batch wrapper's per-lane scalar fixup handles those); NaN
// self-propagates through the polynomial path (vcvtq of NaN is defined on
// aarch64, so it is safe going in). Quadrant table: q=0:+c 1:-s 2:-c 3:+s ->
// swap core on bit0, sign on bit1 XOR bit0 (two's-complement low bits of n
// give n mod 4).
[[nodiscard]] static inline float64x2_t cos_pd(float64x2_t x) noexcept {
    float64x2_t r, rlo;
    int64x2_t n;
    trig_reduce_2pd(x, r, rlo, n);
    float64x2_t s_core, c_core;
    trig_cores_2pd(r, rlo, s_core, c_core);

    const uint64x2_t qu = vreinterpretq_u64_s64(n);
    const uint64x2_t swap = vtstq_u64(qu, vdupq_n_u64(1));
    const uint64x2_t sgn =
        vshlq_n_u64(vandq_u64(veorq_u64(vshrq_n_u64(qu, 1), qu), vdupq_n_u64(1)), 63);
    const float64x2_t cv = vbslq_f64(swap, s_core, c_core);
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(cv), sgn));
}

// sin(x) for |x| <= kTrigDMax (2^23) only — see cos_pd's domain-contract
// comment above; identical caveats apply. Quadrant table: q=0:+s 1:+c 2:-s
// 3:-c -> swap core on bit0 (opposite selection order from cos_pd), sign on
// bit1 alone. Computed from the quadrant table directly, NOT cos(x - pi/2).
[[nodiscard]] static inline float64x2_t sin_pd(float64x2_t x) noexcept {
    float64x2_t r, rlo;
    int64x2_t n;
    trig_reduce_2pd(x, r, rlo, n);
    float64x2_t s_core, c_core;
    trig_cores_2pd(r, rlo, s_core, c_core);

    const uint64x2_t qu = vreinterpretq_u64_s64(n);
    const uint64x2_t swap = vtstq_u64(qu, vdupq_n_u64(1));
    const uint64x2_t sgn = vshlq_n_u64(vandq_u64(vshrq_n_u64(qu, 1), vdupq_n_u64(1)), 63);
    const float64x2_t sv = vbslq_f64(swap, c_core, s_core);
    const float64x2_t result = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(sv), sgn));
    // IEEE sign-of-zero (issue #81): for x = -0 the core computes
    // (-0) + (+0) = +0, dropping the sign; sin(+/-0) must be +/-0 exactly.
    // x == 0 matches both zeros and no other double, so blend x itself
    // back in.
    return vbslq_f64(vceqq_f64(x, vdupq_n_f64(0.0)), x, result);
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
