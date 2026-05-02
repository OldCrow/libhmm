// src/performance/transcendental_kernels.cpp
//
// SIMD implementations of TranscendentalKernels methods plus free-function
// vector log/exp primitives used by Tier-2 distribution kernels.
//
// Compiled with LIBHMM_BEST_SIMD_FLAGS, activating the ISA cascade:
//   AVX-512  8-wide __m512d
//   AVX/AVX2 4-wide __m256d   (AVX-1 compatible; compiler fuses FMA under AVX2)
//   SSE2     2-wide __m128d
//   NEON     2-wide float64x2_t
//   scalar   tail and portable fallback
//
// Vector exp(double) design (exp_pd_*):
//   Range reduction : x = N*ln2 + r,  |r| <= ln2/2  (Cephes split)
//   Polynomial      : 13-term Horner of sum(r^k/k!); ~1 ulp
//   2^N             : (n + 1023) << 52 via integer bit manipulation
//   Underflow guard : clamp x >= MIN_LOG_PROBABILITY; mask to 0 below threshold
//   No +inf / NaN:   FB/BW callers guarantee finite or LOG_ZERO inputs
//
// Vector log(double) design (log_pd_*):
//   Range reduction : extract IEEE754 exponent e and mantissa m; x = 2^e * m
//                     m in [1, 2).  If m > sqrt(2): e += 1, m *= 0.5
//                     so m in [1/sqrt(2), sqrt(2)].
//   Substitution    : y = (m - 1) / (m + 1),  |y| <= 0.172
//   Polynomial      : log(m) = 2y * (1 + y^2/3 + y^4/5 + ... + y^12/13)
//                     7-term Horner; truncation < ~5 ulp at |y|_max
//                     Distribution callers need < 1e-10 absolute; well covered.
//   Reconstruction  : log(x) = e * LN2_HI + e * LN2_LO + log(m)  (split)
//   Guard           : x <= 0 lanes are masked to -inf
//   No NaN:          distribution callers validate x > 0 before calling batch

#include "libhmm/performance/transcendental_kernels.h"
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
// Shared polynomial coefficients (double precision exp Taylor, k=0..12)
// ---------------------------------------------------------------------------
// c[k] = 1/k! stored as double literals for maximum precision.
static constexpr double EXP_C0  = 1.0;
static constexpr double EXP_C1  = 1.0;
static constexpr double EXP_C2  = 0.5;
static constexpr double EXP_C3  = 1.6666666666666666e-1;
static constexpr double EXP_C4  = 4.1666666666666664e-2;
static constexpr double EXP_C5  = 8.3333333333333332e-3;
static constexpr double EXP_C6  = 1.3888888888888889e-3;
static constexpr double EXP_C7  = 1.9841269841269841e-4;
static constexpr double EXP_C8  = 2.4801587301587302e-5;
static constexpr double EXP_C9  = 2.7557319223985888e-6;
static constexpr double EXP_C10 = 2.7557319223985888e-7;
static constexpr double EXP_C11 = 2.5052108385441720e-8;
static constexpr double EXP_C12 = 2.0876756987868099e-9;

// Cephes ln2 split: ln2 = LN2_HI + LN2_LO exactly in double arithmetic.
static constexpr double LN2_HI = 6.93147180369123816490e-1;
static constexpr double LN2_LO = 1.90821492927058770002e-10;
static constexpr double LOG2E  = 1.44269504088896338700; // 1/ln(2)

// Underflow clamp: inputs <= this map to exp() output of 0.
static constexpr double EXP_UNDERFLOW = constants::probability::MIN_LOG_PROBABILITY; // -700.0

// Double-exponent bias.
static constexpr double EXPONENT_BIAS = 1023.0;

// ---------------------------------------------------------------------------
// AVX-512: 8-wide exp(double)
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_AVX512)

static inline __m512d exp_pd_avx512(__m512d x) noexcept {
    const __m512d underflow_v = _mm512_set1_pd(EXP_UNDERFLOW);
    const __m512d log2e_v     = _mm512_set1_pd(LOG2E);
    const __m512d half_v      = _mm512_set1_pd(0.5);
    const __m512d ln2hi_v     = _mm512_set1_pd(LN2_HI);
    const __m512d ln2lo_v     = _mm512_set1_pd(LN2_LO);
    const __m512d zero_v      = _mm512_setzero_pd();

    // Remember which lanes underflow.
    const __mmask8 underflow_mask = _mm512_cmp_pd_mask(x, underflow_v, _CMP_LE_OS);

    // Clamp to prevent polynomial divergence.
    x = _mm512_max_pd(x, underflow_v);

    // n = floor(x * log2e + 0.5);  r = x - n*ln2 (Cephes 2-part subtraction)
    __m512d n = _mm512_floor_pd(_mm512_fmadd_pd(x, log2e_v, half_v));
    __m512d r = _mm512_fnmadd_pd(n, ln2hi_v, x);
    r = _mm512_fnmadd_pd(n, ln2lo_v, r);

    // Horner evaluation of exp(r), 13 terms.
    __m512d p = _mm512_set1_pd(EXP_C12);
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C11));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C10));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C9));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C8));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C7));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C6));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C5));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C4));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C3));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C2));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C1));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(EXP_C0));

    // 2^n via integer bit manipulation: (n + 1023) << 52.
    __m256i ni = _mm512_cvtpd_epi32(n); // 8 x int32 in 256-bit
    __m512i ni64 = _mm512_cvtepi32_epi64(ni); // widen to 8 x int64
    ni64 = _mm512_add_epi64(ni64, _mm512_set1_epi64(static_cast<long long>(EXPONENT_BIAS)));
    ni64 = _mm512_slli_epi64(ni64, 52);
    __m512d pow2n;
    // reinterpret int64 bits as double
    pow2n = _mm512_castsi512_pd(ni64);

    __m512d result = _mm512_mul_pd(p, pow2n);

    // Zero out underflow lanes.
    result = _mm512_mask_blend_pd(underflow_mask, result, zero_v);
    return result;
}

#endif // LIBHMM_HAS_AVX512

// ---------------------------------------------------------------------------
// AVX (covers AVX-1 and AVX2): 4-wide exp(double)
// The 2^n integer step uses two 128-bit halves to stay AVX-1 compatible
// (avoids AVX2-only _mm256_cvtepi32_epi64).
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)

static inline __m256d exp_pd_avx(__m256d x) noexcept {
    const __m256d underflow_v = _mm256_set1_pd(EXP_UNDERFLOW);
    const __m256d log2e_v     = _mm256_set1_pd(LOG2E);
    const __m256d half_v      = _mm256_set1_pd(0.5);
    const __m256d ln2hi_v     = _mm256_set1_pd(LN2_HI);
    const __m256d ln2lo_v     = _mm256_set1_pd(LN2_LO);
    const __m256d zero_v      = _mm256_setzero_pd();

    // Remember underflow lanes.
    const __m256d underflow_mask = _mm256_cmp_pd(x, underflow_v, _CMP_LE_OS);

    // Clamp.
    x = _mm256_max_pd(x, underflow_v);

    // n = floor(x * log2e + 0.5)
    __m256d n = _mm256_floor_pd(_mm256_add_pd(_mm256_mul_pd(x, log2e_v), half_v));

    // r = x - n*ln2_hi - n*ln2_lo
    __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(n, ln2hi_v));
    r = _mm256_sub_pd(r, _mm256_mul_pd(n, ln2lo_v));

    // Horner for exp(r).
    __m256d p = _mm256_set1_pd(EXP_C12);
#define MUL_ADD(a, b, c) _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C11));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C10));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C9));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C8));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C7));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C6));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C5));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C4));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C3));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C2));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C1));
    p = MUL_ADD(p, r, _mm256_set1_pd(EXP_C0));
#undef MUL_ADD

    // 2^n: split into two 128-bit halves to avoid AVX2-only _mm256_cvtepi32_epi64.
    // Convert n to int32 via 128-bit SSE, then build the IEEE754 exponent field.
    __m128d n_lo = _mm256_castpd256_pd128(n);
    __m128d n_hi = _mm256_extractf128_pd(n, 1);

    auto build_pow2 = [](__m128d nd) -> __m128d {
        // cvttpd_epi32 gives 2 int32 in a 128-bit lane (upper 64 bits zero).
        __m128i ni32 = _mm_cvttpd_epi32(nd);
        // Widen int32 -> int64 via arithmetic: shift up 32, then sign-extend? No:
        // cvtepi32_epi64 is SSE4.1. Use unpacklo + shift instead (pure SSE2):
        //   int64 = (int32 + 1023) << 52
        // Since n is in [-1022, 1023] for valid doubles, n+1023 fits in int32.
        __m128i bias128 = _mm_set1_epi32(static_cast<int>(EXPONENT_BIAS));
        ni32 = _mm_add_epi32(ni32, bias128);
        // Widen int32 -> int64: interleave with zeros so each int32 occupies
        // the low 32 bits of a 64-bit slot, then shift left 52.
        __m128i zero128 = _mm_setzero_si128();
        __m128i i64 = _mm_unpacklo_epi32(ni32, zero128); // [i32[0], 0, i32[1], 0]
        i64 = _mm_slli_epi64(i64, 52);
        return _mm_castsi128_pd(i64);
    };

    __m128d pow2_lo = build_pow2(n_lo);
    __m128d pow2_hi = build_pow2(n_hi);
    __m256d pow2n = _mm256_set_m128d(pow2_hi, pow2_lo);

    __m256d result = _mm256_mul_pd(p, pow2n);
    result = _mm256_blendv_pd(result, zero_v, underflow_mask);
    return result;
}

#endif // LIBHMM_HAS_AVX || LIBHMM_HAS_AVX2

// ---------------------------------------------------------------------------
// SSE2: 2-wide exp(double)
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_SSE2)

static inline __m128d exp_pd_sse2(__m128d x) noexcept {
    const __m128d underflow_v = _mm_set1_pd(EXP_UNDERFLOW);
    const __m128d log2e_v     = _mm_set1_pd(LOG2E);
    const __m128d half_v      = _mm_set1_pd(0.5);
    const __m128d ln2hi_v     = _mm_set1_pd(LN2_HI);
    const __m128d ln2lo_v     = _mm_set1_pd(LN2_LO);
    const __m128d zero_v      = _mm_setzero_pd();

    // Underflow mask (all-1s in lane where x <= threshold).
    const __m128d underflow_mask = _mm_cmple_pd(x, underflow_v);

    // Clamp.
    x = _mm_max_pd(x, underflow_v);

    // n = floor(x * log2e + 0.5)  — SSE2 has no floor_pd; use cvtpd_epi32 truncation trick.
    // floor(v) = trunc(v) when v>=0, trunc(v)-1 when v<0 and not integer.
    // Simpler: convert to int via _mm_cvttpd_epi32 (truncation), then correct.
    __m128d t = _mm_add_pd(_mm_mul_pd(x, log2e_v), half_v);
    __m128i ni32 = _mm_cvttpd_epi32(t); // 2 int32 in lower 64 bits
    __m128d n = _mm_cvtepi32_pd(ni32);
    // If we truncated toward zero and t was negative, n may be 1 too large.
    // Correction: if n > t, n -= 1.
    __m128d mask_corr = _mm_cmpgt_pd(n, t);
    n = _mm_sub_pd(n, _mm_and_pd(mask_corr, _mm_set1_pd(1.0)));

    // r = x - n*ln2_hi - n*ln2_lo
    __m128d r = _mm_sub_pd(x, _mm_mul_pd(n, ln2hi_v));
    r = _mm_sub_pd(r, _mm_mul_pd(n, ln2lo_v));

    // Horner.
    __m128d p = _mm_set1_pd(EXP_C12);
#define MUL_ADD(a, b, c) _mm_add_pd(_mm_mul_pd((a), (b)), (c))
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C11));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C10));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C9));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C8));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C7));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C6));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C5));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C4));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C3));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C2));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C1));
    p = MUL_ADD(p, r, _mm_set1_pd(EXP_C0));
#undef MUL_ADD

    // 2^n via integer bit manipulation (same SSE2 unpack trick as AVX build_pow2).
    __m128i ni32b = _mm_cvttpd_epi32(n);
    __m128i bias128 = _mm_set1_epi32(static_cast<int>(EXPONENT_BIAS));
    ni32b = _mm_add_epi32(ni32b, bias128);
    __m128i zero128 = _mm_setzero_si128();
    __m128i i64 = _mm_unpacklo_epi32(ni32b, zero128);
    i64 = _mm_slli_epi64(i64, 52);
    __m128d pow2n = _mm_castsi128_pd(i64);

    __m128d result = _mm_mul_pd(p, pow2n);
    // Zero underflow lanes: SSE2 has no blendv; use andnot/or.
    result = _mm_or_pd(_mm_andnot_pd(underflow_mask, result),
                       _mm_and_pd(underflow_mask, zero_v));
    return result;
}

#endif // LIBHMM_HAS_SSE2

// ---------------------------------------------------------------------------
// NEON: 2-wide exp(double)
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_NEON)

static inline float64x2_t exp_pd_neon(float64x2_t x) noexcept {
    const float64x2_t underflow_v = vdupq_n_f64(EXP_UNDERFLOW);
    const float64x2_t log2e_v     = vdupq_n_f64(LOG2E);
    const float64x2_t half_v      = vdupq_n_f64(0.5);
    const float64x2_t ln2hi_v     = vdupq_n_f64(LN2_HI);
    const float64x2_t ln2lo_v     = vdupq_n_f64(LN2_LO);
    const float64x2_t zero_v      = vdupq_n_f64(0.0);

    // Underflow mask: valid = (x > threshold).
    const uint64x2_t valid_mask = vcgtq_f64(x, underflow_v);

    // Clamp.
    x = vmaxq_f64(x, underflow_v);

    // n = floor(x * log2e + 0.5)  — use vrndmq_f64 (floor, AArch64).
    float64x2_t n = vrndmq_f64(vfmaq_f64(half_v, x, log2e_v));

    // r = x - n*ln2_hi - n*ln2_lo
    float64x2_t r = vfmsq_f64(x, n, ln2hi_v); // r = x - n*ln2_hi
    r = vfmsq_f64(r, n, ln2lo_v);             // r = r - n*ln2_lo

    // Horner.
    float64x2_t p = vdupq_n_f64(EXP_C12);
    p = vfmaq_f64(vdupq_n_f64(EXP_C11), p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C10), p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C9),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C8),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C7),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C6),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C5),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C4),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C3),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C2),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C1),  p, r);
    p = vfmaq_f64(vdupq_n_f64(EXP_C0),  p, r);

    // 2^n via integer bit manipulation.
    // vcvtq_s64_f64 converts float64x2 -> int64x2.
    int64x2_t ni64 = vcvtq_s64_f64(n);
    ni64 = vaddq_s64(ni64, vdupq_n_s64(static_cast<int64_t>(EXPONENT_BIAS)));
    ni64 = vshlq_n_s64(ni64, 52);
    float64x2_t pow2n = vreinterpretq_f64_s64(ni64);

    float64x2_t result = vmulq_f64(p, pow2n);
    // Zero lanes where original x was <= underflow threshold.
    result = vbslq_f64(valid_mask, result, zero_v);
    return result;
}

#endif // LIBHMM_HAS_NEON

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
    __m128d m  = _mm_max_pd(lo, hi);
    return hmax_pd_sse2(m);
}
static inline double hadd_pd_avx(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    return hadd_pd_sse2(s);
}
#endif

// ---------------------------------------------------------------------------
// Vector log(double) helpers — used by Tier-2 distribution kernels.
//
// Shared constants.
// ---------------------------------------------------------------------------
// log polynomial: log(m) = 2y*(c0 + c1*y^2 + c2*y^4 + ... + c6*y^12)
// where y = (m-1)/(m+1), m in [1/sqrt(2), sqrt(2)].  c_k = 1/(2k+1).
static constexpr double LOG_C0 = 1.0;                        // 1/1
static constexpr double LOG_C1 = 3.3333333333333333e-1;      // 1/3
static constexpr double LOG_C2 = 2.0000000000000000e-1;      // 1/5
static constexpr double LOG_C3 = 1.4285714285714285e-1;      // 1/7
static constexpr double LOG_C4 = 1.1111111111111111e-1;      // 1/9
static constexpr double LOG_C5 = 9.0909090909090909e-2;      // 1/11
static constexpr double LOG_C6 = 7.6923076923076923e-2;      // 1/13
static constexpr double SQRT2  = 1.41421356237309504880168872420969807;

// ---------------------------------------------------------------------------
// AVX-512: 8-wide log(double)
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_AVX512)

// Exposed in the anonymous namespace so lognormal/pareto TUs can call it.
static inline __m512d log_pd_avx512(__m512d x) noexcept {
    const __m512d neg_inf_v  = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    const __m512d sqrt2_v    = _mm512_set1_pd(SQRT2);
    const __m512d one_v      = _mm512_set1_pd(1.0);
    const __m512d half_v     = _mm512_set1_pd(0.5);
    const __m512d two_v      = _mm512_set1_pd(2.0);
    const __m512d ln2hi_v    = _mm512_set1_pd(LN2_HI);
    const __m512d ln2lo_v    = _mm512_set1_pd(LN2_LO);

    // Guard: x <= 0 -> -inf.
    const __mmask8 invalid_mask = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_LE_OS);

    // Extract exponent e and mantissa m: x = 2^e * m, m in [1,2).
    // bits = reinterpret as int64; e_biased = bits >> 52 (upper 11 bits)
    __m512i bits = _mm512_castpd_si512(x);
    // Exponent field: e_biased = (bits >> 52) & 0x7FF
    __m512i e_biased = _mm512_srli_epi64(bits, 52);
    // Clear exponent bits; set exponent to 1023 (= exponent of 1.0) to get m.
    const __m512i mantissa_mask  = _mm512_set1_epi64(0x000FFFFFFFFFFFFFLL);
    const __m512i exponent_one   = _mm512_set1_epi64(0x3FF0000000000000LL);
    __m512i mbits = _mm512_or_si512(_mm512_and_si512(bits, mantissa_mask), exponent_one);
    __m512d m = _mm512_castsi512_pd(mbits);
    // Unbiased exponent as double: e = e_biased - 1023
    // _mm512_cvtepi64_pd requires AVX-512 DQ; scalar workaround via store/convert.
    // Since e_biased is in [0, 2046] for normal doubles, the subtract fits in int32.
    // Truncate to int32 (upper 32 bits of each int64 are zero after srli_epi64),
    // then use the existing cvtpd path: extract as two 256-bit halves of int32.
    __m512i e_unbiased64 = _mm512_sub_epi64(e_biased, _mm512_set1_epi64(1023LL));
    // Convert int64 exponents to double via scalar (8 lanes).
    alignas(64) long long e_arr[8];
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(e_arr), e_unbiased64);
    __m512d e = _mm512_set_pd(
        static_cast<double>(e_arr[7]), static_cast<double>(e_arr[6]),
        static_cast<double>(e_arr[5]), static_cast<double>(e_arr[4]),
        static_cast<double>(e_arr[3]), static_cast<double>(e_arr[2]),
        static_cast<double>(e_arr[1]), static_cast<double>(e_arr[0]));

    // If m > sqrt(2): e += 1, m *= 0.5  (so m in [1/sqrt(2), sqrt(2)])
    __mmask8 adj_mask = _mm512_cmp_pd_mask(m, sqrt2_v, _CMP_GT_OS);
    e = _mm512_mask_add_pd(e, adj_mask, e, one_v);
    m = _mm512_mask_mul_pd(m, adj_mask, m, half_v);

    // y = (m - 1) / (m + 1)
    __m512d y = _mm512_div_pd(_mm512_sub_pd(m, one_v), _mm512_add_pd(m, one_v));
    __m512d y2 = _mm512_mul_pd(y, y);

    // Horner: p = c0 + y2*(c1 + y2*(c2 + ... y2*c6))
    __m512d p = _mm512_set1_pd(LOG_C6);
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(LOG_C5));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(LOG_C4));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(LOG_C3));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(LOG_C2));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(LOG_C1));
    p = _mm512_fmadd_pd(p, y2, _mm512_set1_pd(LOG_C0));
    // log(m) = 2 * y * p
    __m512d log_m = _mm512_mul_pd(_mm512_mul_pd(two_v, y), p);

    // log(x) = e * LN2_HI + e * LN2_LO + log(m)
    __m512d result = _mm512_fmadd_pd(e, ln2hi_v,
                       _mm512_fmadd_pd(e, ln2lo_v, log_m));

    // Apply invalid guard.
    result = _mm512_mask_blend_pd(invalid_mask, result, neg_inf_v);
    return result;
}

#endif // LIBHMM_HAS_AVX512

// ---------------------------------------------------------------------------
// AVX: 4-wide log(double)
// Uses _mm256_cvtepi64_pd (AVX-512 DQ), so falls back to two 128-bit extracts
// for AVX-1 / AVX2 compatibility.
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)

static inline __m256d log_pd_avx(__m256d x) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    const __m256d neg_inf_v = _mm256_set1_pd(neg_inf);
    const __m256d sqrt2_v   = _mm256_set1_pd(SQRT2);
    const __m256d one_v     = _mm256_set1_pd(1.0);
    const __m256d half_v    = _mm256_set1_pd(0.5);
    const __m256d two_v     = _mm256_set1_pd(2.0);
    const __m256d ln2hi_v   = _mm256_set1_pd(LN2_HI);
    const __m256d ln2lo_v   = _mm256_set1_pd(LN2_LO);

    // Guard: x <= 0 -> -inf.
    const __m256d invalid_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LE_OS);

    // Extract exponent and mantissa via two 128-bit halves (AVX-1 compatible).
    auto extract_em = [](__m128d xh, __m128d &mh, __m128d &eh) {
        // Reinterpret as int64.
        __m128i bits   = _mm_castpd_si128(xh);
        // e_biased = bits >> 52  (signed shift via arithmetic right)
        __m128i eb = _mm_srli_epi64(bits, 52);
        // mantissa bits: clear exponent, set to 1.0 exponent
        __m128i mant_mask = _mm_set1_epi64x(0x000FFFFFFFFFFFFFLL);
        __m128i exp_one   = _mm_set1_epi64x(0x3FF0000000000000LL);
        __m128i mbits = _mm_or_si128(_mm_and_si128(bits, mant_mask), exp_one);
        mh = _mm_castsi128_pd(mbits);
        // Convert e_biased to double and subtract bias 1023.
        // e_biased is in [0, 2046] for normal doubles; fits in 32-bit.
        // Use unpack trick: put e_biased into low 32 bits of 64-bit int, convert to float.
        // Actually: e_biased is already in the right int64 lane from srli_epi64.
        // Simpler: store, load as int64, subtract 1023, store, load as double.
        // Pure SIMD: broadcast 1023, subtract.
        __m128i bias = _mm_set1_epi64x(1023LL);
        __m128i eu   = _mm_sub_epi64(eb, bias);
        // Convert int64 to double: no direct SSE2/AVX instruction.
        // Use scalar workaround for the 2 lanes.
        long long e0, e1;
        _mm_storel_epi64(reinterpret_cast<__m128i*>(&e0), eu);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(&e1),
                         _mm_unpackhi_epi64(eu, eu));
        eh = _mm_set_pd(static_cast<double>(e1), static_cast<double>(e0));
    };

    __m128d x_lo = _mm256_castpd256_pd128(x);
    __m128d x_hi = _mm256_extractf128_pd(x, 1);
    __m128d m_lo, e_lo, m_hi, e_hi;
    extract_em(x_lo, m_lo, e_lo);
    extract_em(x_hi, m_hi, e_hi);
    __m256d m = _mm256_set_m128d(m_hi, m_lo);
    __m256d e = _mm256_set_m128d(e_hi, e_lo);

    // If m > sqrt(2): e += 1, m *= 0.5.
    __m256d adj_mask = _mm256_cmp_pd(m, sqrt2_v, _CMP_GT_OS);
    e = _mm256_add_pd(e, _mm256_and_pd(adj_mask, one_v));
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half_v), adj_mask);

    // y = (m-1)/(m+1),  y2 = y*y.
    __m256d y  = _mm256_div_pd(_mm256_sub_pd(m, one_v), _mm256_add_pd(m, one_v));
    __m256d y2 = _mm256_mul_pd(y, y);

#define FMA256(a, b, c) _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))
    __m256d p = _mm256_set1_pd(LOG_C6);
    p = FMA256(p, y2, _mm256_set1_pd(LOG_C5));
    p = FMA256(p, y2, _mm256_set1_pd(LOG_C4));
    p = FMA256(p, y2, _mm256_set1_pd(LOG_C3));
    p = FMA256(p, y2, _mm256_set1_pd(LOG_C2));
    p = FMA256(p, y2, _mm256_set1_pd(LOG_C1));
    p = FMA256(p, y2, _mm256_set1_pd(LOG_C0));
#undef FMA256
    __m256d log_m = _mm256_mul_pd(_mm256_mul_pd(two_v, y), p);

    __m256d result = _mm256_add_pd(
        _mm256_mul_pd(e, ln2hi_v),
        _mm256_add_pd(_mm256_mul_pd(e, ln2lo_v), log_m));
    result = _mm256_blendv_pd(result, neg_inf_v, invalid_mask);
    return result;
}

#endif // LIBHMM_HAS_AVX || LIBHMM_HAS_AVX2

// ---------------------------------------------------------------------------
// SSE2: 2-wide log(double)
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_SSE2)

static inline __m128d log_pd_sse2(__m128d x) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    const __m128d neg_inf_v = _mm_set1_pd(neg_inf);
    const __m128d sqrt2_v   = _mm_set1_pd(SQRT2);
    const __m128d one_v     = _mm_set1_pd(1.0);
    const __m128d half_v    = _mm_set1_pd(0.5);
    const __m128d two_v     = _mm_set1_pd(2.0);
    const __m128d ln2hi_v   = _mm_set1_pd(LN2_HI);
    const __m128d ln2lo_v   = _mm_set1_pd(LN2_LO);

    const __m128d invalid_mask = _mm_cmple_pd(x, _mm_setzero_pd());

    __m128i bits     = _mm_castpd_si128(x);
    __m128i eb       = _mm_srli_epi64(bits, 52);
    __m128i mant_mask = _mm_set1_epi64x(0x000FFFFFFFFFFFFFLL);
    __m128i exp_one   = _mm_set1_epi64x(0x3FF0000000000000LL);
    __m128i mbits     = _mm_or_si128(_mm_and_si128(bits, mant_mask), exp_one);
    __m128d m         = _mm_castsi128_pd(mbits);
    // Convert int64 exponent to double via scalar.
    __m128i bias = _mm_set1_epi64x(1023LL);
    __m128i eu   = _mm_sub_epi64(eb, bias);
    long long e0, e1;
    _mm_storel_epi64(reinterpret_cast<__m128i*>(&e0), eu);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(&e1),
                     _mm_unpackhi_epi64(eu, eu));
    __m128d e = _mm_set_pd(static_cast<double>(e1), static_cast<double>(e0));

    __m128d adj_mask = _mm_cmpgt_pd(m, sqrt2_v);
    e = _mm_add_pd(e, _mm_and_pd(adj_mask, one_v));
    m = _mm_or_pd(_mm_andnot_pd(adj_mask, m),
                  _mm_and_pd(adj_mask, _mm_mul_pd(m, half_v)));

    __m128d y  = _mm_div_pd(_mm_sub_pd(m, one_v), _mm_add_pd(m, one_v));
    __m128d y2 = _mm_mul_pd(y, y);

#define FMA128(a, b, c) _mm_add_pd(_mm_mul_pd((a), (b)), (c))
    __m128d p = _mm_set1_pd(LOG_C6);
    p = FMA128(p, y2, _mm_set1_pd(LOG_C5));
    p = FMA128(p, y2, _mm_set1_pd(LOG_C4));
    p = FMA128(p, y2, _mm_set1_pd(LOG_C3));
    p = FMA128(p, y2, _mm_set1_pd(LOG_C2));
    p = FMA128(p, y2, _mm_set1_pd(LOG_C1));
    p = FMA128(p, y2, _mm_set1_pd(LOG_C0));
#undef FMA128
    __m128d log_m = _mm_mul_pd(_mm_mul_pd(two_v, y), p);

    __m128d result = _mm_add_pd(_mm_mul_pd(e, ln2hi_v),
                       _mm_add_pd(_mm_mul_pd(e, ln2lo_v), log_m));
    result = _mm_or_pd(_mm_andnot_pd(invalid_mask, result),
                       _mm_and_pd(invalid_mask, neg_inf_v));
    return result;
}

#endif // LIBHMM_HAS_SSE2

// ---------------------------------------------------------------------------
// NEON: 2-wide log(double)
// ---------------------------------------------------------------------------
#if defined(LIBHMM_HAS_NEON)

static inline float64x2_t log_pd_neon(float64x2_t x) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    const float64x2_t neg_inf_v = vdupq_n_f64(neg_inf);
    const float64x2_t sqrt2_v   = vdupq_n_f64(SQRT2);
    const float64x2_t one_v     = vdupq_n_f64(1.0);
    const float64x2_t half_v    = vdupq_n_f64(0.5);
    const float64x2_t two_v     = vdupq_n_f64(2.0);
    const float64x2_t ln2hi_v   = vdupq_n_f64(LN2_HI);
    const float64x2_t ln2lo_v   = vdupq_n_f64(LN2_LO);

    const uint64x2_t invalid_mask = vcleq_f64(x, vdupq_n_f64(0.0));

    // Extract exponent and mantissa.
    uint64x2_t bits  = vreinterpretq_u64_f64(x);
    uint64x2_t eb    = vshrq_n_u64(bits, 52);
    const uint64x2_t mant_mask = vdupq_n_u64(0x000FFFFFFFFFFFFFULL);
    const uint64x2_t exp_one   = vdupq_n_u64(0x3FF0000000000000ULL);
    uint64x2_t mbits = vorrq_u64(vandq_u64(bits, mant_mask), exp_one);
    float64x2_t m    = vreinterpretq_f64_u64(mbits);
    // e = (int64)(e_biased - 1023) -> double
    int64x2_t ei = vsubq_s64(vreinterpretq_s64_u64(eb), vdupq_n_s64(1023LL));
    float64x2_t e = vcvtq_f64_s64(ei);

    // If m > sqrt(2): e += 1, m *= 0.5.
    uint64x2_t adj_mask = vcgtq_f64(m, sqrt2_v);
    e = vbslq_f64(adj_mask, vaddq_f64(e, one_v), e);
    m = vbslq_f64(adj_mask, vmulq_f64(m, half_v), m);

    float64x2_t y  = vdivq_f64(vsubq_f64(m, one_v), vaddq_f64(m, one_v));
    float64x2_t y2 = vmulq_f64(y, y);

    float64x2_t p = vdupq_n_f64(LOG_C6);
    p = vfmaq_f64(vdupq_n_f64(LOG_C5), p, y2);
    p = vfmaq_f64(vdupq_n_f64(LOG_C4), p, y2);
    p = vfmaq_f64(vdupq_n_f64(LOG_C3), p, y2);
    p = vfmaq_f64(vdupq_n_f64(LOG_C2), p, y2);
    p = vfmaq_f64(vdupq_n_f64(LOG_C1), p, y2);
    p = vfmaq_f64(vdupq_n_f64(LOG_C0), p, y2);
    float64x2_t log_m = vmulq_f64(vmulq_f64(two_v, y), p);

    float64x2_t result = vfmaq_f64(vfmaq_f64(log_m, e, ln2lo_v), e, ln2hi_v);
    result = vbslq_f64(invalid_mask, neg_inf_v, result);
    return result;
}

#endif // LIBHMM_HAS_NEON

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
        maxVal = _mm512_reduce_max_pd(vmax);
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
        if (t > maxVal) maxVal = t;
    }
    return maxVal;
}

// -----------------------------------------------------------------------------
// sum_exp_sum2_minus_max: sum of exp(a[i]+b[i] - maxVal)
// -----------------------------------------------------------------------------
double TranscendentalKernels::sum_exp_sum2_minus_max(const double *a, const double *b,
                                                     std::size_t size, double maxVal) noexcept {
    if (!std::isfinite(maxVal)) return 0.0;
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
            vsum = _mm512_add_pd(vsum, exp_pd_avx512(term));
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
            vsum = _mm256_add_pd(vsum, exp_pd_avx(term));
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
            vsum = _mm_add_pd(vsum, exp_pd_sse2(term));
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
            vsum = vaddq_f64(vsum, exp_pd_neon(term));
        }
        sum += vaddvq_f64(vsum);
    }
#endif

    // Scalar tail.
    for (; i < size; ++i) {
        const double t = a[i] + b[i];
        if (std::isfinite(t)) sum += std::exp(t - maxVal);
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
        maxVal = _mm512_reduce_max_pd(vmax);
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
        if (t > maxVal) maxVal = t;
    }
    return maxVal;
}

// -----------------------------------------------------------------------------
// sum_exp_sum3_minus_max: sum of exp(a[i]+b[i]+c[i] - maxVal)
// -----------------------------------------------------------------------------
double TranscendentalKernels::sum_exp_sum3_minus_max(const double *a, const double *b,
                                                     const double *c, std::size_t size,
                                                     double maxVal) noexcept {
    if (!std::isfinite(maxVal)) return 0.0;
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
            vsum = _mm512_add_pd(vsum, exp_pd_avx512(term));
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
            vsum = _mm256_add_pd(vsum, exp_pd_avx(term));
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
            vsum = _mm_add_pd(vsum, exp_pd_sse2(term));
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
            vsum = vaddq_f64(vsum, exp_pd_neon(term));
        }
        sum += vaddvq_f64(vsum);
    }
#endif

    // Scalar tail.
    for (; i < size; ++i) {
        const double t = a[i] + b[i] + c[i];
        if (std::isfinite(t)) sum += std::exp(t - maxVal);
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
            __m512d vd  = _mm512_loadu_pd(dst + i);
            __m512d va  = _mm512_loadu_pd(a + i);
            __m512d vb  = _mm512_loadu_pd(b + i);
            __m512d arg = _mm512_add_pd(_mm512_add_pd(va, vb), vbias);
            vd = _mm512_add_pd(vd, exp_pd_avx512(arg));
            _mm512_storeu_pd(dst + i, vd);
        }
    }
#endif

#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    {
        const __m256d vbias = _mm256_set1_pd(bias);
        for (; i + 4 <= size; i += 4) {
            __m256d vd  = _mm256_loadu_pd(dst + i);
            __m256d va  = _mm256_loadu_pd(a + i);
            __m256d vb  = _mm256_loadu_pd(b + i);
            __m256d arg = _mm256_add_pd(_mm256_add_pd(va, vb), vbias);
            vd = _mm256_add_pd(vd, exp_pd_avx(arg));
            _mm256_storeu_pd(dst + i, vd);
        }
    }
#endif

#if defined(LIBHMM_HAS_SSE2)
    {
        const __m128d vbias = _mm_set1_pd(bias);
        for (; i + 2 <= size; i += 2) {
            __m128d vd  = _mm_loadu_pd(dst + i);
            __m128d va  = _mm_loadu_pd(a + i);
            __m128d vb  = _mm_loadu_pd(b + i);
            __m128d arg = _mm_add_pd(_mm_add_pd(va, vb), vbias);
            vd = _mm_add_pd(vd, exp_pd_sse2(arg));
            _mm_storeu_pd(dst + i, vd);
        }
    }
#endif

#if defined(LIBHMM_HAS_NEON)
    {
        const float64x2_t vbias = vdupq_n_f64(bias);
        for (; i + 2 <= size; i += 2) {
            float64x2_t vd  = vld1q_f64(dst + i);
            float64x2_t va  = vld1q_f64(a + i);
            float64x2_t vb  = vld1q_f64(b + i);
            float64x2_t arg = vaddq_f64(vaddq_f64(va, vb), vbias);
            vd = vaddq_f64(vd, exp_pd_neon(arg));
            vst1q_f64(dst + i, vd);
        }
    }
#endif

    for (; i < size; ++i) {
        dst[i] += std::exp(a[i] + b[i] + bias);
    }
}

} // namespace detail
} // namespace performance
} // namespace libhmm
