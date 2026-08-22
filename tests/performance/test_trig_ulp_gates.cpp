// tests/performance/test_trig_ulp_gates.cpp
//
// Per-tier ULP accuracy gate for the clean-room quadrant-reduction cos_pd/
// sin_pd SIMD kernels (issue #74, commit 61e7347). Ground truth comes from
// tests/performance/trig_ulp_vectors.inc: cos/sin evaluated at 320-bit
// mpmath precision, each rounded once to nearest double
// (scripts/gen_trig_ulp_vectors.py).
//
// This mirrors the idiom in test_transcendental_kernels.cpp: forward
// declarations of the per-ISA cos_batch_<tier>/sin_batch_<tier> free
// functions, guarded by the LIBHMM_BUILD_*_KERNEL compile definitions this
// target receives from LIBHMM_DISPATCH_DEFINES (tests/CMakeLists.txt), and
// runtime-gated by libhmm::platform::supports_sse2/avx2/avx512(). Each
// compiled-in tier's vector body is exercised directly, bypassing the
// DoubleVecOps dispatch table, so a CPU that happens to prefer a higher
// tier doesn't hide a lower tier's regression.
//
// The kernels themselves are DONE (already landed, already pass smoke
// tests) — this file only builds the validation half: reference vectors +
// gate. Do not loosen any budget below without a matching kernel fix; a
// budget miss here is a kernel bug for the orchestrator, not a test-tuning
// problem.

#include "libhmm/performance/simd_double_ops.h"
#include "libhmm/platform/cpu_detection.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

// Correctly-rounded cos()/sin() reference vectors (input_bits, cos_bits,
// sin_bits), evaluated at 320-bit precision with mpmath then each rounded
// once to nearest double. Defines struct TrigUlpVector, kTrigUlpVectors[]
// (main gate budget) and kTrigUlpSpecials[] (domain-edge/NaN/Inf, gated
// separately). See scripts/gen_trig_ulp_vectors.py.
#include "trig_ulp_vectors.inc"

// The vectorized domain bound the cos_batch/sin_batch wrappers gate their
// per-lane scalar-libm fixup on. Must match kTrigDMax in
// include/libhmm/detail/trig_cleanroom_data.inc (not included wholesale here
// -- its other constants, e.g. kTrigPio2/kTrigSinC/kTrigCosC, are unused in
// this TU and would trip -Wunused-variable under GCC/Clang -Wall).
constexpr double kTrigDMax = 0x1.0000000000000p+23; // 2^23

double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

// Sign-aware ULP distance on the integer lattice; ported from libstats'
// cosUlpError (scripts/gen_cos_ulp_vectors.py /
// tests/test_simd_neon_cos_accuracy.cpp). Both cos and sin span both signs
// and both have zero crossings, so cross-zero distance is charged at full
// weight rather than being treated as "near enough". inf/NaN handled
// explicitly so the metric never has to trust IEEE comparison operators on
// non-finite values.
double trigUlpError(double got, double ref) {
    if (std::isnan(ref))
        return std::isnan(got) ? 0.0 : 1e18;
    if (std::isinf(ref))
        return (got == ref) ? 0.0 : 1e18;
    if (!std::isfinite(got))
        return 1e18;
    const auto ordered = [](double v) -> std::int64_t {
        std::int64_t i;
        std::memcpy(&i, &v, sizeof i);
        return i < 0 ? static_cast<std::int64_t>(0x8000000000000000ULL) - i : i;
    };
    const std::int64_t g = ordered(got), r = ordered(ref);
    return static_cast<double>(g > r ? g - r : r - g);
}

} // namespace

// =========================================================================
// Unit self-tests for trigUlpError. A gate that cannot fail is worthless —
// these pin the metric's own behaviour before it is trusted to gate kernels.
// =========================================================================

TEST(TrigUlpErrorSelfTest, AdjacentDoublesAreOneUlp) {
    const double a = 1.0;
    const double b = std::nextafter(a, 2.0);
    EXPECT_DOUBLE_EQ(trigUlpError(a, b), 1.0);
    EXPECT_DOUBLE_EQ(trigUlpError(b, a), 1.0);
}

TEST(TrigUlpErrorSelfTest, EqualIsZero) {
    EXPECT_DOUBLE_EQ(trigUlpError(0.5, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(trigUlpError(-3.25, -3.25), 0.0);
    EXPECT_DOUBLE_EQ(trigUlpError(0.0, 0.0), 0.0);
}

TEST(TrigUlpErrorSelfTest, CrossZeroChargedFullWeight) {
    // Smallest positive and smallest negative subnormals are each exactly
    // one representable step from zero; crossing zero must cost 2 ULP, not
    // be treated as "close" by an unsigned bit-pattern distance.
    const double tiny_pos = std::numeric_limits<double>::denorm_min();
    const double tiny_neg = -tiny_pos;
    EXPECT_DOUBLE_EQ(trigUlpError(tiny_pos, tiny_neg), 2.0);
    EXPECT_DOUBLE_EQ(trigUlpError(tiny_neg, tiny_pos), 2.0);
}

TEST(TrigUlpErrorSelfTest, NanVsNumberIsHuge) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_GT(trigUlpError(nan, 1.0), 1e17); // got=NaN, ref finite
    EXPECT_GT(trigUlpError(1.0, nan), 1e17); // got finite, ref=NaN
}

TEST(TrigUlpErrorSelfTest, InfVsNumberIsHuge) {
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_GT(trigUlpError(inf, 1.0), 1e17); // got=Inf, ref finite
    EXPECT_GT(trigUlpError(1.0, inf), 1e17); // got finite, ref=Inf
}

TEST(TrigUlpErrorSelfTest, NanVsNanIsZero) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_DOUBLE_EQ(trigUlpError(nan, nan), 0.0);
}

// =========================================================================
// Per-ISA symbol forward declarations (idiom: test_transcendental_kernels.cpp
// section 8). Defined in src/performance/simd_double_ops_{scalar,sse2,avx2,
// avx512,neon}.cpp; scalar is always compiled in, the others only when the
// corresponding LIBHMM_BUILD_*_KERNEL compile definition is set (wired onto
// this test target from LIBHMM_DISPATCH_DEFINES in tests/CMakeLists.txt).
// =========================================================================
namespace libhmm::performance::detail {

void cos_batch_scalar(const double *in, double *out, std::size_t n) noexcept;
void sin_batch_scalar(const double *in, double *out, std::size_t n) noexcept;

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
void cos_batch_sse2(const double *in, double *out, std::size_t n) noexcept;
void sin_batch_sse2(const double *in, double *out, std::size_t n) noexcept;
#endif

#if defined(LIBHMM_BUILD_AVX2_KERNEL)
void cos_batch_avx2(const double *in, double *out, std::size_t n) noexcept;
void sin_batch_avx2(const double *in, double *out, std::size_t n) noexcept;
#endif

#if defined(LIBHMM_BUILD_AVX512_KERNEL)
void cos_batch_avx512(const double *in, double *out, std::size_t n) noexcept;
void sin_batch_avx512(const double *in, double *out, std::size_t n) noexcept;
#endif

#if defined(LIBHMM_BUILD_NEON_KERNEL)
void cos_batch_neon(const double *in, double *out, std::size_t n) noexcept;
void sin_batch_neon(const double *in, double *out, std::size_t n) noexcept;
#endif

} // namespace libhmm::performance::detail

namespace {

namespace pd = libhmm::performance::detail;

using CosSinFn = void (*)(const double *, double *, std::size_t) noexcept;

// -------------------------------------------------------------------------
// Budgets. FMA tiers (avx2/avx512/neon) get the tight budget since the
// (r, rlo) compensated reduction and fused cores are designed for sub-ULP
// accuracy; sse2 has no FMA in the polynomial cores so gets a looser floor;
// scalar is whatever the platform libm delivers. PROVISIONAL until measured
// values are recorded back into the design doc — do not loosen without a
// kernel fix; see the file banner.
// -------------------------------------------------------------------------
constexpr double kBudgetFma = 1.0; // avx2 / avx512 / neon
// [[maybe_unused]]: the SSE2 gate compiles out on ARM builds (AppleClang
// -Wunused-const-variable fires there); every x86 build uses it.
[[maybe_unused]] constexpr double kBudgetSse2 = 2.0; // no FMA in the polynomial cores
constexpr double kBudgetScalar = 4.0;                // platform libm (UCRT on Windows)
constexpr double kBudgetSpecials = 4.0;              // libm fixup path, every tier
constexpr double kBudgetDispatched = 4.0;            // loosest applicable; tight budgets live above

struct GateResult {
    double cos_max = 0.0, cos_mean = 0.0;
    double sin_max = 0.0, sin_mean = 0.0;
    double cos_worst_x = 0.0, sin_worst_x = 0.0;
};

// Runs cos_fn/sin_fn as ONE batch call each over the full vector set,
// computes max/mean ULP vs. the mpmath references, and prints a
// machine-readable-ish one-liner per function (consumed by the orchestrator
// afterward to record measured values in docs).
GateResult run_gate(const char *tier, const char *set_name, const TrigUlpVector *vecs,
                    std::size_t n, CosSinFn cos_fn, CosSinFn sin_fn) {
    std::vector<double> in(n), cos_out(n), sin_out(n);
    for (std::size_t i = 0; i < n; ++i)
        in[i] = bitsToF64(vecs[i].x_bits);

    cos_fn(in.data(), cos_out.data(), n);
    sin_fn(in.data(), sin_out.data(), n);

    GateResult r;
    double cos_sum = 0.0, sin_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ce = trigUlpError(cos_out[i], bitsToF64(vecs[i].cos_bits));
        cos_sum += ce;
        if (ce > r.cos_max) {
            r.cos_max = ce;
            r.cos_worst_x = in[i];
        }
        const double se = trigUlpError(sin_out[i], bitsToF64(vecs[i].sin_bits));
        sin_sum += se;
        if (se > r.sin_max) {
            r.sin_max = se;
            r.sin_worst_x = in[i];
        }
    }
    r.cos_mean = cos_sum / static_cast<double>(n);
    r.sin_mean = sin_sum / static_cast<double>(n);

    std::cout << std::setprecision(17) << "cos " << tier << " " << set_name
              << " max_ulp=" << r.cos_max << " mean_ulp=" << r.cos_mean
              << " worst_x=" << r.cos_worst_x << "\n";
    std::cout << "sin " << tier << " " << set_name << " max_ulp=" << r.sin_max
              << " mean_ulp=" << r.sin_mean << " worst_x=" << r.sin_worst_x << "\n";
    return r;
}

constexpr std::size_t kMainN = sizeof(kTrigUlpVectors) / sizeof(kTrigUlpVectors[0]);
constexpr std::size_t kSpecialsN = sizeof(kTrigUlpSpecials) / sizeof(kTrigUlpSpecials[0]);

void run_main_gate(const char *tier, CosSinFn cos_fn, CosSinFn sin_fn, double budget) {
    const GateResult r = run_gate(tier, "main", kTrigUlpVectors, kMainN, cos_fn, sin_fn);
    EXPECT_LE(r.cos_max, budget) << tier << " cos max ULP over budget (worst x=" << r.cos_worst_x
                                 << ")";
    EXPECT_LE(r.sin_max, budget) << tier << " sin max ULP over budget (worst x=" << r.sin_worst_x
                                 << ")";
}

// Specials gate: beyond-kTrigDMax finite points route through the batch
// wrappers' per-lane scalar-libm fixup, so every tier is held to the libm
// budget here (not the tight FMA budget). ±Inf and NaN are additionally
// required to produce NaN EXACTLY, independent of the ULP metric (which
// already scores NaN-vs-NaN as 0, but an explicit check makes a silent
// "kernel returns something finite for Inf" regression fail loudly).
// kTrigUlpSpecials leads with +inf, -inf, NaN (lanes 0-2) so the "must be NaN"
// assertions below exercise the vector body on every tier; with them at the
// tail of an 11-entry table the 4/8-wide tiers would hand them to the scalar
// libm fixup instead and the kernel's own NaN handling would go untested.
void run_specials_gate(const char *tier, CosSinFn cos_fn, CosSinFn sin_fn, double budget) {
    std::vector<double> in(kSpecialsN), cos_out(kSpecialsN), sin_out(kSpecialsN);
    for (std::size_t i = 0; i < kSpecialsN; ++i)
        in[i] = bitsToF64(kTrigUlpSpecials[i].x_bits);

    cos_fn(in.data(), cos_out.data(), kSpecialsN);
    sin_fn(in.data(), sin_out.data(), kSpecialsN);

    for (std::size_t i = 0; i < kSpecialsN; ++i) {
        if (!std::isfinite(in[i])) {
            EXPECT_TRUE(std::isnan(cos_out[i]))
                << tier << " specials: cos(" << in[i] << ") must be NaN, got " << cos_out[i];
            EXPECT_TRUE(std::isnan(sin_out[i]))
                << tier << " specials: sin(" << in[i] << ") must be NaN, got " << sin_out[i];
        }
    }

    double cos_max = 0.0, cos_sum = 0.0, sin_max = 0.0, sin_sum = 0.0;
    for (std::size_t i = 0; i < kSpecialsN; ++i) {
        const double ce = trigUlpError(cos_out[i], bitsToF64(kTrigUlpSpecials[i].cos_bits));
        cos_sum += ce;
        cos_max = std::max(cos_max, ce);
        const double se = trigUlpError(sin_out[i], bitsToF64(kTrigUlpSpecials[i].sin_bits));
        sin_sum += se;
        sin_max = std::max(sin_max, se);
    }
    const double cos_mean = cos_sum / static_cast<double>(kSpecialsN);
    const double sin_mean = sin_sum / static_cast<double>(kSpecialsN);
    std::cout << std::setprecision(10) << "cos " << tier << " specials max_ulp=" << cos_max
              << " mean_ulp=" << cos_mean << "\n";
    std::cout << "sin " << tier << " specials max_ulp=" << sin_max << " mean_ulp=" << sin_mean
              << "\n";
    EXPECT_LE(cos_max, budget) << tier << " specials cos max ULP over budget";
    EXPECT_LE(sin_max, budget) << tier << " specials sin max ULP over budget";
}

} // namespace

// =========================================================================
// Self-check: the generator's own domain assertion, re-verified here so a
// stale/hand-edited .inc can't silently violate the vectorized-domain
// contract the per-tier gates below depend on.
// =========================================================================

TEST(TrigUlpGates, MainVectorsRespectDomainBound) {
    for (std::size_t i = 0; i < kMainN; ++i) {
        const double x = bitsToF64(kTrigUlpVectors[i].x_bits);
        ASSERT_LE(std::fabs(x), kTrigDMax) << "main-bucket vector " << i << " outside kTrigDMax";
    }
}

// =========================================================================
// Per-tier gates. Each calls cos_batch_<tier>/sin_batch_<tier> directly (not
// through the DoubleVecOps dispatch table) on the full main vector set in
// one batch call.
// =========================================================================

TEST(TrigUlpGates, Scalar) {
    run_main_gate("scalar", pd::cos_batch_scalar, pd::sin_batch_scalar, kBudgetScalar);
}

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
TEST(TrigUlpGates, Sse2) {
    if (!libhmm::platform::supports_sse2()) {
        GTEST_SKIP() << "SSE2 not supported on this CPU";
    }
    run_main_gate("sse2", pd::cos_batch_sse2, pd::sin_batch_sse2, kBudgetSse2);
}
#endif

#if defined(LIBHMM_BUILD_AVX2_KERNEL)
TEST(TrigUlpGates, Avx2) {
    if (!libhmm::platform::supports_avx2()) {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }
    run_main_gate("avx2", pd::cos_batch_avx2, pd::sin_batch_avx2, kBudgetFma);
}
#endif

#if defined(LIBHMM_BUILD_AVX512_KERNEL)
TEST(TrigUlpGates, Avx512) {
    if (!libhmm::platform::supports_avx512()) {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }
    run_main_gate("avx512", pd::cos_batch_avx512, pd::sin_batch_avx512, kBudgetFma);
}
#endif

#if defined(LIBHMM_BUILD_NEON_KERNEL)
TEST(TrigUlpGates, Neon) {
    // NEON is the mandatory AArch64 baseline ISA — always available when this
    // TU is compiled in (mirrors TranscendentalKernelsTierParity's NEON test).
    run_main_gate("neon", pd::cos_batch_neon, pd::sin_batch_neon, kBudgetFma);
}
#endif

// =========================================================================
// Specials gate: domain-edge / beyond-domain / ±Inf / NaN, at the libm
// budget for every tier (the beyond-domain points route through the same
// per-lane scalar std::cos/std::sin fixup regardless of which tier's vector
// core ran the in-domain lanes).
// =========================================================================

TEST(TrigUlpSpecialsGates, Scalar) {
    run_specials_gate("scalar", pd::cos_batch_scalar, pd::sin_batch_scalar, kBudgetSpecials);
}

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
TEST(TrigUlpSpecialsGates, Sse2) {
    if (!libhmm::platform::supports_sse2()) {
        GTEST_SKIP() << "SSE2 not supported on this CPU";
    }
    run_specials_gate("sse2", pd::cos_batch_sse2, pd::sin_batch_sse2, kBudgetSpecials);
}
#endif

#if defined(LIBHMM_BUILD_AVX2_KERNEL)
TEST(TrigUlpSpecialsGates, Avx2) {
    if (!libhmm::platform::supports_avx2()) {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }
    run_specials_gate("avx2", pd::cos_batch_avx2, pd::sin_batch_avx2, kBudgetSpecials);
}
#endif

#if defined(LIBHMM_BUILD_AVX512_KERNEL)
TEST(TrigUlpSpecialsGates, Avx512) {
    if (!libhmm::platform::supports_avx512()) {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }
    run_specials_gate("avx512", pd::cos_batch_avx512, pd::sin_batch_avx512, kBudgetSpecials);
}
#endif

#if defined(LIBHMM_BUILD_NEON_KERNEL)
TEST(TrigUlpSpecialsGates, Neon) {
    run_specials_gate("neon", pd::cos_batch_neon, pd::sin_batch_neon, kBudgetSpecials);
}
#endif

// =========================================================================
// Dispatched-path gate: same main-set check, but through
// get_double_vec_ops().cos_batch/.sin_batch at whichever tier CPUID selects
// on this machine. Budget is the loosest applicable (4.0) since the active
// tier is not known at compile time here — the per-tier tests above carry
// the tight, tier-specific budgets.
// =========================================================================

TEST(TrigUlpGates, DispatchedPath) {
    const auto &ops = libhmm::performance::get_double_vec_ops();
    run_main_gate("dispatched", ops.cos_batch, ops.sin_batch, kBudgetDispatched);
}

// =========================================================================
// Non-lane-multiple sub-span: exercises the masked-tail/scalar-tail path by
// running a sub-span whose length (4999) is not a multiple of any lane
// count (2/4/8), through the runtime-dispatched path so it's meaningful
// regardless of which tier this machine selects.
// =========================================================================

TEST(TrigUlpGates, SubSpanNonLaneMultipleTailPath) {
    constexpr std::size_t n = 4999;
    static_assert(n < kMainN, "sub-span must fit inside the main vector set");
    static_assert(n % 2 != 0, "sub-span length must not be a multiple of any lane count");

    const auto &ops = libhmm::performance::get_double_vec_ops();
    const GateResult r = run_gate("dispatched-subspan", "subspan_4999", kTrigUlpVectors, n,
                                  ops.cos_batch, ops.sin_batch);
    EXPECT_LE(r.cos_max, kBudgetDispatched)
        << "sub-span cos max ULP over budget (worst x=" << r.cos_worst_x << ")";
    EXPECT_LE(r.sin_max, kBudgetDispatched)
        << "sub-span sin max ULP over budget (worst x=" << r.sin_worst_x << ")";
}
