// tests/performance/test_transcendental_kernels.cpp
//
// Parity tests for TranscendentalKernels: verify that each kernel
// method agrees with a stdlib-based scalar reference to within
// 1e-12 relative / 1e-15 absolute tolerance.
//
// Ground truth is always computed inline here using stdlib functions directly — NOT
// by calling the kernel's internal scalar variant — so the test is
// independent of any internal refactor.
//
// The test binary is compiled with LIBHMM_BEST_SIMD_FLAGS (see CMakeLists.txt
// Performance Primitives section), so the active SIMD path matches the production library.
//
// Section 8 below (TranscendentalKernelsTierParity) additionally calls the per-ISA
// symbols in libhmm::performance::detail (<name>_scalar/_sse2/_avx2/_avx512/_neon,
// relocated from this TU into the runtime-dispatch TUs under issue #58) DIRECTLY,
// bypassing TranscendentalKernels and the DoubleVecOps dispatch table entirely, so
// each compiled-in tier's vector body is exercised regardless of which tier the
// CPUID-selected dispatch table would pick at runtime. Each tier is compared
// against the scalar tier's own output (not against a fresh stdlib computation),
// since summation trees legitimately differ across tiers and only their overall
// agreement with the scalar path is being checked here.

#include "libhmm/performance/transcendental_kernels.h"
#include "libhmm/math/constants.h"
#include "libhmm/platform/cpu_detection.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declarations of the per-ISA transcendental/recurrence kernels.
// Defined in src/performance/simd_double_ops_{scalar,sse2,avx2,avx512,neon}.cpp;
// the scalar tier is always compiled in, the others only when the corresponding
// LIBHMM_BUILD_*_KERNEL compile definition is set (tests/CMakeLists.txt wires
// this test target's compile definitions from LIBHMM_DISPATCH_DEFINES).
// ---------------------------------------------------------------------------
namespace libhmm::performance::detail {

double reduce_max_sum2_scalar(const double *a, const double *b, std::size_t size) noexcept;
double sum_exp_sum2_minus_max_scalar(const double *a, const double *b, std::size_t size,
                                     double maxVal) noexcept;
double reduce_max_sum3_scalar(const double *a, const double *b, const double *c,
                              std::size_t size) noexcept;
double sum_exp_sum3_minus_max_scalar(const double *a, const double *b, const double *c,
                                     std::size_t size, double maxVal) noexcept;
void accumulate_exp_sum2_bias_scalar(double *dst, const double *a, const double *b,
                                     std::size_t size, double bias) noexcept;
void log1p_inplace_scalar(double *data, std::size_t size) noexcept;
// log_batch_<tier>: forward-declared here (not above with the recurrence
// kernels) so section 9's subnormal-prescale regression (issue #85) can call
// each compiled-in tier's log_pd wrapper directly.
void log_batch_scalar(const double *in, double *out, std::size_t n) noexcept;

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
double reduce_max_sum2_sse2(const double *a, const double *b, std::size_t size) noexcept;
double sum_exp_sum2_minus_max_sse2(const double *a, const double *b, std::size_t size,
                                   double maxVal) noexcept;
double reduce_max_sum3_sse2(const double *a, const double *b, const double *c,
                            std::size_t size) noexcept;
double sum_exp_sum3_minus_max_sse2(const double *a, const double *b, const double *c,
                                   std::size_t size, double maxVal) noexcept;
void accumulate_exp_sum2_bias_sse2(double *dst, const double *a, const double *b, std::size_t size,
                                   double bias) noexcept;
void log1p_inplace_sse2(double *data, std::size_t size) noexcept;
void log_batch_sse2(const double *in, double *out, std::size_t n) noexcept;
#endif

#if defined(LIBHMM_BUILD_AVX2_KERNEL)
double reduce_max_sum2_avx2(const double *a, const double *b, std::size_t size) noexcept;
double sum_exp_sum2_minus_max_avx2(const double *a, const double *b, std::size_t size,
                                   double maxVal) noexcept;
double reduce_max_sum3_avx2(const double *a, const double *b, const double *c,
                            std::size_t size) noexcept;
double sum_exp_sum3_minus_max_avx2(const double *a, const double *b, const double *c,
                                   std::size_t size, double maxVal) noexcept;
void accumulate_exp_sum2_bias_avx2(double *dst, const double *a, const double *b, std::size_t size,
                                   double bias) noexcept;
void log1p_inplace_avx2(double *data, std::size_t size) noexcept;
void log_batch_avx2(const double *in, double *out, std::size_t n) noexcept;
#endif

#if defined(LIBHMM_BUILD_AVX512_KERNEL)
double reduce_max_sum2_avx512(const double *a, const double *b, std::size_t size) noexcept;
double sum_exp_sum2_minus_max_avx512(const double *a, const double *b, std::size_t size,
                                     double maxVal) noexcept;
double reduce_max_sum3_avx512(const double *a, const double *b, const double *c,
                              std::size_t size) noexcept;
double sum_exp_sum3_minus_max_avx512(const double *a, const double *b, const double *c,
                                     std::size_t size, double maxVal) noexcept;
void accumulate_exp_sum2_bias_avx512(double *dst, const double *a, const double *b,
                                     std::size_t size, double bias) noexcept;
void log1p_inplace_avx512(double *data, std::size_t size) noexcept;
void log_batch_avx512(const double *in, double *out, std::size_t n) noexcept;
#endif

#if defined(LIBHMM_BUILD_NEON_KERNEL)
double reduce_max_sum2_neon(const double *a, const double *b, std::size_t size) noexcept;
double sum_exp_sum2_minus_max_neon(const double *a, const double *b, std::size_t size,
                                   double maxVal) noexcept;
double reduce_max_sum3_neon(const double *a, const double *b, const double *c,
                            std::size_t size) noexcept;
double sum_exp_sum3_minus_max_neon(const double *a, const double *b, const double *c,
                                   std::size_t size, double maxVal) noexcept;
void accumulate_exp_sum2_bias_neon(double *dst, const double *a, const double *b, std::size_t size,
                                   double bias) noexcept;
void log1p_inplace_neon(double *data, std::size_t size) noexcept;
void log_batch_neon(const double *in, double *out, std::size_t n) noexcept;
#endif

} // namespace libhmm::performance::detail

namespace {

using TK = libhmm::performance::detail::TranscendentalKernels;

constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();
constexpr double REL_TOL = 1e-12;
constexpr double ABS_TOL = 1e-15;
constexpr double LOG1P_REL_TOL = 1e-10;
constexpr double LOG1P_ABS_TOL = 1e-14;

// Sizes chosen to cover: scalar-only (1), below SSE2 width (1,3), single
// SSE2 block (2), single AVX block (4), non-multiple-of-4 (7,15,31),
// exact AVX-512 block (8), exact double-block (16,32), and large (64).
const std::vector<std::size_t> TEST_SIZES = {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64};

// -------------------------------------------------------------------------
// Helper: build test input vectors
// -------------------------------------------------------------------------

// "Normal" log-probabilities in the range (-50, 0).
static std::vector<double> make_log_probs(std::size_t n, double offset = 0.0) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = -1.0 - static_cast<double>(i % 20) * 2.3 + offset;
    }
    return v;
}

// Mix of normal log-probs and LOG_ZERO sentinels (every 5th element).
static std::vector<double> make_mixed(std::size_t n, double offset = 0.0) {
    std::vector<double> v = make_log_probs(n, offset);
    for (std::size_t i = 4; i < n; i += 5) {
        v[i] = LOG_ZERO;
    }
    return v;
}

// Comparison helpers.
static void check_scalar(double got, double ref, const char *label) {
    if (std::isinf(ref) && std::isinf(got))
        return; // both -inf is fine
    const double diff = std::abs(got - ref);
    if (ref != 0.0) {
        EXPECT_LE(diff / std::abs(ref), REL_TOL)
            << label << ": relative error too large  got=" << got << " ref=" << ref;
    } else {
        EXPECT_LE(diff, ABS_TOL) << label << ": absolute error too large  got=" << got
                                 << " ref=" << ref;
    }
}

static void check_log1p_array(const std::vector<double> &got, const std::vector<double> &ref,
                              const char *label) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double diff = std::abs(got[i] - ref[i]);
        if (std::abs(ref[i]) > LOG1P_ABS_TOL) {
            EXPECT_LE(diff / std::abs(ref[i]), LOG1P_REL_TOL)
                << label << ": relative error too large at i=" << i << " got=" << got[i]
                << " ref=" << ref[i];
        } else {
            EXPECT_LE(diff, LOG1P_ABS_TOL) << label << ": absolute error too large at i=" << i
                                           << " got=" << got[i] << " ref=" << ref[i];
        }
    }
}

static void check_array(const std::vector<double> &got, const std::vector<double> &ref,
                        const char *label) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        check_scalar(got[i], ref[i], label);
    }
}

// =========================================================================
// 1. reduce_max_sum2
// =========================================================================

static double ref_reduce_max_sum2(const std::vector<double> &a, const std::vector<double> &b) {
    double m = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < a.size(); ++i) {
        double t = a[i] + b[i];
        if (t > m)
            m = t;
    }
    return m;
}

TEST(TranscendentalKernels, ReduceMaxSum2_NormalInputs) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_log_probs(n, 0.0);
        auto b = make_log_probs(n, -3.7);
        double got = TK::reduce_max_sum2(a.data(), b.data(), n);
        double ref = ref_reduce_max_sum2(a, b);
        check_scalar(got, ref, "reduce_max_sum2/normal");
    }
}

TEST(TranscendentalKernels, ReduceMaxSum2_WithLogZero) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_mixed(n, 0.0);
        auto b = make_mixed(n, -1.5);
        double got = TK::reduce_max_sum2(a.data(), b.data(), n);
        double ref = ref_reduce_max_sum2(a, b);
        // -inf + anything is -inf; max may be -inf if all are LOG_ZERO pairs.
        if (std::isinf(ref) && std::isinf(got)) {
            EXPECT_EQ(std::signbit(ref), std::signbit(got));
        } else {
            check_scalar(got, ref, "reduce_max_sum2/mixed");
        }
    }
}

// =========================================================================
// 2. sum_exp_sum2_minus_max
// =========================================================================

static double ref_sum_exp_sum2_minus_max(const std::vector<double> &a, const std::vector<double> &b,
                                         double maxVal) {
    if (!std::isfinite(maxVal))
        return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double t = a[i] + b[i];
        if (std::isfinite(t))
            s += std::exp(t - maxVal);
    }
    return s;
}

TEST(TranscendentalKernels, SumExpSum2MinusMax_NormalInputs) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_log_probs(n, 0.0);
        auto b = make_log_probs(n, -3.7);
        double maxVal = ref_reduce_max_sum2(a, b);
        double got = TK::sum_exp_sum2_minus_max(a.data(), b.data(), n, maxVal);
        double ref = ref_sum_exp_sum2_minus_max(a, b, maxVal);
        check_scalar(got, ref, "sum_exp_sum2_minus_max/normal");
    }
}

TEST(TranscendentalKernels, SumExpSum2MinusMax_WithLogZero) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_mixed(n, 0.0);
        auto b = make_mixed(n, -1.5);
        double maxVal = ref_reduce_max_sum2(a, b);
        double got = TK::sum_exp_sum2_minus_max(a.data(), b.data(), n, maxVal);
        double ref = ref_sum_exp_sum2_minus_max(a, b, maxVal);
        check_scalar(got, ref, "sum_exp_sum2_minus_max/mixed");
    }
}

TEST(TranscendentalKernels, SumExpSum2MinusMax_InfiniteMax) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_log_probs(n);
        auto b = make_log_probs(n);
        double got = TK::sum_exp_sum2_minus_max(a.data(), b.data(), n,
                                                -std::numeric_limits<double>::infinity());
        EXPECT_EQ(got, 0.0) << "should return 0 when maxVal is -inf";
    }
}

// =========================================================================
// 3. reduce_max_sum3
// =========================================================================

static double ref_reduce_max_sum3(const std::vector<double> &a, const std::vector<double> &b,
                                  const std::vector<double> &c) {
    double m = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < a.size(); ++i) {
        double t = a[i] + b[i] + c[i];
        if (t > m)
            m = t;
    }
    return m;
}

TEST(TranscendentalKernels, ReduceMaxSum3_NormalInputs) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_log_probs(n, 0.0);
        auto b = make_log_probs(n, -2.1);
        auto c = make_log_probs(n, -5.3);
        double got = TK::reduce_max_sum3(a.data(), b.data(), c.data(), n);
        double ref = ref_reduce_max_sum3(a, b, c);
        check_scalar(got, ref, "reduce_max_sum3/normal");
    }
}

TEST(TranscendentalKernels, ReduceMaxSum3_WithLogZero) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_mixed(n, 0.0);
        auto b = make_mixed(n, -2.1);
        auto c = make_mixed(n, -5.3);
        double got = TK::reduce_max_sum3(a.data(), b.data(), c.data(), n);
        double ref = ref_reduce_max_sum3(a, b, c);
        if (std::isinf(ref) && std::isinf(got)) {
            EXPECT_EQ(std::signbit(ref), std::signbit(got));
        } else {
            check_scalar(got, ref, "reduce_max_sum3/mixed");
        }
    }
}

// =========================================================================
// 4. sum_exp_sum3_minus_max
// =========================================================================

static double ref_sum_exp_sum3_minus_max(const std::vector<double> &a, const std::vector<double> &b,
                                         const std::vector<double> &c, double maxVal) {
    if (!std::isfinite(maxVal))
        return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double t = a[i] + b[i] + c[i];
        if (std::isfinite(t))
            s += std::exp(t - maxVal);
    }
    return s;
}

TEST(TranscendentalKernels, SumExpSum3MinusMax_NormalInputs) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_log_probs(n, 0.0);
        auto b = make_log_probs(n, -2.1);
        auto c = make_log_probs(n, -5.3);
        double maxVal = ref_reduce_max_sum3(a, b, c);
        double got = TK::sum_exp_sum3_minus_max(a.data(), b.data(), c.data(), n, maxVal);
        double ref = ref_sum_exp_sum3_minus_max(a, b, c, maxVal);
        check_scalar(got, ref, "sum_exp_sum3_minus_max/normal");
    }
}

TEST(TranscendentalKernels, SumExpSum3MinusMax_WithLogZero) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_mixed(n, 0.0);
        auto b = make_mixed(n, -2.1);
        auto c = make_mixed(n, -5.3);
        double maxVal = ref_reduce_max_sum3(a, b, c);
        double got = TK::sum_exp_sum3_minus_max(a.data(), b.data(), c.data(), n, maxVal);
        double ref = ref_sum_exp_sum3_minus_max(a, b, c, maxVal);
        check_scalar(got, ref, "sum_exp_sum3_minus_max/mixed");
    }
}

TEST(TranscendentalKernels, SumExpSum3MinusMax_InfiniteMax) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_log_probs(n);
        auto b = make_log_probs(n);
        auto c = make_log_probs(n);
        double got = TK::sum_exp_sum3_minus_max(a.data(), b.data(), c.data(), n,
                                                -std::numeric_limits<double>::infinity());
        EXPECT_EQ(got, 0.0) << "should return 0 when maxVal is -inf";
    }
}

// =========================================================================
// 5. accumulate_exp_sum2_bias
// =========================================================================

static void ref_accumulate_exp_sum2_bias(std::vector<double> &dst, const std::vector<double> &a,
                                         const std::vector<double> &b, double bias) {
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] += std::exp(a[i] + b[i] + bias);
    }
}

TEST(TranscendentalKernels, AccumulateExpSum2Bias_NormalInputs) {
    for (std::size_t n : TEST_SIZES) {
        auto a = make_log_probs(n, 0.0);
        auto b = make_log_probs(n, -3.7);
        const double bias = -12.5;

        std::vector<double> got_dst(n, 0.5);
        std::vector<double> ref_dst(n, 0.5);

        TK::accumulate_exp_sum2_bias(got_dst.data(), a.data(), b.data(), n, bias);
        ref_accumulate_exp_sum2_bias(ref_dst, a, b, bias);

        check_array(got_dst, ref_dst, "accumulate_exp_sum2_bias/normal");
    }
}

TEST(TranscendentalKernels, AccumulateExpSum2Bias_LogZeroInputs) {
    // LOG_ZERO inputs: exp(-inf + ...) = 0; dst[i] should be unchanged.
    for (std::size_t n : TEST_SIZES) {
        std::vector<double> a(n, LOG_ZERO);
        std::vector<double> b(n, 0.0);
        const double bias = 0.0;

        std::vector<double> got_dst(n, 1.0);
        std::vector<double> ref_dst(n, 1.0);

        TK::accumulate_exp_sum2_bias(got_dst.data(), a.data(), b.data(), n, bias);
        ref_accumulate_exp_sum2_bias(ref_dst, a, b, bias);

        check_array(got_dst, ref_dst, "accumulate_exp_sum2_bias/log_zero");
    }
}

TEST(TranscendentalKernels, AccumulateExpSum2Bias_SmallBias) {
    // Verify behaviour near the underflow threshold.
    // The SIMD kernel intentionally returns 0 for arg <= MIN_LOG_PROBABILITY
    // (branch-free mask). std::exp does not underflow to 0 until ~-708.4, so
    // inputs in the range (-708.4, -700] produce a discrepancy between raw
    // std::exp and the SIMD. The reference must apply the same underflow
    // contract as the kernel so the comparison is against the specified
    // behaviour, not against an unclamped std::exp.
    constexpr double EXP_UNDERFLOW = libhmm::constants::probability::MIN_LOG_PROBABILITY;
    for (std::size_t n : TEST_SIZES) {
        auto a = make_log_probs(n, 0.0);
        auto b = make_log_probs(n, 0.0);
        const double bias = EXP_UNDERFLOW + 5.0; // -695

        std::vector<double> got_dst(n, 0.0);
        std::vector<double> ref_dst(n, 0.0);

        TK::accumulate_exp_sum2_bias(got_dst.data(), a.data(), b.data(), n, bias);

        // Reference: zero for arg <= EXP_UNDERFLOW, std::exp otherwise.
        for (std::size_t k = 0; k < n; ++k) {
            const double arg = a[k] + b[k] + bias;
            if (arg > EXP_UNDERFLOW)
                ref_dst[k] += std::exp(arg);
        }

        check_array(got_dst, ref_dst, "accumulate_exp_sum2_bias/small_bias");
    }
}

// =========================================================================
// 6. log1p_inplace
// =========================================================================

TEST(TranscendentalKernels, Log1pInplace_NormalInputs) {
    for (std::size_t n : TEST_SIZES) {
        std::vector<double> got(n);
        for (std::size_t i = 0; i < n; ++i) {
            got[i] = 0.01 * static_cast<double>(i + 1);
        }
        std::vector<double> ref = got;
        for (double &v : ref) {
            v = std::log1p(v);
        }

        TK::log1p_inplace(std::span<double>(got.data(), got.size()));
        check_log1p_array(got, ref, "log1p_inplace/normal");
    }
}

TEST(TranscendentalKernels, Log1pInplace_SmallAndZeroInputs) {
    std::vector<double> got = {0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5};
    std::vector<double> ref = got;
    for (double &v : ref) {
        v = std::log1p(v);
    }

    TK::log1p_inplace(std::span<double>(got.data(), got.size()));
    check_log1p_array(got, ref, "log1p_inplace/small");
}

TEST(TranscendentalKernels, Log1pInplace_LargeInputs) {
    std::vector<double> got = {0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e6, 1e12};
    std::vector<double> ref = got;
    for (double &v : ref) {
        v = std::log1p(v);
    }

    TK::log1p_inplace(std::span<double>(got.data(), got.size()));
    check_log1p_array(got, ref, "log1p_inplace/large");
}

// =========================================================================
// 7. Consistency: max-reduce round-trip
//    reduce_max then sum_exp should reproduce log-sum-exp.
// =========================================================================

TEST(TranscendentalKernels, RoundTrip_LogSumExp2) {
    // For finite inputs: log(sum_exp(a+b - max)) + max == log_sum_exp(a, b).
    // Just check the intermediate values are consistent with each other.
    for (std::size_t n : TEST_SIZES) {
        if (n == 0)
            continue;
        auto a = make_log_probs(n, 0.0);
        auto b = make_log_probs(n, -2.0);

        double maxVal = TK::reduce_max_sum2(a.data(), b.data(), n);
        double scaledSum = TK::sum_exp_sum2_minus_max(a.data(), b.data(), n, maxVal);

        EXPECT_TRUE(std::isfinite(maxVal))
            << "reduce_max_sum2 should return finite max for normal inputs (n=" << n << ")";
        EXPECT_GT(scaledSum, 0.0) << "scaled sum should be positive (n=" << n << ")";

        double logSumExp = maxVal + std::log(scaledSum);
        EXPECT_TRUE(std::isfinite(logSumExp))
            << "reconstructed log-sum-exp should be finite (n=" << n << ")";
    }
}

// =========================================================================
// 8. Per-tier equivalence: each compiled-in ISA's free functions vs. scalar.
//
// Sizes are chosen to be non-multiples of any lane count (7, 13, 33) plus the
// 0 and 1 edge cases, so every tier's scalar-tail path is exercised alongside
// its vector body. Summation trees legitimately differ across tiers, so
// comparison uses a relative tolerance of ~1e-12, not bitwise equality.
// =========================================================================

namespace pd = libhmm::performance::detail;

const std::vector<std::size_t> TIER_TEST_SIZES = {0, 1, 7, 13, 33};

// Relative-tolerance comparison used throughout this section (1e-12), distinct
// from check_scalar's absolute-fallback semantics: -inf vs -inf is always fine,
// otherwise relative error against the scalar-tier value must be <= 1e-12.
static void check_tier_scalar(double got, double scalarRef, const char *label) {
    if (std::isinf(scalarRef) && std::isinf(got)) {
        EXPECT_EQ(std::signbit(scalarRef), std::signbit(got)) << label << ": sign mismatch at -inf";
        return;
    }
    const double diff = std::abs(got - scalarRef);
    if (scalarRef != 0.0) {
        EXPECT_LE(diff / std::abs(scalarRef), 1e-12)
            << label << ": relative error too large  got=" << got << " scalarRef=" << scalarRef;
    } else {
        EXPECT_LE(diff, 1e-15) << label << ": absolute error too large  got=" << got
                               << " scalarRef=" << scalarRef;
    }
}

// Runs all six kernels' tier-vs-scalar comparison for one compiled-in tier.
// Template on the six function pointers so the SSE2/AVX2/AVX512/NEON blocks
// below are each a short, uniform instantiation rather than five copies of
// this ~40-line body.
template <typename ReduceMax2Fn, typename SumExp2Fn, typename ReduceMax3Fn, typename SumExp3Fn,
          typename AccumFn, typename Log1pFn>
static void run_tier_parity(const char *tier_name, ReduceMax2Fn reduce_max_sum2_tier,
                            SumExp2Fn sum_exp_sum2_minus_max_tier,
                            ReduceMax3Fn reduce_max_sum3_tier,
                            SumExp3Fn sum_exp_sum3_minus_max_tier,
                            AccumFn accumulate_exp_sum2_bias_tier, Log1pFn log1p_inplace_tier) {
    for (std::size_t n : TIER_TEST_SIZES) {
        auto a = make_mixed(n, 0.0);
        auto b = make_mixed(n, -1.5);
        auto c = make_mixed(n, -2.7);

        // reduce_max_sum2
        {
            double ref = pd::reduce_max_sum2_scalar(a.data(), b.data(), n);
            double got = reduce_max_sum2_tier(a.data(), b.data(), n);
            check_tier_scalar(got, ref, (std::string(tier_name) + "/reduce_max_sum2").c_str());
        }
        // sum_exp_sum2_minus_max
        {
            double maxVal = pd::reduce_max_sum2_scalar(a.data(), b.data(), n);
            double ref = pd::sum_exp_sum2_minus_max_scalar(a.data(), b.data(), n, maxVal);
            double got = sum_exp_sum2_minus_max_tier(a.data(), b.data(), n, maxVal);
            check_tier_scalar(got, ref,
                              (std::string(tier_name) + "/sum_exp_sum2_minus_max").c_str());
        }
        // reduce_max_sum3
        {
            double ref = pd::reduce_max_sum3_scalar(a.data(), b.data(), c.data(), n);
            double got = reduce_max_sum3_tier(a.data(), b.data(), c.data(), n);
            check_tier_scalar(got, ref, (std::string(tier_name) + "/reduce_max_sum3").c_str());
        }
        // sum_exp_sum3_minus_max
        {
            double maxVal = pd::reduce_max_sum3_scalar(a.data(), b.data(), c.data(), n);
            double ref = pd::sum_exp_sum3_minus_max_scalar(a.data(), b.data(), c.data(), n, maxVal);
            double got = sum_exp_sum3_minus_max_tier(a.data(), b.data(), c.data(), n, maxVal);
            check_tier_scalar(got, ref,
                              (std::string(tier_name) + "/sum_exp_sum3_minus_max").c_str());
        }
        // accumulate_exp_sum2_bias
        {
            std::vector<double> dst_ref(n, 0.25);
            std::vector<double> dst_got(n, 0.25);
            pd::accumulate_exp_sum2_bias_scalar(dst_ref.data(), a.data(), b.data(), n, -3.0);
            accumulate_exp_sum2_bias_tier(dst_got.data(), a.data(), b.data(), n, -3.0);
            check_array(dst_got, dst_ref,
                        (std::string(tier_name) + "/accumulate_exp_sum2_bias").c_str());
        }
        // log1p_inplace: domain requires x > -1; production callers pass finite x >= 0.
        // Spread covers both the small-|x| polynomial path and the general path.
        {
            std::vector<double> src(n);
            for (std::size_t i = 0; i < n; ++i) {
                src[i] = (i % 7 == 0) ? 1e-10 : 0.01 * static_cast<double>(i + 1);
            }
            std::vector<double> ref = src;
            std::vector<double> got = src;
            pd::log1p_inplace_scalar(ref.data(), n);
            log1p_inplace_tier(got.data(), n);
            check_log1p_array(got, ref, (std::string(tier_name) + "/log1p_inplace").c_str());
        }
    }
}

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
TEST(TranscendentalKernelsTierParity, Sse2MatchesScalar) {
    if (!libhmm::platform::supports_sse2()) {
        GTEST_SKIP() << "SSE2 not supported on this CPU";
    }
    run_tier_parity("sse2", pd::reduce_max_sum2_sse2, pd::sum_exp_sum2_minus_max_sse2,
                    pd::reduce_max_sum3_sse2, pd::sum_exp_sum3_minus_max_sse2,
                    pd::accumulate_exp_sum2_bias_sse2, pd::log1p_inplace_sse2);
}
#endif

#if defined(LIBHMM_BUILD_AVX2_KERNEL)
TEST(TranscendentalKernelsTierParity, Avx2MatchesScalar) {
    if (!libhmm::platform::supports_avx2()) {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }
    run_tier_parity("avx2", pd::reduce_max_sum2_avx2, pd::sum_exp_sum2_minus_max_avx2,
                    pd::reduce_max_sum3_avx2, pd::sum_exp_sum3_minus_max_avx2,
                    pd::accumulate_exp_sum2_bias_avx2, pd::log1p_inplace_avx2);
}
#endif

#if defined(LIBHMM_BUILD_AVX512_KERNEL)
TEST(TranscendentalKernelsTierParity, Avx512MatchesScalar) {
    if (!libhmm::platform::supports_avx512()) {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }
    run_tier_parity("avx512", pd::reduce_max_sum2_avx512, pd::sum_exp_sum2_minus_max_avx512,
                    pd::reduce_max_sum3_avx512, pd::sum_exp_sum3_minus_max_avx512,
                    pd::accumulate_exp_sum2_bias_avx512, pd::log1p_inplace_avx512);
}
#endif

#if defined(LIBHMM_BUILD_NEON_KERNEL)
TEST(TranscendentalKernelsTierParity, NeonMatchesScalar) {
    // NEON is the mandatory AArch64 baseline ISA — always available when this
    // TU is compiled in, so no libhmm::platform::supports_neon() runtime gate
    // is needed (unlike the x86 tiers above, which are CPUID-conditional).
    run_tier_parity("neon", pd::reduce_max_sum2_neon, pd::sum_exp_sum2_minus_max_neon,
                    pd::reduce_max_sum3_neon, pd::sum_exp_sum3_minus_max_neon,
                    pd::accumulate_exp_sum2_bias_neon, pd::log1p_inplace_neon);
}
#endif

// =========================================================================
// 9. log_pd subnormal-prescale regression (issue #85).
//
// The AVX-512/AVX2/NEON log_pd all scale a subnormal input by 2^54 and
// subtract 54 from the extracted exponent before the SLEEF core runs; the
// SSE2 log_pd lacked that prescale, so its exponent-extraction path treated
// a subnormal's biased exponent (0) as though it belonged to a normal
// number, compressing the entire subnormal range [4.9e-324, 2.2e-308] to a
// single wrong result (log_pd(5e-324) landed at -709.09 instead of
// -744.44). Runs the SAME input vector through EVERY compiled-in tier's
// log_batch_<tier> directly (bypassing DoubleVecOps) so a future per-tier
// asymmetry of this shape cannot recur silently. Pre-fix this failed on the
// SSE2 tier only.
// =========================================================================

double logUlpDistance(double got, double ref) {
    if (std::isnan(ref))
        return std::isnan(got) ? 0.0 : 1e18;
    if (!std::isfinite(ref))
        return (got == ref) ? 0.0 : 1e18;
    if (!std::isfinite(got))
        return 1e18;
    // Sign-magnitude -> monotonic-order int64 (same trick as
    // test_trig_ulp_gates.cpp's trigUlpError): plain integer subtraction is
    // then the ULP distance, including across the sign of the result.
    const auto ordered = [](double v) -> std::int64_t {
        std::int64_t i;
        std::memcpy(&i, &v, sizeof i);
        return i < 0 ? static_cast<std::int64_t>(0x8000000000000000ULL) - i : i;
    };
    const std::int64_t g = ordered(got);
    const std::int64_t r = ordered(ref);
    return static_cast<double>(g > r ? g - r : r - g);
}

using LogBatchFn = void (*)(const double *, double *, std::size_t) noexcept;

// {5e-324, 1e-310} are subnormal; 2.2250738585072014e-308 is DBL_MIN (the
// smallest normal, i.e. the boundary the prescale must switch off at); 1.0
// is an ordinary normal input included so the fix is checked not to disturb
// the already-correct normal path.
void run_log_subnormal_gate(const char *tier, LogBatchFn log_fn) {
    static constexpr double kInputs[] = {5e-324, 1e-310, 2.2250738585072014e-308, 1.0};
    constexpr std::size_t n = sizeof(kInputs) / sizeof(kInputs[0]);
    double out[n];
    log_fn(kInputs, out, n);
    for (std::size_t i = 0; i < n; ++i) {
        const double ref = std::log(kInputs[i]);
        const double ulp = logUlpDistance(out[i], ref);
        EXPECT_LE(ulp, 1.0) << tier << " log_batch(" << kInputs[i] << ") = " << out[i]
                            << ", expected " << ref << " (" << ulp << " ULP)";
    }
}

TEST(LogSubnormalPrescale, Scalar) {
    run_log_subnormal_gate("scalar", pd::log_batch_scalar);
}

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
TEST(LogSubnormalPrescale, Sse2) {
    if (!libhmm::platform::supports_sse2()) {
        GTEST_SKIP() << "SSE2 not supported on this CPU";
    }
    run_log_subnormal_gate("sse2", pd::log_batch_sse2);
}
#endif

#if defined(LIBHMM_BUILD_AVX2_KERNEL)
TEST(LogSubnormalPrescale, Avx2) {
    if (!libhmm::platform::supports_avx2()) {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }
    run_log_subnormal_gate("avx2", pd::log_batch_avx2);
}
#endif

#if defined(LIBHMM_BUILD_AVX512_KERNEL)
TEST(LogSubnormalPrescale, Avx512) {
    if (!libhmm::platform::supports_avx512()) {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }
    run_log_subnormal_gate("avx512", pd::log_batch_avx512);
}
#endif

#if defined(LIBHMM_BUILD_NEON_KERNEL)
TEST(LogSubnormalPrescale, Neon) {
    // NEON is the mandatory AArch64 baseline ISA -- always available when
    // this TU is compiled in.
    run_log_subnormal_gate("neon", pd::log_batch_neon);
}
#endif

} // anonymous namespace
