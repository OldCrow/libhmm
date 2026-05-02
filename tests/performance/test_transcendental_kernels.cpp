// tests/performance/test_transcendental_kernels.cpp
//
// Parity tests for TranscendentalKernels: verify that each of the five
// kernel methods agrees with a std::exp-based scalar reference to within
// 1e-12 relative / 1e-15 absolute tolerance.
//
// Ground truth is always computed inline here using std::exp directly — NOT
// by calling the kernel's internal scalar variant — so the test is
// independent of any internal refactor.
//
// The test binary is compiled with LIBHMM_BEST_SIMD_FLAGS (see CMakeLists.txt
// Level 8 section), so the active SIMD path matches the production library.

#include "libhmm/performance/transcendental_kernels.h"
#include "libhmm/math/constants.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace {

using TK = libhmm::performance::detail::TranscendentalKernels;

constexpr double LOG_ZERO  = -std::numeric_limits<double>::infinity();
constexpr double REL_TOL   = 1e-12;
constexpr double ABS_TOL   = 1e-15;

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
    if (std::isinf(ref) && std::isinf(got)) return; // both -inf is fine
    const double diff = std::abs(got - ref);
    if (ref != 0.0) {
        EXPECT_LE(diff / std::abs(ref), REL_TOL)
            << label << ": relative error too large  got=" << got << " ref=" << ref;
    } else {
        EXPECT_LE(diff, ABS_TOL)
            << label << ": absolute error too large  got=" << got << " ref=" << ref;
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

static double ref_reduce_max_sum2(const std::vector<double> &a,
                                  const std::vector<double> &b) {
    double m = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < a.size(); ++i) {
        double t = a[i] + b[i];
        if (t > m) m = t;
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

static double ref_sum_exp_sum2_minus_max(const std::vector<double> &a,
                                         const std::vector<double> &b,
                                         double maxVal) {
    if (!std::isfinite(maxVal)) return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double t = a[i] + b[i];
        if (std::isfinite(t)) s += std::exp(t - maxVal);
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

static double ref_reduce_max_sum3(const std::vector<double> &a,
                                  const std::vector<double> &b,
                                  const std::vector<double> &c) {
    double m = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < a.size(); ++i) {
        double t = a[i] + b[i] + c[i];
        if (t > m) m = t;
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

static double ref_sum_exp_sum3_minus_max(const std::vector<double> &a,
                                         const std::vector<double> &b,
                                         const std::vector<double> &c,
                                         double maxVal) {
    if (!std::isfinite(maxVal)) return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double t = a[i] + b[i] + c[i];
        if (std::isfinite(t)) s += std::exp(t - maxVal);
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

static void ref_accumulate_exp_sum2_bias(std::vector<double> &dst,
                                         const std::vector<double> &a,
                                         const std::vector<double> &b,
                                         double bias) {
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
// 6. Consistency: max-reduce round-trip
//    reduce_max then sum_exp should reproduce log-sum-exp.
// =========================================================================

TEST(TranscendentalKernels, RoundTrip_LogSumExp2) {
    // For finite inputs: log(sum_exp(a+b - max)) + max == log_sum_exp(a, b).
    // Just check the intermediate values are consistent with each other.
    for (std::size_t n : TEST_SIZES) {
        if (n == 0) continue;
        auto a = make_log_probs(n, 0.0);
        auto b = make_log_probs(n, -2.0);

        double maxVal = TK::reduce_max_sum2(a.data(), b.data(), n);
        double scaledSum = TK::sum_exp_sum2_minus_max(a.data(), b.data(), n, maxVal);

        EXPECT_TRUE(std::isfinite(maxVal))
            << "reduce_max_sum2 should return finite max for normal inputs (n=" << n << ")";
        EXPECT_GT(scaledSum, 0.0)
            << "scaled sum should be positive (n=" << n << ")";

        double logSumExp = maxVal + std::log(scaledSum);
        EXPECT_TRUE(std::isfinite(logSumExp))
            << "reconstructed log-sum-exp should be finite (n=" << n << ")";
    }
}

} // anonymous namespace
