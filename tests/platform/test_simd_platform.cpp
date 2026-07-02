// tests/platform/test_simd_platform.cpp
//
// Consistency checks for libhmm/platform/simd_platform.h.
//
// Two layers of verification:
//
//  1. Compile-time (#error) — ISA hierarchy invariants that can only fail if
//     simd_platform.h emits a broken macro combination.  A violation here is
//     a build error, not a test failure.
//
//  2. Runtime (GTest) — contracts on the utility functions:
//       feature_string()        non-null, non-empty, agrees with active macros
//       double_vector_width()   power-of-two >= 1
//       float_vector_width()    == 2 * double_vector_width()
//       optimal_alignment()     power-of-two >= 8, covers one SIMD register
//       has_simd_support()      consistent with double_vector_width()
//       supports_vectorization()consistent with has_simd_support()
//       compile-time constants  DOUBLE_SIMD_WIDTH / FLOAT_SIMD_WIDTH /
//                               SIMD_ALIGNMENT each agree with their function
//
// Not compiled with LIBHMM_BEST_SIMD_FLAGS — tests the detection
// infrastructure, not the intrinsics.

#include <gtest/gtest.h>
#include "libhmm/platform/simd_platform.h"
#include "libhmm/platform/cpu_detection.h"

#include <cstring>

// ============================================================================
// Compile-time ISA hierarchy invariants
// A #error here means simd_platform.h has emitted a broken macro combination.
// ============================================================================

#if defined(LIBHMM_HAS_AVX512) && !defined(LIBHMM_HAS_AVX)
#error "LIBHMM_HAS_AVX512 requires LIBHMM_HAS_AVX"
#endif
#if defined(LIBHMM_HAS_AVX512) && !defined(LIBHMM_HAS_SSE2)
#error "LIBHMM_HAS_AVX512 requires LIBHMM_HAS_SSE2"
#endif
#if defined(LIBHMM_HAS_AVX2) && !defined(LIBHMM_HAS_AVX)
#error "LIBHMM_HAS_AVX2 requires LIBHMM_HAS_AVX"
#endif
#if defined(LIBHMM_HAS_AVX2) && !defined(LIBHMM_HAS_SSE2)
#error "LIBHMM_HAS_AVX2 requires LIBHMM_HAS_SSE2"
#endif
#if defined(LIBHMM_HAS_AVX) && !defined(LIBHMM_HAS_SSE2)
#error "LIBHMM_HAS_AVX requires LIBHMM_HAS_SSE2"
#endif
#if defined(LIBHMM_HAS_SSE4_1) && !defined(LIBHMM_HAS_SSE2)
#error "LIBHMM_HAS_SSE4_1 requires LIBHMM_HAS_SSE2"
#endif
#if defined(LIBHMM_HAS_NEON) && defined(LIBHMM_HAS_SSE2)
#error "LIBHMM_HAS_NEON and x86 SIMD macros are mutually exclusive"
#endif

// ============================================================================
// Helpers
// ============================================================================

using namespace libhmm::performance::simd;

namespace {

constexpr bool is_power_of_two(std::size_t n) noexcept {
    return n >= 1 && (n & (n - 1)) == 0;
}

} // namespace

// ============================================================================
// feature_string
// ============================================================================

TEST(SimdPlatformFeatureString, NonNull) {
    EXPECT_NE(feature_string(), nullptr);
}

TEST(SimdPlatformFeatureString, NonEmpty) {
    EXPECT_GT(std::strlen(feature_string()), 0u);
}

// The reported string must match the highest active ISA macro.
TEST(SimdPlatformFeatureString, ConsistentWithMacros) {
#if defined(LIBHMM_HAS_AVX512)
    EXPECT_STREQ(feature_string(), "AVX-512");
#elif defined(LIBHMM_HAS_AVX2)
    EXPECT_STREQ(feature_string(), "AVX2");
#elif defined(LIBHMM_HAS_AVX)
    EXPECT_STREQ(feature_string(), "AVX");
#elif defined(LIBHMM_HAS_SSE4_1)
    EXPECT_STREQ(feature_string(), "SSE4.1");
#elif defined(LIBHMM_HAS_SSE2)
    EXPECT_STREQ(feature_string(), "SSE2");
#elif defined(LIBHMM_HAS_NEON)
    // Accepts both "ARM NEON" and "ARM NEON (Apple Silicon)".
    EXPECT_EQ(std::strncmp(feature_string(), "ARM NEON", 8), 0);
#else
    EXPECT_STREQ(feature_string(), "Scalar (No SIMD)");
#endif
}

// ============================================================================
// double_vector_width / float_vector_width
// ============================================================================

TEST(SimdPlatformVectorWidth, DoubleWidthAtLeastOne) {
    EXPECT_GE(double_vector_width(), 1u);
}

TEST(SimdPlatformVectorWidth, DoubleWidthIsPowerOfTwo) {
    EXPECT_TRUE(is_power_of_two(double_vector_width()));
}

// float is 32-bit, double is 64-bit: a register holds twice as many floats.
TEST(SimdPlatformVectorWidth, FloatWidthIsTwiceDoubleWidth) {
    EXPECT_EQ(float_vector_width(), 2u * double_vector_width());
}

// ============================================================================
// optimal_alignment
// ============================================================================

TEST(SimdPlatformAlignment, AtLeastEightBytes) {
    EXPECT_GE(optimal_alignment(), 8u);
}

TEST(SimdPlatformAlignment, IsPowerOfTwo) {
    EXPECT_TRUE(is_power_of_two(optimal_alignment()));
}

// Alignment must be at least enough to hold one full SIMD register of doubles.
TEST(SimdPlatformAlignment, CoversOneSimdRegister) {
    EXPECT_GE(optimal_alignment(), double_vector_width() * sizeof(double));
}

// ============================================================================
// has_simd_support / supports_vectorization
// ============================================================================

TEST(SimdPlatformSupport, HasSimdConsistentWithWidth) {
    if (has_simd_support()) {
        EXPECT_GE(double_vector_width(), 2u);
    } else {
        EXPECT_EQ(double_vector_width(), 1u);
    }
}

TEST(SimdPlatformSupport, SupportsVectorizationRequiresHasSimd) {
    if (supports_vectorization()) {
        EXPECT_TRUE(has_simd_support());
        EXPECT_GE(double_vector_width(), 2u);
    }
}

// ============================================================================
// Compile-time constants agree with their corresponding functions
// ============================================================================

TEST(SimdPlatformConstants, DoubleSimdWidthMatchesFunction) {
    EXPECT_EQ(DOUBLE_SIMD_WIDTH, double_vector_width());
}

TEST(SimdPlatformConstants, FloatSimdWidthMatchesFunction) {
    EXPECT_EQ(FLOAT_SIMD_WIDTH, float_vector_width());
}

TEST(SimdPlatformConstants, SimdAlignmentMatchesFunction) {
    EXPECT_EQ(SIMD_ALIGNMENT, optimal_alignment());
}

// ============================================================================
// Runtime CPUID detection logical invariants (H-3)
//
// Tests the logical consistency of the detection API itself: ISA tier
// hierarchy, mutual exclusivity of x86 and ARM, and that AVX implies SSE2.
// Intentionally independent of compile-time LIBHMM_HAS_* macros — those are
// only defined in TUs compiled with SIMD flags, not in this test binary.
// ============================================================================

TEST(CpuDetectionConsistency, X86AndNEONAreMutuallyExclusive) {
    // SSE2 (x86) and NEON (AArch64) cannot both be present on the same CPU.
    EXPECT_FALSE(libhmm::platform::supports_sse2() && libhmm::platform::supports_neon());
}

TEST(CpuDetectionConsistency, AVX2ImpliesSSE2) {
    // AVX2 is a superset of SSE2 — if AVX2 is available, SSE2 must be too.
    if (libhmm::platform::supports_avx2()) {
        EXPECT_TRUE(libhmm::platform::supports_sse2());
    }
}

TEST(CpuDetectionConsistency, AVX512ImpliesAVX2) {
    // AVX-512F is a superset of AVX2.
    if (libhmm::platform::supports_avx512()) {
        EXPECT_TRUE(libhmm::platform::supports_avx2());
    }
}

TEST(CpuDetectionConsistency, AArch64AlwaysReportsNEON) {
#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_TRUE(libhmm::platform::supports_neon());
    EXPECT_FALSE(libhmm::platform::supports_sse2());
    EXPECT_FALSE(libhmm::platform::supports_avx2());
    EXPECT_FALSE(libhmm::platform::supports_avx512());
#endif
}

TEST(CpuDetectionConsistency, X86_64AlwaysReportsSSE2) {
#if defined(__x86_64__) || defined(_M_X64)
    // SSE2 is mandated by the x86-64 ABI — all x86-64 CPUs have it.
    EXPECT_TRUE(libhmm::platform::supports_sse2());
    EXPECT_FALSE(libhmm::platform::supports_neon());
#endif
}
