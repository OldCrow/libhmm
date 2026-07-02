// simd_dispatch.cpp — runtime ISA selection and dispatch table construction.
//
// Compiled WITHOUT any SIMD flags. Receives COMPILE_DEFINITIONS from CMakeLists
// (LIBHMM_BUILD_*_KERNEL) indicating which per-ISA TUs were actually compiled in.
// Only those tiers are forward-declared and referenced here; taking the address
// of a symbol that was not linked would cause a linker error.
//
// build_table() runs exactly once (on the first call to get_double_vec_ops()),
// stored in a function-local static — thread-safe per C++11.

#include "libhmm/performance/simd_double_ops.h"
#include "libhmm/platform/cpu_detection.h"

namespace libhmm::performance {

// ============================================================================
// Forward declarations of per-ISA kernel functions.
// Each block is compiled only when the corresponding TU is compiled in.
// ============================================================================

namespace detail {

// Scalar — always present
void gaussian_batch_scalar(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_scalar(const double *, double *, std::size_t, double, double) noexcept;

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
void gaussian_batch_sse2(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_sse2(const double *, double *, std::size_t, double, double) noexcept;
#endif

#if defined(LIBHMM_BUILD_AVX2_KERNEL)
void gaussian_batch_avx2(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_avx2(const double *, double *, std::size_t, double, double) noexcept;
#endif

#if defined(LIBHMM_BUILD_AVX512_KERNEL)
void gaussian_batch_avx512(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_avx512(const double *, double *, std::size_t, double, double) noexcept;
#endif

#if defined(LIBHMM_BUILD_NEON_KERNEL)
void gaussian_batch_neon(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_neon(const double *, double *, std::size_t, double, double) noexcept;
#endif

} // namespace detail

// ============================================================================
// Table builder
// ============================================================================

static DoubleVecOps build_table() noexcept {
    DoubleVecOps t{};

#if defined(LIBHMM_BUILD_NEON_KERNEL)
    // AArch64: NEON is mandatory and the only SIMD tier compiled in.
    // Assign directly — no scalar overwrite needed.
    t.gaussian_batch = &detail::gaussian_batch_neon;
    t.exponential_batch = &detail::exponential_batch_neon;
#else
    // x86: start at scalar, overwrite up to the highest CPUID-detected tier.
    t.gaussian_batch = &detail::gaussian_batch_scalar;
    t.exponential_batch = &detail::exponential_batch_scalar;

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
    if (libhmm::platform::supports_sse2()) {
        t.gaussian_batch = &detail::gaussian_batch_sse2;
        t.exponential_batch = &detail::exponential_batch_sse2;
    }
#endif
#if defined(LIBHMM_BUILD_AVX2_KERNEL)
    if (libhmm::platform::supports_avx2()) {
        t.gaussian_batch = &detail::gaussian_batch_avx2;
        t.exponential_batch = &detail::exponential_batch_avx2;
    }
#endif
#if defined(LIBHMM_BUILD_AVX512_KERNEL)
    if (libhmm::platform::supports_avx512()) {
        t.gaussian_batch = &detail::gaussian_batch_avx512;
        t.exponential_batch = &detail::exponential_batch_avx512;
    }
#endif
#endif // LIBHMM_BUILD_NEON_KERNEL

    return t;
}

// ============================================================================
// Public accessor — function-local static, built once, thread-safe per C++11.
// ============================================================================

const DoubleVecOps &get_double_vec_ops() noexcept {
    static const DoubleVecOps ops = build_table();
    return ops;
}

} // namespace libhmm::performance
