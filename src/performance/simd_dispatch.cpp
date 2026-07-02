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

// ---- Tier-2 kernels (Gaussian, Exponential) ----
void gaussian_batch_scalar(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_scalar(const double *, double *, std::size_t, double, double) noexcept;

// ---- Generic math primitives ----
void log_batch_scalar(const double *, double *, std::size_t) noexcept;
void exp_batch_scalar(const double *, double *, std::size_t) noexcept;
void cos_batch_scalar(const double *, double *, std::size_t) noexcept;
void log1p_batch_scalar(const double *, double *, std::size_t) noexcept;

// ---- Tier-1 distribution kernels ----
void lognormal_batch_scalar(const double *, double *, std::size_t, double, double, double) noexcept;
void gamma_batch_scalar(const double *, double *, std::size_t, double, double, double) noexcept;
void chisq_batch_scalar(const double *, double *, std::size_t, double, double) noexcept;
void rayleigh_batch_scalar(const double *, double *, std::size_t, double, double) noexcept;
void pareto_batch_scalar(const double *, double *, std::size_t, double, double, double) noexcept;
void weibull_batch_scalar(const double *, double *, std::size_t, double, double, double,
                          double) noexcept;
void beta_batch_scalar(const double *, double *, std::size_t, double, double, double) noexcept;
void student_t_batch_scalar(const double *, double *, std::size_t, double, double, double, double,
                            double) noexcept;
void vonmises_batch_scalar(const double *, double *, std::size_t, double, double, double) noexcept;

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
void gaussian_batch_sse2(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_sse2(const double *, double *, std::size_t, double, double) noexcept;
void log_batch_sse2(const double *, double *, std::size_t) noexcept;
void exp_batch_sse2(const double *, double *, std::size_t) noexcept;
void cos_batch_sse2(const double *, double *, std::size_t) noexcept;
void log1p_batch_sse2(const double *, double *, std::size_t) noexcept;
void lognormal_batch_sse2(const double *, double *, std::size_t, double, double, double) noexcept;
void gamma_batch_sse2(const double *, double *, std::size_t, double, double, double) noexcept;
void chisq_batch_sse2(const double *, double *, std::size_t, double, double) noexcept;
void rayleigh_batch_sse2(const double *, double *, std::size_t, double, double) noexcept;
void pareto_batch_sse2(const double *, double *, std::size_t, double, double, double) noexcept;
void weibull_batch_sse2(const double *, double *, std::size_t, double, double, double,
                        double) noexcept;
void beta_batch_sse2(const double *, double *, std::size_t, double, double, double) noexcept;
void student_t_batch_sse2(const double *, double *, std::size_t, double, double, double, double,
                          double) noexcept;
void vonmises_batch_sse2(const double *, double *, std::size_t, double, double, double) noexcept;
#endif

#if defined(LIBHMM_BUILD_AVX2_KERNEL)
void gaussian_batch_avx2(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_avx2(const double *, double *, std::size_t, double, double) noexcept;
void log_batch_avx2(const double *, double *, std::size_t) noexcept;
void exp_batch_avx2(const double *, double *, std::size_t) noexcept;
void cos_batch_avx2(const double *, double *, std::size_t) noexcept;
void log1p_batch_avx2(const double *, double *, std::size_t) noexcept;
void lognormal_batch_avx2(const double *, double *, std::size_t, double, double, double) noexcept;
void gamma_batch_avx2(const double *, double *, std::size_t, double, double, double) noexcept;
void chisq_batch_avx2(const double *, double *, std::size_t, double, double) noexcept;
void rayleigh_batch_avx2(const double *, double *, std::size_t, double, double) noexcept;
void pareto_batch_avx2(const double *, double *, std::size_t, double, double, double) noexcept;
void weibull_batch_avx2(const double *, double *, std::size_t, double, double, double,
                        double) noexcept;
void beta_batch_avx2(const double *, double *, std::size_t, double, double, double) noexcept;
void student_t_batch_avx2(const double *, double *, std::size_t, double, double, double, double,
                          double) noexcept;
void vonmises_batch_avx2(const double *, double *, std::size_t, double, double, double) noexcept;
#endif

#if defined(LIBHMM_BUILD_AVX512_KERNEL)
void gaussian_batch_avx512(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_avx512(const double *, double *, std::size_t, double, double) noexcept;
void log_batch_avx512(const double *, double *, std::size_t) noexcept;
void exp_batch_avx512(const double *, double *, std::size_t) noexcept;
void cos_batch_avx512(const double *, double *, std::size_t) noexcept;
void log1p_batch_avx512(const double *, double *, std::size_t) noexcept;
void lognormal_batch_avx512(const double *, double *, std::size_t, double, double, double) noexcept;
void gamma_batch_avx512(const double *, double *, std::size_t, double, double, double) noexcept;
void chisq_batch_avx512(const double *, double *, std::size_t, double, double) noexcept;
void rayleigh_batch_avx512(const double *, double *, std::size_t, double, double) noexcept;
void pareto_batch_avx512(const double *, double *, std::size_t, double, double, double) noexcept;
void weibull_batch_avx512(const double *, double *, std::size_t, double, double, double,
                          double) noexcept;
void beta_batch_avx512(const double *, double *, std::size_t, double, double, double) noexcept;
void student_t_batch_avx512(const double *, double *, std::size_t, double, double, double, double,
                            double) noexcept;
void vonmises_batch_avx512(const double *, double *, std::size_t, double, double, double) noexcept;
#endif

#if defined(LIBHMM_BUILD_NEON_KERNEL)
void gaussian_batch_neon(const double *, double *, std::size_t, double, double, double) noexcept;
void exponential_batch_neon(const double *, double *, std::size_t, double, double) noexcept;
void log_batch_neon(const double *, double *, std::size_t) noexcept;
void exp_batch_neon(const double *, double *, std::size_t) noexcept;
void cos_batch_neon(const double *, double *, std::size_t) noexcept;
void log1p_batch_neon(const double *, double *, std::size_t) noexcept;
void lognormal_batch_neon(const double *, double *, std::size_t, double, double, double) noexcept;
void gamma_batch_neon(const double *, double *, std::size_t, double, double, double) noexcept;
void chisq_batch_neon(const double *, double *, std::size_t, double, double) noexcept;
void rayleigh_batch_neon(const double *, double *, std::size_t, double, double) noexcept;
void pareto_batch_neon(const double *, double *, std::size_t, double, double, double) noexcept;
void weibull_batch_neon(const double *, double *, std::size_t, double, double, double,
                        double) noexcept;
void beta_batch_neon(const double *, double *, std::size_t, double, double, double) noexcept;
void student_t_batch_neon(const double *, double *, std::size_t, double, double, double, double,
                          double) noexcept;
void vonmises_batch_neon(const double *, double *, std::size_t, double, double, double) noexcept;
#endif

} // namespace detail
// Table builder
// ============================================================================

static DoubleVecOps build_table() noexcept {
    DoubleVecOps t{};

// Helper macro: assign all 13 new function pointers for a given ISA suffix.
// Keeps build_table() readable without 156 lines of explicit assignments.
#define LIBHMM_ASSIGN_TIER(SUFFIX)                                                                 \
    t.gaussian_batch = &detail::gaussian_batch_##SUFFIX;                                           \
    t.exponential_batch = &detail::exponential_batch_##SUFFIX;                                     \
    t.log_batch = &detail::log_batch_##SUFFIX;                                                     \
    t.exp_batch = &detail::exp_batch_##SUFFIX;                                                     \
    t.cos_batch = &detail::cos_batch_##SUFFIX;                                                     \
    t.log1p_batch = &detail::log1p_batch_##SUFFIX;                                                 \
    t.lognormal_batch = &detail::lognormal_batch_##SUFFIX;                                         \
    t.gamma_batch = &detail::gamma_batch_##SUFFIX;                                                 \
    t.chisq_batch = &detail::chisq_batch_##SUFFIX;                                                 \
    t.rayleigh_batch = &detail::rayleigh_batch_##SUFFIX;                                           \
    t.pareto_batch = &detail::pareto_batch_##SUFFIX;                                               \
    t.weibull_batch = &detail::weibull_batch_##SUFFIX;                                             \
    t.beta_batch = &detail::beta_batch_##SUFFIX;                                                   \
    t.student_t_batch = &detail::student_t_batch_##SUFFIX;                                         \
    t.vonmises_batch = &detail::vonmises_batch_##SUFFIX

#if defined(LIBHMM_BUILD_NEON_KERNEL)
    // AArch64: NEON is mandatory and the only SIMD tier compiled in.
    LIBHMM_ASSIGN_TIER(neon);
#else
    // x86: start at scalar, overwrite up to the highest CPUID-detected tier.
    LIBHMM_ASSIGN_TIER(scalar);

#if defined(LIBHMM_BUILD_SSE2_KERNEL)
    if (libhmm::platform::supports_sse2()) {
        LIBHMM_ASSIGN_TIER(sse2);
    }
#endif
#if defined(LIBHMM_BUILD_AVX2_KERNEL)
    if (libhmm::platform::supports_avx2()) {
        LIBHMM_ASSIGN_TIER(avx2);
    }
#endif
#if defined(LIBHMM_BUILD_AVX512_KERNEL)
    if (libhmm::platform::supports_avx512()) {
        LIBHMM_ASSIGN_TIER(avx512);
    }
#endif
#endif // LIBHMM_BUILD_NEON_KERNEL

#undef LIBHMM_ASSIGN_TIER

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
