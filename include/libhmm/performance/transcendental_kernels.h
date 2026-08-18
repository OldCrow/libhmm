#pragma once

#include <cstddef>
#include <span>

/**
 * @file transcendental_kernels.h
 * @brief SIMD-accelerated inner-loop kernels for FB max-reduce and BW xi accumulation.
 *
 * Declares six static methods on TranscendentalKernels. Implementations live in
 * src/performance/transcendental_kernels.cpp, which is now a thin wrapper: each
 * method forwards through the runtime-dispatched DoubleVecOps table
 * (libhmm::performance::get_double_vec_ops(), see performance/simd_double_ops.h)
 * selected once at startup via CPUID (issue #58). The actual per-ISA kernel
 * bodies live in the five simd_double_ops_{scalar,sse2,avx2,avx512,neon}.cpp
 * TUs, each compiled with a targeted per-file flag:
 *   AVX-512  8-wide __m512d
 *   AVX/AVX2 4-wide __m256d  (AVX-1 compatible; AVX2 compiler fuses FMA)
 *   SSE2     2-wide __m128d
 *   NEON     2-wide float64x2_t
 *   scalar   tail / fallback
 * This header itself declares no SIMD types and stays std-only; TranscendentalKernels'
 * TU compiles at the platform baseline ISA rather than under LIBHMM_BEST_SIMD_FLAGS.
 *
 * Active ISA diagnostics use libhmm::performance::simd::feature_string() and
 * double_vector_width() from simd_platform.h — consistent with the rest of the library.
 */

namespace libhmm {
namespace performance {
namespace detail {

/**
 * @brief Vectorised inner-loop kernels shared by ForwardBackwardCalculator (max-reduce
 *        recurrence) and BaumWelchTrainer (dense-xi accumulation).
 *
 * All methods are noexcept and operate on raw double pointers.  Inputs are
 * expected to be either finite log-probabilities or LOG_ZERO (-inf); +inf and
 * NaN are not produced by any production caller and are not guarded.
 */
class TranscendentalKernels {
public:
    /// Element-wise max of (a[i]+b[i]) over [0, size).  No exp calls.
    [[nodiscard]] static double reduce_max_sum2(const double *a, const double *b,
                                                std::size_t size) noexcept;

    /// Sum of exp(a[i]+b[i] - maxVal) for finite terms, over [0, size).
    /// Returns 0 when maxVal is not finite.
    [[nodiscard]] static double sum_exp_sum2_minus_max(const double *a, const double *b,
                                                       std::size_t size, double maxVal) noexcept;

    /// Element-wise max of (a[i]+b[i]+c[i]) over [0, size).  No exp calls.
    [[nodiscard]] static double reduce_max_sum3(const double *a, const double *b, const double *c,
                                                std::size_t size) noexcept;

    /// Sum of exp(a[i]+b[i]+c[i] - maxVal) for finite terms, over [0, size).
    /// Returns 0 when maxVal is not finite.
    [[nodiscard]] static double sum_exp_sum3_minus_max(const double *a, const double *b,
                                                       const double *c, std::size_t size,
                                                       double maxVal) noexcept;

    /// dst[i] += exp(a[i] + b[i] + bias) for i in [0, size).
    static void accumulate_exp_sum2_bias(double *dst, const double *a, const double *b,
                                         std::size_t size, double bias) noexcept;

    /// In-place log1p(v[i]) for i in [0, size).
    /// Production callers pass finite values >= 0.0.
    static void log1p_inplace(std::span<double> values) noexcept;
};

} // namespace detail
} // namespace performance
} // namespace libhmm
