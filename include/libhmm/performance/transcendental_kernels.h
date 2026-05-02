#pragma once

#include <cstddef>

/**
 * @file transcendental_kernels.h
 * @brief SIMD-accelerated inner-loop kernels for FB max-reduce and BW xi accumulation.
 *
 * Declares five static methods on TranscendentalKernels. Implementations live in
 * src/performance/transcendental_kernels.cpp and are compiled with
 * LIBHMM_BEST_SIMD_FLAGS, activating the appropriate #if LIBHMM_HAS_* cascade:
 *   AVX-512  8-wide __m512d
 *   AVX/AVX2 4-wide __m256d  (AVX-1 compatible; AVX2 compiler fuses FMA)
 *   SSE2     2-wide __m128d
 *   NEON     2-wide float64x2_t
 *   scalar   tail / fallback
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
                                                       std::size_t size,
                                                       double maxVal) noexcept;

    /// Element-wise max of (a[i]+b[i]+c[i]) over [0, size).  No exp calls.
    [[nodiscard]] static double reduce_max_sum3(const double *a, const double *b,
                                                const double *c,
                                                std::size_t size) noexcept;

    /// Sum of exp(a[i]+b[i]+c[i] - maxVal) for finite terms, over [0, size).
    /// Returns 0 when maxVal is not finite.
    [[nodiscard]] static double sum_exp_sum3_minus_max(const double *a, const double *b,
                                                       const double *c,
                                                       std::size_t size,
                                                       double maxVal) noexcept;

    /// dst[i] += exp(a[i] + b[i] + bias) for i in [0, size).
    static void accumulate_exp_sum2_bias(double *dst, const double *a, const double *b,
                                         std::size_t size, double bias) noexcept;
};

} // namespace detail
} // namespace performance
} // namespace libhmm
