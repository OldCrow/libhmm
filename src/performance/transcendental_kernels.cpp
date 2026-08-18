// src/performance/transcendental_kernels.cpp
//
// TranscendentalKernels: thin wrappers over the runtime-dispatched DoubleVecOps
// table (libhmm::performance::get_double_vec_ops()).
//
// Issue #58: the six per-ISA cascades that used to live in this TU (each a
// compile-time #if LIBHMM_HAS_* block activated by LIBHMM_BEST_SIMD_FLAGS) were
// relocated verbatim into the five per-ISA dispatch TUs
// (src/performance/simd_double_ops_{scalar,sse2,avx2,avx512,neon}.cpp) as free
// functions <name>_<tier>, registered in simd_dispatch.cpp's DoubleVecOps table.
// This TU now contains no SIMD intrinsics and compiles at the platform baseline
// ISA — each method call is a single indirect call through the dispatch table
// built once at startup via CPUID.
//
// log1p_inplace forwards to the dedicated log1p_inplace table entry, which uses
// the log1p_pd small-|x| polynomial path — NOT log1p_batch, which is add-then-log
// with no small-x accuracy path and would silently lose precision here.

#include "libhmm/performance/transcendental_kernels.h"
#include "libhmm/performance/simd_double_ops.h"

namespace libhmm {
namespace performance {
namespace detail {

double TranscendentalKernels::reduce_max_sum2(const double *a, const double *b,
                                              std::size_t size) noexcept {
    return get_double_vec_ops().reduce_max_sum2(a, b, size);
}

double TranscendentalKernels::sum_exp_sum2_minus_max(const double *a, const double *b,
                                                     std::size_t size, double maxVal) noexcept {
    return get_double_vec_ops().sum_exp_sum2_minus_max(a, b, size, maxVal);
}

double TranscendentalKernels::reduce_max_sum3(const double *a, const double *b, const double *c,
                                              std::size_t size) noexcept {
    return get_double_vec_ops().reduce_max_sum3(a, b, c, size);
}

double TranscendentalKernels::sum_exp_sum3_minus_max(const double *a, const double *b,
                                                     const double *c, std::size_t size,
                                                     double maxVal) noexcept {
    return get_double_vec_ops().sum_exp_sum3_minus_max(a, b, c, size, maxVal);
}

void TranscendentalKernels::accumulate_exp_sum2_bias(double *dst, const double *a, const double *b,
                                                     std::size_t size, double bias) noexcept {
    get_double_vec_ops().accumulate_exp_sum2_bias(dst, a, b, size, bias);
}

void TranscendentalKernels::log1p_inplace(std::span<double> values) noexcept {
    get_double_vec_ops().log1p_inplace(values.data(), values.size());
}

} // namespace detail
} // namespace performance
} // namespace libhmm
