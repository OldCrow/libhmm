// Explicit instantiation of BasicForwardBackwardCalculator<double> (scalar path).
// Compiled with LIBHMM_BEST_SIMD_FLAGS to enable SIMD in the recurrence kernels.

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicForwardBackwardCalculator<double>;
} // namespace libhmm
