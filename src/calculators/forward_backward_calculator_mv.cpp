// Explicit instantiation of BasicForwardBackwardCalculator<ObservationVectorView> (MV path).
// Compiled with LIBHMM_BEST_SIMD_FLAGS so the shared transition recurrence
// benefits from SIMD acceleration alongside the scalar specialisation.

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicForwardBackwardCalculator<ObservationVectorView>;
} // namespace libhmm
