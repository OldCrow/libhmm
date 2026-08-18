// Explicit instantiation of BasicForwardBackwardCalculator<ObservationVectorView> (MV path).
// The shared transition recurrence it calls (TranscendentalKernels) routes through
// the runtime-dispatched DoubleVecOps table (issue #58); this TU itself compiles
// at the platform baseline ISA.

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicForwardBackwardCalculator<ObservationVectorView>;
} // namespace libhmm
