// Explicit instantiation of BasicForwardBackwardCalculator<double> (scalar path).
// The recurrence kernels it calls (TranscendentalKernels) route through the
// runtime-dispatched DoubleVecOps table (issue #58); this TU itself compiles at
// the platform baseline ISA.

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicForwardBackwardCalculator<double>;
} // namespace libhmm
