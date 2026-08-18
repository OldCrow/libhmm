// Explicit instantiation of BasicBaumWelchTrainer<ObservationVectorView> (MV path).
// The shared xi accumulation kernel it calls (TranscendentalKernels) routes
// through the runtime-dispatched DoubleVecOps table (issue #58); this TU itself
// compiles at the platform baseline ISA.

#include "libhmm/training/basic_baum_welch_trainer.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicBaumWelchTrainer<ObservationVectorView>;
} // namespace libhmm
