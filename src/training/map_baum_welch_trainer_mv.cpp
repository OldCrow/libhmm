// Explicit instantiation of BasicMapBaumWelchTrainer<ObservationVectorView> (MV path).
// Calls accumulate_exp_sum2_bias, which routes through the runtime-dispatched
// DoubleVecOps table (issue #58); this TU itself compiles at the platform
// baseline ISA.

#include "libhmm/training/basic_map_baum_welch_trainer.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicMapBaumWelchTrainer<ObservationVectorView>;
} // namespace libhmm
