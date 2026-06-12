// Explicit instantiation of BasicBaumWelchTrainer<ObservationVectorView> (MV path).
// Compiled with LIBHMM_BEST_SIMD_FLAGS so the shared xi accumulation kernel
// benefits from SIMD alongside the scalar specialisation.

#include "libhmm/training/basic_baum_welch_trainer.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicBaumWelchTrainer<ObservationVectorView>;
} // namespace libhmm
