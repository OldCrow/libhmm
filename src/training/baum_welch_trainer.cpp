// Explicit instantiation of BasicBaumWelchTrainer<double> (scalar path).
// Compiled with LIBHMM_BEST_SIMD_FLAGS to activate SIMD in the xi accumulation kernel.

#include "libhmm/training/basic_baum_welch_trainer.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicBaumWelchTrainer<double>;
} // namespace libhmm
