// Explicit instantiation of BasicMapBaumWelchTrainer<double> (scalar path).
// Compiled with LIBHMM_BEST_SIMD_FLAGS (calls accumulate_exp_sum2_bias).

#include "libhmm/training/basic_map_baum_welch_trainer.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {
template class BasicMapBaumWelchTrainer<double>;
} // namespace libhmm
