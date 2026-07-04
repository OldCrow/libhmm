// Explicit instantiation of BasicSegmentalKMeansTrainer<double> (scalar path).
//
// All method definitions are in basic_segmental_kmeans_trainer.h.
// Compiled without LIBHMM_BEST_SIMD_FLAGS — the trainer body contains no
// vectorisable inner loops, and SIMD acceleration is provided by
// BasicViterbiCalculator (included transitively).

#include "libhmm/training/basic_segmental_kmeans_trainer.h"

namespace libhmm {

template class BasicSegmentalKMeansTrainer<double>;

} // namespace libhmm
