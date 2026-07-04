// Explicit instantiation of BasicSegmentalKMeansTrainer<ObservationVectorView> (MV path).
//
// All method definitions are in basic_segmental_kmeans_trainer.h.

#include "libhmm/training/basic_segmental_kmeans_trainer.h"

namespace libhmm {

template class BasicSegmentalKMeansTrainer<ObservationVectorView>;

} // namespace libhmm
