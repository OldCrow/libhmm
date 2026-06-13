// Explicit instantiation of BasicViterbiTrainer<ObservationVectorView> (MV path).

#include "libhmm/training/basic_viterbi_trainer.h"

namespace libhmm {
template class BasicViterbiTrainer<ObservationVectorView>;
} // namespace libhmm
