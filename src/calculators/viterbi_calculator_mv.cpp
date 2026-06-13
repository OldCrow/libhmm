// Explicit instantiation of BasicViterbiCalculator<ObservationVectorView> (MV path).

#include "libhmm/calculators/basic_viterbi_calculator.h"

namespace libhmm {
template class BasicViterbiCalculator<ObservationVectorView>;
} // namespace libhmm
