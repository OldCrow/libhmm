#include "libhmm/common/common_complete.h"

namespace libhmm {

void clear_matrix( Matrix& m ) {
    // Custom BasicMatrix class already zeros itself on construction
    // This function provides explicit clearing for API compatibility
    m.clear();
}

void clear_vector( Vector& v ) {
    // Custom BasicVector class already zeros itself on construction
    // This function provides explicit clearing for API compatibility  
    v.clear();
}

void clear_vector( StateSequence& v ) {
    // StateSequence is BasicVector<StateIndex> = BasicVector<int>
    // This function provides explicit clearing for API compatibility
    v.clear();
}

} // namespace libhmm
