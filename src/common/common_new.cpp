#include "libhmm/common/common_new.h"

namespace libhmm {

void clear_matrix( Matrix& m ) {
    // Custom Matrix class already zeros itself on construction
    // This function provides explicit clearing for compatibility
    m.clear();
}

void clear_vector( Vector& v ) {
    // Custom Vector class already zeros itself on construction
    // This function provides explicit clearing for compatibility  
    v.clear();
}

void clear_vector( StateSequence& v ) {
    // StateSequence is Vector<StateIndex> = Vector<int>
    // This function provides explicit clearing for compatibility
    v.clear();
}

} // namespace libhmm
