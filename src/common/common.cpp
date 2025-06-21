#include "libhmm/common/common.h"
#include <algorithm>

namespace libhmm{

void clear_matrix( Matrix& m )
{
    for (std::size_t i = 0; i < m.size1(); ++i) {
        for (std::size_t j = 0; j < m.size2(); ++j) {
            m(i, j) = 0.0;
        }
    }
}

void clear_vector( Vector& v )
{
    std::fill(v.begin(), v.end(), 0.0);
}

void clear_vector( StateSequence& v )
{
    std::fill(v.begin(), v.end(), 0);
}

} //namespace
