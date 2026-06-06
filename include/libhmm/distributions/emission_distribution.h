#pragma once

// v4 compatibility header.
//
// EmissionDistribution is now a type alias for BasicEmissionDistribution<double>.
// All v3 code that uses EmissionDistribution continues to compile unchanged:
// virtual signatures, unique_ptr<EmissionDistribution>, from_json return types,
// and Hmm::setDistribution / getDistribution are all unaffected by the alias.

#include "libhmm/distributions/basic_emission_distribution.h"

namespace libhmm {

/// @brief Scalar emission distribution interface (v3 compatibility alias).
using EmissionDistribution = BasicEmissionDistribution<double>;

} // namespace libhmm
