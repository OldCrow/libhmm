#pragma once

// v4 compatibility header.
// Calculator is now a type alias for BasicCalculator<double>.
// All v3 concrete calculators (ForwardBackwardCalculator, ViterbiCalculator)
// derive from Calculator and continue to work unchanged.

#include "libhmm/calculators/basic_calculator.h"
#include "libhmm/hmm.h"   // provides 'Hmm = BasicHmm<double>' alias

namespace libhmm {

/// @brief Scalar calculator base (v3 compatibility alias).
using Calculator = BasicCalculator<double>;

} // namespace libhmm
