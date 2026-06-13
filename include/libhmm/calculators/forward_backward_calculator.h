#pragma once

// v4 Phase H alias header.
// ForwardBackwardCalculator is now a type alias for BasicForwardBackwardCalculator<double>.
// All v3 code using ForwardBackwardCalculator continues to compile unchanged.

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/hmm.h" // provides Hmm = BasicHmm<double> and HmmMV aliases
