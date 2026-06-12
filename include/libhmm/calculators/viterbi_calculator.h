#pragma once

// v4 Phase H alias header.
// ViterbiCalculator is now a type alias for BasicViterbiCalculator<double>.
// All v3 code using ViterbiCalculator continues to compile unchanged.

#include "libhmm/calculators/basic_viterbi_calculator.h"
#include "libhmm/hmm.h" // provides Hmm = BasicHmm<double> and HmmMV aliases
