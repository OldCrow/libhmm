#pragma once

// v4 Phase H alias header.
// BaumWelchTrainer is now a type alias for BasicBaumWelchTrainer<double>.
// All v3 code using BaumWelchTrainer continues to compile unchanged.

#include "libhmm/training/basic_baum_welch_trainer.h"
#include "libhmm/hmm.h"   // provides Hmm = BasicHmm<double> and HmmMV aliases
