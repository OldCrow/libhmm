#pragma once

// v4 Phase H alias header.
// MapBaumWelchTrainer is now a type alias for BasicMapBaumWelchTrainer<double>.
// All v3 code using MapBaumWelchTrainer continues to compile unchanged.

#include "libhmm/training/basic_map_baum_welch_trainer.h"
#include "libhmm/hmm.h" // provides Hmm = BasicHmm<double> and HmmMV aliases
