#pragma once

// v4 Phase H alias header.
// ViterbiTrainer is now a type alias for BasicViterbiTrainer<double>.
// TrainingConfig and training_presets are defined in basic_viterbi_trainer.h.
// All v3 code using ViterbiTrainer/TrainingConfig/training_presets compiles unchanged.

#include "libhmm/training/basic_viterbi_trainer.h"
#include "libhmm/hmm.h" // provides Hmm = BasicHmm<double> and HmmMV aliases
