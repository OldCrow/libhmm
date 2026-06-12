#pragma once

// v4 compatibility header.
// Trainer is now a type alias for BasicTrainer<double>.
//
// v4 breaking change: the observation list is now held by const reference,
// not copied by value. Pass a named variable; passing a temporary will fail
// to compile. See MIGRATION.md.
//
// All existing concrete trainers (BaumWelchTrainer, ViterbiTrainer, etc.)
// derive from Trainer and continue to work unchanged.

#include "libhmm/training/basic_trainer.h"
#include "libhmm/hmm.h" // provides 'Hmm = BasicHmm<double>' alias

namespace libhmm {

/// @brief Scalar trainer base (v3 compatibility alias).
using Trainer = BasicTrainer<double>;

} // namespace libhmm
