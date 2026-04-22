#pragma once
// HmmTrainer is the legacy name for Trainer — kept so SegmentedKMeansTrainer
// and any other legacy subclass compiles without changes to its header.
#include "libhmm/training/trainer.h"

namespace libhmm {
    using HmmTrainer = Trainer;
} // namespace libhmm

