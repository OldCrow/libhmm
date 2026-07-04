#pragma once

// SegmentalKMeansTrainer: scalar (v3-compatible) alias for
// BasicSegmentalKMeansTrainer<double>.
//
// The v3 API — construction from Hmm& or Hmm*, isTerminated(), train() — is
// fully preserved. The discrete-only restriction is lifted: any scalar
// EmissionDistribution works via the generic fit() interface.

#include "libhmm/training/basic_segmental_kmeans_trainer.h"

namespace libhmm {

/// @brief Scalar (v3-compatible) segmental k-means trainer alias.
using SegmentalKMeansTrainer = BasicSegmentalKMeansTrainer<double>;

} // namespace libhmm
