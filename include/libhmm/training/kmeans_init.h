#pragma once

#include <cstddef>
#include <random>

#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {

/**
 * @brief Initialise a multivariate HMM's emission distributions via k-means.
 *
 * Runs Lloyd's algorithm with k-means++ seeding on all observation vectors in
 * @p data (sequence order is ignored).  The resulting k = hmm.getNumStates()
 * cluster centres are assigned to HMM states in the order they emerge from the
 * algorithm, and each state's emission distribution is reset by calling
 *
 *   hmm.getDistribution(k).fit(span<const ObservationVectorView>)
 *
 * with the cluster members.  This provides a data-driven starting point before
 * running BasicBaumWelchTrainer<ObservationVectorView>.
 *
 * k-means++ seeding:
 *   1. Choose the first centroid uniformly at random from all observations.
 *   2. Choose each subsequent centroid with probability proportional to the
 *      squared distance from the nearest already-chosen centroid.
 * This significantly reduces sensitivity to initialisation compared to purely
 * random centroid selection.
 *
 * Empty clusters (states that received no observations in a Lloyd iteration)
 * are silently left at their previous centroid.  If a cluster remains empty
 * at the end, reset() is called on its emission distribution.
 *
 * @param hmm       MV HMM whose emission distributions will be initialised.
 * @param data      Observation sequences; must contain at least one observation.
 * @param rng       Seeded RNG for reproducible k-means++ seeding.
 * @param max_iter  Maximum number of Lloyd iterations (default 100).
 *
 * @throws std::invalid_argument if data is empty or hmm has zero states.
 * @throws std::invalid_argument if the dimensionality of observations is zero.
 */
void kmeans_init(HmmMV& hmm, const MultiObservationLists& data,
                 std::mt19937_64& rng, std::size_t max_iter = 100);

} // namespace libhmm
