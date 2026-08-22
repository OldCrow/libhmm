#pragma once

#include <cstddef>
#include <stdexcept>

#include "libhmm/basic_hmm.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/math/constants.h"

namespace libhmm {

/**
 * @brief Structural transition-matrix topologies (issue #46).
 *
 * A topology declares which transitions are structurally possible; invalid
 * transitions are held at exactly zero. Tied states (shared emission
 * distributions) are a separate concern and deliberately out of scope here.
 */
enum class HmmTopology {
    /// Fully connected: every transition valid (the default for a BasicHmm).
    Ergodic,
    /// Left-to-right (Bakis): A[i,j] = 0 for j < i; any forward jump and the
    /// self-loop are valid (full upper triangle). max_skip is ignored.
    LeftToRight,
    /// Left-to-right with bounded skip: valid j in [i, i+max_skip], clamped
    /// to the last state. max_skip = 1 is the strict sequential chain.
    LeftToRightSkip,
    /// Band-diagonal: valid j in [i-max_skip, i+max_skip] — each state
    /// reaches only its neighbours (and itself).
    Banded,
};

namespace detail {

/// True when the transition i -> j is structurally valid under topo.
[[nodiscard]] inline bool topology_allows(HmmTopology topo, std::size_t i, std::size_t j,
                                          std::size_t max_skip) noexcept {
    switch (topo) {
        case HmmTopology::LeftToRight:
            return j >= i;
        case HmmTopology::LeftToRightSkip:
            return j >= i && j <= i + max_skip;
        case HmmTopology::Banded:
            return (j >= i ? j - i : i - j) <= max_skip;
        case HmmTopology::Ergodic:
        default:
            return true;
    }
}

inline void validate_topology_args(HmmTopology topo, int max_skip) {
    if (max_skip < 1 && (topo == HmmTopology::LeftToRightSkip || topo == HmmTopology::Banded)) {
        throw std::invalid_argument("topology: max_skip must be >= 1");
    }
}

} // namespace detail

/**
 * @brief Initialise a topology-constrained transition matrix.
 *
 * Overwrites the HMM's transition matrix: A[i,j] = 1/(valid transitions
 * from i) where the topology allows i -> j, and exactly 0 elsewhere. Every
 * row has at least one valid transition (the self-loop is valid in all
 * topologies), so each row is a proper stochastic vector.
 *
 * Only the transition matrix is touched. Set pi separately — a left-to-right
 * model conventionally starts with a point mass on state 0. A model whose pi
 * was never set is rejected at training/scoring entry (issue #78).
 *
 * @param hmm       HMM whose transition matrix is replaced.
 * @param topo      Structural topology (see HmmTopology).
 * @param max_skip  Band half-width for LeftToRightSkip / Banded (>= 1).
 *                  Ignored for Ergodic and LeftToRight.
 * @throws std::invalid_argument if max_skip < 1 for a topology that uses it.
 */
template <typename Obs>
void initialize_topology(BasicHmm<Obs> &hmm, HmmTopology topo, int max_skip = 1) {
    detail::validate_topology_args(topo, max_skip);
    const std::size_t N = hmm.getNumStatesModern();
    // A negative max_skip converts modulo 2^64 ([conv.integral], well-defined, not UB);
    // validate_topology_args() has already rejected it for the topologies that read `skip`.
    const auto skip = static_cast<std::size_t>(max_skip);

    Matrix trans(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t valid = 0;
        for (std::size_t j = 0; j < N; ++j)
            if (detail::topology_allows(topo, i, j, skip))
                ++valid;
        const double p = 1.0 / static_cast<double>(valid);
        for (std::size_t j = 0; j < N; ++j)
            trans(i, j) = detail::topology_allows(topo, i, j, skip) ? p : 0.0;
    }
    hmm.setTrans(trans);
}

/**
 * @brief Re-impose a topology on the transition matrix after an M-step.
 *
 * Zeroes every structurally invalid entry and renormalises each row over its
 * valid entries. A row whose valid mass is (numerically) zero is reset to
 * uniform over its valid entries — this is precisely the case where the
 * Baum-Welch M-step fell back to a uniform 1/N row for an unvisited state,
 * which would otherwise silently break the constraint.
 *
 * EM preserves structural zeros on its own (a zero transition accumulates
 * zero xi mass), so in typical training this call only repairs that
 * uniform-reset fallback; calling it after every train() iteration is cheap
 * (O(N^2)) and makes the constraint unconditional.
 *
 * @param hmm       HMM whose transition matrix is masked in place.
 * @param topo      Structural topology (must match the one initialised).
 * @param max_skip  Band half-width for LeftToRightSkip / Banded (>= 1).
 * @throws std::invalid_argument if max_skip < 1 for a topology that uses it.
 */
template <typename Obs>
void enforce_topology(BasicHmm<Obs> &hmm, HmmTopology topo, int max_skip = 1) {
    detail::validate_topology_args(topo, max_skip);
    if (topo == HmmTopology::Ergodic)
        return; // nothing is invalid; leave the M-step result untouched
    const std::size_t N = hmm.getNumStatesModern();
    // A negative max_skip converts modulo 2^64 ([conv.integral], well-defined, not UB);
    // validate_topology_args() has already rejected it for the topologies that read `skip`.
    const auto skip = static_cast<std::size_t>(max_skip);

    Matrix trans = hmm.getTrans();
    for (std::size_t i = 0; i < N; ++i) {
        double validMass = 0.0;
        std::size_t valid = 0;
        for (std::size_t j = 0; j < N; ++j) {
            if (detail::topology_allows(topo, i, j, skip)) {
                validMass += trans(i, j);
                ++valid;
            } else {
                trans(i, j) = 0.0;
            }
        }
        // Same denormal-rejecting threshold as the trainers' M-steps.
        const bool degenerate = validMass < constants::precision::ZERO;
        const double uniform = 1.0 / static_cast<double>(valid);
        for (std::size_t j = 0; j < N; ++j) {
            if (!detail::topology_allows(topo, i, j, skip))
                continue;
            trans(i, j) = degenerate ? uniform : trans(i, j) / validMass;
        }
    }
    hmm.setTrans(trans);
}

} // namespace libhmm
