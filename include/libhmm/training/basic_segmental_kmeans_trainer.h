#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "libhmm/calculators/basic_viterbi_calculator.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/basic_trainer.h"

namespace libhmm {

/**
 * @brief Segmental K-means (hard-assignment EM) trainer, parameterised on observation type.
 *
 * @tparam Obs  `double` (scalar) or `ObservationVectorView` (v4 MV).
 *
 * Implements the segmental k-means algorithm for HMM parameter estimation.
 * Each iteration:
 *   1. learnPi()    — estimate π from the first assigned state of each sequence.
 *   2. learnTrans() — estimate A from consecutive assigned-state transitions.
 *   3. learnEmis()  — refit emission distributions via unweighted fit() calls
 *                     on hard-assigned observations.
 *   4. optimizeCluster() — re-run Viterbi; update per-sequence state assignments;
 *                          return true if any assignment changed.
 * Convergence: optimizeCluster() returning false terminates train(). A
 * maxIterations cap prevents non-termination on continuous data.
 *
 * Initial assignment: observation at global index i (sequenced over all sequences
 * in list order) is assigned to state floor(i * N / totalObs), giving an
 * approximately uniform partition without requiring data statistics. For
 * continuous/MV data, call kmeans_init() on the HMM before training to obtain
 * a data-driven starting point that improves convergence.
 *
 * State assignments are stored per-sequence, per-timestep:
 *   assignments_[s][t] = state for sequence s at time t.
 * This correctly handles the case where the same observation value maps to
 * different states at different times — the hash-map approach used for
 * pure discrete training cannot represent this.
 *
 * Two explicit instantiations are compiled:
 *   - src/training/segmental_kmeans_trainer.cpp    → BasicSegmentalKMeansTrainer<double>
 *   - src/training/segmental_kmeans_trainer_mv.cpp → BasicSegmentalKMeansTrainer<ObservationVectorView>
 */
template <typename Obs>
class BasicSegmentalKMeansTrainer : public BasicTrainer<Obs> {
public:
    using Base = BasicTrainer<Obs>;
    using HmmType = typename Base::HmmType;
    using ListType = typename Base::ListType;
    using SeqType = typename ObsSeqTraits<Obs>::SeqType;

    /**
     * @brief Construct from an HMM reference and observation lists.
     *
     * @param maxIterations  Hard cap on training iterations per train() call.
     *                       Default 100. Use a larger value for difficult continuous
     *                       problems; convergence is the primary termination criterion.
     * @throws std::invalid_argument if obsLists is empty.
     */
    BasicSegmentalKMeansTrainer(HmmType &hmm, const ListType &obsLists,
                                std::size_t maxIterations = 100);

    /** @brief Legacy pointer constructor for backward compatibility. */
    BasicSegmentalKMeansTrainer(HmmType *hmm, const ListType &obsLists,
                                std::size_t maxIterations = 100);

    ~BasicSegmentalKMeansTrainer() override = default;

    /**
     * @brief Run segmental k-means to convergence or maxIterations.
     *
     * State assignments are reinitialised on each call via index-partition,
     * so train() is an independent training run when called multiple times.
     */
    void train() override;

    /**
     * @return true if the last train() call converged (no state-assignment
     *         change in the final Viterbi pass). false if maxIterations was
     *         reached without convergence, or if train() has not been called.
     */
    [[nodiscard]] bool isTerminated() const noexcept { return terminated_; }

private:
    std::size_t maxIterations_;
    bool terminated_{false};

    // Per-sequence, per-timestep hard state assignments.
    // assignments_[s][t] = state index for sequence s at time t.
    std::vector<std::vector<std::size_t>> assignments_;

    // Observation element type adaptor (same pattern as BasicViterbiTrainer).
    using EmisElem = std::conditional_t<std::is_same_v<Obs, double>, double, ObservationVectorView>;

    /// Partition all observations into N clusters by global index position.
    void initAssignments();

    /// Estimate π from the first assigned state of each sequence.
    void learnPi();

    /// Estimate A from consecutive assigned-state transitions.
    void learnTrans();

    /// Refit each emission distribution from its hard-assigned observations.
    void learnEmis();

    /**
     * @brief Re-run Viterbi; update assignments; return true if any changed.
     *
     * Sequences with zero probability under the current model are skipped:
     * their assignments remain unchanged.
     */
    [[nodiscard]] bool optimizeCluster();
};

// =============================================================================
// Inline method definitions
// =============================================================================

template <typename Obs>
BasicSegmentalKMeansTrainer<Obs>::BasicSegmentalKMeansTrainer(HmmType &hmm,
                                                              const ListType &obsLists,
                                                              std::size_t maxIterations)
    : Base(hmm, obsLists), maxIterations_(maxIterations) {}

template <typename Obs>
BasicSegmentalKMeansTrainer<Obs>::BasicSegmentalKMeansTrainer(HmmType *hmm,
                                                              const ListType &obsLists,
                                                              std::size_t maxIterations)
    : Base(hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"), obsLists),
      maxIterations_(maxIterations) {}

// ---------------------------------------------------------------------------
// train()
// ---------------------------------------------------------------------------

template <typename Obs>
void BasicSegmentalKMeansTrainer<Obs>::train() {
    terminated_ = false;
    initAssignments();

    for (std::size_t iter = 0; iter < maxIterations_; ++iter) {
        learnPi();
        learnTrans();
        learnEmis();
        if (!optimizeCluster()) {
            terminated_ = true;
            return;
        }
    }
    // maxIterations_ reached without convergence; terminated_ stays false.
}

// ---------------------------------------------------------------------------
// initAssignments — global-index partition
// ---------------------------------------------------------------------------

template <typename Obs>
void BasicSegmentalKMeansTrainer<Obs>::initAssignments() {
    const std::size_t N = this->getHmmRef().getNumStatesModern();
    const auto &lists = this->getObservationLists();

    std::size_t totalObs = 0;
    for (const auto &seq : lists)
        totalObs += ObsSeqTraits<Obs>::sequence_length(seq);

    if (totalObs == 0)
        return;

    assignments_.resize(lists.size());
    std::size_t globalIdx = 0;
    for (std::size_t s = 0; s < lists.size(); ++s) {
        const std::size_t T = ObsSeqTraits<Obs>::sequence_length(lists[s]);
        assignments_[s].resize(T);
        for (std::size_t t = 0; t < T; ++t, ++globalIdx)
            assignments_[s][t] = (globalIdx * N) / totalObs;
    }
}

// ---------------------------------------------------------------------------
// learnPi — estimate initial state distribution
// ---------------------------------------------------------------------------

template <typename Obs>
void BasicSegmentalKMeansTrainer<Obs>::learnPi() {
    HmmType &hmm = this->getHmmRef();
    const std::size_t N = hmm.getNumStatesModern();
    const auto &lists = this->getObservationLists();

    Vector pi(N);
    clear_vector(pi);

    for (std::size_t s = 0; s < lists.size(); ++s) {
        if (!assignments_[s].empty())
            pi(assignments_[s][0]) += 1.0;
    }

    const double total = static_cast<double>(lists.size());
    for (std::size_t i = 0; i < N; ++i)
        pi(i) /= total;

    hmm.setPi(pi);
}

// ---------------------------------------------------------------------------
// learnTrans — estimate transition matrix
// ---------------------------------------------------------------------------

template <typename Obs>
void BasicSegmentalKMeansTrainer<Obs>::learnTrans() {
    HmmType &hmm = this->getHmmRef();
    const std::size_t N = hmm.getNumStatesModern();

    Matrix trans(N, N);
    clear_matrix(trans);

    for (const auto &asgn : assignments_) {
        const std::size_t T = asgn.size();
        for (std::size_t t = 0; t + 1 < T; ++t)
            trans(asgn[t], asgn[t + 1]) += 1.0;
    }

    // Row-normalise; uniform fallback for zero-count rows.
    for (std::size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < N; ++j)
            sum += trans(i, j);
        if (sum == 0.0) {
            const double uniform = 1.0 / static_cast<double>(N);
            for (std::size_t j = 0; j < N; ++j)
                trans(i, j) = uniform;
        } else {
            for (std::size_t j = 0; j < N; ++j)
                trans(i, j) /= sum;
        }
    }

    hmm.setTrans(trans);
}

// ---------------------------------------------------------------------------
// learnEmis — refit emission distributions from hard-assigned observations
// ---------------------------------------------------------------------------

template <typename Obs>
void BasicSegmentalKMeansTrainer<Obs>::learnEmis() {
    HmmType &hmm = this->getHmmRef();
    const std::size_t N = hmm.getNumStatesModern();
    const auto &lists = this->getObservationLists();

    std::vector<std::vector<EmisElem>> accum(N);

    for (std::size_t s = 0; s < lists.size(); ++s) {
        const auto &seq = lists[s];
        const std::size_t T = ObsSeqTraits<Obs>::sequence_length(seq);
        for (std::size_t t = 0; t < T; ++t) {
            const std::size_t state = assignments_[s][t];
            if constexpr (std::is_same_v<Obs, double>)
                accum[state].push_back(seq(t));
            else
                accum[state].push_back(row_view(seq, t));
        }
    }

    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t M = accum[k].size();
        if (M == 0) {
            hmm.getDistribution(k).reset();
            continue;
        }
        hmm.getDistribution(k).fit(std::span<const EmisElem>(accum[k].data(), M));
    }
}

// ---------------------------------------------------------------------------
// optimizeCluster — Viterbi re-assignment
// ---------------------------------------------------------------------------

template <typename Obs>
bool BasicSegmentalKMeansTrainer<Obs>::optimizeCluster() {
    bool modified = false;
    const auto &lists = this->getObservationLists();
    HmmType &hmm = this->getHmmRef();

    for (std::size_t s = 0; s < lists.size(); ++s) {
        const std::size_t T = ObsSeqTraits<Obs>::sequence_length(lists[s]);
        if (T == 0)
            continue;
        try {
            BasicViterbiCalculator<Obs> vc(hmm, lists[s]);
            if (!std::isfinite(vc.getLogProbability()))
                continue;
            const StateSequence &states = vc.getStateSequence();
            for (std::size_t t = 0; t < T; ++t) {
                const std::size_t newState = static_cast<std::size_t>(states(t));
                if (assignments_[s][t] != newState) {
                    modified = true;
                    assignments_[s][t] = newState;
                }
            }
        } catch (...) {
            // Skip sequence on error — leave existing assignments unchanged.
        }
    }

    return modified;
}

// =============================================================================
// Explicit instantiation declarations.
// Suppress implicit instantiation in consumer TUs.
// Definitions are in segmental_kmeans_trainer.cpp (scalar) and
// segmental_kmeans_trainer_mv.cpp (multivariate).
// =============================================================================
extern template class BasicSegmentalKMeansTrainer<double>;
extern template class BasicSegmentalKMeansTrainer<ObservationVectorView>;

/// @brief Multivariate alias (v4 addition).
using SegmentalKMeansTrainerMV = BasicSegmentalKMeansTrainer<ObservationVectorView>;

} // namespace libhmm
