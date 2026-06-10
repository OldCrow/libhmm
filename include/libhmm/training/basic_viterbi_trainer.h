#pragma once

#include <cmath>
#include <deque>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "libhmm/calculators/basic_viterbi_calculator.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/basic_trainer.h"

namespace libhmm {

/**
 * @brief Convergence and iteration settings for ViterbiTrainer.
 *
 * Convergence is declared when the absolute change in total log-probability
 * between successive iterations stays below convergenceTolerance for
 * convergenceWindow consecutive iterations.
 */
struct TrainingConfig {
    double      convergenceTolerance{1e-6};
    std::size_t maxIterations{500};
    std::size_t convergenceWindow{3};
    bool        enableProgressReporting{false};
};

/// Named preset configurations for common training scenarios.
namespace training_presets {
/// Fast convergence, loose tolerance — suitable for initialisation.
[[nodiscard]] inline TrainingConfig fast() noexcept { return {1e-5, 100, 2, false}; }
/// Default balanced settings.
[[nodiscard]] inline TrainingConfig balanced() noexcept { return {1e-6, 500, 3, false}; }
/// Tight tolerance for high-accuracy offline training.
[[nodiscard]] inline TrainingConfig precise() noexcept { return {1e-8, 2000, 5, false}; }
} // namespace training_presets

/**
 * @brief Log-space Viterbi (hard-assignment EM) trainer, parameterised on observation type.
 *
 * @tparam Obs  `double` (scalar, v3-compatible) or `ObservationVectorView` (v4 MV).
 *
 * Each iteration:
 *   1. Runs BasicViterbiCalculator<Obs> on every observation sequence.
 *   2. Assigns each observation to the decoded state (hard assignment).
 *   3. Refits emission distributions via fit(data) (unweighted MLE).
 *   4. Re-estimates π and the transition matrix from the Viterbi paths.
 *
 * For the MV path, observed row-views (non-owning spans) are accumulated per state
 * and passed to fit(span<const ObservationVectorView>).
 *
 * train() runs multiple iterations to convergence (or maxIterations).
 * Use hasConverged() / reachedMaxIterations() to inspect termination.
 *
 * Two explicit instantiations are provided:
 *   - src/training/viterbi_trainer.cpp    → BasicViterbiTrainer<double>
 *   - src/training/viterbi_trainer_mv.cpp → BasicViterbiTrainer<ObservationVectorView>
 */
template<typename Obs>
class BasicViterbiTrainer : public BasicTrainer<Obs> {
public:
    using Base     = BasicTrainer<Obs>;
    using HmmType  = typename Base::HmmType;
    using ListType = typename Base::ListType;
    using SeqType  = typename ObsSeqTraits<Obs>::SeqType;

    explicit BasicViterbiTrainer(HmmType& hmm, const ListType& obsLists,
                                  const TrainingConfig& config = {});

    explicit BasicViterbiTrainer(HmmType* hmm, const ListType& obsLists,
                                  const TrainingConfig& config = {});

    ~BasicViterbiTrainer() override = default;

    /** @brief Run Viterbi training to convergence or maxIterations. */
    void train() override;

    /** @return true if the last train() call converged. */
    [[nodiscard]] bool hasConverged() const noexcept { return converged_; }
    /** @return true if the last train() call exhausted maxIterations. */
    [[nodiscard]] bool reachedMaxIterations() const noexcept { return maxItersReached_; }
    /** @return Total log-probability from the final iteration. */
    [[nodiscard]] double getLastLogProbability() const noexcept { return lastLogProb_; }

    [[nodiscard]] const TrainingConfig& getConfig() const noexcept { return config_; }
    void setConfig(const TrainingConfig& config) { config_ = config; }

private:
    TrainingConfig config_;
    bool   converged_{false};
    bool   maxItersReached_{false};
    double lastLogProb_{-std::numeric_limits<double>::infinity()};

    /// Run one Viterbi pass. Returns total log-probability; -∞ on failure.
    double runIteration();

    /// Normalise pi/trans counts and commit to the HMM.
    static void normalize_and_commit(HmmType& hmm, std::size_t N,
                                      Vector& pi, Matrix& trans);
};

// =============================================================================
// Inline method definitions
// =============================================================================

template<typename Obs>
BasicViterbiTrainer<Obs>::BasicViterbiTrainer(
        HmmType& hmm, const ListType& obsLists, const TrainingConfig& config)
    : Base(hmm, obsLists), config_(config) {}

template<typename Obs>
BasicViterbiTrainer<Obs>::BasicViterbiTrainer(
        HmmType* hmm, const ListType& obsLists, const TrainingConfig& config)
    : Base(hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"),
           obsLists),
      config_(config) {}

template<typename Obs>
void BasicViterbiTrainer<Obs>::train() {
    converged_       = false;
    maxItersReached_ = false;
    lastLogProb_     = -std::numeric_limits<double>::infinity();

    std::deque<double> history;

    for (std::size_t iter = 0; iter < config_.maxIterations; ++iter) {
        const double logProb = runIteration();

        history.push_back(logProb);
        if (history.size() > config_.convergenceWindow) history.pop_front();

        if (history.size() == config_.convergenceWindow) {
            bool stable = true;
            for (std::size_t k = 1; k < history.size(); ++k) {
                if (std::abs(history[k] - history[k - 1]) >= config_.convergenceTolerance) {
                    stable = false;
                    break;
                }
            }
            if (stable) {
                converged_   = true;
                lastLogProb_ = logProb;
                return;
            }
        }
        lastLogProb_ = logProb;
    }
    maxItersReached_ = true;
}

template<typename Obs>
double BasicViterbiTrainer<Obs>::runIteration() {
    HmmType& hmm = this->getHmmRef();
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());

    Vector pi(N);
    clear_vector(pi);
    Matrix trans(N, N);
    clear_matrix(trans);

    double      totalLogProb = 0.0;
    std::size_t validSeqs    = 0;

    if constexpr (std::is_same_v<Obs, double>) {
        // ----------------------------------------------------------------
        // Scalar path: accumulate observation values per state.
        // ----------------------------------------------------------------
        std::vector<std::vector<double>> emisData(N);

        for (const auto& obs : this->getObservationLists()) {
            if (obs.size() == 0) continue;
            try {
                BasicViterbiCalculator<double> vc(hmm, obs);
                const double lp = vc.getLogProbability();
                if (!std::isfinite(lp)) continue;
                const StateSequence& seq = vc.getStateSequence();
                const std::size_t T = obs.size();
                pi(static_cast<std::size_t>(seq(0))) += 1.0;
                for (std::size_t t = 0; t < T; ++t) {
                    const std::size_t s = static_cast<std::size_t>(seq(t));
                    emisData[s].push_back(obs(t));
                    if (t + 1 < T)
                        trans(s, static_cast<std::size_t>(seq(t + 1))) += 1.0;
                }
                totalLogProb += lp;
                ++validSeqs;
            } catch (...) {}
        }

        if (validSeqs == 0) return lastLogProb_;

        normalize_and_commit(hmm, N, pi, trans);
        this->apply_emission_fits(hmm, N, emisData);
    } else {
        // ----------------------------------------------------------------
        // Multivariate path: accumulate non-owning row views per state.
        // ----------------------------------------------------------------
        std::vector<std::vector<ObservationVectorView>> emisViews(N);

        for (const auto& obs : this->getObservationLists()) {
            const std::size_t T = ObsSeqTraits<Obs>::sequence_length(obs);
            if (T == 0) continue;
            try {
                BasicViterbiCalculator<Obs> vc(hmm, obs);
                const double lp = vc.getLogProbability();
                if (!std::isfinite(lp)) continue;
                const StateSequence& seq = vc.getStateSequence();
                pi(static_cast<std::size_t>(seq(0))) += 1.0;
                for (std::size_t t = 0; t < T; ++t) {
                    const std::size_t s = static_cast<std::size_t>(seq(t));
                    emisViews[s].push_back(row_view(obs, t));
                    if (t + 1 < T)
                        trans(s, static_cast<std::size_t>(seq(t + 1))) += 1.0;
                }
                totalLogProb += lp;
                ++validSeqs;
            } catch (...) {}
        }

        if (validSeqs == 0) return lastLogProb_;

        normalize_and_commit(hmm, N, pi, trans);

        // Unweighted fit from hard-assigned row views.
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t M = emisViews[i].size();
            if (M == 0) { hmm.getDistribution(i).reset(); continue; }
            hmm.getDistribution(i).fit(
                std::span<const ObservationVectorView>(emisViews[i].data(), M));
        }
    }

    return totalLogProb;
}

template<typename Obs>
void BasicViterbiTrainer<Obs>::normalize_and_commit(
        HmmType& hmm, std::size_t N, Vector& pi, Matrix& trans)
{
    double piSum = 0.0;
    for (std::size_t i = 0; i < N; ++i) piSum += pi(i);
    if (piSum > 0.0) {
        for (std::size_t i = 0; i < N; ++i) pi(i) /= piSum;
    } else {
        for (std::size_t i = 0; i < N; ++i) pi(i) = 1.0 / static_cast<double>(N);
    }
    hmm.setPi(pi);

    for (std::size_t i = 0; i < N; ++i) {
        double rowSum = 0.0;
        for (std::size_t j = 0; j < N; ++j) rowSum += trans(i, j);
        if (rowSum > 0.0) {
            for (std::size_t j = 0; j < N; ++j) trans(i, j) /= rowSum;
        } else {
            for (std::size_t j = 0; j < N; ++j) trans(i, j) = 1.0 / static_cast<double>(N);
        }
    }
    hmm.setTrans(trans);
}

// =============================================================================
// Explicit instantiation declarations.
// =============================================================================
extern template class BasicViterbiTrainer<double>;
extern template class BasicViterbiTrainer<ObservationVectorView>;

/// @brief Scalar alias (v3-compatible name).
using ViterbiTrainer = BasicViterbiTrainer<double>;

} // namespace libhmm
