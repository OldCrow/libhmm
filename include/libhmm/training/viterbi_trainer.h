#pragma once

#include "libhmm/training/trainer.h"
#include <deque>
#include <limits>

namespace libhmm {

/**
 * @brief Configuration for ViterbiTrainer.
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
    [[nodiscard]] TrainingConfig fast() noexcept;
    /// Default balanced settings.
    [[nodiscard]] TrainingConfig balanced() noexcept;
    /// Tight tolerance for high-accuracy offline training.
    [[nodiscard]] TrainingConfig precise() noexcept;
} // namespace training_presets

/**
 * @brief Log-space Viterbi trainer.
 *
 * Each iteration:
 *   1. Runs ViterbiCalculator on every observation sequence.
 *   2. Assigns each observation to the decoded state.
 *   3. Refits emission distributions via EmissionDistribution::fit(data).
 *   4. Re-estimates pi and the transition matrix from the Viterbi paths.
 *
 * Works with any EmissionDistribution (continuous or discrete).
 */
class ViterbiTrainer : public Trainer {
public:
    explicit ViterbiTrainer(Hmm& hmm, const ObservationLists& obsLists,
                            TrainingConfig config = {});
    explicit ViterbiTrainer(Hmm* hmm, const ObservationLists& obsLists,
                            TrainingConfig config = {});
    ~ViterbiTrainer() override = default;

    void train() override;

    /// True if the last train() call terminated via convergence criterion.
    [[nodiscard]] bool hasConverged()          const noexcept { return converged_; }
    /// True if the last train() call exhausted maxIterations.
    [[nodiscard]] bool reachedMaxIterations()  const noexcept { return maxItersReached_; }
    /// Total log-probability from the final iteration.
    [[nodiscard]] double getLastLogProbability() const noexcept { return lastLogProb_; }

    [[nodiscard]] const TrainingConfig& getConfig() const noexcept { return config_; }
    void setConfig(const TrainingConfig& config) { config_ = config; }

private:
    TrainingConfig config_;
    bool   converged_{false};
    bool   maxItersReached_{false};
    double lastLogProb_{-std::numeric_limits<double>::infinity()};

    /// Run one Viterbi iteration; returns total log-probability over all sequences.
    double runIteration();
}; // class ViterbiTrainer

} // namespace libhmm
