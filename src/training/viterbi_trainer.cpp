#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/hmm.h"
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace libhmm {

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

namespace training_presets {
TrainingConfig fast() noexcept {
    return {1e-5, 100, 2, false};
}
TrainingConfig balanced() noexcept {
    return {1e-6, 500, 3, false};
}
TrainingConfig precise() noexcept {
    return {1e-8, 2000, 5, false};
}
} // namespace training_presets

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ViterbiTrainer::ViterbiTrainer(Hmm &hmm, const ObservationLists &obsLists,
                               const TrainingConfig &config)
    : Trainer(hmm, obsLists), config_(config) {}

ViterbiTrainer::ViterbiTrainer(Hmm *hmm, const ObservationLists &obsLists,
                               const TrainingConfig &config)
    : Trainer(hmm, obsLists), config_(config) {}

// ---------------------------------------------------------------------------
// train()
// ---------------------------------------------------------------------------

void ViterbiTrainer::train() {
    converged_ = false;
    maxItersReached_ = false;
    lastLogProb_ = -std::numeric_limits<double>::infinity();

    std::deque<double> history;

    for (std::size_t iter = 0; iter < config_.maxIterations; ++iter) {
        const double logProb = runIteration();

        history.push_back(logProb);
        if (history.size() > config_.convergenceWindow) {
            history.pop_front();
        }

        // Declare convergence when every successive pair in the window is
        // within tolerance.
        if (history.size() == config_.convergenceWindow) {
            bool stable = true;
            for (std::size_t k = 1; k < history.size(); ++k) {
                if (std::abs(history[k] - history[k - 1]) >= config_.convergenceTolerance) {
                    stable = false;
                    break;
                }
            }
            if (stable) {
                converged_ = true;
                lastLogProb_ = logProb;
                return;
            }
        }

        lastLogProb_ = logProb;
    }

    maxItersReached_ = true;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

double ViterbiTrainer::accum_sequence(const Hmm &hmm, const ObservationSet &obs, Vector &pi,
                                      Matrix &trans,
                                      std::vector<std::vector<double>> &emisData) noexcept {
    try {
        ViterbiCalculator vc(hmm, obs);
        const double lp = vc.getLogProbability();
        if (!std::isfinite(lp))
            return lp;
        const StateSequence &seq = vc.getStateSequence();
        const std::size_t T = obs.size();
        pi(static_cast<std::size_t>(seq(0))) += 1.0;
        for (std::size_t t = 0; t < T; ++t) {
            const std::size_t s = static_cast<std::size_t>(seq(t));
            emisData[s].push_back(obs(t));
            if (t + 1 < T)
                trans(s, static_cast<std::size_t>(seq(t + 1))) += 1.0;
        }
        return lp;
    } catch (...) {
        return -std::numeric_limits<double>::infinity();
    }
}

void ViterbiTrainer::normalize_and_commit(Hmm &hmm, std::size_t N, Vector &pi,
                                          Matrix &trans) noexcept {
    // Normalize pi.
    double piSum = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        piSum += pi(i);
    if (piSum > 0.0) {
        for (std::size_t i = 0; i < N; ++i)
            pi(i) /= piSum;
    } else {
        for (std::size_t i = 0; i < N; ++i)
            pi(i) = 1.0 / static_cast<double>(N);
    }
    hmm.setPi(pi);
    // Normalize each transition row.
    for (std::size_t i = 0; i < N; ++i) {
        double rowSum = 0.0;
        for (std::size_t j = 0; j < N; ++j)
            rowSum += trans(i, j);
        if (rowSum > 0.0) {
            for (std::size_t j = 0; j < N; ++j)
                trans(i, j) /= rowSum;
        } else {
            for (std::size_t j = 0; j < N; ++j)
                trans(i, j) = 1.0 / static_cast<double>(N);
        }
    }
    hmm.setTrans(trans);
}

// ---------------------------------------------------------------------------
// runIteration() — one Viterbi pass
// ---------------------------------------------------------------------------

double ViterbiTrainer::runIteration() {
    Hmm &hmm = hmm_ref_.get();
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());

    Vector pi(N);
    clear_vector(pi);
    Matrix trans(N, N);
    clear_matrix(trans);
    std::vector<std::vector<double>> emisData(N);

    double totalLogProb = 0.0;
    std::size_t validSeqs = 0;

    for (const auto &obs : obsLists_) {
        if (obs.size() == 0)
            continue;
        const double lp = accum_sequence(hmm, obs, pi, trans, emisData);
        if (!std::isfinite(lp))
            continue;
        totalLogProb += lp;
        ++validSeqs;
    }

    if (validSeqs == 0)
        return lastLogProb_;

    normalize_and_commit(hmm, N, pi, trans);
    apply_emission_fits(hmm, N, emisData);
    return totalLogProb;
}

} // namespace libhmm
