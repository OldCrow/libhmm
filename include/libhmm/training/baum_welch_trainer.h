#pragma once

#include "libhmm/training/trainer.h"

namespace libhmm {

/**
 * @brief Log-space Baum-Welch (Forward-Backward) trainer.
 *
 * Implements one EM iteration per train() call:
 *   E-step: compute log-gamma and log-xi using ForwardBackwardCalculator.
 *   M-step: update pi, transition matrix, and call
 *           EmissionDistribution::fit(data, weights) per state.
 *
 * Works with any EmissionDistribution that implements the weighted fit
 * overload. Accumulates statistics across all supplied observation sequences
 * before applying the M-step update.
 *
 * @throws std::runtime_error from train() if no sequence has finite
 *         log-probability under the current model.
 */
class BaumWelchTrainer : public Trainer {
public:
    BaumWelchTrainer(Hmm &hmm, const ObservationLists &obsLists);
    BaumWelchTrainer(Hmm *hmm, const ObservationLists &obsLists);
    ~BaumWelchTrainer() override = default;

    void train() override;

private:
    /// Numerically stable log(exp(a) + exp(b)).
    static double logSumExp(double a, double b) noexcept;
}; // class BaumWelchTrainer

} // namespace libhmm
