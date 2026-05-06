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
    /// Accumulates gamma statistics (emission data/weights, pi numerator, trans
    /// denominator) for one observation sequence.
    static void accumulate_gamma(const Matrix &logAlpha, const Matrix &logBeta,
                                 const ObservationSet &obs, double logP, std::size_t N,
                                 std::vector<std::vector<double>> &emisData,
                                 std::vector<std::vector<double>> &emisWts,
                                 std::vector<double> &piNum,
                                 std::vector<double> &transDen) noexcept;

    /// Accumulates xi statistics (expected transition counts) for one sequence.
    /// Dispatches to sparse or dense path based on hasZeroTransitions.
    static void accumulate_xi(const double *logAlphaData, const double *logBetaData,
                              const std::vector<double> &logEmitByTime,
                              const std::vector<double> &logTransT, double logP, std::size_t T,
                              std::size_t N, bool hasZeroTransitions,
                              std::vector<double> &transNumT) noexcept;

    /// M-step: normalize piNum and update the HMM initial distribution.
    static void m_step_pi(Hmm &hmm, std::size_t N, const std::vector<double> &piNum);

    /// M-step: normalize transNumT/transDen and update the HMM transition matrix.
    static void m_step_transitions(Hmm &hmm, std::size_t N, const std::vector<double> &transNumT,
                                   const std::vector<double> &transDen);

    /// Numerically stable log(exp(a) + exp(b)).
    static double logSumExp(double a, double b) noexcept;
}; // class BaumWelchTrainer

} // namespace libhmm
