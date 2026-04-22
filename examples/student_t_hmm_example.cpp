/**
 * student_t_hmm_example — financial risk regime detection.
 *
 * Models asset return volatility regimes using Student's t distributions,
 * which capture the heavy tails common in financial data.
 *
 *   State 0: Low-volatility regime  (ν=10, μ=0.0, σ=0.5)
 *   State 1: High-volatility regime (ν= 3, μ=0.0, σ=2.0)
 *
 * The low degrees-of-freedom in the high-volatility state allow the
 * distribution to assign meaningful probability to extreme returns.
 *
 * Training uses Baum-Welch (the canonical log-space trainer), which
 * works directly with StudentTDistribution via weighted fit().
 */
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include "libhmm/libhmm.h"

using namespace libhmm;

static std::unique_ptr<Hmm> make_volatility_hmm() {
    auto hmm = std::make_unique<Hmm>(2);

    // Regime transition: low-vol is persistent, high-vol is transient
    Matrix trans(2, 2);
    trans(0, 0) = 0.92; trans(0, 1) = 0.08;  // Low-vol: mostly stays
    trans(1, 0) = 0.30; trans(1, 1) = 0.70;  // High-vol: mean-reverts faster
    hmm->setTrans(trans);

    Vector pi(2); pi(0) = 0.70; pi(1) = 0.30;
    hmm->setPi(pi);

    // Student's t: heavier tails than Gaussian
    hmm->setDistribution(0, std::make_unique<StudentTDistribution>(10.0, 0.0, 0.5));
    hmm->setDistribution(1, std::make_unique<StudentTDistribution>( 3.0, 0.0, 2.0));
    return hmm;
}

int main() {
    std::cout << "Student's t HMM: Financial Risk Regime Detection\n";
    std::cout << "==================================================\n\n";

    auto hmm = make_volatility_hmm();
    std::cout << "Initial model:\n" << *hmm << "\n";

    // -------------------------------------------------------------------------
    // Synthetic return data: quiet period, then stress, then recovery
    // -------------------------------------------------------------------------
    ObservationLists obs;

    // Quiet returns (low-vol regime)
    ObservationSet quiet(20);
    for (std::size_t i = 0; i < 20; ++i)
        quiet(i) = 0.3 * std::sin(i * 0.5);
    obs.push_back(quiet);

    // Stress period (high-vol: fat-tail returns)
    ObservationSet stress(15);
    double fat_tail_data[] = {0.4, -2.1, 1.8, -3.5, 0.7, 2.9, -1.6, 0.3,
                              -4.2, 1.1, 3.3, -0.8, 2.0, -1.9, 0.5};
    for (std::size_t i = 0; i < 15; ++i) stress(i) = fat_tail_data[i];
    obs.push_back(stress);

    // Recovery (mixed)
    ObservationSet recovery(12);
    for (std::size_t i = 0; i < 12; ++i)
        recovery(i) = 0.5 * std::sin(i * 0.3) - 0.1 * static_cast<double>(i % 3);
    obs.push_back(recovery);

    // -------------------------------------------------------------------------
    // Train with Baum-Welch (works with any EmissionDistribution via weighted fit)
    // -------------------------------------------------------------------------
    std::cout << "Training with Baum-Welch (5 iterations)...\n";
    BaumWelchTrainer trainer(hmm.get(), obs);
    for (int i = 0; i < 5; ++i) trainer.train();

    std::cout << "Trained model:\n" << *hmm << "\n";

    // -------------------------------------------------------------------------
    // Evaluate and decode
    // -------------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Inference on stress period:\n";
    ForwardBackwardCalculator fbc(*hmm, stress);
    std::cout << "  log P(O|λ) = " << fbc.getLogProbability() << "\n";

    ViterbiCalculator vc(*hmm, stress);
    auto path = vc.decode();
    std::cout << "  Viterbi path: ";
    for (std::size_t t = 0; t < path.size(); ++t)
        std::cout << path(t) << (t + 1 < path.size() ? "-" : "");
    std::cout << "\n  (0=low-vol, 1=high-vol)\n\n";

    // Count high-vol assignments
    int high_vol = 0;
    for (std::size_t t = 0; t < path.size(); ++t)
        if (path(t) == 1) ++high_vol;
    std::cout << "  High-volatility steps: " << high_vol << " / " << path.size() << "\n";
    std::cout << "  (expect majority in high-vol during stress period)\n";

    return 0;
}
