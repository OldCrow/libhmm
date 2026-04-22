/**
 * baum_welch_example — Baum-Welch (EM) training demonstration.
 *
 * Baum-Welch is the primary training algorithm for HMMs. It runs the
 * EM (Expectation-Maximisation) algorithm: the E-step computes state
 * occupation probabilities (γ) using ForwardBackward; the M-step
 * re-estimates transition and emission parameters using weighted fit().
 *
 * This example shows:
 *   1. Training a Gaussian HMM on synthetic two-cluster data.
 *   2. Log-likelihood progression demonstrating EM monotonic improvement.
 *   3. Final parameter recovery: learned means should approximate 0 and 5.
 *   4. Comparison with Viterbi training on the same data.
 */
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>
#include "libhmm/libhmm.h"

using namespace libhmm;

// Sum log P(O_k | λ) across all sequences (total log-likelihood)
static double total_ll(const Hmm& hmm, const ObservationLists& seqs) {
    double ll = 0.0;
    for (const auto& seq : seqs) {
        ForwardBackwardCalculator fbc(hmm, seq);
        ll += fbc.getLogProbability();
    }
    return ll;
}

int main() {
    std::cout << "Baum-Welch (EM) Training Example\n";
    std::cout << "=================================\n\n";

    // -------------------------------------------------------------------------
    // Generative model: 2 states, N(0,1) and N(5,1), self-transition 0.8
    // -------------------------------------------------------------------------
    auto hmm = std::make_unique<Hmm>(2);

    Matrix trans(2, 2);
    trans(0, 0) = 0.7; trans(0, 1) = 0.3;
    trans(1, 0) = 0.3; trans(1, 1) = 0.7;
    hmm->setTrans(trans);

    Vector pi(2); pi(0) = 0.5; pi(1) = 0.5;
    hmm->setPi(pi);

    // Deliberately offset initial parameters to show training effect
    hmm->setDistribution(0, std::make_unique<GaussianDistribution>(1.0, 2.0));
    hmm->setDistribution(1, std::make_unique<GaussianDistribution>(4.0, 2.0));

    std::cout << "Initial parameters (offset from true values):\n";
    std::cout << "  State 0: N(1.0, 2.0)  [true: N(0, 1)]\n";
    std::cout << "  State 1: N(4.0, 2.0)  [true: N(5, 1)]\n\n";

    // -------------------------------------------------------------------------
    // Synthetic training data: alternating clusters
    // -------------------------------------------------------------------------
    ObservationLists obs;

    // Sequence 1: starts near 0, transitions to 5
    ObservationSet s1(20);
    for (std::size_t i = 0; i < 10; ++i) s1(i) = static_cast<double>(i % 3) * 0.4 - 0.2;
    for (std::size_t i = 10; i < 20; ++i) s1(i) = 5.0 + static_cast<double>(i % 3) * 0.4 - 0.2;
    obs.push_back(s1);

    // Sequence 2: all near 0
    ObservationSet s2(15);
    for (std::size_t i = 0; i < 15; ++i) s2(i) = static_cast<double>(i % 5) * 0.2 - 0.4;
    obs.push_back(s2);

    // Sequence 3: mixed
    ObservationSet s3(18);
    for (std::size_t i = 0; i < 18; ++i)
        s3(i) = (i % 2 == 0) ? static_cast<double>(i % 3) * 0.3
                              : 5.0 + static_cast<double>(i % 3) * 0.3;
    obs.push_back(s3);

    // -------------------------------------------------------------------------
    // Train and track log-likelihood at each iteration
    // -------------------------------------------------------------------------
    std::cout << "Training log-likelihood progression:\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(10) << "Iteration"
              << std::setw(16) << "log-likelihood\n";
    std::cout << std::string(26, '-') << "\n";

    BaumWelchTrainer trainer(hmm.get(), obs);
    double prev_ll = total_ll(*hmm, obs);
    std::cout << std::setw(10) << 0 << std::setw(16) << prev_ll << "\n";

    for (int iter = 1; iter <= 10; ++iter) {
        trainer.train();
        const double ll = total_ll(*hmm, obs);
        std::cout << std::setw(10) << iter << std::setw(16) << ll;
        if (ll < prev_ll - 1e-6)
            std::cout << "  WARNING: decreased by " << (prev_ll - ll);
        std::cout << "\n";
        prev_ll = ll;
    }

    std::cout << "\nLearned parameters after 10 iterations:\n";
    const auto& d0 = static_cast<const GaussianDistribution&>(hmm->getDistribution(0));
    const auto& d1 = static_cast<const GaussianDistribution&>(hmm->getDistribution(1));
    std::cout << "  State 0: N(" << d0.getMean() << ", "
              << d0.getStandardDeviation() << ")  [target ~N(0, 1)]\n";
    std::cout << "  State 1: N(" << d1.getMean() << ", "
              << d1.getStandardDeviation() << ")  [target ~N(5, 1)]\n\n";

    // -------------------------------------------------------------------------
    // Contrast: Viterbi training on the same data
    // -------------------------------------------------------------------------
    std::cout << "Comparison: Viterbi training (hard assignments, 5 iterations)\n";
    std::cout << "--------------------------------------------------------------\n";

    auto hmm_vt = std::make_unique<Hmm>(2);
    hmm_vt->setTrans(trans);
    hmm_vt->setPi(pi);
    hmm_vt->setDistribution(0, std::make_unique<GaussianDistribution>(1.0, 2.0));
    hmm_vt->setDistribution(1, std::make_unique<GaussianDistribution>(4.0, 2.0));

    ViterbiTrainer vt(hmm_vt.get(), obs);
    for (int i = 0; i < 5; ++i) vt.train();

    const auto& vd0 = static_cast<const GaussianDistribution&>(hmm_vt->getDistribution(0));
    const auto& vd1 = static_cast<const GaussianDistribution&>(hmm_vt->getDistribution(1));
    std::cout << "  State 0: N(" << vd0.getMean() << ", "
              << vd0.getStandardDeviation() << ")\n";
    std::cout << "  State 1: N(" << vd1.getMean() << ", "
              << vd1.getStandardDeviation() << ")\n\n";

    std::cout << "Baum-Welch log-likelihood: " << total_ll(*hmm, obs) << "\n";
    std::cout << "Viterbi  log-likelihood:   " << total_ll(*hmm_vt, obs) << "\n";
    std::cout << "(Baum-Welch optimises P(O|λ) directly; Viterbi maximises the best path)\n";

    return 0;
}
