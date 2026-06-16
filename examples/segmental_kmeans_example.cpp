/**
 * segmental_kmeans_example — Segmental k-means training for discrete HMMs.
 *
 * SegmentalKMeansTrainer is a hard-assignment training algorithm for HMMs
 * with DiscreteDistribution emissions on every state. Relative to Baum-Welch,
 * it is faster but cruder: each observation is assigned to a single state via
 * Viterbi decoding, and parameters are re-estimated from those hard assignments
 * directly. It is most useful as an initialisation step before further
 * refinement with BaumWelchTrainer.
 *
 * This example shows:
 *   1. A 2-state discrete HMM modelling biased dice (one fair, one loaded).
 *   2. Path A — segmental k-means alone.
 *   3. Path B — segmental k-means warm-start followed by Baum-Welch refinement
 *               (the "init then refine" pattern).
 *   4. Constraint: SegmentalKMeansTrainer requires DiscreteDistribution on every
 *               state and throws std::runtime_error otherwise.
 */
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "libhmm/libhmm.h"

using namespace libhmm;

// Sum log P(O_k | λ) across all sequences (total log-likelihood)
static double total_ll(const Hmm &hmm, const ObservationLists &seqs) {
    double ll = 0.0;
    for (const auto &seq : seqs) {
        ForwardBackwardCalculator fbc(hmm, seq);
        ll += fbc.getLogProbability();
    }
    return ll;
}

// Build a 2-state discrete HMM over a 6-symbol alphabet (faces of a die).
// Emissions start uniform; training will discover one "fair" and one "loaded"
// state from the observation patterns.
static std::unique_ptr<Hmm> make_die_hmm() {
    auto hmm = std::make_unique<Hmm>(2);

    Matrix trans(2, 2);
    trans(0, 0) = 0.6;
    trans(0, 1) = 0.4;
    trans(1, 0) = 0.4;
    trans(1, 1) = 0.6;
    hmm->setTrans(trans);

    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm->setPi(pi);

    auto fair = std::make_unique<DiscreteDistribution>(6);
    auto loaded = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 6; ++i) {
        fair->setProbability(static_cast<double>(i), 1.0 / 6.0);
        loaded->setProbability(static_cast<double>(i), 1.0 / 6.0);
    }
    hmm->setDistribution(0, std::move(fair));
    hmm->setDistribution(1, std::move(loaded));

    return hmm;
}

static void print_emissions(const Hmm &hmm, const char *label) {
    std::cout << "  " << label << "\n";
    for (std::size_t s = 0; s < hmm.getNumStatesModern(); ++s) {
        const auto &d = static_cast<const DiscreteDistribution &>(hmm.getDistribution(s));
        std::cout << "    State " << s << ": ";
        for (std::size_t sym = 0; sym < d.getNumSymbols(); ++sym) {
            std::cout << std::fixed << std::setprecision(3) << d.getSymbolProbability(sym);
            if (sym + 1 < d.getNumSymbols())
                std::cout << " | ";
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "Segmental K-Means Training Example\n";
    std::cout << "==================================\n\n";

    // -------------------------------------------------------------------------
    // Synthetic training data: alternating runs of fair and loaded rolls.
    // Symbol space {0..5}; loaded sequences over-emit 4 and 5.
    // -------------------------------------------------------------------------
    ObservationLists obs;
    {
        ObservationSet s(20); // Fair: cycles through all six faces
        for (std::size_t i = 0; i < 20; ++i)
            s(i) = static_cast<double>(i % 6);
        obs.push_back(s);
    }
    {
        ObservationSet s(20); // Loaded: mostly 5, occasionally 4
        for (std::size_t i = 0; i < 20; ++i)
            s(i) = (i % 5 == 0) ? 4.0 : 5.0;
        obs.push_back(s);
    }
    {
        ObservationSet s(30); // Mixed: fair half, loaded half
        for (std::size_t i = 0; i < 15; ++i)
            s(i) = static_cast<double>(i % 6);
        for (std::size_t i = 15; i < 30; ++i)
            s(i) = (i % 4 == 0) ? 4.0 : 5.0;
        obs.push_back(s);
    }

    std::cout << std::fixed << std::setprecision(4);

    // -------------------------------------------------------------------------
    // Path A: SegmentalKMeansTrainer alone.
    // -------------------------------------------------------------------------
    std::cout << "Path A: SegmentalKMeansTrainer alone\n";
    std::cout << "------------------------------------\n";
    auto hmm_a = make_die_hmm();
    print_emissions(*hmm_a, "Initial (uniform):");
    std::cout << "  Initial log-likelihood: " << total_ll(*hmm_a, obs) << "\n";

    SegmentalKMeansTrainer skm(hmm_a.get(), obs);
    skm.train();

    std::cout << "  Converged: " << (skm.isTerminated() ? "yes" : "no") << "\n";
    print_emissions(*hmm_a, "After segmental k-means:");
    std::cout << "  Log-likelihood: " << total_ll(*hmm_a, obs) << "\n\n";

    // -------------------------------------------------------------------------
    // Path B: SegmentalKMeansTrainer warm-start + BaumWelchTrainer refinement.
    // The "init then refine" pattern: hard-assignment k-means converges quickly
    // to a reasonable local optimum, then EM smooths the assignments and
    // optimises P(O|λ) directly.
    // -------------------------------------------------------------------------
    std::cout << "Path B: Segmental k-means warm-start + Baum-Welch refinement\n";
    std::cout << "-------------------------------------------------------------\n";
    auto hmm_b = make_die_hmm();
    SegmentalKMeansTrainer skm_b(hmm_b.get(), obs);
    skm_b.train();
    const double ll_after_skm = total_ll(*hmm_b, obs);
    std::cout << "  After segmental k-means: " << ll_after_skm << "\n";

    BaumWelchTrainer bw(hmm_b.get(), obs);
    for (int i = 0; i < 5; ++i)
        bw.train();
    const double ll_after_bw = total_ll(*hmm_b, obs);
    std::cout << "  After 5 BW refinements:  " << ll_after_bw << "\n";
    std::cout << "  Improvement from BW:     " << (ll_after_bw - ll_after_skm) << "\n";
    print_emissions(*hmm_b, "Final emissions (Path B):");

    // -------------------------------------------------------------------------
    // Constraint: SegmentalKMeansTrainer requires DiscreteDistribution.
    // Constructing it on a Gaussian-emission HMM throws.
    // -------------------------------------------------------------------------
    std::cout << "\nConstraint demonstration: Gaussian emissions are rejected\n";
    std::cout << "---------------------------------------------------------\n";
    auto hmm_c = std::make_unique<Hmm>(2);
    hmm_c->setTrans(hmm_a->getTrans());
    hmm_c->setPi(hmm_a->getPi());
    hmm_c->setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm_c->setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.0));

    try {
        SegmentalKMeansTrainer skm_c(hmm_c.get(), obs);
        std::cout << "  ERROR: expected std::runtime_error but none was thrown\n";
        return 1;
    } catch (const std::runtime_error &e) {
        std::cout << "  Caught expected error: " << e.what() << "\n";
    }

    return 0;
}
