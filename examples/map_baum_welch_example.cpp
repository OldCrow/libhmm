/**
 * map_baum_welch_example — MAP-EM Baum-Welch with Dirichlet priors.
 *
 * Demonstrates MapBaumWelchTrainer on the dishonest casino HMM:
 *
 *   c = 0  → standard MLE (identical to BaumWelchTrainer)
 *   c = 1  → Laplace smoothing: adds one virtual observation per outcome
 *
 * Key points illustrated:
 *   1. c = 0 and BaumWelchTrainer produce identical updates.
 *   2. c > 0 keeps all transitions and emission probabilities strictly
 *      positive, preventing degenerate sparse solutions.
 *   3. The correct convergence criterion is the full MAP objective
 *      log P(O|λ) + log P(λ|c), not the likelihood alone.
 *   4. Discrete emissions are Dirichlet-smoothed; continuous distributions
 *      are fitted by MLE and unaffected by c.
 */
#include <iomanip>
#include <iostream>
#include <memory>
#include "libhmm/libhmm.h"

using namespace libhmm;

// Build a 2-state, 6-symbol casino HMM from uniform starting parameters.
static std::unique_ptr<Hmm> make_casino_hmm() {
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
    for (int s = 0; s < 2; ++s)
        hmm->setDistribution(s, std::make_unique<DiscreteDistribution>(6));
    return hmm;
}

// Total log-likelihood across all sequences.
static double total_logL(const Hmm &hmm, const ObservationLists &obs) {
    double ll = 0.0;
    for (const auto &seq : obs)
        ll += ForwardBackwardCalculator(hmm, seq).getLogProbability();
    return ll;
}

int main() {
    std::cout << "MAP Baum-Welch Example\n";
    std::cout << "======================\n\n";

    // -------------------------------------------------------------------------
    // Training data: structured to produce a clear two-state pattern.
    // First half of each sequence: symbols 0-3 (fair-looking).
    // Second half: symbol 5 (six, loaded-looking).
    // -------------------------------------------------------------------------
    ObservationLists obs;
    for (int s = 0; s < 8; ++s) {
        ObservationSet seq(30);
        for (int t = 0; t < 15; ++t)
            seq(t) = static_cast<double>(t % 4);
        for (int t = 15; t < 30; ++t)
            seq(t) = 5.0;
        obs.push_back(seq);
    }

    // -------------------------------------------------------------------------
    // Compare: c = 0 (standard MLE) vs c = 1 (Laplace smoothing)
    // -------------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(4);

    for (double c : {0.0, 1.0}) {
        auto hmm = make_casino_hmm();
        MapBaumWelchTrainer trainer(*hmm, obs, c);

        std::cout << "--- c = " << c
                  << (c == 0.0 ? " (standard MLE, identical to BaumWelchTrainer)" : " (Laplace)")
                  << " ---\n";
        std::cout << std::setw(6) << "iter" << std::setw(14) << "log P(O|λ)" << std::setw(14)
                  << "log P(λ|c)" << std::setw(14) << "MAP objective"
                  << "\n";
        std::cout << std::string(48, '-') << "\n";

        for (int iter = 1; iter <= 10; ++iter) {
            trainer.train();
            const double logL = total_logL(*hmm, obs);
            const double logPr = trainer.computeLogPrior();
            const double mapObj = logL + logPr;
            std::cout << std::setw(6) << iter << std::setw(14) << logL << std::setw(14) << logPr
                      << std::setw(14) << mapObj << "\n";
        }

        // Print final transition matrix.
        std::cout << "\nFinal transition matrix:\n";
        const Matrix &A = hmm->getTrans();
        for (int i = 0; i < 2; ++i) {
            std::cout << "  [";
            for (int j = 0; j < 2; ++j)
                std::cout << std::setw(8) << A(i, j);
            std::cout << "  ]\n";
        }

        // Print final emission probabilities for each state.
        std::cout << "Final emission probabilities (symbols 1-6):\n";
        for (int s = 0; s < 2; ++s) {
            const auto &dd = static_cast<const DiscreteDistribution &>(hmm->getDistribution(s));
            std::cout << "  State " << s << ":";
            for (std::size_t k = 0; k < dd.getNumSymbols(); ++k)
                std::cout << std::setw(8) << dd.getSymbolProbability(k);
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // -------------------------------------------------------------------------
    // Note on convergence criterion
    // -------------------------------------------------------------------------
    std::cout << "Notes on convergence:\n";
    std::cout << "  With c > 0, log P(O|λ) alone is NOT guaranteed to be monotone.\n";
    std::cout << "  Use  logL + trainer.computeLogPrior()  as the convergence signal.\n";
    std::cout << "  With c = 0, log P(λ|c) = 0 and the MAP objective equals logL.\n\n";

    std::cout << "Notes on scope:\n";
    std::cout << "  Dirichlet priors apply to transitions, π, and DiscreteDistribution\n";
    std::cout << "  emissions. Continuous distributions (Gaussian, etc.) use MLE\n";
    std::cout << "  regardless of c — conjugate priors for them are not implemented.\n";

    return 0;
}
