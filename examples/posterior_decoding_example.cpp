/**
 * posterior_decoding_example — posterior decoding vs Viterbi.
 *
 * Two decoding strategies are available after running ForwardBackward:
 *
 *   Viterbi decoding  — finds the single most probable state sequence
 *                       argmax_{q_1..q_T} P(q_1..q_T | O, λ).
 *                       Globally optimal joint path; may contain a state at
 *                       time t that is individually unlikely at that step.
 *
 *   Posterior decoding — at each time step independently picks the most
 *                        probable state: argmax_i P(q_t=i | O, λ).
 *                        Minimises the per-step state error rate; consecutive
 *                        transitions in the result need not be in the model.
 *
 * This example uses the classic "occasionally dishonest casino" HMM
 * (Durbin et al. 1998) and shows where the two decodings agree or diverge,
 * and why each is appropriate in different applications.
 */
#include <iomanip>
#include <iostream>
#include <memory>
#include "libhmm/libhmm.h"

using namespace libhmm;

// Build the casino HMM: state 0 = fair die, state 1 = loaded die.
static std::unique_ptr<Hmm> make_casino_hmm() {
    auto hmm = std::make_unique<Hmm>(2);

    Matrix trans(2, 2);
    trans(0, 0) = 0.95;
    trans(0, 1) = 0.05; // mostly stays fair
    trans(1, 0) = 0.10;
    trans(1, 1) = 0.90; // mostly stays loaded
    hmm->setTrans(trans);

    Vector pi(2);
    pi(0) = 0.9;
    pi(1) = 0.1;
    hmm->setPi(pi);

    auto fair = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 6; ++i)
        fair->setProbability(i, 1.0 / 6.0);
    hmm->setDistribution(0, std::move(fair));

    auto loaded = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 5; ++i)
        loaded->setProbability(i, 0.1);
    loaded->setProbability(5, 0.5); // loaded die strongly favours 6
    hmm->setDistribution(1, std::move(loaded));

    return hmm;
}

int main() {
    std::cout << "Posterior Decoding vs Viterbi Example\n";
    std::cout << "======================================\n\n";

    auto hmm = make_casino_hmm();

    // -------------------------------------------------------------------------
    // Observation sequence: fair section, then loaded section, then mixed.
    // "Rolls" are 0-indexed (0 = face 1, 5 = face 6).
    // -------------------------------------------------------------------------
    ObservationSet obs(30);
    // t=0..9: fair-looking (low numbers)
    const double fairSec[] = {0, 2, 1, 3, 0, 2, 4, 1, 3, 2};
    // t=10..19: loaded (lots of 6s)
    const double loadedSec[] = {5, 5, 4, 5, 5, 3, 5, 5, 5, 4};
    // t=20..29: ambiguous (mixed)
    const double mixedSec[] = {5, 0, 5, 2, 5, 1, 5, 3, 5, 0};

    for (int i = 0; i < 10; ++i)
        obs(i) = fairSec[i];
    for (int i = 0; i < 10; ++i)
        obs(10 + i) = loadedSec[i];
    for (int i = 0; i < 10; ++i)
        obs(20 + i) = mixedSec[i];

    std::cout << "HMM: 2-state casino — fair die (0) / loaded die (1)\n";
    std::cout << "  Transitions: fair→loaded = 0.05, loaded→fair = 0.10\n";
    std::cout << "  Loaded die: p(6) = 0.50, p(1..5) = 0.10 each\n\n";

    // -------------------------------------------------------------------------
    // Run FB for posterior decoding, Viterbi for MAP path.
    // -------------------------------------------------------------------------
    ForwardBackwardCalculator fbc(*hmm, obs);
    ViterbiCalculator vc(*hmm, obs);

    const StateSequence posterior = fbc.decodePosterior();
    const StateSequence &viterbi = vc.getStateSequence();

    std::cout << "Sequence decoding (F = fair, L = loaded, * = decodings differ):\n\n";
    std::cout << " t  obs  posterior  viterbi\n";
    std::cout << std::string(32, '-') << "\n";

    int nDiffer = 0;
    for (std::size_t t = 0; t < obs.size(); ++t) {
        const bool diff = (posterior(t) != viterbi(t));
        if (diff)
            ++nDiffer;
        const char pChar = (posterior(t) == 0) ? 'F' : 'L';
        const char vChar = (viterbi(t) == 0) ? 'F' : 'L';
        std::cout << std::setw(2) << t << "   " << static_cast<int>(obs(t)) + 1 // 1-indexed face
                  << "       " << pChar << "          " << vChar << (diff ? "  *" : "") << "\n";
    }

    std::cout << "\nDecodings differ at " << nDiffer << "/" << obs.size() << " time steps.\n\n";

    // -------------------------------------------------------------------------
    // Log-probabilities
    // -------------------------------------------------------------------------
    std::cout << "log P(O | λ)  [total sequence probability from FB]: " << std::fixed
              << std::setprecision(4) << fbc.getLogProbability() << "\n";
    std::cout << "log P(O, q* | λ) [best-path probability from Viterbi]: " << vc.getLogProbability()
              << "\n\n";

    std::cout << "Notes:\n";
    std::cout << "  Viterbi log-prob ≤ FB log-prob (best path ≤ sum over all paths).\n";
    std::cout << "  Posterior decoding minimises per-step error rate;\n";
    std::cout << "  Viterbi minimises joint path error.\n";
    std::cout << "  Use posterior when per-step annotation accuracy matters most\n";
    std::cout << "  (e.g. gene prediction). Use Viterbi when whole-sequence\n";
    std::cout << "  structural coherence is required (e.g. speech alignment).\n";

    // -------------------------------------------------------------------------
    // Model criteria (AIC / BIC / AICc) from the same forward-backward pass.
    //
    // evaluate_model() takes the log-likelihood already computed above and the
    // HMM's parameter count; no extra computation is required.
    //
    // Free parameters for this 2-state, 6-symbol discrete HMM:
    //   Transitions: 2*(2-1) = 2
    //   Initial π:   2-1     = 1
    //   Emissions:   2*(6-1) = 10   (K-1 free per state, simplex constraint)
    //   Total k = 13
    // -------------------------------------------------------------------------
    std::cout << "\nModel criteria for this HMM on the 30-observation sequence:\n";
    std::cout << std::string(52, '-') << "\n";
    std::cout << std::fixed << std::setprecision(2);

    const HmmModelCriteria criteria = evaluate_model(*hmm, fbc.getLogProbability(), obs.size());
    const std::size_t k = count_free_parameters(*hmm);

    std::cout << "  Free parameters k: " << k << "\n";
    std::cout << "  AIC:               " << criteria.aic << "\n";
    std::cout << "  BIC:               " << criteria.bic << "\n";
    std::cout << "  AICc:              " << criteria.aicc << "  (n=" << obs.size() << ")\n";
    std::cout << "\n  To select between models of different state counts, fit each\n";
    std::cout << "  via Baum-Welch, compute evaluate_model() for each, and choose\n";
    std::cout << "  the model with the lowest criterion.\n";

    return 0;
}
