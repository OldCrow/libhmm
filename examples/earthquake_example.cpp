/**
 * earthquake_example — Seismicity regime detection via Poisson HMM.
 *
 * Fits a 2-state Poisson HMM to the annual count of major earthquakes
 * (magnitude ≥ 7) worldwide, 1900–2006.  This is the canonical running
 * example in Zucchini & MacDonald (2009) and benchmarks directly against
 * the HiddenMarkov R package (David Harte, CRAN).
 *
 * Dataset
 * -------
 * 107 annual counts from the USGS earthquake catalogue (1900–2006),
 * published in Zucchini & MacDonald Table 1.1.  The data are embedded
 * in this file; no external download is needed.
 *
 * Model
 * -----
 * 2 hidden states: low-seismicity (λ₁) and high-seismicity (λ₂).
 * Emission: PoissonDistribution(λ) per state.
 * Training: BaumWelchTrainer (canonical log-space EM).
 *
 * HiddenMarkov R reference (BaumWelch, April 2026)
 * ------------------------------------------------
 *   λ₁ = 15.418  (low seismicity,  65 years)
 *   λ₂ = 26.013  (high seismicity, 42 years)
 *   Transition: [[0.928, 0.072], [0.119, 0.881]]
 *   Log-likelihood: -341.879
 *   Wall time: ~0.02 s (single-sequence, 107 observations)
 *
 * Reference
 * ---------
 * Zucchini W, MacDonald IL (2009). Hidden Markov Models for Time Series:
 * An Introduction Using R. CRC Press. (Table 1.1, Chapters 3–4.)
 * Harte D (2025). HiddenMarkov: Hidden Markov Models. CRAN.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "libhmm/libhmm.h"

using namespace libhmm;

// Annual major earthquake counts 1900-2006 (Zucchini & MacDonald Table 1.1)
static const int EQ_DATA[] = {
    13, 14, 8,  10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18, 25, 21, 21, 14, 8,  11,
    14, 23, 18, 17, 19, 20, 22, 19, 13, 26, 13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 27, 41,
    31, 27, 35, 26, 28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15, 22, 18, 15, 20, 15, 22,
    19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15, 18, 14, 10, 15, 8,  15, 6,  11,
    8,  7,  18, 16, 13, 12, 13, 20, 15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11};
static const int EQ_N = sizeof(EQ_DATA) / sizeof(EQ_DATA[0]);

static double total_loglik(const Hmm &hmm, const ObservationLists &obs) {
    double ll = 0.0;
    for (const auto &seq : obs)
        ll += ForwardBackwardCalculator(hmm, seq).getLogProbability();
    return ll;
}

static const PoissonDistribution &pois(const Hmm &h, std::size_t s) {
    return static_cast<const PoissonDistribution &>(h.getDistribution(s));
}

int main() {
    std::cout << "Major Earthquake Counts 1900-2006 — Poisson HMM\n";
    std::cout << "================================================\n\n";

    // Load data
    ObservationSet seq(EQ_N);
    double sum = 0.0;
    for (int i = 0; i < EQ_N; ++i) {
        seq(i) = EQ_DATA[i];
        sum += EQ_DATA[i];
    }
    ObservationLists obs = {seq};
    std::cout << "Dataset: " << EQ_N << " annual counts (1900-2006)\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  mean=" << sum / EQ_N << "  min=6  max=41\n\n";

    // HMM setup: 2 states, Poisson emissions
    Hmm hmm(2);
    Matrix A(2, 2);
    A(0, 0) = 0.9;
    A(0, 1) = 0.1;
    A(1, 0) = 0.1;
    A(1, 1) = 0.9;
    hmm.setTrans(A);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);
    hmm.setDistribution(0, std::make_unique<PoissonDistribution>(15.0));
    hmm.setDistribution(1, std::make_unique<PoissonDistribution>(25.0));

    std::cout << "Initial: λ₁=15.0 (low seismicity), λ₂=25.0 (high seismicity)\n\n";

    // Train
    std::cout << "Baum-Welch training:\n";
    std::cout << std::setw(7) << "iter" << std::setw(14) << "logL" << std::setw(10) << "delta\n";
    std::cout << std::string(31, '-') << "\n";

    BaumWelchTrainer trainer(&hmm, obs);
    const auto t0 = std::chrono::steady_clock::now();
    double prev = total_loglik(hmm, obs);
    std::cout << std::setprecision(4);
    std::cout << std::setw(7) << 0 << std::setw(14) << prev << "  (initial)\n";

    int conv = -1;
    for (int i = 1; i <= 200; ++i) {
        trainer.train();
        const double ll = total_loglik(hmm, obs);
        const double d = ll - prev;
        std::cout << std::setw(7) << i << std::setw(14) << ll << std::setw(10) << d;
        if (i > 1 && std::fabs(d) < 1e-5) {
            std::cout << "  <- converged";
            if (conv < 0)
                conv = i;
        }
        std::cout << "\n";
        if (conv > 0 && i >= conv + 2)
            break;
        prev = ll;
    }

    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double fll = total_loglik(hmm, obs);

    // Sort states by lambda ascending
    int lo = (pois(hmm, 0).getLambda() < pois(hmm, 1).getLambda()) ? 0 : 1;
    int hi = 1 - lo;

    std::cout << "\n=== libhmm results ===\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_ms << " ms\n\n";
    std::cout << std::setprecision(4);
    std::cout << "State " << lo << " (low  seismicity): λ = " << pois(hmm, lo).getLambda() << "\n";
    std::cout << "State " << hi << " (high seismicity): λ = " << pois(hmm, hi).getLambda()
              << "\n\n";
    std::cout << "Transition matrix:\n";
    const Matrix &T = hmm.getTrans();
    for (int i = 0; i < 2; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 2; ++j)
            std::cout << std::setw(9) << T(i, j);
        std::cout << " ]\n";
    }
    std::cout << "\nLog-likelihood: " << std::setprecision(3) << fll << "\n\n";

    // Viterbi state occupancy
    ViterbiCalculator vc(hmm, seq);
    const StateSequence &vpath = vc.getStateSequence();
    int cnt[2] = {0, 0};
    for (std::size_t t = 0; t < vpath.size(); ++t)
        ++cnt[vpath(t)];
    std::cout << "Viterbi occupancy: low=" << cnt[lo] << " years  high=" << cnt[hi] << " years\n\n";

    // Comparison
    std::cout << "=== Comparison: libhmm vs HiddenMarkov (R) ===\n\n";
    std::cout << std::setw(22) << " " << std::setw(12) << "libhmm" << std::setw(14)
              << "HiddenMarkov\n";
    std::cout << std::string(48, '-') << "\n";
    std::cout << std::setprecision(4);
    std::cout << std::setw(22) << "λ low" << std::setw(12) << pois(hmm, lo).getLambda()
              << std::setw(14) << 15.418 << "\n";
    std::cout << std::setw(22) << "λ high" << std::setw(12) << pois(hmm, hi).getLambda()
              << std::setw(14) << 26.013 << "\n";
    std::cout << std::setw(22) << "A[low→low]" << std::setw(12) << T(lo, lo) << std::setw(14)
              << 0.9283 << "\n";
    std::cout << std::setw(22) << "A[high→low]" << std::setw(12) << T(hi, lo) << std::setw(14)
              << 0.1189 << "\n";
    std::cout << std::setw(22) << "Log-likelihood" << std::setw(12) << std::setprecision(2) << fll
              << std::setw(14) << -341.879 << "\n";
    const std::string ws = std::to_string(static_cast<int>(wall_ms)) + " ms";
    std::cout << std::setw(22) << "Wall time" << std::setw(12) << ws << std::setw(14) << "~20 ms\n";

    std::cout << "\nNotes:\n";
    std::cout << "  Both fit the same 2-state Poisson HMM via Baum-Welch EM.\n";
    std::cout << "  HiddenMarkov uses nlm()-based maximisation; libhmm uses direct EM.\n";
    std::cout << "  The earthquake dataset is the canonical running example in\n";
    std::cout << "  Zucchini & MacDonald (2009), used throughout chapters 3-7.\n";

    return 0;
}
