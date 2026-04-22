/**
 * batch_performance — ForwardBackward and Viterbi timing benchmark.
 *
 * Builds Gaussian HMMs at varied (N states, T observations) and reports
 * throughput in state-steps per millisecond. Larger N amplifies the SIMD
 * benefit since getBatchLogProbabilities is called N times per time step.
 *
 * Usage: batch_performance
 *
 * The benchmark runs 5 warm-up iterations then 10 timed iterations
 * and reports the median.
 */
#include "libhmm/hmm.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace libhmm;
using Clock     = std::chrono::high_resolution_clock;
using Millis    = std::chrono::duration<double, std::milli>;

static std::unique_ptr<Hmm> make_hmm(int N) {
    auto hmm = std::make_unique<Hmm>(N);
    // Normalised random-ish transition matrix
    Matrix trans(N, N);
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            trans(i, j) = 0.1 + 0.8 * (0.5 + 0.5 * std::sin(i * 0.7 + j * 1.3));
            sum += trans(i, j);
        }
        for (int j = 0; j < N; ++j) trans(i, j) /= sum;
    }
    hmm->setTrans(trans);
    Vector pi(N);
    for (int i = 0; i < N; ++i) pi(i) = 1.0 / N;
    hmm->setPi(pi);
    for (int i = 0; i < N; ++i)
        hmm->setDistribution(i, std::make_unique<GaussianDistribution>(i * 2.0, 1.0));
    return hmm;
}

static ObservationSet make_obs(int T, int N) {
    ObservationSet obs(T);
    for (int t = 0; t < T; ++t)
        obs(t) = std::sin(t * 0.1) * static_cast<double>(N);
    return obs;
}

// Returns median of timings (ms)
static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main() {
    std::cout << "libhmm Batch Performance Benchmark\n";
    std::cout << "====================================\n";
    std::cout << "Gaussian HMM, log-space ForwardBackward and Viterbi.\n";
    std::cout << "Throughput = N\u00b2 \u00d7 T / ms  (dominant O(N\u00b2T) cost).\n\n";

    struct Config { int N; int T; };
    const std::vector<Config> configs = {
        {  2,   200},
        {  4,   500},
        {  8,  1000},
        { 16,  1000},
        { 32,   500},
        { 64,   200},
        {128,   100},
    };

    const int warmup = 5;
    const int runs   = 10;

    // Header
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(5)  << "N"
              << std::setw(7)  << "T"
              << std::setw(12) << "FB (ms)"
              << std::setw(16) << "FB (ss/ms)"
              << std::setw(12) << "V (ms)"
              << std::setw(16) << "V (ss/ms)\n";
    std::cout << std::string(68, '-') << "\n";

    for (const auto& cfg : configs) {
        auto hmm = make_hmm(cfg.N);
        auto obs = make_obs(cfg.T, cfg.N);
        const double state_steps = static_cast<double>(cfg.N) * cfg.N * cfg.T;

        // Warm up
        for (int i = 0; i < warmup; ++i) {
            ForwardBackwardCalculator fbc(*hmm, obs);
            (void)fbc.getLogProbability();
        }

        // Time ForwardBackward
        std::vector<double> fb_times;
        fb_times.reserve(runs);
        for (int r = 0; r < runs; ++r) {
            auto t0 = Clock::now();
            ForwardBackwardCalculator fbc(*hmm, obs);
            (void)fbc.getLogProbability();
            fb_times.push_back(Millis(Clock::now() - t0).count());
        }

        // Warm up Viterbi
        for (int i = 0; i < warmup; ++i) {
            ViterbiCalculator vc(*hmm, obs);
            (void)vc.decode();
        }

        // Time Viterbi
        std::vector<double> vt_times;
        vt_times.reserve(runs);
        for (int r = 0; r < runs; ++r) {
            auto t0 = Clock::now();
            ViterbiCalculator vc(*hmm, obs);
            (void)vc.decode();
            vt_times.push_back(Millis(Clock::now() - t0).count());
        }

        const double fb_ms = median(fb_times);
        const double vt_ms = median(vt_times);

        std::cout << std::setw(5) << cfg.N
                  << std::setw(7) << cfg.T
                  << std::setw(10) << fb_ms << " ms"
                  << std::setw(12) << static_cast<long long>(state_steps / fb_ms) << " ss/ms"
                  << std::setw(10) << vt_ms << " ms"
                  << std::setw(12) << static_cast<long long>(state_steps / vt_ms) << " ss/ms\n";
    }

    std::cout << "\nNote: ss/ms = state-steps per millisecond (N\u00b2\u00d7T / time).\n";
    std::cout << "Throughput should increase with N as SIMD batch width is amortised.\n";
    return 0;
}
