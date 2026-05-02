// tools/fb_crossover_sweep.cpp
//
// Measures ForwardBackwardCalculator runtime for Pairwise vs MaxReduce modes
// at a range of N values using the production calculator (which has SIMD
// transcendental kernels active in the MaxReduce path).
//
// Output: tab-separated table of N, pairwise_ms, maxreduce_ms, ratio.

#include "libhmm/calculators/fb_recurrence_policy.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/platform/simd_platform.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace libhmm;
using Clock   = std::chrono::high_resolution_clock;
using Millis  = std::chrono::duration<double, std::milli>;

namespace {

constexpr int WARMUP_RUNS = 2;
constexpr int TIMED_RUNS  = 8;
// T large enough that measurement is stable; small enough to finish quickly.
constexpr int T_DEFAULT   = 1000;

std::unique_ptr<Hmm> make_hmm(int n) {
    auto hmm = std::make_unique<Hmm>(n);
    Matrix trans(n, n);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
            trans(i, j) = 0.1 + 0.8 * (0.5 + 0.5 * std::sin(i * 0.7 + j * 1.3));
            s += trans(i, j);
        }
        for (int j = 0; j < n; ++j) trans(i, j) /= s;
    }
    hmm->setTrans(trans);
    Vector pi(n);
    for (int i = 0; i < n; ++i) pi(i) = 1.0 / n;
    hmm->setPi(pi);
    for (int i = 0; i < n; ++i)
        hmm->setDistribution(i, std::make_unique<GaussianDistribution>(i * 2.0, 1.0));
    return hmm;
}

ObservationSet make_obs(int t, int n) {
    ObservationSet obs(t);
    for (int i = 0; i < t; ++i)
        obs(i) = std::sin(i * 0.1) * n;
    return obs;
}

double time_mode(Hmm &hmm, const ObservationSet &obs, FbRecurrenceMode mode) {
    ForwardBackwardCalculator fbc(hmm, obs);
    fbc.setRecurrenceModeOverride(mode);

    // Warmup.
    for (int r = 0; r < WARMUP_RUNS; ++r)
        fbc.compute();

    // Timed runs.
    std::vector<double> samples;
    samples.reserve(TIMED_RUNS);
    for (int r = 0; r < TIMED_RUNS; ++r) {
        auto t0 = Clock::now();
        fbc.compute();
        samples.push_back(Millis(Clock::now() - t0).count());
    }

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2]; // median
}

} // anonymous namespace

int main() {
    const std::vector<int> N_VALUES = {2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64};
    const int T = T_DEFAULT;

    std::cout << "FB mode crossover sweep  (T=" << T
              << ", median of " << TIMED_RUNS << " runs, " << WARMUP_RUNS << " warmup)\n";
    std::cout << "Active ISA: " << libhmm::performance::simd::feature_string() << "\n\n";

    std::cout << std::setw(6)  << "N"
              << std::setw(14) << "Pairwise(ms)"
              << std::setw(14) << "MaxReduce(ms)"
              << std::setw(10) << "MR/PW"
              << std::setw(12) << "Winner"
              << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (int n : N_VALUES) {
        auto hmm = make_hmm(n);
        auto obs = make_obs(T, n);

        const double pw = time_mode(*hmm, obs, FbRecurrenceMode::Pairwise);
        const double mr = time_mode(*hmm, obs, FbRecurrenceMode::MaxReduce);
        const double ratio = mr / pw;
        const char *winner = (mr < pw) ? "MaxReduce" : "Pairwise";
        const char *current =
            (selectFbRecurrenceMode(n, T) == FbRecurrenceMode::MaxReduce) ? " [current]" : "";

        std::cout << std::setw(6)  << n
                  << std::setw(14) << std::fixed << std::setprecision(3) << pw
                  << std::setw(14) << std::fixed << std::setprecision(3) << mr
                  << std::setw(10) << std::fixed << std::setprecision(3) << ratio
                  << "  " << winner << current
                  << "\n";
    }

    std::cout << "\n(ratio < 1 = MaxReduce faster; > 1 = Pairwise faster)\n";
    std::cout << "[current] = what selectFbRecurrenceMode() currently picks for this N\n";
    return 0;
}
