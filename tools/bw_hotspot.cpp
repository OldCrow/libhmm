/**
 * @file bw_hotspot.cpp
 * @brief Baum-Welch inner-loop cost breakdown.
 *
 * Profiles the three separable cost centres of one BW E-step:
 *   1. FB computation (delegated to ForwardBackwardCalculator)
 *   2. Gamma accumulation  — N*T  exp() calls
 *   3. Xi accumulation     — N^2*(T-1) exp() calls  (dominant for N>1)
 *
 * Implemented inline here (not through BaumWelchTrainer) so each phase
 * can be timed independently without modifying the library.
 *
 * Usage:
 *   bw_hotspot                        (default configs)
 *   bw_hotspot <N> <T> [runs] [warmup]
 */

#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/distributions.h"
#include "libhmm/performance/transcendental_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace libhmm;
using Clock = std::chrono::high_resolution_clock;
using Millis = std::chrono::duration<double, std::milli>;

namespace {

constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

// Prevent dead-code elimination on accumulated values.
volatile double g_sink = 0.0;

// ---------------------------------------------------------------------------

double elapsed_ms(const Clock::time_point start) {
    return Millis(Clock::now() - start).count();
}

template <typename T>
double median(std::vector<T> v) {
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    return static_cast<double>(v[v.size() / 2]);
}

// ---------------------------------------------------------------------------

std::unique_ptr<Hmm> make_hmm(int n) {
    auto hmm = std::make_unique<Hmm>(n);
    Matrix trans(n, n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            trans(i, j) = 0.1 + 0.8 * (0.5 + 0.5 * std::sin(i * 0.7 + j * 1.3));
            sum += trans(i, j);
        }
        for (int j = 0; j < n; ++j)
            trans(i, j) /= sum;
    }
    hmm->setTrans(trans);

    Vector pi(n);
    for (int i = 0; i < n; ++i)
        pi(i) = 1.0 / static_cast<double>(n);
    hmm->setPi(pi);

    for (int i = 0; i < n; ++i)
        hmm->setDistribution(i, std::make_unique<GaussianDistribution>(i * 2.0, 1.0));
    return hmm;
}

ObservationSet make_obs(int t, int n) {
    ObservationSet obs(t);
    for (int i = 0; i < t; ++i)
        obs(i) = std::sin(i * 0.1) * static_cast<double>(n);
    return obs;
}

// ---------------------------------------------------------------------------
// One E-step with independent phase timers.
// ---------------------------------------------------------------------------

struct BwBreakdown {
    double fb_ms = 0.0;    // ForwardBackwardCalculator (construct + compute)
    double gamma_ms = 0.0; // gamma accumulation: N*T   exp() calls
    double xi_ms = 0.0;    // xi accumulation:    N^2*(T-1) exp() calls
    std::uint64_t gamma_exp_calls = 0;
    std::uint64_t xi_exp_calls = 0;
};

BwBreakdown profile_bw(const Hmm &hmm, const ObservationSet &obs, int warmup, int runs) {
    const std::size_t N = hmm.getNumStatesModern();
    const std::size_t T = obs.size();

    // Precompute flat log-transition (row-major N×N) once — same as trainer would do.
    std::vector<double> logTrans(N * N);
    bool hasZeroTransitions = false;
    {
        const Matrix &t = hmm.getTrans();
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                const double a = t(i, j);
                if (a > 0.0) {
                    logTrans[i * N + j] = std::log(a);
                } else {
                    logTrans[i * N + j] = LOG_ZERO;
                    hasZeroTransitions = true;
                }
            }
    }

    // Log-emission: time-major logEmitByTime[t*N+j] = log b_j(O_t).
    std::vector<double> logEmitByTime(T * N);
    {
        std::vector<double> stateMajor(N * T);
        const std::span<const double> obsSpan(obs.data(), T);
        for (std::size_t i = 0; i < N; ++i)
            hmm.getDistribution(i).getBatchLogProbabilities(
                obsSpan, std::span<double>(stateMajor.data() + i * T, T));
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t t2 = 0; t2 < T; ++t2)
                logEmitByTime[t2 * N + i] = stateMajor[i * T + t2];
    }

    std::vector<double> fb_ms_v, gamma_ms_v, xi_ms_v;
    fb_ms_v.reserve(static_cast<std::size_t>(runs));
    gamma_ms_v.reserve(static_cast<std::size_t>(runs));
    xi_ms_v.reserve(static_cast<std::size_t>(runs));

    // Accumulators (reset per run to prevent dead-code elim).
    std::vector<double> piNum(N);
    std::vector<double> transDen(N);
    std::vector<double> transNum(N * N);
    std::vector<double> emisWts(N * T);

    for (int iter = 0; iter < warmup + runs; ++iter) {
        // Phase 1: FB
        auto t0 = Clock::now();
        ForwardBackwardCalculator fbc(hmm, obs);
        const double logP = fbc.getLogProbability();
        const double fb_time = elapsed_ms(t0);

        if (!std::isfinite(logP))
            continue;

        const Matrix &logAlpha = fbc.getLogForwardVariables();
        const Matrix &logBeta = fbc.getLogBackwardVariables();

        // Phase 2: gamma accumulation (N*T exp() calls)
        std::fill(piNum.begin(), piNum.end(), 0.0);
        std::fill(transDen.begin(), transDen.end(), 0.0);

        t0 = Clock::now();
        for (std::size_t t2 = 0; t2 < T; ++t2) {
            for (std::size_t i = 0; i < N; ++i) {
                const double g = std::exp(logAlpha(t2, i) + logBeta(t2, i) - logP);
                emisWts[t2 * N + i] = g;
                if (t2 == 0)
                    piNum[i] += g;
                if (t2 < T - 1)
                    transDen[i] += g;
            }
        }
        const double gamma_time = elapsed_ms(t0);

        // Phase 3: xi accumulation (N^2*(T-1) exp() calls)
        std::fill(transNum.begin(), transNum.end(), 0.0);

        t0 = Clock::now();
        if (hasZeroTransitions) {
            for (std::size_t t2 = 0; t2 + 1 < T; ++t2) {
                const double *emitNext = logEmitByTime.data() + (t2 + 1) * N;
                for (std::size_t i = 0; i < N; ++i) {
                    const double logAlphaI = logAlpha(t2, i);
                    const double *logTransRow = logTrans.data() + i * N;
                    for (std::size_t j = 0; j < N; ++j) {
                        if (logTransRow[j] == LOG_ZERO) {
                            continue;
                        }
                        const double logXi =
                            logAlphaI + logTransRow[j] + emitNext[j] + logBeta(t2 + 1, j) - logP;
                        transNum[i * N + j] += std::exp(logXi);
                    }
                }
            }
        } else {
            for (std::size_t t2 = 0; t2 + 1 < T; ++t2) {
                const double *emitNext = logEmitByTime.data() + (t2 + 1) * N;
                for (std::size_t i = 0; i < N; ++i) {
                    const double logAlphaI = logAlpha(t2, i);
                    const double *logTransRow = logTrans.data() + i * N;
                    const double bias = -logP;
                    // The hotspot tool keeps the same dense-xi shape as the trainer:
                    // exp(alpha[i] + trans[i,j] + (emitNext[j] + betaNext[j] - logP)).
                    // Since this tool stores row-major transNum, keep the scalar loop
                    // here rather than inventing a second helper shape prematurely.
                    for (std::size_t j = 0; j < N; ++j) {
                        const double logXi =
                            logAlphaI + logTransRow[j] + emitNext[j] + logBeta(t2 + 1, j) + bias;
                        transNum[i * N + j] += std::exp(logXi);
                    }
                }
            }
        }
        const double xi_time = elapsed_ms(t0);

        // Sink to prevent elision.
        g_sink = g_sink + piNum[0] + transDen[0] + transNum[0] + emisWts[0];

        if (iter >= warmup) {
            fb_ms_v.push_back(fb_time);
            gamma_ms_v.push_back(gamma_time);
            xi_ms_v.push_back(xi_time);
        }
    }

    BwBreakdown r;
    r.fb_ms = median(fb_ms_v);
    r.gamma_ms = median(gamma_ms_v);
    r.xi_ms = median(xi_ms_v);
    r.gamma_exp_calls = static_cast<std::uint64_t>(N) * T;
    r.xi_exp_calls = static_cast<std::uint64_t>(N) * N * (T > 0 ? T - 1 : 0);
    return r;
}

int parse_pos(const char *v, const char *name) {
    try {
        const int x = std::stoi(v);
        if (x <= 0)
            throw std::invalid_argument("non-positive");
        return x;
    } catch (...) {
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + v);
    }
}

} // namespace

int main(int argc, char *argv[]) {
    struct Config {
        int n;
        int t;
    };
    std::vector<Config> configs = {{4, 500}, {8, 1000}, {16, 500}, {32, 2000}};
    int warmup = 2, runs = 8;

    if (argc == 3 || argc == 4 || argc == 5) {
        try {
            configs = {{parse_pos(argv[1], "N"), parse_pos(argv[2], "T")}};
            if (argc >= 4)
                runs = parse_pos(argv[3], "runs");
            if (argc == 5)
                warmup = parse_pos(argv[4], "warmup");
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    } else if (argc != 1) {
        std::cerr << "Usage: bw_hotspot [N T [runs [warmup]]]\n";
        return 1;
    }

    try {
        std::cout << "libhmm BW Hotspot Breakdown  (median of " << runs << " runs, " << warmup
                  << " warmup)\n";
        std::cout << std::string(66, '=') << "\n\n";
        std::cout << std::fixed << std::setprecision(3);

        for (const auto &cfg : configs) {
            auto hmm = make_hmm(cfg.n);
            auto obs = make_obs(cfg.t, cfg.n);
            const auto bw = profile_bw(*hmm, obs, warmup, runs);

            const double total = bw.fb_ms + bw.gamma_ms + bw.xi_ms;
            auto pct = [&](double v) {
                return (total > 0.0) ? 100.0 * v / total : 0.0;
            };

            std::cout << "N=" << cfg.n << "  T=" << cfg.t << "\n";
            std::cout << "  exp() call volume:  gamma="
                      << static_cast<double>(bw.gamma_exp_calls) / 1e3 << "K"
                      << "  xi=" << static_cast<double>(bw.xi_exp_calls) / 1e6 << "M"
                      << "  ratio xi/gamma="
                      << (bw.gamma_exp_calls > 0 ? static_cast<double>(bw.xi_exp_calls) /
                                                       static_cast<double>(bw.gamma_exp_calls)
                                                 : 0.0)
                      << "x\n";

            auto row = [&](const char *label, double ms, std::uint64_t calls) {
                std::cout << "  " << std::left << std::setw(24) << label << std::right
                          << std::setw(8) << ms << " ms"
                          << "  " << std::setw(6) << std::setprecision(1) << pct(ms) << "%";
                if (calls > 0) {
                    const double ns_per = (ms * 1e6) / static_cast<double>(calls);
                    std::cout << "  " << std::setprecision(1) << ns_per << " ns/exp()";
                }
                std::cout << "\n";
                std::cout << std::setprecision(3);
            };

            row("FB (fwd+bwd)", bw.fb_ms, 0);
            row("Gamma accum", bw.gamma_ms, bw.gamma_exp_calls);
            row("Xi accum", bw.xi_ms, bw.xi_exp_calls);
            std::cout << "  " << std::left << std::setw(24) << "TOTAL (1 BW iter)" << std::right
                      << std::setw(8) << total << " ms\n";
            std::cout << "\n";
        }

        if (g_sink == 1.23456789)
            std::cout << "sink=" << g_sink << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
