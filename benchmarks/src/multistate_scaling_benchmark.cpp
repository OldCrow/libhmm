/**
 * @file multistate_scaling_benchmark.cpp
 * @brief Multi-state N×T scaling benchmark for the v4.0.3 MaxReduce hot-path.
 *
 * Sweeps N ∈ {2, 4, 8, 16, 32} states and T ∈ {500, 2000, 10000} time-steps.
 * For each cell, runs Forward-Backward with:
 *   1. libhmm Gaussian (continuous, SIMD emission + MaxReduce recurrence)
 *   2. libhmm Discrete (discrete table lookup + MaxReduce recurrence; baseline)
 *   3. HMMLib Discrete (if compiled with -DLIBHMM_BENCH_HAS_HMMLIB)
 *   4. GHMM Discrete   (if compiled with -DLIBHMM_BENCH_HAS_GHMM; POSIX/macOS only)
 *
 * The N≥4 cells exercise the MaxReduce path added in v4.0.3.
 * N=2 cells remain on the Pairwise path and serve as the within-library baseline.
 *
 * Comparators use discrete HMMs with a fixed 8-symbol alphabet.
 * Median of 5 trials is reported to reduce measurement noise.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// libhmm
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/performance/fb_recurrence_policy.h"

// HMMLib (optional; provided when -DLIBHMM_BENCH_HAS_HMMLIB is set)
#ifdef LIBHMM_BENCH_HAS_HMMLIB
#include "HMMlib/hmm.hpp"
#endif

// GHMM (optional; provided when -DLIBHMM_BENCH_HAS_GHMM is set; POSIX/macOS only)
#ifdef LIBHMM_BENCH_HAS_GHMM
extern "C" {
#include <ghmm/foba.h>
#include <ghmm/ghmm.h>
#include <ghmm/model.h>
#include <ghmm/sequence.h>
#include <ghmm/viterbi.h>
}
#endif

using namespace std::chrono;

// ============================================================
// Constants and helpers
// ============================================================

namespace {

static constexpr int DISCRETE_ALPHABET = 8; ///< Alphabet size for discrete comparators
static constexpr int NUM_TRIALS = 5;        ///< Median of this many timed runs per cell

using RNG = std::mt19937;

/// Return a random row-stochastic vector of length N (minimum weight 0.1 per entry).
[[nodiscard]] std::vector<double> randomRow(int N, RNG &rng) {
    std::uniform_real_distribution<double> u(0.1, 1.0);
    std::vector<double> row(static_cast<std::size_t>(N));
    double sum = 0.0;
    for (auto &v : row) {
        v = u(rng);
        sum += v;
    }
    for (auto &v : row) {
        v /= sum;
    }
    return row;
}

/// Median of a vector (makes an internal copy; does not modify caller's data).
[[nodiscard]] double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
    return v[m];
}

} // anonymous namespace

// ============================================================
// Result type
// ============================================================

struct BenchResult {
    std::string library;
    int N{0};
    int T{0};
    double fb_ms{-1.0}; ///< Median forward-backward wall time (ms); -1 if failed
    double log_lik{0.0};
    bool success{false};
    std::string mode; ///< "pairwise" | "max-reduce" (libhmm only) or "" (other libraries)
};

// ============================================================
// libhmm — Gaussian continuous
// ============================================================

[[nodiscard]] BenchResult benchGaussian(int N, int T, RNG &rng) {
    BenchResult r{.library = "libhmm-Gauss", .N = N, .T = T};
    try {
        auto hmm = std::make_unique<libhmm::Hmm>(N);

        // Initial probabilities (uniform)
        {
            libhmm::Vector pi(static_cast<std::size_t>(N));
            const double w = 1.0 / N;
            for (int i = 0; i < N; ++i) {
                pi(static_cast<std::size_t>(i)) = w;
            }
            hmm->setPi(pi);
        }

        // Random row-stochastic transition matrix
        {
            libhmm::Matrix A(static_cast<std::size_t>(N), static_cast<std::size_t>(N));
            for (int i = 0; i < N; ++i) {
                auto row = randomRow(N, rng);
                for (int j = 0; j < N; ++j) {
                    A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                        row[static_cast<std::size_t>(j)];
                }
            }
            hmm->setTrans(A);
        }

        // Gaussian emissions: state i → N(i×2, 1)
        for (int i = 0; i < N; ++i) {
            hmm->setDistribution(static_cast<std::size_t>(i),
                                 std::make_unique<libhmm::GaussianDistribution>(i * 2.0, 1.0));
        }

        // Observation sequence: draw from N(N-1, max(1,N/2))
        std::normal_distribution<double> obs_dist(static_cast<double>(N - 1),
                                                  std::max(1.0, static_cast<double>(N) / 2.0));
        libhmm::ObservationSet obs(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) {
            obs(static_cast<std::size_t>(t)) = obs_dist(rng);
        }

        std::vector<double> times(NUM_TRIALS);
        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            auto t0 = high_resolution_clock::now();
            libhmm::ForwardBackwardCalculator fb(*hmm, obs);
            auto t1 = high_resolution_clock::now();
            times[static_cast<std::size_t>(trial)] =
                duration_cast<microseconds>(t1 - t0).count() / 1000.0;
            if (trial == 0) {
                r.log_lik = fb.getLogProbability();
                r.mode = libhmm::toString(fb.getRecurrenceMode());
            }
        }
        r.fb_ms = median(times);
        r.success = true;
    } catch (const std::exception &e) {
        std::cerr << "libhmm-Gauss error (N=" << N << " T=" << T << "): " << e.what() << "\n";
    }
    return r;
}

// ============================================================
// libhmm — Discrete (recurrence baseline)
// ============================================================

[[nodiscard]] BenchResult benchLibhmmDiscrete(int N, int T, RNG &rng) {
    BenchResult r{.library = "libhmm-Disc", .N = N, .T = T};
    const int M = DISCRETE_ALPHABET;
    try {
        auto hmm = std::make_unique<libhmm::Hmm>(N);

        {
            libhmm::Vector pi(static_cast<std::size_t>(N));
            const double w = 1.0 / N;
            for (int i = 0; i < N; ++i) {
                pi(static_cast<std::size_t>(i)) = w;
            }
            hmm->setPi(pi);
        }
        {
            libhmm::Matrix A(static_cast<std::size_t>(N), static_cast<std::size_t>(N));
            for (int i = 0; i < N; ++i) {
                auto row = randomRow(N, rng);
                for (int j = 0; j < N; ++j) {
                    A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                        row[static_cast<std::size_t>(j)];
                }
            }
            hmm->setTrans(A);
        }
        for (int i = 0; i < N; ++i) {
            auto disc = std::make_unique<libhmm::DiscreteDistribution>(M);
            auto emitRow = randomRow(M, rng);
            for (int k = 0; k < M; ++k) {
                disc->setProbability(k, emitRow[static_cast<std::size_t>(k)]);
            }
            hmm->setDistribution(static_cast<std::size_t>(i), std::move(disc));
        }

        std::uniform_int_distribution<int> obs_dist(0, M - 1);
        libhmm::ObservationSet obs(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) {
            obs(static_cast<std::size_t>(t)) = static_cast<double>(obs_dist(rng));
        }

        std::vector<double> times(NUM_TRIALS);
        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            auto t0 = high_resolution_clock::now();
            libhmm::ForwardBackwardCalculator fb(*hmm, obs);
            auto t1 = high_resolution_clock::now();
            times[static_cast<std::size_t>(trial)] =
                duration_cast<microseconds>(t1 - t0).count() / 1000.0;
            if (trial == 0) {
                r.log_lik = fb.getLogProbability();
                r.mode = libhmm::toString(fb.getRecurrenceMode());
            }
        }
        r.fb_ms = median(times);
        r.success = true;
    } catch (const std::exception &e) {
        std::cerr << "libhmm-Disc error (N=" << N << " T=" << T << "): " << e.what() << "\n";
    }
    return r;
}

// ============================================================
// HMMLib — Discrete (optional)
// ============================================================

#ifdef LIBHMM_BENCH_HAS_HMMLIB

[[nodiscard]] BenchResult benchHMMLib(int N, int T, RNG &rng) {
    BenchResult r{.library = "HMMLib", .N = N, .T = T};
    const int M = DISCRETE_ALPHABET;
    try {
        using HMMVec = hmmlib::HMMVector<double>;
        using HMMMat = hmmlib::HMMMatrix<double>;

        // HMMLib requires boost::shared_ptr (part of its public API)
        auto pi_ptr = boost::shared_ptr<HMMVec>(new HMMVec(N));
        auto A_ptr = boost::shared_ptr<HMMMat>(new HMMMat(N, N));
        auto E_ptr = boost::shared_ptr<HMMMat>(new HMMMat(M, N)); // E(symbol, state)

        const double w = 1.0 / N;
        for (int i = 0; i < N; ++i) {
            (*pi_ptr)(i) = w;
        }
        for (int i = 0; i < N; ++i) {
            auto row = randomRow(N, rng);
            for (int j = 0; j < N; ++j) {
                (*A_ptr)(i, j) = row[static_cast<std::size_t>(j)];
            }
        }
        // HMMLib emission indexing: E(symbol, state)
        for (int s = 0; s < N; ++s) {
            auto emitRow = randomRow(M, rng);
            for (int k = 0; k < M; ++k) {
                (*E_ptr)(k, s) = emitRow[static_cast<std::size_t>(k)];
            }
        }

        hmmlib::HMM<double> hmm(pi_ptr, A_ptr, E_ptr);

        std::uniform_int_distribution<unsigned int> obs_dist(0, static_cast<unsigned int>(M - 1));
        std::vector<unsigned int> obs_seq(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) {
            obs_seq[static_cast<std::size_t>(t)] = obs_dist(rng);
        }

        HMMMat F(T, N);
        HMMMat B(T, N);
        HMMVec scales(T);

        std::vector<double> times(NUM_TRIALS);
        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            auto t0 = high_resolution_clock::now();
            hmm.forward(obs_seq, scales, F);
            hmm.backward(obs_seq, scales, B);
            double log_lik = hmm.likelihood(scales);
            auto t1 = high_resolution_clock::now();
            times[static_cast<std::size_t>(trial)] =
                duration_cast<microseconds>(t1 - t0).count() / 1000.0;
            if (trial == 0) {
                r.log_lik = log_lik; // HMMLib::likelihood() returns log-probability
            }
        }
        r.fb_ms = median(times);
        r.success = true;
    } catch (const std::exception &e) {
        std::cerr << "HMMLib error (N=" << N << " T=" << T << "): " << e.what() << "\n";
    }
    return r;
}

#endif // LIBHMM_BENCH_HAS_HMMLIB

// ============================================================
// GHMM — Discrete (optional; POSIX/macOS only)
// ============================================================

#ifdef LIBHMM_BENCH_HAS_GHMM

[[nodiscard]] BenchResult benchGHMM(int N, int T, RNG &rng) {
    BenchResult r{.library = "GHMM", .N = N, .T = T};
    const int M = DISCRETE_ALPHABET;
    try {
        // Build random parameters
        const double w = 1.0 / N;
        std::vector<std::vector<double>> A(static_cast<std::size_t>(N));
        for (auto &row : A) {
            row = randomRow(N, rng);
        }
        std::vector<std::vector<double>> B(static_cast<std::size_t>(N));
        for (auto &row : B) {
            row = randomRow(M, rng);
        }

        // Allocate GHMM model state arrays
        struct StateArrays {
            double *emit{nullptr};
            int *out_id{nullptr};
            double *out_a{nullptr};
            int *in_id{nullptr};
            double *in_a{nullptr};
        };
        std::vector<StateArrays> sa(static_cast<std::size_t>(N));
        auto *model_states = new ghmm_dstate[static_cast<std::size_t>(N)];

        for (int i = 0; i < N; ++i) {
            auto &s = sa[static_cast<std::size_t>(i)];
            s.emit = new double[static_cast<std::size_t>(M)];
            s.out_id = new int[static_cast<std::size_t>(N)];
            s.out_a = new double[static_cast<std::size_t>(N)];
            s.in_id = new int[static_cast<std::size_t>(N)];
            s.in_a = new double[static_cast<std::size_t>(N)];

            for (int k = 0; k < M; ++k) {
                s.emit[k] = B[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            }
            for (int j = 0; j < N; ++j) {
                s.out_id[j] = j;
                s.out_a[j] = A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                s.in_id[j] = j;
                s.in_a[j] = A[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)]; // reverse
            }

            auto &ms = model_states[i];
            ms.pi = w;
            ms.b = s.emit;
            ms.out_states = N;
            ms.out_a = s.out_a;
            ms.out_id = s.out_id;
            ms.in_states = N;
            ms.in_id = s.in_id;
            ms.in_a = s.in_a;
            ms.fix = 1;
        }

        auto *silent_arr = new int[static_cast<std::size_t>(N)]();

        ghmm_dmodel model{};
        model.N = N;
        model.M = M;
        model.s = model_states;
        model.prior = -1;
        model.model_type = 0;
        model.silent = silent_arr;

        // Observation sequence
        std::uniform_int_distribution<int> obs_dist(0, M - 1);
        auto *ghmm_seq = new int[static_cast<std::size_t>(T)];
        for (int t = 0; t < T; ++t) {
            ghmm_seq[t] = obs_dist(rng);
        }

        std::vector<double> times(NUM_TRIALS);
        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            double fwd_prob{};
            auto t0 = high_resolution_clock::now();
            if (ghmm_dmodel_logp(&model, ghmm_seq, T, &fwd_prob) != 0) {
                throw std::runtime_error("ghmm_dmodel_logp failed");
            }
            auto t1 = high_resolution_clock::now();
            times[static_cast<std::size_t>(trial)] =
                duration_cast<microseconds>(t1 - t0).count() / 1000.0;
            if (trial == 0) {
                r.log_lik = fwd_prob;
            }
        }
        r.fb_ms = median(times);
        r.success = true;

        // Cleanup
        delete[] ghmm_seq;
        delete[] silent_arr;
        for (int i = 0; i < N; ++i) {
            auto &s = sa[static_cast<std::size_t>(i)];
            delete[] s.emit;
            delete[] s.out_id;
            delete[] s.out_a;
            delete[] s.in_id;
            delete[] s.in_a;
        }
        delete[] model_states;

    } catch (const std::exception &e) {
        std::cerr << "GHMM error (N=" << N << " T=" << T << "): " << e.what() << "\n";
    }
    return r;
}

#endif // LIBHMM_BENCH_HAS_GHMM

// ============================================================
// Reporting
// ============================================================

/// Group of results for a single (N, T) cell
using ResultGroup = std::vector<BenchResult>;

/// Find a result by library name; returns nullptr if not present or failed
const BenchResult *findResult(const ResultGroup &g, const std::string &lib) {
    for (const auto &r : g) {
        if (r.library == lib) {
            return r.success ? &r : nullptr;
        }
    }
    return nullptr;
}

void printMainTable(const std::vector<ResultGroup> &rows) {
    constexpr int CW = 14; // column width

    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "MULTISTATE SCALING BENCHMARK — Forward-Backward, median of " << NUM_TRIALS
              << " trials\n";
    std::cout << "MaxReduce path active for N≥4 (pairwise for N<4)\n";
    std::cout << std::string(100, '=') << "\n";

    // Header
    std::cout << std::left << std::setw(4) << "N" << std::setw(7) << "T" << std::setw(12) << "Mode"
              << std::setw(CW) << "Gauss(ms)" << std::setw(CW) << "Disc(ms)";
#ifdef LIBHMM_BENCH_HAS_HMMLIB
    std::cout << std::setw(CW) << "HMMLib(ms)";
#endif
#ifdef LIBHMM_BENCH_HAS_GHMM
    std::cout << std::setw(CW) << "GHMM(ms)";
#endif
    std::cout << std::setw(CW) << "obs·N²/ms" << "\n";
    std::cout << std::string(100, '-') << "\n";

    int prev_N = -1;
    for (const auto &group : rows) {
        const BenchResult *gauss = findResult(group, "libhmm-Gauss");
        const BenchResult *disc = findResult(group, "libhmm-Disc");
#ifdef LIBHMM_BENCH_HAS_HMMLIB
        const BenchResult *hmmlib = findResult(group, "HMMLib");
#endif
#ifdef LIBHMM_BENCH_HAS_GHMM
        const BenchResult *ghmm = findResult(group, "GHMM");
#endif

        if (!gauss) {
            continue;
        }

        // Separator between N-groups
        if (gauss->N != prev_N && prev_N != -1) {
            std::cout << "\n";
        }
        prev_N = gauss->N;

        std::cout << std::left << std::setw(4) << gauss->N << std::setw(7) << gauss->T
                  << std::setw(12) << gauss->mode << std::fixed << std::setprecision(3);

        auto printCell = [&](const BenchResult *rp) {
            if (rp != nullptr && rp->fb_ms >= 0.0) {
                std::cout << std::setw(CW) << rp->fb_ms;
            } else {
                std::cout << std::setw(CW) << "N/A";
            }
        };

        printCell(gauss);
        printCell(disc);
#ifdef LIBHMM_BENCH_HAS_HMMLIB
        printCell(hmmlib);
#endif
#ifdef LIBHMM_BENCH_HAS_GHMM
        printCell(ghmm);
#endif

        // Normalized throughput: T × N² / time_ms (work units per ms)
        const double work = static_cast<double>(gauss->T) * static_cast<double>(gauss->N) *
                            static_cast<double>(gauss->N);
        const double norm = (gauss->fb_ms > 0.0) ? (work / gauss->fb_ms) : 0.0;
        std::cout << std::setw(CW) << std::setprecision(0) << norm << "\n";
    }

    std::cout << std::string(100, '=') << "\n";
}

void printScalingTable(const std::vector<ResultGroup> &rows) {
    // For each (library, T), show time(N=X) / time(N=2) as the scaling factor
    const std::vector<int> T_vals = {500, 2000, 10000};
    const std::vector<int> N_vals = {2, 4, 8, 16, 32};
    const std::vector<std::string> libs = {
        "libhmm-Gauss",
        "libhmm-Disc",
#ifdef LIBHMM_BENCH_HAS_HMMLIB
        "HMMLib",
#endif
#ifdef LIBHMM_BENCH_HAS_GHMM
        "GHMM",
#endif
    };

    std::cout << "\nN-SCALING RATIO (time(N) / time(N=2)); 1.0 = linear, "
                 "higher = super-linear cost\n";
    std::cout << "Ideal O(N²): ratio at N=4→4x, N=8→16x, N=16→64x, N=32→256x\n";
    std::cout << std::string(90, '-') << "\n";

    for (int T : T_vals) {
        std::cout << "\nT=" << T << "\n";
        std::cout << std::left << std::setw(15) << "Library";
        for (int N : N_vals) {
            std::cout << std::setw(10) << ("N=" + std::to_string(N));
        }
        std::cout << "\n" << std::string(65, '-') << "\n";

        for (const auto &lib : libs) {
            // Find baseline time at N=2
            double base_ms = -1.0;
            for (const auto &group : rows) {
                const BenchResult *rp = findResult(group, lib);
                if (rp && rp->N == 2 && rp->T == T) {
                    base_ms = rp->fb_ms;
                    break;
                }
            }
            if (base_ms <= 0.0) {
                continue;
            }

            std::cout << std::left << std::setw(15) << lib;
            for (int N : N_vals) {
                double this_ms = -1.0;
                for (const auto &group : rows) {
                    const BenchResult *rp = findResult(group, lib);
                    if (rp && rp->N == N && rp->T == T) {
                        this_ms = rp->fb_ms;
                        break;
                    }
                }
                if (this_ms >= 0.0) {
                    std::cout << std::fixed << std::setprecision(1) << std::setw(10)
                              << (this_ms / base_ms);
                } else {
                    std::cout << std::setw(10) << "N/A";
                }
            }
            std::cout << "\n";
        }
    }
    std::cout << std::string(90, '=') << "\n";
}

// ============================================================
// main
// ============================================================

int main() {
    std::cout << "Multistate Scaling Benchmark — libhmm v4.0.3 MaxReduce Path\n";
    std::cout << "==============================================================\n";
    std::cout << "Discrete alphabet: " << DISCRETE_ALPHABET << " symbols\n";
    std::cout << "Trials per cell:   " << NUM_TRIALS << " (median reported)\n";
    std::cout << "RNG seed:          42 (fixed for reproducibility)\n\n";
    std::cout << "Comparators:\n";
    std::cout << "  libhmm-Gauss  always (Gaussian emissions, SIMD + MaxReduce)\n";
    std::cout << "  libhmm-Disc   always (discrete lookup, MaxReduce recurrence)\n";
#ifdef LIBHMM_BENCH_HAS_HMMLIB
    std::cout << "  HMMLib        enabled\n";
#else
    std::cout << "  HMMLib        disabled\n";
#endif
#ifdef LIBHMM_BENCH_HAS_GHMM
    std::cout << "  GHMM          enabled\n";
#else
    std::cout << "  GHMM          disabled\n";
#endif

    const std::vector<int> N_vals = {2, 4, 8, 16, 32};
    const std::vector<int> T_vals = {500, 2000, 10000};

    RNG rng(42);

    std::vector<ResultGroup> all_rows;
    all_rows.reserve(N_vals.size() * T_vals.size());

    for (int N : N_vals) {
        for (int T : T_vals) {
            std::cout << "  N=" << std::setw(2) << N << "  T=" << std::setw(6) << T << " ...";
            std::cout.flush();

            ResultGroup group;
            group.push_back(benchGaussian(N, T, rng));
            group.push_back(benchLibhmmDiscrete(N, T, rng));
#ifdef LIBHMM_BENCH_HAS_HMMLIB
            group.push_back(benchHMMLib(N, T, rng));
#endif
#ifdef LIBHMM_BENCH_HAS_GHMM
            group.push_back(benchGHMM(N, T, rng));
#endif
            all_rows.push_back(std::move(group));
            std::cout << " done\n";
        }
    }

    printMainTable(all_rows);
    printScalingTable(all_rows);

    return 0;
}
