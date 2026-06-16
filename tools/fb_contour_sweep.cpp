#include "libhmm/hmm.h"
#include "libhmm/distributions/distributions.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
namespace fs = std::filesystem;

namespace {

constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();
[[maybe_unused]] constexpr std::size_t FB_MAX_REDUCE_FORCE_PAIRWISE_MAX_STATES = 2;
volatile double g_sink_double = 0.0;

struct Config {
    int n;
    int t;
};

struct Timings {
    double transition_ms = 0.0;
    double obs_copy_ms = 0.0;
    double emission_ms = 0.0;
    double alloc_ms = 0.0;
    double forward_ms = 0.0;
    double backward_ms = 0.0;
    double reduction_ms = 0.0;
    double total_ms = 0.0;
};

double elapsed_ms(const Clock::time_point start) {
    return Millis(Clock::now() - start).count();
}

bool should_use_max_reduce(const std::size_t n, const std::size_t t) noexcept {
#if defined(LIBHMM_EXPERIMENT_FB_MAX_REDUCE)
    (void)n;
    (void)t;
    return true;
#elif defined(LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR)
    (void)t;
    return n > FB_MAX_REDUCE_FORCE_PAIRWISE_MAX_STATES;
#else
    (void)n;
    (void)t;
    return false;
#endif
}

double log_sum_exp_pairwise(const double a, const double b) noexcept {
    if (a == LOG_ZERO) {
        return b;
    }
    if (b == LOG_ZERO) {
        return a;
    }
    if (a > b) {
        return a + std::log1p(std::exp(b - a));
    }
    return b + std::log1p(std::exp(a - b));
}

template <typename T>
double median(std::vector<T> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return static_cast<double>(values[values.size() / 2]);
}

std::unique_ptr<Hmm> make_hmm(const int n) {
    auto hmm = std::make_unique<Hmm>(n);
    Matrix trans(n, n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            trans(i, j) = 0.1 + 0.8 * (0.5 + 0.5 * std::sin(i * 0.7 + j * 1.3));
            sum += trans(i, j);
        }
        for (int j = 0; j < n; ++j) {
            trans(i, j) /= sum;
        }
    }
    hmm->setTrans(trans);

    Vector pi(n);
    for (int i = 0; i < n; ++i) {
        pi(i) = 1.0 / static_cast<double>(n);
    }
    hmm->setPi(pi);

    for (int i = 0; i < n; ++i) {
        hmm->setDistribution(i, std::make_unique<GaussianDistribution>(i * 2.0, 1.0));
    }
    return hmm;
}

ObservationSet make_obs(const int t, const int n) {
    ObservationSet obs(t);
    for (int i = 0; i < t; ++i) {
        obs(i) = std::sin(i * 0.1) * static_cast<double>(n);
    }
    return obs;
}

Timings run_once(const Hmm &hmm, const ObservationSet &obs) {
    Timings out;
    const std::size_t n = hmm.getNumStatesModern();
    const std::size_t t = obs.size();

    auto total_start = Clock::now();

    auto stage_start = Clock::now();
    Matrix log_trans(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double a = hmm.getTrans()(i, j);
            log_trans(i, j) = (a > 0.0) ? std::log(a) : LOG_ZERO;
        }
    }
    out.transition_ms = elapsed_ms(stage_start);

    stage_start = Clock::now();
    std::vector<double> obs_copy(t);
    for (std::size_t i = 0; i < t; ++i) {
        obs_copy[i] = obs(i);
    }
    const std::span<const double> obs_span(obs_copy.data(), t);
    out.obs_copy_ms = elapsed_ms(stage_start);

    stage_start = Clock::now();
    std::vector<double> log_emit_buf(n * t);
    for (std::size_t i = 0; i < n; ++i) {
        hmm.getDistribution(i).getBatchLogProbabilities(
            obs_span, std::span<double>(log_emit_buf.data() + i * t, t));
    }
    out.emission_ms = elapsed_ms(stage_start);

    stage_start = Clock::now();
    Matrix log_alpha(t, n);
    Matrix log_beta(t, n);
    out.alloc_ms = elapsed_ms(stage_start);

    stage_start = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        const double pi = hmm.getPi()(i);
        const double log_pi = (pi > 0.0) ? std::log(pi) : LOG_ZERO;
        log_alpha(0, i) = log_pi + log_emit_buf[i * t];
    }
    const bool use_max_reduce = should_use_max_reduce(n, t);
    if (use_max_reduce) {
        for (std::size_t ti = 1; ti < t; ++ti) {
            for (std::size_t j = 0; j < n; ++j) {
                double max_term = LOG_ZERO;
                for (std::size_t i = 0; i < n; ++i) {
                    const double term = log_alpha(ti - 1, i) + log_trans(i, j);
                    if (term > max_term) {
                        max_term = term;
                    }
                }
                double log_sum = LOG_ZERO;
                if (std::isfinite(max_term)) {
                    double scaled_sum = 0.0;
                    for (std::size_t i = 0; i < n; ++i) {
                        const double term = log_alpha(ti - 1, i) + log_trans(i, j);
                        if (std::isfinite(term)) {
                            scaled_sum += std::exp(term - max_term);
                        }
                    }
                    if (scaled_sum > 0.0) {
                        log_sum = max_term + std::log(scaled_sum);
                    }
                }
                log_alpha(ti, j) = log_emit_buf[j * t + ti] + log_sum;
            }
        }
    } else {
        for (std::size_t ti = 1; ti < t; ++ti) {
            for (std::size_t j = 0; j < n; ++j) {
                double log_sum = LOG_ZERO;
                for (std::size_t i = 0; i < n; ++i) {
                    log_sum = log_sum_exp_pairwise(log_sum, log_alpha(ti - 1, i) + log_trans(i, j));
                }
                log_alpha(ti, j) = log_emit_buf[j * t + ti] + log_sum;
            }
        }
    }
    out.forward_ms = elapsed_ms(stage_start);

    stage_start = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        log_beta(t - 1, i) = 0.0;
    }
    if (t > 1) {
        if (use_max_reduce) {
            for (std::size_t ti = t - 2;; --ti) {
                for (std::size_t i = 0; i < n; ++i) {
                    double max_term = LOG_ZERO;
                    for (std::size_t j = 0; j < n; ++j) {
                        const double term =
                            log_trans(i, j) + log_emit_buf[j * t + (ti + 1)] + log_beta(ti + 1, j);
                        if (term > max_term) {
                            max_term = term;
                        }
                    }
                    double log_sum = LOG_ZERO;
                    if (std::isfinite(max_term)) {
                        double scaled_sum = 0.0;
                        for (std::size_t j = 0; j < n; ++j) {
                            const double term = log_trans(i, j) + log_emit_buf[j * t + (ti + 1)] +
                                                log_beta(ti + 1, j);
                            if (std::isfinite(term)) {
                                scaled_sum += std::exp(term - max_term);
                            }
                        }
                        if (scaled_sum > 0.0) {
                            log_sum = max_term + std::log(scaled_sum);
                        }
                    }
                    log_beta(ti, i) = log_sum;
                }
                if (ti == 0) {
                    break;
                }
            }
        } else {
            for (std::size_t ti = t - 2;; --ti) {
                for (std::size_t i = 0; i < n; ++i) {
                    double log_sum = LOG_ZERO;
                    for (std::size_t j = 0; j < n; ++j) {
                        log_sum = log_sum_exp_pairwise(log_sum, log_trans(i, j) +
                                                                    log_emit_buf[j * t + (ti + 1)] +
                                                                    log_beta(ti + 1, j));
                    }
                    log_beta(ti, i) = log_sum;
                }
                if (ti == 0) {
                    break;
                }
            }
        }
    }
    out.backward_ms = elapsed_ms(stage_start);

    stage_start = Clock::now();
    double log_probability = LOG_ZERO;
    for (std::size_t i = 0; i < n; ++i) {
        log_probability = log_sum_exp_pairwise(log_probability, log_alpha(t - 1, i));
    }
    out.reduction_ms = elapsed_ms(stage_start);
    g_sink_double += log_probability;

    out.total_ms = elapsed_ms(total_start);
    return out;
}

Timings profile_config(const Hmm &hmm, const ObservationSet &obs, const int runs,
                       const int warmup) {
    std::vector<double> transition_ms;
    std::vector<double> obs_copy_ms;
    std::vector<double> emission_ms;
    std::vector<double> alloc_ms;
    std::vector<double> forward_ms;
    std::vector<double> backward_ms;
    std::vector<double> reduction_ms;
    std::vector<double> total_ms;

    transition_ms.reserve(static_cast<std::size_t>(runs));
    obs_copy_ms.reserve(static_cast<std::size_t>(runs));
    emission_ms.reserve(static_cast<std::size_t>(runs));
    alloc_ms.reserve(static_cast<std::size_t>(runs));
    forward_ms.reserve(static_cast<std::size_t>(runs));
    backward_ms.reserve(static_cast<std::size_t>(runs));
    reduction_ms.reserve(static_cast<std::size_t>(runs));
    total_ms.reserve(static_cast<std::size_t>(runs));

    for (int iter = 0; iter < warmup + runs; ++iter) {
        const Timings t = run_once(hmm, obs);
        if (iter >= warmup) {
            transition_ms.push_back(t.transition_ms);
            obs_copy_ms.push_back(t.obs_copy_ms);
            emission_ms.push_back(t.emission_ms);
            alloc_ms.push_back(t.alloc_ms);
            forward_ms.push_back(t.forward_ms);
            backward_ms.push_back(t.backward_ms);
            reduction_ms.push_back(t.reduction_ms);
            total_ms.push_back(t.total_ms);
        }
    }

    return {
        median(transition_ms), median(obs_copy_ms), median(emission_ms),  median(alloc_ms),
        median(forward_ms),    median(backward_ms), median(reduction_ms), median(total_ms),
    };
}

int parse_positive_int(const char *value, const char *name) {
    try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) {
            throw std::invalid_argument("non-positive");
        }
        return parsed;
    } catch (...) {
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + value);
    }
}

std::string mode_name() {
#if defined(LIBHMM_EXPERIMENT_FB_MAX_REDUCE)
    return "max_reduce";
#elif defined(LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR)
    return "adaptive_static_v1";
#else
    return "pairwise";
#endif
}

} // namespace

int main(int argc, char *argv[]) {
    int runs = 5;
    int warmup = 1;

    fs::path output_path =
        fs::path("benchmark-analysis") / ("fb_contour_sweep_" + mode_name() + ".csv");

    if (argc >= 2) {
        output_path = argv[1];
    }
    try {
        if (argc >= 3) {
            runs = parse_positive_int(argv[2], "runs");
        }
        if (argc >= 4) {
            warmup = parse_positive_int(argv[3], "warmup");
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    if (argc > 4) {
        std::cerr << "Usage:\n";
        std::cerr << "  fb_contour_sweep [output_csv] [runs] [warmup]\n";
        return 1;
    }

    try {
        const std::vector<Config> configs = {
            {2, 1000},   {2, 10000}, {2, 100000}, {2, 1000000}, {4, 1000},  {4, 10000},
            {4, 100000}, {8, 1000},  {8, 5000},   {8, 10000},   {16, 1000}, {16, 2000},
            {16, 5000},  {32, 500},  {32, 1000},  {32, 2000},   {64, 200},  {64, 500},
            {64, 1000},  {128, 100}, {128, 250},  {128, 500},
        };

        const fs::path output_dir = output_path.parent_path();
        if (!output_dir.empty()) {
            fs::create_directories(output_dir);
        }
        std::ofstream csv(output_path);
        if (!csv) {
            std::cerr << "Failed to open output file: " << output_path << "\n";
            return 1;
        }

        csv << "mode,n,t,runs,warmup,recurrence_work,emission_work,transition_ms,obs_copy_ms,"
               "emission_ms,alloc_ms,forward_ms,backward_ms,reduction_ms,total_ms\n";

        std::cout << "libhmm FB contour sweep\n";
        std::cout << "Mode: " << mode_name() << "\n";
        std::cout << "Runs: " << runs << " (warmup " << warmup << ")\n";
        std::cout << "Output: " << output_path << "\n\n";
        std::cout << std::fixed << std::setprecision(3);

        for (const auto &cfg : configs) {
            auto hmm = make_hmm(cfg.n);
            auto obs = make_obs(cfg.t, cfg.n);
            const Timings timed = profile_config(*hmm, obs, runs, warmup);

            const std::uint64_t recurrence_work =
                static_cast<std::uint64_t>(cfg.n) * cfg.n * static_cast<std::uint64_t>(cfg.t - 1);
            const std::uint64_t emission_work =
                static_cast<std::uint64_t>(cfg.n) * static_cast<std::uint64_t>(cfg.t);

            csv << mode_name() << "," << cfg.n << "," << cfg.t << "," << runs << "," << warmup
                << "," << recurrence_work << "," << emission_work << "," << timed.transition_ms
                << "," << timed.obs_copy_ms << "," << timed.emission_ms << "," << timed.alloc_ms
                << "," << timed.forward_ms << "," << timed.backward_ms << "," << timed.reduction_ms
                << "," << timed.total_ms << "\n";

            const double recurrence_pct =
                (timed.total_ms > 0.0)
                    ? ((timed.forward_ms + timed.backward_ms) * 100.0 / timed.total_ms)
                    : 0.0;
            std::cout << "N=" << std::setw(3) << cfg.n << " T=" << std::setw(8) << cfg.t
                      << " total=" << std::setw(9) << timed.total_ms << " ms"
                      << " recur=" << std::setw(6) << recurrence_pct << "%\n";
        }

        csv.close();
        if (g_sink_double == 42.0) {
            std::cout << "sink=" << g_sink_double << "\n";
        }
        std::cout << "\nDone.\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
