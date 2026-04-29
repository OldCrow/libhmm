#include "libhmm/hmm.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/math/constants.h"
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
constexpr std::size_t FB_MAX_REDUCE_FORCE_PAIRWISE_MAX_STATES = 2;
volatile double g_sink_double = 0.0;
volatile int g_sink_int = 0;

struct Config {
    int num_states;
    int sequence_length;
};

struct ForwardBreakdown {
    double transition_ms = 0.0;
    double obs_copy_ms = 0.0;
    double emission_ms = 0.0;
    double buffer_alloc_ms = 0.0;
    double forward_ms = 0.0;
    double backward_ms = 0.0;
    double reduction_ms = 0.0;
};

struct ViterbiBreakdown {
    double transition_ms = 0.0;
    double emission_ms = 0.0;
    double emission_relayout_ms = 0.0;
    double buffer_alloc_ms = 0.0;
    double recursion_ms = 0.0;
    double backtrack_ms = 0.0;
};

template <typename T>
double median(std::vector<T> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return static_cast<double>(values[values.size() / 2]);
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

double elapsed_ms(const Clock::time_point start) {
    return Millis(Clock::now() - start).count();
}

double log_sum_exp(const double a, const double b) noexcept {
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

ForwardBreakdown profile_forward_backward(const Hmm &hmm, const ObservationSet &obs, const int warmup,
                                          const int runs) {
    const std::size_t n = static_cast<std::size_t>(hmm.getNumStates());
    const std::size_t t = obs.size();

    std::vector<double> transition_ms;
    std::vector<double> obs_copy_ms;
    std::vector<double> emission_ms;
    std::vector<double> buffer_alloc_ms;
    std::vector<double> forward_ms;
    std::vector<double> backward_ms;
    std::vector<double> reduction_ms;

    transition_ms.reserve(static_cast<std::size_t>(runs));
    obs_copy_ms.reserve(static_cast<std::size_t>(runs));
    emission_ms.reserve(static_cast<std::size_t>(runs));
    buffer_alloc_ms.reserve(static_cast<std::size_t>(runs));
    forward_ms.reserve(static_cast<std::size_t>(runs));
    backward_ms.reserve(static_cast<std::size_t>(runs));
    reduction_ms.reserve(static_cast<std::size_t>(runs));

    for (int iter = 0; iter < warmup + runs; ++iter) {
        auto stage_start = Clock::now();
        Matrix log_trans(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double a = hmm.getTrans()(i, j);
                log_trans(i, j) = (a > 0.0) ? std::log(a) : LOG_ZERO;
            }
        }
        const double trans_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        std::vector<double> obs_copy(t);
        for (std::size_t i = 0; i < t; ++i) {
            obs_copy[i] = obs(i);
        }
        const std::span<const double> obs_span(obs_copy.data(), t);
        const double obs_copy_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        std::vector<double> log_emit_buf(n * t);
        for (std::size_t i = 0; i < n; ++i) {
            hmm.getDistribution(i).getBatchLogProbabilities(
                obs_span, std::span<double>(log_emit_buf.data() + i * t, t));
        }
        const double emission_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        Matrix log_alpha(t, n);
        Matrix log_beta(t, n);
        const double buffer_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            const double pi = hmm.getPi()(i);
            const double log_pi = (pi > 0.0) ? std::log(pi) : LOG_ZERO;
            log_alpha(0, i) = log_pi + log_emit_buf[i * t];
        }
        const bool use_max_reduce = should_use_max_reduce(n, t);
        for (std::size_t ti = 1; ti < t; ++ti) {
            for (std::size_t j = 0; j < n; ++j) {
                double log_sum = LOG_ZERO;
                if (use_max_reduce) {
                    double max_term = LOG_ZERO;
                    for (std::size_t i = 0; i < n; ++i) {
                        const double term = log_alpha(ti - 1, i) + log_trans(i, j);
                        if (term > max_term) {
                            max_term = term;
                        }
                    }
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
                } else {
                    for (std::size_t i = 0; i < n; ++i) {
                        log_sum = log_sum_exp(log_sum, log_alpha(ti - 1, i) + log_trans(i, j));
                    }
                }
                log_alpha(ti, j) = log_emit_buf[j * t + ti] + log_sum;
            }
        }
        const double forward_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            log_beta(t - 1, i) = 0.0;
        }
        if (t > 1) {
            for (std::size_t ti = t - 2;; --ti) {
                for (std::size_t i = 0; i < n; ++i) {
                    double log_sum = LOG_ZERO;
                    if (use_max_reduce) {
                        double max_term = LOG_ZERO;
                        for (std::size_t j = 0; j < n; ++j) {
                            const double term = log_trans(i, j) + log_emit_buf[j * t + (ti + 1)] +
                                                log_beta(ti + 1, j);
                            if (term > max_term) {
                                max_term = term;
                            }
                        }
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
                    } else {
                        for (std::size_t j = 0; j < n; ++j) {
                            log_sum = log_sum_exp(log_sum, log_trans(i, j) +
                                                               log_emit_buf[j * t + (ti + 1)] +
                                                               log_beta(ti + 1, j));
                        }
                    }
                    log_beta(ti, i) = log_sum;
                }
                if (ti == 0) {
                    break;
                }
            }
        }
        const double backward_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        double log_probability = LOG_ZERO;
        for (std::size_t i = 0; i < n; ++i) {
            log_probability = log_sum_exp(log_probability, log_alpha(t - 1, i));
        }
        const double reduction_time = elapsed_ms(stage_start);
        g_sink_double += log_probability;

        if (iter >= warmup) {
            transition_ms.push_back(trans_time);
            obs_copy_ms.push_back(obs_copy_time);
            emission_ms.push_back(emission_time);
            buffer_alloc_ms.push_back(buffer_time);
            forward_ms.push_back(forward_time);
            backward_ms.push_back(backward_time);
            reduction_ms.push_back(reduction_time);
        }
    }

    return {
        median(transition_ms), median(obs_copy_ms), median(emission_ms),   median(buffer_alloc_ms),
        median(forward_ms),    median(backward_ms), median(reduction_ms),
    };
}

ViterbiBreakdown profile_viterbi(const Hmm &hmm, const ObservationSet &obs, const int warmup,
                                 const int runs) {
    const std::size_t n = static_cast<std::size_t>(hmm.getNumStates());
    const std::size_t t = obs.size();

    std::vector<double> transition_ms;
    std::vector<double> emission_ms;
    std::vector<double> emission_relayout_ms;
    std::vector<double> buffer_alloc_ms;
    std::vector<double> recursion_ms;
    std::vector<double> backtrack_ms;

    transition_ms.reserve(static_cast<std::size_t>(runs));
    emission_ms.reserve(static_cast<std::size_t>(runs));
    emission_relayout_ms.reserve(static_cast<std::size_t>(runs));
    buffer_alloc_ms.reserve(static_cast<std::size_t>(runs));
    recursion_ms.reserve(static_cast<std::size_t>(runs));
    backtrack_ms.reserve(static_cast<std::size_t>(runs));

    for (int iter = 0; iter < warmup + runs; ++iter) {
        auto stage_start = Clock::now();
        Matrix log_trans(n, n);
        Matrix log_trans_t(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double a = hmm.getTrans()(i, j);
                const double log_a = (a > 0.0) ? std::log(a) : LOG_ZERO;
                log_trans(i, j) = log_a;
                log_trans_t(j, i) = log_a;
            }
        }
        const double trans_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        std::vector<double> log_emit_buf(n * t);
        const std::span<const double> obs_span(obs.data(), t);
        for (std::size_t i = 0; i < n; ++i) {
            hmm.getDistribution(i).getBatchLogProbabilities(
                obs_span, std::span<double>(log_emit_buf.data() + i * t, t));
        }
        const double emission_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        std::vector<double> log_emit_by_time(n * t);
        for (std::size_t i = 0; i < n; ++i) {
            const double *state_row = log_emit_buf.data() + i * t;
            for (std::size_t ti = 0; ti < t; ++ti) {
                log_emit_by_time[ti * n + i] = state_row[ti];
            }
        }
        const double relayout_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        Matrix log_delta(t, n);
        std::vector<int> psi(t * n, 0);
        std::vector<int> sequence(t, 0);
        const double buffer_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        const double *log_trans_t_data = log_trans_t.data();
        const double *log_emit_by_time_data = log_emit_by_time.data();
        double *log_delta_data = log_delta.data();

        const double *emit_row_0 = log_emit_by_time_data;
        for (std::size_t i = 0; i < n; ++i) {
            const double pi = hmm.getPi()(i);
            const double log_pi = (pi > 0.0) ? std::log(pi) : LOG_ZERO;
            log_delta_data[i] = log_pi + emit_row_0[i];
        }

        for (std::size_t ti = 1; ti < t; ++ti) {
            const double *prev_delta_row = log_delta_data + (ti - 1) * n;
            double *delta_row = log_delta_data + ti * n;
            const double *emit_row = log_emit_by_time_data + ti * n;
            for (std::size_t j = 0; j < n; ++j) {
                double max_val = LOG_ZERO;
                int max_from = 0;
                const double *trans_col = log_trans_t_data + j * n;
                for (std::size_t i = 0; i < n; ++i) {
                    const double value = prev_delta_row[i] + trans_col[i];
                    if (value > max_val) {
                        max_val = value;
                        max_from = static_cast<int>(i);
                    }
                }
                delta_row[j] = max_val + emit_row[j];
                psi[ti * n + j] = max_from;
            }
        }

        double best_val = LOG_ZERO;
        int best_last = 0;
        const double *final_delta_row = log_delta_data + (t - 1) * n;
        for (std::size_t i = 0; i < n; ++i) {
            if (final_delta_row[i] > best_val) {
                best_val = final_delta_row[i];
                best_last = static_cast<int>(i);
            }
        }
        sequence[t - 1] = best_last;
        const double recursion_time = elapsed_ms(stage_start);

        stage_start = Clock::now();
        if (t > 1) {
            for (std::size_t ti = t - 2;; --ti) {
                sequence[ti] = psi[(ti + 1) * n + static_cast<std::size_t>(sequence[ti + 1])];
                if (ti == 0) {
                    break;
                }
            }
        }
        const double backtrack_time = elapsed_ms(stage_start);
        g_sink_double += best_val;
        g_sink_int += sequence[0];

        if (iter >= warmup) {
            transition_ms.push_back(trans_time);
            emission_ms.push_back(emission_time);
            emission_relayout_ms.push_back(relayout_time);
            buffer_alloc_ms.push_back(buffer_time);
            recursion_ms.push_back(recursion_time);
            backtrack_ms.push_back(backtrack_time);
        }
    }

    return {
        median(transition_ms),      median(emission_ms), median(emission_relayout_ms),
        median(buffer_alloc_ms),    median(recursion_ms), median(backtrack_ms),
    };
}

std::size_t estimate_forward_working_set_bytes(const std::size_t n, const std::size_t t) {
    const std::size_t doubles = (n * n) + (3 * n * t) + t;
    return doubles * sizeof(double);
}

std::size_t estimate_viterbi_working_set_bytes(const std::size_t n, const std::size_t t) {
    const std::size_t double_count = (2 * n * n) + (3 * n * t);
    const std::size_t int_count = (2 * n * t);
    return double_count * sizeof(double) + int_count * sizeof(int);
}

double bytes_to_mib(const std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_phase(const std::string &label, const double value_ms, const double total_ms) {
    const double pct = (total_ms > 0.0) ? (100.0 * value_ms / total_ms) : 0.0;
    std::cout << "  " << std::left << std::setw(28) << label << std::right << std::setw(10)
              << value_ms << " ms  " << std::setw(6) << pct << "%\n";
}

int parse_positive_int(const char *value, const char *arg_name) {
    try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) {
            throw std::invalid_argument("non-positive");
        }
        return parsed;
    } catch (...) {
        throw std::invalid_argument(std::string("Invalid ") + arg_name + ": " + value);
    }
}

} // namespace

int main(int argc, char *argv[]) {
    std::vector<Config> configs = {
        {8, 1000},
        {32, 2000},
        {64, 1000},
    };

    int warmup = 2;
    int runs = 8;

    if (argc == 3 || argc == 4 || argc == 5) {
        const int n = parse_positive_int(argv[1], "N");
        const int t = parse_positive_int(argv[2], "T");
        configs = {{n, t}};
        if (argc >= 4) {
            runs = parse_positive_int(argv[3], "runs");
        }
        if (argc == 5) {
            warmup = parse_positive_int(argv[4], "warmup");
        }
    } else if (argc != 1) {
        std::cerr << "Usage:\n";
        std::cerr << "  hotspot_breakdown\n";
        std::cerr << "  hotspot_breakdown <N> <T> [runs] [warmup]\n";
        return 1;
    }

    std::cout << "libhmm Hotspot Breakdown Tool\n";
    std::cout << "============================\n";
    std::cout << "Median of " << runs << " timed runs (" << warmup << " warmup).\n\n";
#if defined(LIBHMM_EXPERIMENT_FB_MAX_REDUCE)
    std::cout << "Forward-Backward accumulation mode: max-then-reduce (experimental)\n\n";
#elif defined(LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR)
    std::cout << "Forward-Backward accumulation mode: static adaptive selector (stage-1)\n\n";
#else
    std::cout << "Forward-Backward accumulation mode: pairwise logSumExp (control)\n\n";
#endif

    std::cout << std::fixed << std::setprecision(3);

    for (const auto &cfg : configs) {
        auto hmm = make_hmm(cfg.num_states);
        auto obs = make_obs(cfg.sequence_length, cfg.num_states);

        const auto fb = profile_forward_backward(*hmm, obs, warmup, runs);
        const auto vt = profile_viterbi(*hmm, obs, warmup, runs);

        const double fb_total = fb.transition_ms + fb.obs_copy_ms + fb.emission_ms +
                                fb.buffer_alloc_ms + fb.forward_ms + fb.backward_ms +
                                fb.reduction_ms;
        const double vt_total = vt.transition_ms + vt.emission_ms + vt.emission_relayout_ms +
                                vt.buffer_alloc_ms + vt.recursion_ms + vt.backtrack_ms;

        const std::size_t n = static_cast<std::size_t>(cfg.num_states);
        const std::size_t t = static_cast<std::size_t>(cfg.sequence_length);
        const std::uint64_t emission_work = static_cast<std::uint64_t>(n) * t;
        const std::uint64_t recurrence_work =
            (t > 0) ? static_cast<std::uint64_t>(n) * n * (t - 1) : 0ULL;

        std::cout << "Config: N=" << cfg.num_states << ", T=" << cfg.sequence_length << "\n";
        std::cout << "  Estimated recurrence work per pass: "
                  << static_cast<double>(recurrence_work) / 1.0e6 << " M (N^2*(T-1))\n";
        std::cout << "  Emission evaluations per pass:      "
                  << static_cast<double>(emission_work) / 1.0e6 << " M (N*T)\n";

        std::cout << "\nForward-Backward phase breakdown:\n";
        print_phase("Transition log precompute", fb.transition_ms, fb_total);
        print_phase("Observation copy", fb.obs_copy_ms, fb_total);
        print_phase("Emission batch eval", fb.emission_ms, fb_total);
        print_phase("Alpha/Beta buffer alloc", fb.buffer_alloc_ms, fb_total);
        print_phase("Forward recursion", fb.forward_ms, fb_total);
        print_phase("Backward recursion", fb.backward_ms, fb_total);
        print_phase("Final log-sum-exp reduce", fb.reduction_ms, fb_total);
        std::cout << "  " << std::left << std::setw(28) << "TOTAL" << std::right << std::setw(10)
                  << fb_total << " ms\n";

        std::cout << "  Estimated FB working set: "
                  << bytes_to_mib(estimate_forward_working_set_bytes(n, t)) << " MiB\n";

        std::cout << "\nViterbi phase breakdown:\n";
        print_phase("Transition log precompute", vt.transition_ms, vt_total);
        print_phase("Emission batch eval", vt.emission_ms, vt_total);
        print_phase("Emission relayout (T-major)", vt.emission_relayout_ms, vt_total);
        print_phase("Delta/Psi buffer alloc", vt.buffer_alloc_ms, vt_total);
        print_phase("Viterbi recursion", vt.recursion_ms, vt_total);
        print_phase("Backtrack", vt.backtrack_ms, vt_total);
        std::cout << "  " << std::left << std::setw(28) << "TOTAL" << std::right << std::setw(10)
                  << vt_total << " ms\n";

        std::cout << "  Estimated Viterbi working set: "
                  << bytes_to_mib(estimate_viterbi_working_set_bytes(n, t)) << " MiB\n";
        std::cout << "\n------------------------------------------------------------\n\n";
    }

    if (g_sink_int == 42) {
        std::cout << "sink=" << g_sink_double << "\n";
    }

    return 0;
}
