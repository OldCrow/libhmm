#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/chi_squared_distribution.h"
#include "libhmm/distributions/exponential_distribution.h"
#include "libhmm/distributions/gamma_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/log_normal_distribution.h"
#include "libhmm/distributions/poisson_distribution.h"
#include "libhmm/distributions/student_t_distribution.h"
#include "libhmm/distributions/uniform_distribution.h"
#include "libhmm/hmm.h"

#include "StochHMMlib.h"
#include "PDF.h"

namespace fs = std::filesystem;

namespace {

constexpr double kWarnAbsLogDiff = 1e-8;
constexpr double kBlockAbsLogDiff = 1e-6;
constexpr double kTargetTimingSampleMs = 5.0;
constexpr int kMaxTimingBatchIterations = 1000000;

enum class DistributionKind {
    Gaussian,
    Exponential,
    Poisson,
    Gamma,
    LogNormal,
    StudentT,
    ChiSquared,
    Uniform
};
enum class GateStatus { Pass, Warn, Block };

struct CliOptions {
    bool include_1e6 = false;
    int repeats = 3;
    int warmup = 1;
    bool hmm_stage = true;
    fs::path output_dir;
    bool output_dir_explicit = false;
    bool show_help = false;
};

struct DistributionCase {
    DistributionKind kind;
    std::string distribution;
    std::string case_name;
    double p1;
    double p2;
    std::string parameter_summary;
};

struct DistributionEvalResult {
    bool success = false;
    std::string error;
    double time_ms = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> log_values;
};

struct AccuracyMetrics {
    double max_abs = std::numeric_limits<double>::infinity();
    double max_rel = std::numeric_limits<double>::infinity();
};

struct DistributionComparisonRow {
    std::string distribution;
    std::string case_name;
    int size = 0;
    double libhmm_time_ms = std::numeric_limits<double>::quiet_NaN();
    double stoch_time_ms = std::numeric_limits<double>::quiet_NaN();
    double libhmm_throughput = std::numeric_limits<double>::quiet_NaN();
    double stoch_throughput = std::numeric_limits<double>::quiet_NaN();
    double speedup_stoch_over_libhmm = std::numeric_limits<double>::quiet_NaN();
    double speedup_libhmm_over_stoch = std::numeric_limits<double>::quiet_NaN();
    double max_abs_log_diff = std::numeric_limits<double>::infinity();
    double max_rel_log_diff = std::numeric_limits<double>::infinity();
    GateStatus gate = GateStatus::Block;
    bool success = false;
    std::string notes;
};

struct Phase1AggregateRow {
    std::string distribution;
    int size = 0;
    GateStatus worst_gate = GateStatus::Block;
    double max_abs_log_diff = std::numeric_limits<double>::infinity();
    double mean_speedup_stoch_over_libhmm = std::numeric_limits<double>::quiet_NaN();
    double mean_speedup_libhmm_over_stoch = std::numeric_limits<double>::quiet_NaN();
};

struct DistributionGateSummary {
    std::string distribution;
    GateStatus worst_gate = GateStatus::Pass;
    double max_abs_log_diff = 0.0;
    double mean_speedup_stoch_over_libhmm = std::numeric_limits<double>::quiet_NaN();
    double mean_speedup_libhmm_over_stoch = std::numeric_limits<double>::quiet_NaN();
    int warn_count = 0;
    int block_count = 0;
    int total_rows = 0;
};

struct CanonicalHMMSpec {
    DistributionKind kind;
    std::string distribution;
    double state0_p1;
    double state0_p2;
    double state1_p1;
    double state1_p2;
    std::string parameter_summary;
};

struct HMMRunResult {
    bool success = false;
    std::string error;
    double forward_ms = std::numeric_limits<double>::quiet_NaN();
    double viterbi_ms = std::numeric_limits<double>::quiet_NaN();
    double forward_throughput = std::numeric_limits<double>::quiet_NaN();
    double log_likelihood = std::numeric_limits<double>::quiet_NaN();
};

struct HMMComparisonRow {
    std::string distribution;
    int size = 0;
    double libhmm_forward_ms = std::numeric_limits<double>::quiet_NaN();
    double stoch_forward_ms = std::numeric_limits<double>::quiet_NaN();
    double libhmm_viterbi_ms = std::numeric_limits<double>::quiet_NaN();
    double stoch_viterbi_ms = std::numeric_limits<double>::quiet_NaN();
    double libhmm_forward_throughput = std::numeric_limits<double>::quiet_NaN();
    double stoch_forward_throughput = std::numeric_limits<double>::quiet_NaN();
    double speedup_stoch_over_libhmm = std::numeric_limits<double>::quiet_NaN();
    double speedup_libhmm_over_stoch = std::numeric_limits<double>::quiet_NaN();
    double libhmm_log_likelihood = std::numeric_limits<double>::quiet_NaN();
    double stoch_log_likelihood = std::numeric_limits<double>::quiet_NaN();
    double abs_log_likelihood_diff = std::numeric_limits<double>::infinity();
    bool success = false;
    std::string notes;
};

std::string gateToString(GateStatus status) {
    switch (status) {
        case GateStatus::Pass:
            return "PASS";
        case GateStatus::Warn:
            return "WARN";
        case GateStatus::Block:
            return "BLOCK";
    }
    return "BLOCK";
}

int gateSeverity(GateStatus status) {
    switch (status) {
        case GateStatus::Pass:
            return 0;
        case GateStatus::Warn:
            return 1;
        case GateStatus::Block:
            return 2;
    }
    return 2;
}

GateStatus worstGate(GateStatus a, GateStatus b) {
    return gateSeverity(a) >= gateSeverity(b) ? a : b;
}

GateStatus gateFromAbsDiff(double abs_diff) {
    if (!std::isfinite(abs_diff)) {
        return GateStatus::Block;
    }
    if (abs_diff > kBlockAbsLogDiff) {
        return GateStatus::Block;
    }
    if (abs_diff > kWarnAbsLogDiff) {
        return GateStatus::Warn;
    }
    return GateStatus::Pass;
}

void printUsage(const char *argv0) {
    std::cout << "Usage: " << (argv0 ? argv0 : "libhmm_vs_stochhmm_distribution_suite")
              << " [--include-1e6] [--repeats N] [--warmup N] [--hmm-stage on|off]"
                 " [--output-dir PATH]\n";
}

bool parsePositiveInt(const std::string &text, int &value_out) {
    try {
        std::size_t pos = 0;
        int parsed = std::stoi(text, &pos);
        if (pos != text.size() || parsed <= 0) {
            return false;
        }
        value_out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseNonNegativeInt(const std::string &text, int &value_out) {
    try {
        std::size_t pos = 0;
        int parsed = std::stoi(text, &pos);
        if (pos != text.size() || parsed < 0) {
            return false;
        }
        value_out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseArgs(int argc, char *argv[], CliOptions &options, std::string &error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            options.show_help = true;
            return true;
        }
        if (arg == "--include-1e6") {
            options.include_1e6 = true;
            continue;
        }
        if (arg == "--repeats") {
            if (i + 1 >= argc) {
                error = "--repeats requires an integer argument";
                return false;
            }
            int parsed = 0;
            if (!parsePositiveInt(argv[++i], parsed)) {
                error = "--repeats must be a positive integer";
                return false;
            }
            options.repeats = parsed;
            continue;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) {
                error = "--warmup requires an integer argument";
                return false;
            }
            int parsed = 0;
            if (!parseNonNegativeInt(argv[++i], parsed)) {
                error = "--warmup must be a non-negative integer";
                return false;
            }
            options.warmup = parsed;
            continue;
        }
        if (arg == "--hmm-stage") {
            if (i + 1 >= argc) {
                error = "--hmm-stage requires on|off";
                return false;
            }
            const std::string value = argv[++i];
            if (value == "on") {
                options.hmm_stage = true;
            } else if (value == "off") {
                options.hmm_stage = false;
            } else {
                error = "--hmm-stage must be 'on' or 'off'";
                return false;
            }
            continue;
        }
        if (arg == "--output-dir") {
            if (i + 1 >= argc) {
                error = "--output-dir requires a path argument";
                return false;
            }
            options.output_dir = fs::path(argv[++i]);
            options.output_dir_explicit = true;
            continue;
        }

        error = "Unknown argument: " + arg;
        return false;
    }

    return true;
}

fs::path resolveBenchmarkLogDir(const char *argv0) {
    std::error_code ec;
    fs::path build_dir;

    if (argv0 != nullptr) {
        const fs::path exec_path = fs::weakly_canonical(fs::path(argv0), ec);
        if (!ec && exec_path.has_parent_path()) {
            const fs::path parent = exec_path.parent_path();
            if (parent.has_parent_path()) {
                build_dir = parent.parent_path();
            }
        }
    }

    if (build_dir.empty()) {
        build_dir = fs::current_path(ec);
    }

    fs::path log_dir = build_dir / "benchmark-logs";
    fs::create_directories(log_dir, ec);
    if (ec) {
        return build_dir;
    }
    return log_dir;
}

std::string timestampNow() {
    const std::time_t now = std::time(nullptr);
    std::tm tm_now{};
#if defined(_WIN32)
    localtime_s(&tm_now, &now);
#else
    localtime_r(&now, &tm_now);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    return oss.str();
}

uint64_t fnv1aHash(const std::string &text) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : text) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t makeSeed(const std::string &distribution, const std::string &case_name, int size,
                  uint64_t phase_tag) {
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    seed ^= fnv1aHash(distribution) + 0x9E3779B97F4A7C15ULL + (seed << 6U) + (seed >> 2U);
    seed ^= fnv1aHash(case_name) + 0x9E3779B97F4A7C15ULL + (seed << 6U) + (seed >> 2U);
    seed ^= static_cast<uint64_t>(size) * 0x27D4EB2FULL;
    seed ^= phase_tag;
    return seed;
}

std::vector<int> buildSizes(bool include_1e6) {
    std::vector<int> sizes = {10, 100, 1000, 10000, 100000};
    if (include_1e6) {
        sizes.push_back(1000000);
    }
    return sizes;
}

std::vector<DistributionCase> buildBenchmarkCases() {
    return {
        {DistributionKind::Gaussian, "Gaussian", "standard", 0.0, 1.0, "mean=0.0,std=1.0"},
        {DistributionKind::Gaussian, "Gaussian", "shifted", 2.5, 1.4, "mean=2.5,std=1.4"},
        {DistributionKind::Gaussian, "Gaussian", "high_variance", -1.0, 3.0, "mean=-1.0,std=3.0"},
        {DistributionKind::Exponential, "Exponential", "low_rate", 0.2, 0.0, "lambda=0.2"},
        {DistributionKind::Exponential, "Exponential", "medium_rate", 1.0, 0.0, "lambda=1.0"},
        {DistributionKind::Exponential, "Exponential", "high_rate", 3.0, 0.0, "lambda=3.0"},
        {DistributionKind::Poisson, "Poisson", "sparse", 1.0, 0.0, "lambda=1.0"},
        {DistributionKind::Poisson, "Poisson", "moderate", 5.0, 0.0, "lambda=5.0"},
        {DistributionKind::Poisson, "Poisson", "dense", 20.0, 0.0, "lambda=20.0"},
        {DistributionKind::Gamma, "Gamma", "low_shape", 1.5, 0.7, "shape=1.5,scale=0.7"},
        {DistributionKind::Gamma, "Gamma", "medium_shape", 3.0, 1.2, "shape=3.0,scale=1.2"},
        {DistributionKind::Gamma, "Gamma", "high_shape", 8.0, 0.5, "shape=8.0,scale=0.5"},
        {DistributionKind::LogNormal, "LogNormal", "mild_skew", 0.0, 0.5, "mu=0.0,sigma=0.5"},
        {DistributionKind::LogNormal, "LogNormal", "moderate_skew", 1.0, 0.8, "mu=1.0,sigma=0.8"},
        {DistributionKind::LogNormal, "LogNormal", "heavy_tail", 0.0, 1.2, "mu=0.0,sigma=1.2"},
        {DistributionKind::StudentT, "StudentT", "heavy_tail", 3.0, 0.0, "nu=3.0,loc=0,scale=1"},
        {DistributionKind::StudentT, "StudentT", "moderate_tail", 8.0, 0.0, "nu=8.0,loc=0,scale=1"},
        {DistributionKind::StudentT, "StudentT", "near_normal", 30.0, 0.0, "nu=30.0,loc=0,scale=1"},
        {DistributionKind::ChiSquared, "ChiSquared", "low_df", 2.0, 0.0, "df=2.0"},
        {DistributionKind::ChiSquared, "ChiSquared", "medium_df", 6.0, 0.0, "df=6.0"},
        {DistributionKind::ChiSquared, "ChiSquared", "high_df", 20.0, 0.0, "df=20.0"},
        {DistributionKind::Uniform, "Uniform", "unit_interval", 0.0, 1.0, "a=0.0,b=1.0"},
        {DistributionKind::Uniform, "Uniform", "symmetric", -2.0, 2.0, "a=-2.0,b=2.0"},
        {DistributionKind::Uniform, "Uniform", "shifted", 10.0, 20.0, "a=10.0,b=20.0"}};
}

std::vector<CanonicalHMMSpec> buildCanonicalHMMSpecs() {
    return {{DistributionKind::Gaussian, "Gaussian", 0.0, 1.0, 3.0, 1.5,
             "state0(mean=0,std=1), state1(mean=3,std=1.5)"},
            {DistributionKind::Exponential, "Exponential", 0.8, 0.0, 2.2, 0.0,
             "state0(lambda=0.8), state1(lambda=2.2)"},
            {DistributionKind::Poisson, "Poisson", 2.0, 0.0, 8.0, 0.0,
             "state0(lambda=2), state1(lambda=8)"},
            {DistributionKind::Gamma, "Gamma", 2.0, 0.8, 6.0, 0.5,
             "state0(shape=2,scale=0.8), state1(shape=6,scale=0.5)"},
            {DistributionKind::LogNormal, "LogNormal", 0.0, 0.6, 1.2, 0.9,
             "state0(mu=0,sigma=0.6), state1(mu=1.2,sigma=0.9)"},
            {DistributionKind::StudentT, "StudentT", 4.0, 0.0, 12.0, 0.0,
             "state0(nu=4,loc=0,scale=1), state1(nu=12,loc=0,scale=1)"},
            {DistributionKind::ChiSquared, "ChiSquared", 2.0, 0.0, 8.0, 0.0,
             "state0(df=2), state1(df=8)"},
            {DistributionKind::Uniform, "Uniform", 0.0, 2.0, 1.0, 4.0,
             "state0(a=0,b=2), state1(a=1,b=4)"}};
}

template <typename Fn>
double medianRuntimeMs(Fn &&fn, int warmup, int repeats) {
    const auto timed_run_once_ms = [&]() {
        const auto start = std::chrono::high_resolution_clock::now();
        fn();
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    double calibration_ms = timed_run_once_ms();
    int batch_iterations = 1;
    if (std::isfinite(calibration_ms) && calibration_ms > 0.0 &&
        calibration_ms < kTargetTimingSampleMs) {
        const double desired = std::ceil(kTargetTimingSampleMs / calibration_ms);
        batch_iterations = static_cast<int>(
            std::min(static_cast<double>(kMaxTimingBatchIterations), std::max(1.0, desired)));
    }

    const auto run_batched = [&]() {
        for (int i = 0; i < batch_iterations; ++i) {
            fn();
        }
    };

    for (int i = 0; i < warmup; ++i) {
        run_batched();
    }

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        run_batched();
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(elapsed_ms / static_cast<double>(batch_iterations));
    }

    std::sort(times.begin(), times.end());
    const std::size_t n = times.size();
    if (n == 0) {
        return 0.0;
    }
    if ((n % 2U) == 1U) {
        return times[n / 2U];
    }
    return 0.5 * (times[n / 2U - 1U] + times[n / 2U]);
}

std::vector<double> generateObservations(const DistributionCase &distribution_case, int size,
                                         uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<double> observations;
    observations.reserve(static_cast<std::size_t>(size));

    switch (distribution_case.kind) {
        case DistributionKind::Gaussian: {
            std::normal_distribution<double> dist(distribution_case.p1, distribution_case.p2);
            for (int i = 0; i < size; ++i) {
                observations.push_back(dist(rng));
            }
            break;
        }
        case DistributionKind::Exponential: {
            std::exponential_distribution<double> dist(distribution_case.p1);
            for (int i = 0; i < size; ++i) {
                observations.push_back(dist(rng));
            }
            break;
        }
        case DistributionKind::Poisson: {
            std::poisson_distribution<int> dist(distribution_case.p1);
            for (int i = 0; i < size; ++i) {
                observations.push_back(static_cast<double>(dist(rng)));
            }
            break;
        }
        case DistributionKind::Gamma: {
            std::gamma_distribution<double> dist(distribution_case.p1, distribution_case.p2);
            for (int i = 0; i < size; ++i) {
                observations.push_back(dist(rng));
            }
            break;
        }
        case DistributionKind::LogNormal: {
            std::lognormal_distribution<double> dist(distribution_case.p1, distribution_case.p2);
            for (int i = 0; i < size; ++i) {
                observations.push_back(dist(rng));
            }
            break;
        }
        case DistributionKind::StudentT: {
            std::student_t_distribution<double> dist(distribution_case.p1);
            for (int i = 0; i < size; ++i) {
                observations.push_back(dist(rng));
            }
            break;
        }
        case DistributionKind::ChiSquared: {
            std::chi_squared_distribution<double> dist(distribution_case.p1);
            for (int i = 0; i < size; ++i) {
                observations.push_back(dist(rng));
            }
            break;
        }
        case DistributionKind::Uniform: {
            std::uniform_real_distribution<double> dist(distribution_case.p1, distribution_case.p2);
            for (int i = 0; i < size; ++i) {
                observations.push_back(dist(rng));
            }
            break;
        }
    }

    return observations;
}

template <typename Dist, typename... Args>
DistributionEvalResult evaluateLibhmmBatch(const std::vector<double> &observations, int warmup,
                                           int repeats, Args &&...args) {
    DistributionEvalResult result;
    try {
        Dist distribution(std::forward<Args>(args)...);
        result.log_values.resize(observations.size(), 0.0);
        const auto obs_span = std::span<const double>(observations.data(), observations.size());
        auto out_span = std::span<double>(result.log_values.data(), result.log_values.size());

        auto run = [&] {
            distribution.getBatchLogProbabilities(obs_span, out_span);
        };
        result.time_ms = medianRuntimeMs(run, warmup, repeats);
        run();
        result.success = true;
    } catch (const std::exception &ex) {
        result.success = false;
        result.error = ex.what();
    } catch (...) {
        result.success = false;
        result.error = "Unknown exception while evaluating libhmm distribution";
    }
    return result;
}

DistributionEvalResult evaluateLibhmmDistribution(const DistributionCase &distribution_case,
                                                  const std::vector<double> &observations,
                                                  int warmup, int repeats) {
    switch (distribution_case.kind) {
        case DistributionKind::Gaussian:
            return evaluateLibhmmBatch<libhmm::GaussianDistribution>(
                observations, warmup, repeats, distribution_case.p1, distribution_case.p2);
        case DistributionKind::Exponential:
            return evaluateLibhmmBatch<libhmm::ExponentialDistribution>(
                observations, warmup, repeats, distribution_case.p1);
        case DistributionKind::Poisson:
            return evaluateLibhmmBatch<libhmm::PoissonDistribution>(observations, warmup, repeats,
                                                                    distribution_case.p1);
        case DistributionKind::Gamma:
            return evaluateLibhmmBatch<libhmm::GammaDistribution>(
                observations, warmup, repeats, distribution_case.p1, distribution_case.p2);
        case DistributionKind::LogNormal:
            return evaluateLibhmmBatch<libhmm::LogNormalDistribution>(
                observations, warmup, repeats, distribution_case.p1, distribution_case.p2);
        case DistributionKind::StudentT:
            return evaluateLibhmmBatch<libhmm::StudentTDistribution>(
                observations, warmup, repeats, distribution_case.p1, 0.0, 1.0);
        case DistributionKind::ChiSquared:
            return evaluateLibhmmBatch<libhmm::ChiSquaredDistribution>(
                observations, warmup, repeats, distribution_case.p1);
        case DistributionKind::Uniform:
            return evaluateLibhmmBatch<libhmm::UniformDistribution>(
                observations, warmup, repeats, distribution_case.p1, distribution_case.p2);
    }

    DistributionEvalResult result;
    result.success = false;
    result.error = "Unhandled distribution kind for libhmm evaluator";
    return result;
}

double evaluateStochLogProbability(DistributionKind kind, double value,
                                   const std::vector<double> &params) {
    switch (kind) {
        case DistributionKind::Gaussian:
            return StochHMM::normal_pdf(value, &params);
        case DistributionKind::Exponential:
            return StochHMM::exponential_pdf(value, &params);
        case DistributionKind::Poisson:
            return StochHMM::poisson_pdf(value, &params);
        case DistributionKind::Gamma:
            return StochHMM::gamma_pdf(value, &params);
        case DistributionKind::LogNormal:
            return StochHMM::log_normal_pdf(value, &params);
        case DistributionKind::StudentT:
            return StochHMM::students_t_pdf(value, &params);
        case DistributionKind::ChiSquared:
            return StochHMM::chi_squared_pdf(value, &params);
        case DistributionKind::Uniform:
            return StochHMM::continuous_uniform_pdf(value, &params);
    }
    return -std::numeric_limits<double>::infinity();
}

DistributionEvalResult evaluateStochDistribution(const DistributionCase &distribution_case,
                                                 const std::vector<double> &observations,
                                                 int warmup, int repeats) {
    DistributionEvalResult result;
    try {
        std::vector<double> params;
        if (distribution_case.kind == DistributionKind::Gaussian) {
            params = {distribution_case.p1, distribution_case.p2};
        } else if (distribution_case.kind == DistributionKind::Gamma) {
            if (distribution_case.p2 <= 0.0) {
                throw std::runtime_error("Gamma scale must be > 0 for StochHMM mapping");
            }
            params = {distribution_case.p1, 1.0 / distribution_case.p2}; // alpha, beta(rate)
        } else if (distribution_case.kind == DistributionKind::LogNormal) {
            params = {distribution_case.p1,
                      distribution_case.p2 * distribution_case.p2}; // mu, sigma^2
        } else if (distribution_case.kind == DistributionKind::Uniform) {
            params = {distribution_case.p1, distribution_case.p2}; // a, b
        } else {
            params = {distribution_case.p1};
        }

        result.log_values.resize(observations.size(), 0.0);
        auto run = [&] {
            for (std::size_t i = 0; i < observations.size(); ++i) {
                result.log_values[i] =
                    evaluateStochLogProbability(distribution_case.kind, observations[i], params);
            }
        };
        result.time_ms = medianRuntimeMs(run, warmup, repeats);
        run();
        result.success = true;
    } catch (const std::exception &ex) {
        result.success = false;
        result.error = ex.what();
    } catch (...) {
        result.success = false;
        result.error = "Unknown exception while evaluating StochHMM distribution";
    }
    return result;
}

AccuracyMetrics computeAccuracyMetrics(const std::vector<double> &libhmm_values,
                                       const std::vector<double> &stoch_values) {
    AccuracyMetrics metrics{};
    metrics.max_abs = 0.0;
    metrics.max_rel = 0.0;

    if (libhmm_values.size() != stoch_values.size()) {
        metrics.max_abs = std::numeric_limits<double>::infinity();
        metrics.max_rel = std::numeric_limits<double>::infinity();
        return metrics;
    }

    for (std::size_t i = 0; i < libhmm_values.size(); ++i) {
        const double a = libhmm_values[i];
        const double b = stoch_values[i];

        double abs_diff = 0.0;
        if (std::isfinite(a) && std::isfinite(b)) {
            abs_diff = std::abs(a - b);
        } else if ((std::isinf(a) && std::isinf(b) && (std::signbit(a) == std::signbit(b)))) {
            abs_diff = 0.0;
        } else {
            abs_diff = std::numeric_limits<double>::infinity();
        }

        metrics.max_abs = std::max(metrics.max_abs, abs_diff);

        double rel_diff = std::numeric_limits<double>::infinity();
        if (std::isfinite(abs_diff)) {
            const double denom = std::max(1.0, std::max(std::abs(a), std::abs(b)));
            rel_diff = abs_diff / denom;
        }
        metrics.max_rel = std::max(metrics.max_rel, rel_diff);
    }

    return metrics;
}

double safeThroughput(int size, double time_ms) {
    if (!std::isfinite(time_ms) || time_ms <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(size) / time_ms;
}

double safeSpeedup(double baseline_throughput, double comparison_throughput) {
    if (!std::isfinite(baseline_throughput) || baseline_throughput <= 0.0 ||
        !std::isfinite(comparison_throughput)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return comparison_throughput / baseline_throughput;
}

double safeReciprocal(double value) {
    if (!std::isfinite(value) || value == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 / value;
}

DistributionComparisonRow runDistributionComparison(const DistributionCase &distribution_case,
                                                    int size, int warmup, int repeats) {
    DistributionComparisonRow row;
    row.distribution = distribution_case.distribution;
    row.case_name = distribution_case.case_name;
    row.size = size;
    row.gate = GateStatus::Block;
    row.success = false;

    const uint64_t seed =
        makeSeed(distribution_case.distribution, distribution_case.case_name, size, 0xA11CE);
    const std::vector<double> observations = generateObservations(distribution_case, size, seed);

    const auto libhmm_eval =
        evaluateLibhmmDistribution(distribution_case, observations, warmup, repeats);
    const auto stoch_eval =
        evaluateStochDistribution(distribution_case, observations, warmup, repeats);

    row.libhmm_time_ms = libhmm_eval.time_ms;
    row.stoch_time_ms = stoch_eval.time_ms;
    row.libhmm_throughput = safeThroughput(size, row.libhmm_time_ms);
    row.stoch_throughput = safeThroughput(size, row.stoch_time_ms);
    row.speedup_stoch_over_libhmm = safeSpeedup(row.libhmm_throughput, row.stoch_throughput);
    row.speedup_libhmm_over_stoch = safeReciprocal(row.speedup_stoch_over_libhmm);

    if (!libhmm_eval.success || !stoch_eval.success) {
        std::ostringstream notes;
        if (!libhmm_eval.success) {
            notes << "libhmm_error=" << libhmm_eval.error;
        }
        if (!stoch_eval.success) {
            if (!notes.str().empty()) {
                notes << ";";
            }
            notes << "stochhmm_error=" << stoch_eval.error;
        }
        row.notes = notes.str();
        row.max_abs_log_diff = std::numeric_limits<double>::infinity();
        row.max_rel_log_diff = std::numeric_limits<double>::infinity();
        row.gate = GateStatus::Block;
        row.success = false;
        return row;
    }

    const AccuracyMetrics accuracy =
        computeAccuracyMetrics(libhmm_eval.log_values, stoch_eval.log_values);
    row.max_abs_log_diff = accuracy.max_abs;
    row.max_rel_log_diff = accuracy.max_rel;
    row.gate = gateFromAbsDiff(row.max_abs_log_diff);
    row.success = true;
    return row;
}

std::vector<Phase1AggregateRow>
aggregatePhase1ByDistributionAndSize(const std::vector<DistributionComparisonRow> &rows) {
    std::map<std::pair<std::string, int>, std::vector<const DistributionComparisonRow *>> grouped;
    for (const auto &row : rows) {
        grouped[{row.distribution, row.size}].push_back(&row);
    }

    std::vector<Phase1AggregateRow> aggregates;
    aggregates.reserve(grouped.size());
    for (const auto &entry : grouped) {
        Phase1AggregateRow aggregate;
        aggregate.distribution = entry.first.first;
        aggregate.size = entry.first.second;
        aggregate.worst_gate = GateStatus::Pass;
        aggregate.max_abs_log_diff = 0.0;

        double speedup_sum = 0.0;
        double inverse_speedup_sum = 0.0;
        int speedup_count = 0;
        for (const DistributionComparisonRow *row : entry.second) {
            aggregate.worst_gate = worstGate(aggregate.worst_gate, row->gate);
            aggregate.max_abs_log_diff =
                std::max(aggregate.max_abs_log_diff, row->max_abs_log_diff);
            if (std::isfinite(row->speedup_stoch_over_libhmm)) {
                speedup_sum += row->speedup_stoch_over_libhmm;
                ++speedup_count;
            }
            if (std::isfinite(row->speedup_libhmm_over_stoch)) {
                inverse_speedup_sum += row->speedup_libhmm_over_stoch;
            }
        }
        aggregate.mean_speedup_stoch_over_libhmm =
            (speedup_count > 0) ? (speedup_sum / static_cast<double>(speedup_count))
                                : std::numeric_limits<double>::quiet_NaN();
        aggregate.mean_speedup_libhmm_over_stoch =
            (speedup_count > 0) ? (inverse_speedup_sum / static_cast<double>(speedup_count))
                                : std::numeric_limits<double>::quiet_NaN();

        aggregates.push_back(aggregate);
    }
    return aggregates;
}

std::vector<DistributionGateSummary>
summarizeDistributionGates(const std::vector<DistributionComparisonRow> &rows) {
    std::map<std::string, DistributionGateSummary> summaries;
    std::map<std::string, double> speedup_sums;
    std::map<std::string, double> inverse_speedup_sums;
    std::map<std::string, int> speedup_counts;

    for (const auto &row : rows) {
        auto &summary = summaries[row.distribution];
        summary.distribution = row.distribution;
        summary.worst_gate = worstGate(summary.worst_gate, row.gate);
        summary.max_abs_log_diff = std::max(summary.max_abs_log_diff, row.max_abs_log_diff);
        summary.total_rows += 1;
        if (row.gate == GateStatus::Warn) {
            summary.warn_count += 1;
        } else if (row.gate == GateStatus::Block) {
            summary.block_count += 1;
        }
        if (std::isfinite(row.speedup_stoch_over_libhmm)) {
            speedup_sums[row.distribution] += row.speedup_stoch_over_libhmm;
            speedup_counts[row.distribution] += 1;
        }
        if (std::isfinite(row.speedup_libhmm_over_stoch)) {
            inverse_speedup_sums[row.distribution] += row.speedup_libhmm_over_stoch;
        }
    }

    std::vector<DistributionGateSummary> out;
    out.reserve(summaries.size());
    for (auto &entry : summaries) {
        auto &summary = entry.second;
        const int speedup_count = speedup_counts[summary.distribution];
        if (speedup_count > 0) {
            summary.mean_speedup_stoch_over_libhmm =
                speedup_sums[summary.distribution] / static_cast<double>(speedup_count);
            summary.mean_speedup_libhmm_over_stoch =
                inverse_speedup_sums[summary.distribution] / static_cast<double>(speedup_count);
        } else {
            summary.mean_speedup_stoch_over_libhmm = std::numeric_limits<double>::quiet_NaN();
            summary.mean_speedup_libhmm_over_stoch = std::numeric_limits<double>::quiet_NaN();
        }
        out.push_back(summary);
    }

    std::sort(out.begin(), out.end(),
              [](const DistributionGateSummary &a, const DistributionGateSummary &b) {
                  return a.distribution < b.distribution;
              });
    return out;
}

std::map<std::string, GateStatus>
buildDistributionGateMap(const std::vector<DistributionGateSummary> &summaries) {
    std::map<std::string, GateStatus> result;
    for (const auto &summary : summaries) {
        result[summary.distribution] = summary.worst_gate;
    }
    return result;
}

std::vector<double> generateHMMObservations(const CanonicalHMMSpec &spec, int size, uint64_t seed) {
    static constexpr std::array<double, 2> kPi = {0.6, 0.4};
    static constexpr std::array<std::array<double, 2>, 2> kTrans = {
        std::array<double, 2>{0.85, 0.15}, std::array<double, 2>{0.10, 0.90}};

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
    std::vector<double> observations;
    observations.reserve(static_cast<std::size_t>(size));

    int state = (unit_dist(rng) < kPi[0]) ? 0 : 1;
    std::normal_distribution<double> gauss0(spec.state0_p1, spec.state0_p2);
    std::normal_distribution<double> gauss1(spec.state1_p1, spec.state1_p2);
    std::exponential_distribution<double> exp0(spec.state0_p1);
    std::exponential_distribution<double> exp1(spec.state1_p1);
    std::poisson_distribution<int> poisson0(spec.state0_p1);
    std::poisson_distribution<int> poisson1(spec.state1_p1);
    std::gamma_distribution<double> gamma0(spec.state0_p1, spec.state0_p2);
    std::gamma_distribution<double> gamma1(spec.state1_p1, spec.state1_p2);
    std::lognormal_distribution<double> lognormal0(spec.state0_p1, spec.state0_p2);
    std::lognormal_distribution<double> lognormal1(spec.state1_p1, spec.state1_p2);
    std::student_t_distribution<double> studentt0(spec.state0_p1);
    std::student_t_distribution<double> studentt1(spec.state1_p1);
    std::chi_squared_distribution<double> chisq0(spec.state0_p1);
    std::chi_squared_distribution<double> chisq1(spec.state1_p1);
    std::uniform_real_distribution<double> uniform0(spec.state0_p1, spec.state0_p2);
    std::uniform_real_distribution<double> uniform1(spec.state1_p1, spec.state1_p2);

    for (int t = 0; t < size; ++t) {
        double obs = 0.0;
        switch (spec.kind) {
            case DistributionKind::Gaussian:
                obs = (state == 0) ? gauss0(rng) : gauss1(rng);
                break;
            case DistributionKind::Exponential:
                obs = (state == 0) ? exp0(rng) : exp1(rng);
                break;
            case DistributionKind::Poisson:
                obs = static_cast<double>((state == 0) ? poisson0(rng) : poisson1(rng));
                break;
            case DistributionKind::Gamma:
                obs = (state == 0) ? gamma0(rng) : gamma1(rng);
                break;
            case DistributionKind::LogNormal:
                obs = (state == 0) ? lognormal0(rng) : lognormal1(rng);
                break;
            case DistributionKind::StudentT:
                obs = (state == 0) ? studentt0(rng) : studentt1(rng);
                break;
            case DistributionKind::ChiSquared:
                obs = (state == 0) ? chisq0(rng) : chisq1(rng);
                break;
            case DistributionKind::Uniform:
                obs = (state == 0) ? uniform0(rng) : uniform1(rng);
                break;
        }
        observations.push_back(obs);

        const double u = unit_dist(rng);
        state = (u < kTrans[state][0]) ? 0 : 1;
    }
    return observations;
}

HMMRunResult runLibhmmHMM(const CanonicalHMMSpec &spec, const std::vector<double> &observations,
                          int warmup, int repeats) {
    HMMRunResult result;
    try {
        static constexpr std::array<double, 2> kPi = {0.6, 0.4};
        static constexpr std::array<std::array<double, 2>, 2> kTrans = {
            std::array<double, 2>{0.85, 0.15}, std::array<double, 2>{0.10, 0.90}};

        libhmm::Hmm hmm(2);
        libhmm::Vector pi(2);
        pi(0) = kPi[0];
        pi(1) = kPi[1];
        hmm.setPi(pi);

        libhmm::Matrix trans(2, 2);
        trans(0, 0) = kTrans[0][0];
        trans(0, 1) = kTrans[0][1];
        trans(1, 0) = kTrans[1][0];
        trans(1, 1) = kTrans[1][1];
        hmm.setTrans(trans);

        if (spec.kind == DistributionKind::Gaussian) {
            hmm.setDistribution(
                0, std::make_unique<libhmm::GaussianDistribution>(spec.state0_p1, spec.state0_p2));
            hmm.setDistribution(
                1, std::make_unique<libhmm::GaussianDistribution>(spec.state1_p1, spec.state1_p2));
        } else if (spec.kind == DistributionKind::Exponential) {
            hmm.setDistribution(0,
                                std::make_unique<libhmm::ExponentialDistribution>(spec.state0_p1));
            hmm.setDistribution(1,
                                std::make_unique<libhmm::ExponentialDistribution>(spec.state1_p1));
        } else if (spec.kind == DistributionKind::Poisson) {
            hmm.setDistribution(0, std::make_unique<libhmm::PoissonDistribution>(spec.state0_p1));
            hmm.setDistribution(1, std::make_unique<libhmm::PoissonDistribution>(spec.state1_p1));
        } else if (spec.kind == DistributionKind::Gamma) {
            hmm.setDistribution(
                0, std::make_unique<libhmm::GammaDistribution>(spec.state0_p1, spec.state0_p2));
            hmm.setDistribution(
                1, std::make_unique<libhmm::GammaDistribution>(spec.state1_p1, spec.state1_p2));
        } else if (spec.kind == DistributionKind::LogNormal) {
            hmm.setDistribution(
                0, std::make_unique<libhmm::LogNormalDistribution>(spec.state0_p1, spec.state0_p2));
            hmm.setDistribution(
                1, std::make_unique<libhmm::LogNormalDistribution>(spec.state1_p1, spec.state1_p2));
        } else if (spec.kind == DistributionKind::StudentT) {
            hmm.setDistribution(
                0, std::make_unique<libhmm::StudentTDistribution>(spec.state0_p1, 0.0, 1.0));
            hmm.setDistribution(
                1, std::make_unique<libhmm::StudentTDistribution>(spec.state1_p1, 0.0, 1.0));
        } else if (spec.kind == DistributionKind::ChiSquared) {
            hmm.setDistribution(0,
                                std::make_unique<libhmm::ChiSquaredDistribution>(spec.state0_p1));
            hmm.setDistribution(1,
                                std::make_unique<libhmm::ChiSquaredDistribution>(spec.state1_p1));
        } else {
            hmm.setDistribution(
                0, std::make_unique<libhmm::UniformDistribution>(spec.state0_p1, spec.state0_p2));
            hmm.setDistribution(
                1, std::make_unique<libhmm::UniformDistribution>(spec.state1_p1, spec.state1_p2));
        }

        libhmm::ObservationSet obs_set(observations.size());
        for (std::size_t i = 0; i < observations.size(); ++i) {
            obs_set(i) = observations[i];
        }

        double forward_ll = 0.0;
        auto forward_run = [&] {
            libhmm::ForwardBackwardCalculator fb(&hmm, obs_set);
            forward_ll = fb.getLogProbability();
        };
        result.forward_ms = medianRuntimeMs(forward_run, warmup, repeats);
        forward_run();
        result.log_likelihood = forward_ll;

        auto viterbi_run = [&] {
            libhmm::ViterbiCalculator viterbi(&hmm, obs_set);
            auto path = viterbi.decode();
            if (path.empty() && !observations.empty()) {
                throw std::runtime_error("libhmm viterbi returned empty path");
            }
        };
        result.viterbi_ms = medianRuntimeMs(viterbi_run, warmup, repeats);
        result.forward_throughput =
            safeThroughput(static_cast<int>(observations.size()), result.forward_ms);
        result.success = true;
    } catch (const std::exception &ex) {
        result.success = false;
        result.error = ex.what();
    } catch (...) {
        result.success = false;
        result.error = "Unknown exception while running libhmm HMM stage";
    }

    return result;
}

double logSumExp(double a, double b) {
    if (!std::isfinite(a)) {
        return b;
    }
    if (!std::isfinite(b)) {
        return a;
    }
    const double m = std::max(a, b);
    return m + std::log1p(std::exp(std::min(a, b) - m));
}

HMMRunResult runStochManualHMM(const CanonicalHMMSpec &spec,
                               const std::vector<double> &observations, int warmup, int repeats) {
    HMMRunResult result;
    try {
        static constexpr std::array<double, 2> kPi = {0.6, 0.4};
        static constexpr std::array<std::array<double, 2>, 2> kTrans = {
            std::array<double, 2>{0.85, 0.15}, std::array<double, 2>{0.10, 0.90}};

        std::vector<double> params0;
        std::vector<double> params1;
        if (spec.kind == DistributionKind::Gaussian) {
            params0 = {spec.state0_p1, spec.state0_p2};
            params1 = {spec.state1_p1, spec.state1_p2};
        } else if (spec.kind == DistributionKind::Gamma) {
            if (spec.state0_p2 <= 0.0 || spec.state1_p2 <= 0.0) {
                throw std::runtime_error("Gamma scale must be > 0 for StochHMM HMM mapping");
            }
            params0 = {spec.state0_p1, 1.0 / spec.state0_p2}; // alpha, beta(rate)
            params1 = {spec.state1_p1, 1.0 / spec.state1_p2}; // alpha, beta(rate)
        } else if (spec.kind == DistributionKind::LogNormal) {
            params0 = {spec.state0_p1, spec.state0_p2 * spec.state0_p2}; // mu, sigma^2
            params1 = {spec.state1_p1, spec.state1_p2 * spec.state1_p2}; // mu, sigma^2
        } else if (spec.kind == DistributionKind::Uniform) {
            params0 = {spec.state0_p1, spec.state0_p2}; // a, b
            params1 = {spec.state1_p1, spec.state1_p2}; // a, b
        } else {
            params0 = {spec.state0_p1};
            params1 = {spec.state1_p1};
        }

        const double log_pi0 = std::log(kPi[0]);
        const double log_pi1 = std::log(kPi[1]);
        const double log_t00 = std::log(kTrans[0][0]);
        const double log_t01 = std::log(kTrans[0][1]);
        const double log_t10 = std::log(kTrans[1][0]);
        const double log_t11 = std::log(kTrans[1][1]);

        auto emission = [&](int state, double x) {
            return evaluateStochLogProbability(spec.kind, x, (state == 0) ? params0 : params1);
        };

        double forward_ll = 0.0;
        auto forward_run = [&] {
            const std::size_t T = observations.size();
            std::vector<std::array<double, 2>> alpha(T, {0.0, 0.0});

            alpha[0][0] = log_pi0 + emission(0, observations[0]);
            alpha[0][1] = log_pi1 + emission(1, observations[0]);

            for (std::size_t t = 1; t < T; ++t) {
                const double e0 = emission(0, observations[t]);
                const double e1 = emission(1, observations[t]);
                alpha[t][0] = e0 + logSumExp(alpha[t - 1][0] + log_t00, alpha[t - 1][1] + log_t10);
                alpha[t][1] = e1 + logSumExp(alpha[t - 1][0] + log_t01, alpha[t - 1][1] + log_t11);
            }

            forward_ll = logSumExp(alpha[T - 1][0], alpha[T - 1][1]);
        };

        result.forward_ms = medianRuntimeMs(forward_run, warmup, repeats);
        forward_run();
        result.log_likelihood = forward_ll;

        auto viterbi_run = [&] {
            const std::size_t T = observations.size();
            std::vector<std::array<double, 2>> delta(T, {0.0, 0.0});
            std::vector<std::array<uint8_t, 2>> psi(T, {0U, 0U});

            delta[0][0] = log_pi0 + emission(0, observations[0]);
            delta[0][1] = log_pi1 + emission(1, observations[0]);

            for (std::size_t t = 1; t < T; ++t) {
                const double e0 = emission(0, observations[t]);
                const double e1 = emission(1, observations[t]);

                const double c00 = delta[t - 1][0] + log_t00;
                const double c10 = delta[t - 1][1] + log_t10;
                if (c00 >= c10) {
                    delta[t][0] = c00 + e0;
                    psi[t][0] = 0U;
                } else {
                    delta[t][0] = c10 + e0;
                    psi[t][0] = 1U;
                }

                const double c01 = delta[t - 1][0] + log_t01;
                const double c11 = delta[t - 1][1] + log_t11;
                if (c01 >= c11) {
                    delta[t][1] = c01 + e1;
                    psi[t][1] = 0U;
                } else {
                    delta[t][1] = c11 + e1;
                    psi[t][1] = 1U;
                }
            }

            std::vector<uint8_t> path(T, 0U);
            path[T - 1] = (delta[T - 1][0] >= delta[T - 1][1]) ? 0U : 1U;
            for (std::size_t t = T - 1; t > 0; --t) {
                path[t - 1] = psi[t][path[t]];
            }
            if (path.empty() && T > 0) {
                throw std::runtime_error("stochhmm manual viterbi returned empty path");
            }
        };

        result.viterbi_ms = medianRuntimeMs(viterbi_run, warmup, repeats);
        result.forward_throughput =
            safeThroughput(static_cast<int>(observations.size()), result.forward_ms);
        result.success = true;
    } catch (const std::exception &ex) {
        result.success = false;
        result.error = ex.what();
    } catch (...) {
        result.success = false;
        result.error = "Unknown exception while running StochHMM manual HMM stage";
    }

    return result;
}

std::string csvEscape(const std::string &value) {
    bool needs_quotes = false;
    for (char c : value) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return value;
    }
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(c);
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string formatMaybeNumber(double value, int precision = 10) {
    if (!std::isfinite(value)) {
        return "";
    }
    std::ostringstream oss;
    oss << std::setprecision(precision) << value;
    return oss.str();
}

void writeCSV(const fs::path &csv_path, const std::vector<DistributionComparisonRow> &phase1_rows,
              const std::vector<HMMComparisonRow> &hmm_rows) {
    std::ofstream csv(csv_path);
    csv << "stage,distribution,case_name,size,libhmm_forward_ms,stochhmm_forward_ms,"
           "libhmm_viterbi_ms,stochhmm_viterbi_ms,libhmm_throughput,stochhmm_throughput,"
           "speedup_stoch_over_libhmm,speedup_libhmm_over_stoch,"
           "libhmm_log_likelihood,stochhmm_log_likelihood,"
           "max_abs_log_diff,max_rel_log_diff,gate_status,success,notes\n";

    for (const auto &row : phase1_rows) {
        csv << "distribution_level"
            << "," << csvEscape(row.distribution) << "," << csvEscape(row.case_name) << ","
            << row.size << "," << formatMaybeNumber(row.libhmm_time_ms) << ","
            << formatMaybeNumber(row.stoch_time_ms) << ",,"
            << "," << formatMaybeNumber(row.libhmm_throughput) << ","
            << formatMaybeNumber(row.stoch_throughput) << ","
            << formatMaybeNumber(row.speedup_stoch_over_libhmm) << ","
            << formatMaybeNumber(row.speedup_libhmm_over_stoch) << ",,"
            << "," << formatMaybeNumber(row.max_abs_log_diff) << ","
            << formatMaybeNumber(row.max_rel_log_diff) << "," << gateToString(row.gate) << ","
            << (row.success ? "true" : "false") << "," << csvEscape(row.notes) << "\n";
    }

    for (const auto &row : hmm_rows) {
        csv << "hmm_stage"
            << "," << csvEscape(row.distribution) << ",canonical_hmm," << row.size << ","
            << formatMaybeNumber(row.libhmm_forward_ms) << ","
            << formatMaybeNumber(row.stoch_forward_ms) << ","
            << formatMaybeNumber(row.libhmm_viterbi_ms) << ","
            << formatMaybeNumber(row.stoch_viterbi_ms) << ","
            << formatMaybeNumber(row.libhmm_forward_throughput) << ","
            << formatMaybeNumber(row.stoch_forward_throughput) << ","
            << formatMaybeNumber(row.speedup_stoch_over_libhmm) << ","
            << formatMaybeNumber(row.speedup_libhmm_over_stoch) << ","
            << formatMaybeNumber(row.libhmm_log_likelihood) << ","
            << formatMaybeNumber(row.stoch_log_likelihood) << ","
            << formatMaybeNumber(row.abs_log_likelihood_diff) << ",,"
            << "," << (row.success ? "true" : "false") << "," << csvEscape(row.notes) << "\n";
    }
}

std::string jsonEscape(const std::string &value) {
    std::ostringstream oss;
    for (char c : value) {
        switch (c) {
            case '\\':
                oss << "\\\\";
                break;
            case '\"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << c;
                break;
        }
    }
    return oss.str();
}

void writeJSONNumber(std::ostream &os, double value) {
    if (!std::isfinite(value)) {
        os << "null";
        return;
    }
    os << std::setprecision(12) << value;
}

void writeJSON(const fs::path &json_path, const CliOptions &options, const std::vector<int> &sizes,
               const std::vector<DistributionComparisonRow> &phase1_rows,
               const std::vector<Phase1AggregateRow> &phase1_aggregates,
               const std::vector<DistributionGateSummary> &gate_summaries,
               const std::vector<HMMComparisonRow> &hmm_rows,
               const std::vector<std::string> &skipped_hmm_distributions,
               const std::string &overall_assessment, const fs::path &csv_path) {
    std::ofstream json(json_path);
    json << "{\n";
    json << "  \"run_config\": {\n";
    json << "    \"include_1e6\": " << (options.include_1e6 ? "true" : "false") << ",\n";
    json << "    \"repeats\": " << options.repeats << ",\n";
    json << "    \"warmup\": " << options.warmup << ",\n";
    json << "    \"hmm_stage\": " << (options.hmm_stage ? "true" : "false") << ",\n";
    json << "    \"sizes\": [";
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) {
            json << ", ";
        }
        json << sizes[i];
    }
    json << "]\n";
    json << "  },\n";

    json << "  \"parameter_mapping\": {\n";
    json << "    \"Gaussian\": {\n";
    json << "      \"libhmm\": \"GaussianDistribution(mean, stddev)\",\n";
    json << "      \"stochhmm\": \"normal_pdf(x, [mean, stddev])\"\n";
    json << "    },\n";
    json << "    \"Exponential\": {\n";
    json << "      \"libhmm\": \"ExponentialDistribution(lambda_rate)\",\n";
    json << "      \"stochhmm\": \"exponential_pdf(x, [lambda_rate])\"\n";
    json << "    },\n";
    json << "    \"Poisson\": {\n";
    json << "      \"libhmm\": \"PoissonDistribution(lambda_rate)\",\n";
    json << "      \"stochhmm\": \"poisson_pdf(k, [lambda_rate])\"\n";
    json << "    },\n";
    json << "    \"Gamma\": {\n";
    json << "      \"libhmm\": \"GammaDistribution(shape_k, scale_theta)\",\n";
    json << "      \"stochhmm\": \"gamma_pdf(x, [alpha=shape_k, beta=1/scale_theta])\"\n";
    json << "    },\n";
    json << "    \"LogNormal\": {\n";
    json << "      \"libhmm\": \"LogNormalDistribution(mu, sigma)\",\n";
    json << "      \"stochhmm\": \"log_normal_pdf(x, [mu, sigma_sqrd=sigma^2])\"\n";
    json << "    },\n";
    json << "    \"StudentT\": {\n";
    json << "      \"libhmm\": \"StudentTDistribution(nu, location=0, scale=1)\",\n";
    json << "      \"stochhmm\": \"students_t_pdf(x, [nu])\"\n";
    json << "    },\n";
    json << "    \"ChiSquared\": {\n";
    json << "      \"libhmm\": \"ChiSquaredDistribution(degrees_of_freedom)\",\n";
    json << "      \"stochhmm\": \"chi_squared_pdf(x, [degrees_of_freedom])\"\n";
    json << "    },\n";
    json << "    \"Uniform\": {\n";
    json << "      \"libhmm\": \"UniformDistribution(a, b)\",\n";
    json << "      \"stochhmm\": \"continuous_uniform_pdf(x, [a, b])\"\n";
    json << "    }\n";
    json << "  },\n";
    json << "  \"gate_thresholds\": {\n";
    json << "    \"warn_abs_log_diff\": " << kWarnAbsLogDiff << ",\n";
    json << "    \"block_abs_log_diff\": " << kBlockAbsLogDiff << "\n";
    json << "  },\n";

    json << "  \"phase1_rows\": [\n";
    for (std::size_t i = 0; i < phase1_rows.size(); ++i) {
        const auto &row = phase1_rows[i];
        json << "    {\n";
        json << "      \"distribution\": \"" << jsonEscape(row.distribution) << "\",\n";
        json << "      \"case_name\": \"" << jsonEscape(row.case_name) << "\",\n";
        json << "      \"size\": " << row.size << ",\n";
        json << "      \"libhmm_time_ms\": ";
        writeJSONNumber(json, row.libhmm_time_ms);
        json << ",\n      \"stochhmm_time_ms\": ";
        writeJSONNumber(json, row.stoch_time_ms);
        json << ",\n      \"libhmm_throughput\": ";
        writeJSONNumber(json, row.libhmm_throughput);
        json << ",\n      \"stochhmm_throughput\": ";
        writeJSONNumber(json, row.stoch_throughput);
        json << ",\n      \"speedup_stoch_over_libhmm\": ";
        writeJSONNumber(json, row.speedup_stoch_over_libhmm);
        json << ",\n      \"speedup_libhmm_over_stoch\": ";
        writeJSONNumber(json, row.speedup_libhmm_over_stoch);
        json << ",\n      \"max_abs_log_diff\": ";
        writeJSONNumber(json, row.max_abs_log_diff);
        json << ",\n      \"max_rel_log_diff\": ";
        writeJSONNumber(json, row.max_rel_log_diff);
        json << ",\n      \"gate_status\": \"" << gateToString(row.gate) << "\",\n";
        json << "      \"success\": " << (row.success ? "true" : "false") << ",\n";
        json << "      \"notes\": \"" << jsonEscape(row.notes) << "\"\n";
        json << "    }";
        if (i + 1 < phase1_rows.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ],\n";

    json << "  \"phase1_aggregates\": [\n";
    for (std::size_t i = 0; i < phase1_aggregates.size(); ++i) {
        const auto &row = phase1_aggregates[i];
        json << "    {\n";
        json << "      \"distribution\": \"" << jsonEscape(row.distribution) << "\",\n";
        json << "      \"size\": " << row.size << ",\n";
        json << "      \"worst_gate\": \"" << gateToString(row.worst_gate) << "\",\n";
        json << "      \"max_abs_log_diff\": ";
        writeJSONNumber(json, row.max_abs_log_diff);
        json << ",\n      \"mean_speedup_stoch_over_libhmm\": ";
        writeJSONNumber(json, row.mean_speedup_stoch_over_libhmm);
        json << ",\n      \"mean_speedup_libhmm_over_stoch\": ";
        writeJSONNumber(json, row.mean_speedup_libhmm_over_stoch);
        json << "\n    }";
        if (i + 1 < phase1_aggregates.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ],\n";

    json << "  \"distribution_gate_summary\": [\n";
    for (std::size_t i = 0; i < gate_summaries.size(); ++i) {
        const auto &summary = gate_summaries[i];
        json << "    {\n";
        json << "      \"distribution\": \"" << jsonEscape(summary.distribution) << "\",\n";
        json << "      \"worst_gate\": \"" << gateToString(summary.worst_gate) << "\",\n";
        json << "      \"max_abs_log_diff\": ";
        writeJSONNumber(json, summary.max_abs_log_diff);
        json << ",\n      \"mean_speedup_stoch_over_libhmm\": ";
        writeJSONNumber(json, summary.mean_speedup_stoch_over_libhmm);
        json << ",\n      \"mean_speedup_libhmm_over_stoch\": ";
        writeJSONNumber(json, summary.mean_speedup_libhmm_over_stoch);
        json << ",\n      \"warn_count\": " << summary.warn_count << ",\n";
        json << "      \"block_count\": " << summary.block_count << ",\n";
        json << "      \"total_rows\": " << summary.total_rows << "\n";
        json << "    }";
        if (i + 1 < gate_summaries.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ],\n";

    json << "  \"hmm_rows\": [\n";
    for (std::size_t i = 0; i < hmm_rows.size(); ++i) {
        const auto &row = hmm_rows[i];
        json << "    {\n";
        json << "      \"distribution\": \"" << jsonEscape(row.distribution) << "\",\n";
        json << "      \"size\": " << row.size << ",\n";
        json << "      \"libhmm_forward_ms\": ";
        writeJSONNumber(json, row.libhmm_forward_ms);
        json << ",\n      \"stochhmm_forward_ms\": ";
        writeJSONNumber(json, row.stoch_forward_ms);
        json << ",\n      \"libhmm_viterbi_ms\": ";
        writeJSONNumber(json, row.libhmm_viterbi_ms);
        json << ",\n      \"stochhmm_viterbi_ms\": ";
        writeJSONNumber(json, row.stoch_viterbi_ms);
        json << ",\n      \"libhmm_forward_throughput\": ";
        writeJSONNumber(json, row.libhmm_forward_throughput);
        json << ",\n      \"stochhmm_forward_throughput\": ";
        writeJSONNumber(json, row.stoch_forward_throughput);
        json << ",\n      \"speedup_stoch_over_libhmm\": ";
        writeJSONNumber(json, row.speedup_stoch_over_libhmm);
        json << ",\n      \"speedup_libhmm_over_stoch\": ";
        writeJSONNumber(json, row.speedup_libhmm_over_stoch);
        json << ",\n      \"libhmm_log_likelihood\": ";
        writeJSONNumber(json, row.libhmm_log_likelihood);
        json << ",\n      \"stochhmm_log_likelihood\": ";
        writeJSONNumber(json, row.stoch_log_likelihood);
        json << ",\n      \"abs_log_likelihood_diff\": ";
        writeJSONNumber(json, row.abs_log_likelihood_diff);
        json << ",\n      \"success\": " << (row.success ? "true" : "false") << ",\n";
        json << "      \"notes\": \"" << jsonEscape(row.notes) << "\"\n";
        json << "    }";
        if (i + 1 < hmm_rows.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ],\n";

    json << "  \"skipped_hmm_distributions\": [";
    for (std::size_t i = 0; i < skipped_hmm_distributions.size(); ++i) {
        if (i > 0) {
            json << ", ";
        }
        json << "\"" << jsonEscape(skipped_hmm_distributions[i]) << "\"";
    }
    json << "],\n";

    json << "  \"overall_assessment\": \"" << jsonEscape(overall_assessment) << "\",\n";
    json << "  \"artifacts\": {\n";
    json << "    \"csv\": \"" << jsonEscape(csv_path.string()) << "\",\n";
    json << "    \"json\": \"" << jsonEscape(json_path.string()) << "\"\n";
    json << "  }\n";
    json << "}\n";
}

void printPhase1AggregateTable(const std::vector<Phase1AggregateRow> &aggregates) {
    std::cout << "\nPHASE 1 SUMMARY (distribution-level, aggregated by distribution/size)\n";
    std::cout
        << "----------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(14) << "Distribution" << std::setw(10) << "Size"
              << std::setw(12) << "Gate" << std::setw(18) << "Max|Δlogp|" << std::setw(22)
              << "Speedup(libhmm/stoch)\n";
    std::cout
        << "----------------------------------------------------------------------------------\n";
    for (const auto &row : aggregates) {
        std::cout << std::left << std::setw(14) << row.distribution << std::setw(10) << row.size
                  << std::setw(12) << gateToString(row.worst_gate) << std::setw(18)
                  << std::scientific << std::setprecision(3) << row.max_abs_log_diff
                  << std::setw(22) << std::fixed << std::setprecision(3)
                  << row.mean_speedup_libhmm_over_stoch << "\n";
    }
}

void printHMMSummaryTable(const std::vector<HMMComparisonRow> &hmm_rows) {
    if (hmm_rows.empty()) {
        std::cout << "\nPHASE 2 SUMMARY (HMM stage)\n";
        std::cout << "No HMM-stage rows were executed.\n";
        return;
    }

    std::cout << "\nPHASE 2 SUMMARY (selective HMM stage)\n";
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------------------\n";
    std::cout << std::left << std::setw(14) << "Distribution" << std::setw(10) << "Size"
              << std::setw(14) << "GateResult" << std::setw(22) << "FwdSpd(lib/stoch)"
              << std::setw(18) << "Fwd|ΔlogL|" << std::setw(18) << "libFwd(ms)" << std::setw(18)
              << "stochFwd(ms)\n";
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------------------\n";
    for (const auto &row : hmm_rows) {
        std::cout << std::left << std::setw(14) << row.distribution << std::setw(10) << row.size
                  << std::setw(14) << (row.success ? "RAN" : "FAILED") << std::setw(22)
                  << std::fixed << std::setprecision(3) << row.speedup_libhmm_over_stoch
                  << std::setw(18) << std::scientific << std::setprecision(3)
                  << row.abs_log_likelihood_diff << std::fixed << std::setprecision(3)
                  << std::setw(18) << row.libhmm_forward_ms << std::setw(18) << row.stoch_forward_ms
                  << "\n";
    }
}

std::string buildOverallAssessment(const std::vector<DistributionGateSummary> &gate_summaries,
                                   bool hmm_stage_enabled,
                                   const std::vector<std::string> &skipped_hmm_distributions) {
    int pass_count = 0;
    int warn_count = 0;
    int block_count = 0;
    for (const auto &summary : gate_summaries) {
        if (summary.worst_gate == GateStatus::Pass) {
            ++pass_count;
        } else if (summary.worst_gate == GateStatus::Warn) {
            ++warn_count;
        } else {
            ++block_count;
        }
    }

    std::ostringstream oss;
    oss << "Gate outcome: " << pass_count << " PASS, " << warn_count << " WARN, " << block_count
        << " BLOCK distributions.";

    if (!hmm_stage_enabled) {
        oss << " HMM stage disabled by CLI.";
        return oss.str();
    }

    if (!skipped_hmm_distributions.empty()) {
        oss << " HMM stage skipped for blocked distributions: ";
        for (std::size_t i = 0; i < skipped_hmm_distributions.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << skipped_hmm_distributions[i];
        }
        oss << ".";
    } else {
        oss << " All distributions advanced to HMM stage.";
    }
    return oss.str();
}

} // namespace

int main(int argc, char *argv[]) {
    CliOptions options;
    std::string arg_error;
    if (!parseArgs(argc, argv, options, arg_error)) {
        std::cerr << "Argument error: " << arg_error << "\n";
        printUsage(argc > 0 ? argv[0] : nullptr);
        return 1;
    }
    if (options.show_help) {
        printUsage(argc > 0 ? argv[0] : nullptr);
        return 0;
    }

    const std::vector<int> sizes = buildSizes(options.include_1e6);
    fs::path output_dir = options.output_dir_explicit
                              ? options.output_dir
                              : resolveBenchmarkLogDir((argc > 0) ? argv[0] : nullptr);
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "Could not create output directory " << output_dir << ": " << ec.message()
                  << "\n";
        return 1;
    }

    std::cout << "libhmm vs StochHMM Distribution Suite (Wave 1+2+3)\n";
    std::cout << "================================================\n";
    std::cout << "Distributions: Gaussian, Exponential, Poisson, Gamma, LogNormal, StudentT, "
                 "ChiSquared, Uniform\n";
    std::cout << "Sizes: ";
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << sizes[i];
    }
    std::cout << "\n";
    std::cout << "Gate thresholds: WARN > " << kWarnAbsLogDiff << ", BLOCK > " << kBlockAbsLogDiff
              << "\n";
    std::cout << "Speedup convention: speedup(libhmm/stoch) > 1 means libhmm faster.\n";
    std::cout << "Repeats: " << options.repeats << ", Warmup: " << options.warmup
              << ", HMM stage: " << (options.hmm_stage ? "on" : "off") << "\n";
    std::cout << "Output dir: " << output_dir << "\n";

    const std::vector<DistributionCase> benchmark_cases = buildBenchmarkCases();
    std::vector<DistributionComparisonRow> phase1_rows;
    phase1_rows.reserve(benchmark_cases.size() * sizes.size());

    for (const auto &distribution_case : benchmark_cases) {
        for (int size : sizes) {
            std::cout << "[phase1] " << distribution_case.distribution << "/"
                      << distribution_case.case_name << " n=" << size << " ... ";
            DistributionComparisonRow row =
                runDistributionComparison(distribution_case, size, options.warmup, options.repeats);
            phase1_rows.push_back(row);
            if (row.success) {
                std::cout << "done"
                          << " | gate=" << gateToString(row.gate)
                          << " | max|Δlogp|=" << std::scientific << std::setprecision(3)
                          << row.max_abs_log_diff << std::fixed << std::setprecision(3)
                          << " | speedup(libhmm/stoch)=" << row.speedup_libhmm_over_stoch << "x\n";
            } else {
                std::cout << "failed"
                          << " | " << row.notes << "\n";
            }
        }
    }

    std::vector<Phase1AggregateRow> phase1_aggregates =
        aggregatePhase1ByDistributionAndSize(phase1_rows);
    std::sort(phase1_aggregates.begin(), phase1_aggregates.end(),
              [](const Phase1AggregateRow &a, const Phase1AggregateRow &b) {
                  if (a.distribution == b.distribution) {
                      return a.size < b.size;
                  }
                  return a.distribution < b.distribution;
              });

    const std::vector<DistributionGateSummary> gate_summaries =
        summarizeDistributionGates(phase1_rows);
    const std::map<std::string, GateStatus> gate_map = buildDistributionGateMap(gate_summaries);

    printPhase1AggregateTable(phase1_aggregates);

    std::vector<HMMComparisonRow> hmm_rows;
    std::vector<std::string> skipped_hmm_distributions;
    if (options.hmm_stage) {
        const std::vector<CanonicalHMMSpec> canonical_specs = buildCanonicalHMMSpecs();
        for (const auto &spec : canonical_specs) {
            const auto gate_it = gate_map.find(spec.distribution);
            const GateStatus distribution_gate =
                (gate_it != gate_map.end()) ? gate_it->second : GateStatus::Block;
            if (distribution_gate == GateStatus::Block) {
                skipped_hmm_distributions.push_back(spec.distribution);
                continue;
            }

            for (int size : sizes) {
                std::cout << "[phase2] " << spec.distribution << " canonical_hmm n=" << size
                          << " ... ";
                const uint64_t seed = makeSeed(spec.distribution, "canonical_hmm", size, 0xBEEF);
                const std::vector<double> observations = generateHMMObservations(spec, size, seed);

                const HMMRunResult libhmm_run =
                    runLibhmmHMM(spec, observations, options.warmup, options.repeats);
                const HMMRunResult stoch_run =
                    runStochManualHMM(spec, observations, options.warmup, options.repeats);

                HMMComparisonRow row;
                row.distribution = spec.distribution;
                row.size = size;
                row.libhmm_forward_ms = libhmm_run.forward_ms;
                row.stoch_forward_ms = stoch_run.forward_ms;
                row.libhmm_viterbi_ms = libhmm_run.viterbi_ms;
                row.stoch_viterbi_ms = stoch_run.viterbi_ms;
                row.libhmm_forward_throughput = libhmm_run.forward_throughput;
                row.stoch_forward_throughput = stoch_run.forward_throughput;
                row.speedup_stoch_over_libhmm =
                    safeSpeedup(row.libhmm_forward_throughput, row.stoch_forward_throughput);
                row.speedup_libhmm_over_stoch = safeReciprocal(row.speedup_stoch_over_libhmm);
                row.libhmm_log_likelihood = libhmm_run.log_likelihood;
                row.stoch_log_likelihood = stoch_run.log_likelihood;
                row.abs_log_likelihood_diff =
                    std::abs(row.libhmm_log_likelihood - row.stoch_log_likelihood);
                row.success = libhmm_run.success && stoch_run.success;
                if (!libhmm_run.success || !stoch_run.success) {
                    std::ostringstream notes;
                    if (!libhmm_run.success) {
                        notes << "libhmm_error=" << libhmm_run.error;
                    }
                    if (!stoch_run.success) {
                        if (!notes.str().empty()) {
                            notes << ";";
                        }
                        notes << "stochhmm_error=" << stoch_run.error;
                    }
                    row.notes = notes.str();
                }
                hmm_rows.push_back(row);

                if (row.success) {
                    std::cout << "done"
                              << " | fwd|ΔlogL|=" << std::scientific << std::setprecision(3)
                              << row.abs_log_likelihood_diff << std::fixed << std::setprecision(3)
                              << " | speedup(libhmm/stoch)=" << row.speedup_libhmm_over_stoch
                              << "x\n";
                } else {
                    std::cout << "failed"
                              << " | " << row.notes << "\n";
                }
            }
        }
    }

    printHMMSummaryTable(hmm_rows);

    const std::string overall_assessment =
        buildOverallAssessment(gate_summaries, options.hmm_stage, skipped_hmm_distributions);

    const std::string timestamp = timestampNow();
    const fs::path csv_path =
        output_dir / ("libhmm_vs_stochhmm_distribution_suite_" + timestamp + ".csv");
    const fs::path json_path =
        output_dir / ("libhmm_vs_stochhmm_distribution_suite_" + timestamp + ".json");

    writeCSV(csv_path, phase1_rows, hmm_rows);
    writeJSON(json_path, options, sizes, phase1_rows, phase1_aggregates, gate_summaries, hmm_rows,
              skipped_hmm_distributions, overall_assessment, csv_path);

    std::cout << "\nOVERALL ASSESSMENT\n";
    std::cout << "------------------\n";
    std::cout << overall_assessment << "\n";
    std::cout << "\nArtifacts written:\n";
    std::cout << "  CSV:  " << csv_path << "\n";
    std::cout << "  JSON: " << json_path << "\n";

    return 0;
}
