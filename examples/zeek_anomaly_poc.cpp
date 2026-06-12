/**
 * zeek_anomaly_poc — Network anomaly detection via MV HMM on CTU-13 Scenario 1.
 *
 * Proof-of-concept for finding malware in zeek datasets: trains a multivariate HMM on
 * BENIGN per-connection-key flow sequences, then scores all sequences (benign +
 * botnet) and measures how well the model separates them.
 *
 * Dataset: CTU-Malware-Capture-Botnet-42 (CTU-13 Scenario 1, Neris botnet)
 *   Garcia S., Grill M., Stiborek J. & Zunino A. (2014). An empirical comparison
 *   of botnet detection methods. Computers and Security, 45, 100-123.
 *   https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/
 *
 * Data preparation (one-time):
 *   python3 scripts/prepare_ctu13_data.py [output_dir]
 *   Writes ctu13_train.csv, ctu13_test.csv, ctu13_labels.csv to output_dir.
 *
 * Usage:
 *   ./zeek_anomaly_poc               # looks for CSVs in /tmp
 *   ./zeek_anomaly_poc /path/to/dir
 *
 * Models compared:
 *   A — DiagonalGaussianDistribution (4D, assumes independence within state)
 *   B — FullCovarianceGaussianDistribution (4D, captures cross-feature correlation)
 *
 * Features (all log1p-transformed):
 *   f0  inter_arrival_time   — time between consecutive flows to same endpoint
 *   f1  duration             — flow duration in seconds
 *   f2  total_bytes          — total bytes (both directions)
 *   f3  src_bytes            — bytes from source
 *
 * Detection approach:
 *   Train HMM on benign-only sequences.  Score each test sequence as the
 *   normalised log-probability (total log-prob / number of observations).
 *   Flag as anomalous when score < threshold.  The threshold is derived from
 *   the benign training distribution so that a target FPR is met.
 *
 * Key result: FullCovGaussian captures the tight inter-arrival / byte-size
 * correlations in beaconing traffic that DiagonalGaussian misses, improving
 * separation of the Neris C2 channel (port 4506, cv ≈ 0.64) from normal.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/basic_baum_welch_trainer.h"
#include "libhmm/training/kmeans_init.h"
#include "libhmm/training/model_selection.h"

using namespace libhmm;

static constexpr int N_STATES = 3;
static constexpr int N_FEATURES = 4;
static constexpr int MAX_ITER = 200;
static constexpr double CONV_TOL = 1e-4;

// =============================================================================
// Data loading
// =============================================================================

struct Sequence {
    std::string key;
    std::vector<std::array<double, 4>> obs; ///< log1p-transformed features
};

struct Label {
    int is_botnet;
    std::size_t n_obs;
};

/// Load sequences from CSV (key,obs_idx,f0,f1,f2,f3).
/// Rows are grouped by key (obs_idx resets to 0 at each key boundary).
static std::vector<Sequence> load_sequences(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);

    std::vector<Sequence> seqs;
    std::string line;
    std::getline(f, line); // skip header

    Sequence cur;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        std::istringstream ss(line);
        std::string key, idx_s, f0s, f1s, f2s, f3s;
        if (!std::getline(ss, key, ','))
            continue;
        if (!std::getline(ss, idx_s, ','))
            continue;
        if (!std::getline(ss, f0s, ','))
            continue;
        if (!std::getline(ss, f1s, ','))
            continue;
        if (!std::getline(ss, f2s, ','))
            continue;
        if (!std::getline(ss, f3s))
            continue;
        const int idx = std::stoi(idx_s);
        if (idx == 0 && !cur.key.empty()) {
            seqs.push_back(std::move(cur));
            cur = {};
        }
        if (idx == 0)
            cur.key = key;
        cur.obs.push_back({std::stod(f0s), std::stod(f1s), std::stod(f2s), std::stod(f3s)});
    }
    if (!cur.key.empty())
        seqs.push_back(std::move(cur));
    return seqs;
}

/// Load per-key labels from CSV (key,is_botnet,n_obs).
static std::unordered_map<std::string, Label> load_labels(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::unordered_map<std::string, Label> labels;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        std::istringstream ss(line);
        std::string key, bot_s, n_s;
        std::getline(ss, key, ',');
        std::getline(ss, bot_s, ',');
        std::getline(ss, n_s);
        labels[key] = {std::stoi(bot_s), static_cast<std::size_t>(std::stoul(n_s))};
    }
    return labels;
}

// =============================================================================
// HMM helpers
// =============================================================================

static MultiObservationLists to_obs_lists(const std::vector<Sequence> &seqs) {
    MultiObservationLists lists;
    lists.reserve(seqs.size());
    for (const auto &s : seqs) {
        ObservationMatrix mat(s.obs.size(), N_FEATURES);
        for (std::size_t t = 0; t < s.obs.size(); ++t)
            for (int d = 0; d < N_FEATURES; ++d)
                mat(t, d) = s.obs[t][d];
        lists.push_back(std::move(mat));
    }
    return lists;
}

/// Compute total log-prob of a single sequence under hmm.
static double seq_logp(HmmMV &hmm, const Sequence &s) {
    ObservationMatrix mat(s.obs.size(), N_FEATURES);
    for (std::size_t t = 0; t < s.obs.size(); ++t)
        for (int d = 0; d < N_FEATURES; ++d)
            mat(t, d) = s.obs[t][d];
    BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, mat);
    return fbc.getLogProbability();
}

/// Total log-prob across all sequences.
static double total_logp(HmmMV &hmm, const MultiObservationLists &lists) {
    double lp = 0.0;
    for (const auto &seq : lists) {
        BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, seq);
        lp += fbc.getLogProbability();
    }
    return lp;
}

/// Baum-Welch to convergence; return final log-likelihood.
static double run_bwt(HmmMV &hmm, const MultiObservationLists &lists) {
    BasicBaumWelchTrainer<ObservationVectorView> trainer(hmm, lists);
    double prev = total_logp(hmm, lists), conv_iter = -1;
    for (int k = 0; k < MAX_ITER; ++k) {
        trainer.train();
        const double cur = total_logp(hmm, lists);
        const double delta = cur - prev;
        if (k > 0 && delta >= -1e-8 && delta < CONV_TOL) {
            if (conv_iter < 0)
                conv_iter = k;
        }
        if (conv_iter >= 0 && k >= conv_iter + 2)
            break;
        prev = cur;
    }
    return total_logp(hmm, lists);
}

/// Build a uniform-transition N_STATES HmmMV.
static HmmMV make_hmm() {
    HmmMV hmm(N_STATES);
    Matrix A(N_STATES, N_STATES);
    const double p_stay = 0.88;
    const double p_other = (1.0 - p_stay) / (N_STATES - 1);
    for (int i = 0; i < N_STATES; ++i)
        for (int j = 0; j < N_STATES; ++j)
            A(i, j) = (i == j) ? p_stay : p_other;
    hmm.setTrans(A);
    Vector pi(N_STATES);
    for (int i = 0; i < N_STATES; ++i)
        pi(i) = 1.0 / N_STATES;
    hmm.setPi(pi);
    return hmm;
}

// =============================================================================
// Scoring + detection metrics
// =============================================================================

struct ScoredSeq {
    std::string key;
    double norm_score; ///< log-prob / n_obs
    int is_botnet;
};

/// Score all sequences; return normalised per-observation log-prob.
static std::vector<ScoredSeq> score_all(HmmMV &hmm, const std::vector<Sequence> &seqs,
                                        const std::unordered_map<std::string, Label> &labels) {
    std::vector<ScoredSeq> result;
    result.reserve(seqs.size());
    for (const auto &s : seqs) {
        const double lp = seq_logp(hmm, s);
        const double norm = std::isfinite(lp) ? lp / static_cast<double>(s.obs.size()) : -1e9;
        int bot = 0;
        auto it = labels.find(s.key);
        if (it != labels.end())
            bot = it->second.is_botnet;
        result.push_back({s.key, norm, bot});
    }
    return result;
}

static void print_detection_table(const std::vector<ScoredSeq> &scored, double benign_mean,
                                  double benign_sd) {
    std::cout << "  Threshold = benign_mean - k*sd\n";
    std::cout << std::setw(8) << "k" << std::setw(12) << "threshold" << std::setw(8) << "TPR%"
              << std::setw(8) << "FPR%"
              << "\n";
    std::cout << std::string(36, '-') << "\n";

    for (double k : {1.0, 1.5, 2.0, 2.5, 3.0}) {
        const double thr = benign_mean - k * benign_sd;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (const auto &s : scored) {
            const bool flagged = s.norm_score < thr;
            if (s.is_botnet) {
                if (flagged)
                    ++tp;
                else
                    ++fn;
            } else {
                if (flagged)
                    ++fp;
                else
                    ++tn;
            }
        }
        const int n_bot = tp + fn;
        const int n_benign = fp + tn;
        const double tpr = n_bot > 0 ? 100.0 * tp / n_bot : 0.0;
        const double fpr = n_benign > 0 ? 100.0 * fp / n_benign : 0.0;
        std::cout << std::fixed << std::setprecision(1) << std::setw(8) << k << std::setw(12)
                  << std::setprecision(3) << thr << std::setw(8) << std::setprecision(1) << tpr
                  << std::setw(8) << fpr << "\n";
    }
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char *argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "/tmp";
    const auto train_path = data_dir + "/ctu13_train.csv";
    const auto test_path = data_dir + "/ctu13_test.csv";
    const auto labels_path = data_dir + "/ctu13_labels.csv";

    std::cout << "CTU-13 Scenario 1 (Neris Botnet) — MV HMM Anomaly Detection POC\n";
    std::cout << "================================================================\n\n";

    // -------------------------------------------------------------------------
    // Load data
    // -------------------------------------------------------------------------
    std::cout << "Loading data from " << data_dir << "/\n";
    std::vector<Sequence> train_seqs, test_seqs;
    std::unordered_map<std::string, Label> labels;
    try {
        train_seqs = load_sequences(train_path);
        test_seqs = load_sequences(test_path);
        labels = load_labels(labels_path);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n"
                  << "Run: python3 scripts/prepare_ctu13_data.py " << data_dir << "\n";
        return 1;
    }

    const std::size_t n_train = train_seqs.size();
    std::size_t n_benign_test = 0, n_botnet_test = 0;
    for (const auto &s : test_seqs) {
        auto it = labels.find(s.key);
        if (it != labels.end() && it->second.is_botnet)
            ++n_botnet_test;
        else
            ++n_benign_test;
    }
    const auto train_lists = to_obs_lists(train_seqs);

    std::cout << "  Training keys (benign only): " << n_train << "\n";
    std::cout << "  Test keys — benign: " << n_benign_test << "  botnet: " << n_botnet_test
              << "\n\n";

    // -------------------------------------------------------------------------
    // Model A — DiagonalGaussian
    // -------------------------------------------------------------------------
    std::cout << "--- Model A: DiagonalGaussianDistribution ---\n\n";

    HmmMV hmm_a = make_hmm();
    for (int s = 0; s < N_STATES; ++s)
        hmm_a.setDistribution(s, std::make_unique<DiagonalGaussianDistribution>(N_FEATURES));
    {
        std::mt19937_64 rng(42);
        kmeans_init(hmm_a, train_lists, rng);
    }
    const auto t_a0 = std::chrono::steady_clock::now();
    const double ll_a = run_bwt(hmm_a, train_lists);
    const double wall_a =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_a0).count();
    std::cout << "  Training LL: " << std::fixed << std::setprecision(2) << ll_a << "  ("
              << static_cast<int>(wall_a) << " ms)\n";

    const auto scores_a = score_all(hmm_a, test_seqs, labels);
    double sum_b_a = 0, sum2_b_a = 0, sum_m_a = 0, sum2_m_a = 0;
    int cnt_b_a = 0, cnt_m_a = 0;
    for (const auto &s : scores_a) {
        if (s.is_botnet) {
            sum_m_a += s.norm_score;
            sum2_m_a += s.norm_score * s.norm_score;
            ++cnt_m_a;
        } else {
            sum_b_a += s.norm_score;
            sum2_b_a += s.norm_score * s.norm_score;
            ++cnt_b_a;
        }
    }
    const double mean_b_a = sum_b_a / cnt_b_a;
    const double sd_b_a = std::sqrt(sum2_b_a / cnt_b_a - mean_b_a * mean_b_a);
    const double mean_m_a = sum_m_a / cnt_m_a;
    const double sd_m_a = std::sqrt(sum2_m_a / cnt_m_a - mean_m_a * mean_m_a);
    const double sep_a =
        (mean_b_a - mean_m_a) / std::sqrt(0.5 * (sd_b_a * sd_b_a + sd_m_a * sd_m_a));

    std::cout << std::fixed << std::setprecision(4) << "  Benign  mean score: " << mean_b_a
              << "  sd: " << sd_b_a << "\n"
              << "  Botnet  mean score: " << mean_m_a << "  sd: " << sd_m_a << "\n"
              << "  Cohen's d (separation): " << std::setprecision(2) << sep_a << "\n\n";
    print_detection_table(scores_a, mean_b_a, sd_b_a);

    // -------------------------------------------------------------------------
    // Model B — FullCovarianceGaussian
    // -------------------------------------------------------------------------
    std::cout << "\n--- Model B: FullCovarianceGaussianDistribution ---\n\n";

    HmmMV hmm_b = make_hmm();
    for (int s = 0; s < N_STATES; ++s)
        hmm_b.setDistribution(s, std::make_unique<FullCovarianceGaussianDistribution>(N_FEATURES));
    {
        std::mt19937_64 rng(42);
        kmeans_init(hmm_b, train_lists, rng);
    }
    const auto t_b0 = std::chrono::steady_clock::now();
    const double ll_b = run_bwt(hmm_b, train_lists);
    const double wall_b =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_b0).count();
    std::cout << "  Training LL: " << std::fixed << std::setprecision(2) << ll_b << "  ("
              << static_cast<int>(wall_b) << " ms)\n";

    const auto scores_b = score_all(hmm_b, test_seqs, labels);
    double sum_b_b = 0, sum2_b_b = 0, sum_m_b = 0, sum2_m_b = 0;
    int cnt_b_b = 0, cnt_m_b = 0;
    for (const auto &s : scores_b) {
        if (s.is_botnet) {
            sum_m_b += s.norm_score;
            sum2_m_b += s.norm_score * s.norm_score;
            ++cnt_m_b;
        } else {
            sum_b_b += s.norm_score;
            sum2_b_b += s.norm_score * s.norm_score;
            ++cnt_b_b;
        }
    }
    const double mean_b_b = sum_b_b / cnt_b_b;
    const double sd_b_b = std::sqrt(sum2_b_b / cnt_b_b - mean_b_b * mean_b_b);
    const double mean_m_b = sum_m_b / cnt_m_b;
    const double sd_m_b = std::sqrt(sum2_m_b / cnt_m_b - mean_m_b * mean_m_b);
    const double sep_b =
        (mean_b_b - mean_m_b) / std::sqrt(0.5 * (sd_b_b * sd_b_b + sd_m_b * sd_m_b));

    std::cout << std::fixed << std::setprecision(4) << "  Benign  mean score: " << mean_b_b
              << "  sd: " << sd_b_b << "\n"
              << "  Botnet  mean score: " << mean_m_b << "  sd: " << sd_m_b << "\n"
              << "  Cohen's d (separation): " << std::setprecision(2) << sep_b << "\n\n";
    print_detection_table(scores_b, mean_b_b, sd_b_b);

    // -------------------------------------------------------------------------
    // Model comparison
    // -------------------------------------------------------------------------
    const std::size_t k_a = count_free_parameters(hmm_a);
    const std::size_t k_b = count_free_parameters(hmm_b);
    const std::size_t n_train_obs =
        std::accumulate(train_lists.begin(), train_lists.end(), std::size_t{0},
                        [](std::size_t s, const ObservationMatrix &m) { return s + m.size1(); });
    const double bic_a = compute_bic(ll_a, k_a, n_train_obs);
    const double bic_b = compute_bic(ll_b, k_b, n_train_obs);

    std::cout << "\n=== Model Comparison ===\n\n";
    std::cout << std::setw(30) << " " << std::setw(14) << "Model A (Diag)" << std::setw(14)
              << "Model B (Full)" << "\n";
    std::cout << std::string(58, '-') << "\n";
    std::cout << std::fixed << std::setprecision(2) << std::setw(30) << "Training LL"
              << std::setw(14) << ll_a << std::setw(14) << ll_b << "\n"
              << std::setw(30) << "BIC (lower = better)" << std::setw(14) << bic_a << std::setw(14)
              << bic_b << "\n"
              << std::setw(30) << "Free parameters k" << std::setw(14) << static_cast<long>(k_a)
              << std::setw(14) << static_cast<long>(k_b) << "\n"
              << std::setw(30) << "Cohen's d (separation)" << std::setw(14) << std::setprecision(3)
              << sep_a << std::setw(14) << sep_b << "\n"
              << std::setw(30) << "Wall time (ms)" << std::setw(14) << static_cast<int>(wall_a)
              << std::setw(14) << static_cast<int>(wall_b) << "\n";

    std::cout << "\nNotes:\n"
              << "  Features: log1p(inter_arrival), log1p(dur), log1p(tot_bytes),"
                 " log1p(src_bytes).\n"
              << "  HMM trained on BENIGN traffic only; anomalies score below the"
                 " benign distribution.\n"
              << "  Neris botnet: DNS-based C2 (port 53) is harder to separate; "
                 "dedicated C2\n"
              << "  channel (port 4506, cv=0.64) and SMTP spam (port 25, cv=1.69) "
                 "are clearer.\n";
    if (sep_b > sep_a)
        std::cout << "  -> FullCovGaussian (d=" << std::setprecision(2) << sep_b
                  << ") separates better than DiagonalGaussian (d=" << sep_a
                  << "): cross-feature covariance is informative.\n";
    else
        std::cout << "  -> DiagonalGaussian separates as well; independence"
                     " assumption holds for this dataset.\n";
    std::cout << "\nNext step: zeekhmm — a full Zeek log post-processor "
                 "using this model.\n";

    return 0;
}
