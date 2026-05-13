/**
 * elk_movement_example — Animal movement state detection using libhmm.
 *
 * Fits a 2-state, joint Gamma + von Mises HMM to elk GPS step lengths and
 * turning angles from Morales et al. (2004). Directly comparable to the
 * canonical moveHMM (R package) reference fit.
 *
 * Data preparation:
 *   Rscript scripts/prepare_elk_data.R
 *   Exports elk_<id>_obs.csv per animal to /tmp/ (or pass dir as arg 1).
 *
 * moveHMM reference fit (Gamma + von Mises, 2 states, April 2026):
 *   State 0 (encamped):   step mean=373.8m sd=399.0m; angle kappa=0.592
 *   State 1 (travelling): step mean=3247.3m sd=4393.5m; angle kappa=0.208
 *   Transition: [[0.912, 0.088], [0.200, 0.800]]
 *   moveHMM log-likelihood: -6935.6; wall time: ~2000ms
 *
 * Reference: Michelot T, Langrock R, Patterson TA (2016). moveHMM.
 *   Methods in Ecology and Evolution, 7(11), 1308-1315.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libhmm/libhmm.h"

using namespace libhmm;

struct Sequence {
    std::vector<double> steps;
    std::vector<double> angles;
    std::size_t size() const { return steps.size(); }
};

static Sequence read_obs_csv(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    Sequence seq;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        const std::size_t c = line.find(',');
        if (c == std::string::npos)
            continue;
        seq.steps.push_back(std::stod(line.substr(0, c)));
        seq.angles.push_back(std::stod(line.substr(c + 1)));
    }
    return seq;
}

static double lse2(double a, double b) {
    const double m = std::max(a, b);
    if (!std::isfinite(m))
        return m;
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

// ---------------------------------------------------------------------------
// Joint 2-state Gamma × VonMises HMM with log-space Baum-Welch EM
// ---------------------------------------------------------------------------
struct JointHMM {
    double pi[2] = {0.5, 0.5};
    double A[2][2] = {{0.9, 0.1}, {0.1, 0.9}};
    GammaDistribution gd[2];
    VonMisesDistribution vm[2];

    double seq_loglik(const Sequence &seq) const {
        const std::size_t T = seq.size();
        std::array<double, 2> la;
        for (int s = 0; s < 2; ++s)
            la[s] = std::log(pi[s]) + gd[s].getLogProbability(seq.steps[0]) +
                    vm[s].getLogProbability(seq.angles[0]);
        for (std::size_t t = 1; t < T; ++t) {
            std::array<double, 2> la2;
            for (int s = 0; s < 2; ++s) {
                const double e =
                    gd[s].getLogProbability(seq.steps[t]) + vm[s].getLogProbability(seq.angles[t]);
                la2[s] = lse2(la[0] + std::log(A[0][s]), la[1] + std::log(A[1][s])) + e;
            }
            la = la2;
        }
        return lse2(la[0], la[1]);
    }

    double total_loglik(const std::vector<Sequence> &seqs) const {
        double ll = 0.0;
        for (const auto &s : seqs)
            ll += seq_loglik(s);
        return ll;
    }

    // One EM iteration; returns total log-likelihood
    double train_once(const std::vector<Sequence> &seqs) {
        // Concatenated observations and per-state responsibilities
        std::vector<double> all_steps, all_angles;
        std::vector<double> gamma0, gamma1;
        double xiSum[2][2] = {{0, 0}, {0, 0}};
        double total_ll = 0.0;

        for (const auto &seq : seqs) {
            for (double x : seq.steps)
                all_steps.push_back(x);
            for (double x : seq.angles)
                all_angles.push_back(x);
        }

        std::size_t obs_offset = 0;
        for (const auto &seq : seqs) {
            const std::size_t T = seq.size();

            // Forward
            std::vector<std::array<double, 2>> la(T);
            for (int s = 0; s < 2; ++s)
                la[0][s] = std::log(pi[s]) + gd[s].getLogProbability(seq.steps[0]) +
                           vm[s].getLogProbability(seq.angles[0]);
            for (std::size_t t = 1; t < T; ++t) {
                for (int s = 0; s < 2; ++s) {
                    const double e = gd[s].getLogProbability(seq.steps[t]) +
                                     vm[s].getLogProbability(seq.angles[t]);
                    la[t][s] =
                        lse2(la[t - 1][0] + std::log(A[0][s]), la[t - 1][1] + std::log(A[1][s])) +
                        e;
                }
            }
            const double ll = lse2(la[T - 1][0], la[T - 1][1]);
            total_ll += ll;

            // Backward
            std::vector<std::array<double, 2>> lb(T);
            lb[T - 1][0] = lb[T - 1][1] = 0.0;
            for (int t = static_cast<int>(T) - 2; t >= 0; --t) {
                for (int s = 0; s < 2; ++s) {
                    double v = -std::numeric_limits<double>::infinity();
                    for (int s2 = 0; s2 < 2; ++s2) {
                        const double e = gd[s2].getLogProbability(seq.steps[t + 1]) +
                                         vm[s2].getLogProbability(seq.angles[t + 1]);
                        v = lse2(v, std::log(A[s][s2]) + e + lb[t + 1][s2]);
                    }
                    lb[t][s] = v;
                }
            }

            // γ
            for (std::size_t t = 0; t < T; ++t) {
                const double denom = lse2(la[t][0] + lb[t][0], la[t][1] + lb[t][1]);
                const double g0 = std::exp(la[t][0] + lb[t][0] - denom);
                const double g1 = std::exp(la[t][1] + lb[t][1] - denom);
                gamma0.push_back(g0);
                gamma1.push_back(g1);
            }

            // ξ
            for (std::size_t t = 0; t < T - 1; ++t) {
                for (int s = 0; s < 2; ++s) {
                    for (int s2 = 0; s2 < 2; ++s2) {
                        const double e = gd[s2].getLogProbability(seq.steps[t + 1]) +
                                         vm[s2].getLogProbability(seq.angles[t + 1]);
                        xiSum[s][s2] +=
                            std::exp(la[t][s] + std::log(A[s][s2]) + e + lb[t + 1][s2] - ll);
                    }
                }
            }
            obs_offset += T;
        }

        // M-step: transition
        for (int s = 0; s < 2; ++s) {
            const double row = xiSum[s][0] + xiSum[s][1];
            if (row > 0.0) {
                A[s][0] = xiSum[s][0] / row;
                A[s][1] = xiSum[s][1] / row;
            }
        }

        // M-step: emissions
        const std::span<const double> step_sp(all_steps);
        const std::span<const double> ang_sp(all_angles);
        gd[0].fit(step_sp, std::span<const double>(gamma0));
        gd[1].fit(step_sp, std::span<const double>(gamma1));
        vm[0].fit(ang_sp, std::span<const double>(gamma0));
        vm[1].fit(ang_sp, std::span<const double>(gamma1));

        return total_ll;
    }
};

static double gmean(double k, double t) {
    return k * t;
}
static double gsd(double k, double t) {
    return std::sqrt(k) * t;
}

int main(int argc, char *argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "/tmp";

    std::cout << "Elk Movement State Detection — libhmm (Gamma + von Mises)\n";
    std::cout << "vs moveHMM reference\n";
    std::cout << "============================================================\n\n";

    const std::vector<std::string> elk_ids = {"elk_115", "elk_163", "elk_287", "elk_363"};
    std::vector<Sequence> seqs;
    std::size_t n_total = 0;
    std::cout << "Loading from " << data_dir << "/\n";
    for (const auto &id : elk_ids) {
        try {
            auto s = read_obs_csv(data_dir + "/" + id + "_obs.csv");
            n_total += s.size();
            std::cout << "  " << id << ": " << s.size() << " obs\n";
            seqs.push_back(std::move(s));
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << "\nRun: Rscript scripts/prepare_elk_data.R\n";
            return 1;
        }
    }
    std::cout << "Total: " << n_total << " observations\n\n";

    JointHMM model;
    model.gd[0] = GammaDistribution(1.0, 100.0);
    model.gd[1] = GammaDistribution(1.0, 1000.0);
    model.vm[0] = VonMisesDistribution(0.0, 0.1);
    model.vm[1] = VonMisesDistribution(0.0, 1.0);

    std::cout << "Initial parameters:\n";
    std::cout << "  State 0: Gamma(k=1, θ=100)  VonMises(μ=0, κ=0.1)\n";
    std::cout << "  State 1: Gamma(k=1, θ=1000) VonMises(μ=0, κ=1.0)\n\n";

    std::cout << "Baum-Welch training:\n";
    std::cout << std::setw(8) << "iter" << std::setw(16) << "log-likelihood" << std::setw(12)
              << "delta\n";
    std::cout << std::string(36, '-') << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    double prev_ll = model.total_loglik(seqs);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(8) << 0 << std::setw(16) << prev_ll << "  (initial)\n";

    int conv = -1;
    for (int iter = 1; iter <= 200; ++iter) {
        const double ll = model.train_once(seqs);
        const double d = ll - prev_ll;
        std::cout << std::setw(8) << iter << std::setw(16) << ll << std::setw(12) << d;
        if (iter > 1 && std::fabs(d) < 1e-4) {
            std::cout << "  <- converged";
            if (conv < 0)
                conv = iter;
        }
        std::cout << "\n";
        if (conv > 0 && iter >= conv + 2)
            break;
        prev_ll = ll;
    }

    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const double fll = model.total_loglik(seqs);
    std::cout << "\n=== libhmm results ===\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_ms << " ms\n\n";
    std::cout << std::setprecision(4);

    for (int s = 0; s < 2; ++s) {
        std::cout << "State " << s << " (" << (s == 0 ? "encamped  " : "travelling") << "):\n";
        std::cout << "  Gamma:    k=" << model.gd[s].getK() << "  theta=" << model.gd[s].getTheta()
                  << "  mean=" << gmean(model.gd[s].getK(), model.gd[s].getTheta()) << "m"
                  << "  sd=" << gsd(model.gd[s].getK(), model.gd[s].getTheta()) << "m\n";
        std::cout << "  VonMises: mu=" << model.vm[s].getMu()
                  << "  kappa=" << model.vm[s].getKappa() << "\n";
    }
    std::cout << "\nTransition matrix:\n";
    std::cout << "  [[" << model.A[0][0] << ", " << model.A[0][1] << "],\n";
    std::cout << "   [" << model.A[1][0] << ", " << model.A[1][1] << "]]\n";
    std::cout << "\nLog-likelihood: " << fll << "\n\n";

    // Comparison
    std::cout << "=== Comparison: libhmm vs moveHMM ===\n\n";
    const struct {
        const char *name;
        double lib, ref;
    } rows[] = {
        {"State 0 step mean (m)", gmean(model.gd[0].getK(), model.gd[0].getTheta()), 373.8},
        {"State 0 step sd (m)", gsd(model.gd[0].getK(), model.gd[0].getTheta()), 399.0},
        {"State 0 angle kappa", model.vm[0].getKappa(), 0.592},
        {"State 1 step mean (m)", gmean(model.gd[1].getK(), model.gd[1].getTheta()), 3247.3},
        {"State 1 step sd (m)", gsd(model.gd[1].getK(), model.gd[1].getTheta()), 4393.5},
        {"State 1 angle kappa", model.vm[1].getKappa(), 0.208},
        {"A[0->0]", model.A[0][0], 0.9115},
        {"A[1->0]", model.A[1][0], 0.2002},
    };
    std::cout << std::setw(24) << " " << std::setw(14) << "libhmm" << std::setw(14) << "moveHMM\n";
    std::cout << std::string(52, '-') << "\n";
    for (const auto &r : rows)
        std::cout << std::setw(24) << r.name << std::setw(14) << r.lib << std::setw(14) << r.ref
                  << "\n";
    const std::string wall_str = std::to_string(static_cast<int>(wall_ms)) + " ms";
    std::cout << std::setw(24) << "Wall time" << std::setw(14) << wall_str << std::setw(14)
              << "~2000 ms\n";

    std::cout << "\nNotes:\n";
    std::cout << "  Both models: joint Gamma+VonMises, 4 separate animal tracks,\n";
    std::cout << "  conditional independence of step length and turning angle.\n";
    std::cout << "  moveHMM additionally models the single zero step (zero-mass parameter).\n";

    return 0;
}
