/**
 * dax_regime_example — Market regime detection on DAX log-returns.
 *
 * Fits a 3-state, location-scale Student-t HMM to DAX daily log-returns
 * (2000-01-01 to 2022-12-31) using Baum-Welch EM. Directly comparable
 * to the fHMM R package reference fit.
 *
 * Dataset
 * -------
 * DAX (^GDAXI) daily closing prices, Yahoo Finance, 5838 trading days.
 * Log-returns r_t = log(P_t / P_{t-1}). Covers the 9/11 aftermath,
 * 2008 financial crisis, 2020 COVID crash, and the 2022 bear market.
 *
 * Data preparation
 * ----------------
 * Rscript scripts/prepare_dax_data.R [output_dir]
 * Exports dax_logreturns.csv to /tmp/ (or pass dir as arg 1).
 *
 * Model
 * -----
 * 3 states: Bearish (high volatility, negative drift),
 *           Neutral  (moderate volatility, near-zero drift),
 *           Bullish  (low volatility, positive drift)
 * Emission: StudentTDistribution(nu, mu, sigma) per state
 * Training: BaumWelchTrainer (canonical log-space EM)
 *
 * Initial parameters:
 *   State 0 (bearish):  nu=10, mu=-0.002,  sigma=0.025
 *   State 1 (neutral):  nu=30, mu= 0.000,  sigma=0.012
 *   State 2 (bullish):  nu= 5, mu= 0.001,  sigma=0.006
 *
 * fHMM reference fit (3-state Student-t, fHMM 1.2.0, April 2026)
 * ---------------------------------------------------------------
 * Data: ^GDAXI, 5838 log-returns, 2000-01-01 to 2022-12-31
 * fHMM uses numerical optimizer (nlm) on the full parameter likelihood,
 * not direct EM — this drives the large wall-time difference.
 *
 *   State 1 (bearish):  mu=-0.00180, sigma=0.02629, nu=11.2  | 702 days
 *   State 2 (bullish):  mu=+0.00126, sigma=0.00600, nu= 5.3  | 2362 days
 *   State 3 (neutral):  mu=-0.00031, sigma=0.01330, nu=91.2  | 2774 days
 *   Transition: mostly diagonal; bearish <-> bullish only via neutral
 *   Log-likelihood: 17485.7
 *   fHMM wall time: ~1360s (22+ minutes on Intel Ivy Bridge)
 *
 * Notes on comparison
 * -------------------
 * Both models maximise the same log-likelihood. Results may differ due to:
 *   - Different optimisation algorithms (nlm vs direct EM)
 *   - Different starting values and local optima
 *   - fHMM's nu=91 (near-Gaussian neutral state) is hard to estimate
 *     reliably from kurtosis with a fraction of the data
 *
 * Reference
 * ---------
 * Oelschläger L, Adam T, Michels R (2024). fHMM: Hidden Markov Models for
 * Financial Time Series in R. J. Statistical Software, 109(9).
 * https://doi.org/10.18637/jss.v109.i09
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libhmm/libhmm.h"

using namespace libhmm;

static ObservationSet read_csv_column(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<double> vals;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (!line.empty())
            vals.push_back(std::stod(line));
    }
    ObservationSet seq(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i)
        seq(i) = vals[i];
    return seq;
}

static double total_loglik(const Hmm &hmm, const ObservationLists &obs) {
    double ll = 0.0;
    for (const auto &seq : obs)
        ll += ForwardBackwardCalculator(hmm, seq).getLogProbability();
    return ll;
}

// Return a StudentTDistribution as a const ref for clean parameter access
static const StudentTDistribution &std_t(const Hmm &hmm, std::size_t state) {
    return static_cast<const StudentTDistribution &>(hmm.getDistribution(state));
}

int main(int argc, char *argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "/tmp";

    std::cout << "DAX Market Regime Detection — libhmm vs fHMM\n";
    std::cout << "=============================================\n\n";

    // -----------------------------------------------------------------------
    // Load data
    // -----------------------------------------------------------------------
    const std::string path = data_dir + "/dax_logreturns.csv";
    ObservationSet returns;
    try {
        returns = read_csv_column(path);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\nRun: Rscript scripts/prepare_dax_data.R\n";
        return 1;
    }

    ObservationLists obs = {returns};
    std::cout << "DAX log-returns loaded: " << returns.size()
              << " daily observations (2000-2022)\n";
    {
        double mn = returns(0), mx = returns(0), sum = 0.0;
        for (std::size_t i = 0; i < returns.size(); ++i) {
            mn = std::min(mn, returns(i));
            mx = std::max(mx, returns(i));
            sum += returns(i);
        }
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  min=" << mn << "  max=" << mx
                  << "  mean=" << sum / static_cast<double>(returns.size()) << "\n\n";
    }

    // -----------------------------------------------------------------------
    // HMM setup: 3 states, StudentT emissions
    // -----------------------------------------------------------------------
    Hmm hmm(3);

    Matrix trans(3, 3);
    trans(0, 0) = 0.95;
    trans(0, 1) = 0.03;
    trans(0, 2) = 0.02;
    trans(1, 0) = 0.02;
    trans(1, 1) = 0.95;
    trans(1, 2) = 0.03;
    trans(2, 0) = 0.02;
    trans(2, 1) = 0.03;
    trans(2, 2) = 0.95;
    hmm.setTrans(trans);

    Vector pi(3);
    pi(0) = 0.333;
    pi(1) = 0.334;
    pi(2) = 0.333;
    hmm.setPi(pi);

    // StudentTDistribution(nu, mu, sigma)
    hmm.setDistribution(0, std::make_unique<StudentTDistribution>(10.0, -0.002, 0.025));
    hmm.setDistribution(1, std::make_unique<StudentTDistribution>(30.0, 0.000, 0.012));
    hmm.setDistribution(2, std::make_unique<StudentTDistribution>(5.0, 0.001, 0.006));

    std::cout << "Initial parameters:\n";
    std::cout << "  State 0 (bearish):  nu=10  mu=-0.002  sigma=0.025\n";
    std::cout << "  State 1 (neutral):  nu=30  mu= 0.000  sigma=0.012\n";
    std::cout << "  State 2 (bullish):  nu= 5  mu= 0.001  sigma=0.006\n\n";

    // -----------------------------------------------------------------------
    // Baum-Welch EM
    // -----------------------------------------------------------------------
    std::cout << "Baum-Welch training:\n";
    std::cout << std::setw(8) << "iter" << std::setw(16) << "log-likelihood" << std::setw(12)
              << "delta\n";
    std::cout << std::string(36, '-') << "\n";

    BaumWelchTrainer trainer(&hmm, obs);
    const auto t_start = std::chrono::steady_clock::now();

    double prev_ll = total_loglik(hmm, obs);
    std::cout << std::setprecision(4) << std::fixed;
    std::cout << std::setw(8) << 0 << std::setw(16) << prev_ll << "  (initial)\n";

    int converged_at = -1;
    for (int iter = 1; iter <= 200; ++iter) {
        trainer.train();
        const double ll = total_loglik(hmm, obs);
        const double delta = ll - prev_ll;
        std::cout << std::setw(8) << iter << std::setw(16) << ll << std::setw(12) << delta;
        if (iter > 1 && std::fabs(delta) < 1e-4) {
            std::cout << "  <- converged";
            if (converged_at < 0)
                converged_at = iter;
        }
        std::cout << "\n";
        if (converged_at > 0 && iter >= converged_at + 2)
            break;
        prev_ll = ll;
    }

    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    const double final_ll = total_loglik(hmm, obs);

    // -----------------------------------------------------------------------
    // Results — sort states by sigma descending (bearish first) for readability
    // -----------------------------------------------------------------------
    // Determine labelling: state with highest sigma = bearish,
    // lowest sigma = bullish, middle = neutral
    std::array<int, 3> order = {0, 1, 2};
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return std_t(hmm, a).getScale() > std_t(hmm, b).getScale(); });
    const char *labels[3] = {"bearish ", "neutral ", "bullish "};

    std::cout << "\n=== libhmm results ===\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_s << " s\n\n";
    std::cout << std::setprecision(6);

    for (int i = 0; i < 3; ++i) {
        const int s = order[i];
        const auto &d = std_t(hmm, s);
        std::cout << "State " << s << " (" << labels[i] << "):"
                  << "  nu=" << std::setw(7) << d.getDegreesOfFreedom() << "  mu=" << std::setw(10)
                  << d.getLocation() << "  sigma=" << std::setw(10) << d.getScale() << "\n";
    }

    std::cout << "\nTransition matrix:\n";
    const Matrix &A = hmm.getTrans();
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 3; ++j)
            std::cout << std::setw(9) << std::setprecision(5) << A(i, j);
        std::cout << " ]\n";
    }

    std::cout << "\nLog-likelihood: " << std::setprecision(2) << final_ll << "\n\n";

    // Quick Viterbi state count
    ViterbiCalculator vc(hmm, returns);
    const StateSequence &vpath = vc.getStateSequence();
    std::array<int, 3> counts = {0, 0, 0};
    for (std::size_t t = 0; t < vpath.size(); ++t)
        ++counts[vpath(t)];
    std::cout << "Viterbi state occupancy: "
              << "s0=" << counts[0] << "  s1=" << counts[1] << "  s2=" << counts[2] << "\n\n";

    // -----------------------------------------------------------------------
    // Comparison with fHMM reference (sorted by sigma descending)
    // -----------------------------------------------------------------------
    std::cout << "=== Comparison: libhmm vs fHMM reference ===\n\n";
    std::cout << "(fHMM states sorted by sigma desc to match libhmm order)\n\n";

    // fHMM reference: bearish (sigma=0.02629), neutral (sigma=0.01330), bullish (sigma=0.00600)
    const struct {
        const char *label;
        double mu, sigma, nu;
    } ref[3] = {
        {"bearish ", -0.001803, 0.026290, 11.16},
        {"neutral ", -0.000310, 0.013300, 91.15},
        {"bullish ", 0.001257, 0.006003, 5.316},
    };

    std::cout << std::setw(18) << " " << std::setw(12) << "libhmm" << std::setw(12) << "fHMM\n";
    std::cout << std::string(42, '-') << "\n";
    std::cout << std::setprecision(6);
    for (int i = 0; i < 3; ++i) {
        const int s = order[i];
        const auto &d = std_t(hmm, s);
        std::cout << std::setw(18) << (std::string(ref[i].label) + " mu") << std::setw(12)
                  << d.getLocation() << std::setw(12) << ref[i].mu << "\n";
        std::cout << std::setw(18) << (std::string(ref[i].label) + " sigma") << std::setw(12)
                  << d.getScale() << std::setw(12) << ref[i].sigma << "\n";
        std::cout << std::setw(18) << (std::string(ref[i].label) + " nu") << std::setw(12)
                  << d.getDegreesOfFreedom() << std::setw(12) << ref[i].nu << "\n";
    }
    const std::string wall_str = std::to_string(static_cast<int>(wall_s)) + " s";
    std::cout << std::setw(18) << "Log-likelihood" << std::setw(12) << std::setprecision(1)
              << final_ll << std::setw(12) << "17485.7\n";
    std::cout << std::setw(18) << "Wall time" << std::setw(12) << wall_str << std::setw(12)
              << "~1360 s\n";

    std::cout << "\nNotes:\n";
    std::cout << "  fHMM uses nlm() optimizer (gradient-based); libhmm uses direct EM.\n";
    std::cout << "  mu and sigma typically match well; nu is harder to estimate\n";
    std::cout << "  reliably from kurtosis when a state covers few observations.\n";
    std::cout << "  fHMM's neutral state (nu=91) is near-Gaussian: our kurtosis\n";
    std::cout << "  estimate may not distinguish it sharply from Gaussian.\n";

    return 0;
}
