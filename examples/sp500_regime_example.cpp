/**
 * sp500_regime_example — Market regime detection on S&P 500 log-returns.
 *
 * Fits the same 3-state location-scale Student-t HMM used in
 * dax_regime_example to the S&P 500 index (^GSPC), 2000-2022.
 * Comparing the two sets of results reveals differences in US vs German
 * equity market regimes (crisis depth, volatility clustering, tail weight).
 *
 * Dataset
 * -------
 * ^GSPC daily closing prices, Yahoo Finance, 5786 trading days.
 * Log-returns r_t = log(P_t / P_{t-1}).
 * Covers: dot-com crash, 9/11, 2008 GFC, 2020 COVID, 2022 bear market.
 *
 * Data preparation
 * ----------------
 * Rscript scripts/prepare_dax_data.R [output_dir]
 * (The same script now exports both DAX and S&P 500 log-returns.)
 *
 * Model
 * -----
 * Same as dax_regime_example: 3 states (Bearish / Neutral / Bullish),
 * StudentTDistribution(nu, mu, sigma) per state via ECME.
 *
 * Cross-market comparison (DAX 3.7.0 results for reference)
 * ----------------------------------------------------------
 *   DAX bearish:  mu=-0.00179, sigma=0.02628, nu=11.1
 *   DAX neutral:  mu=-0.00028, sigma=0.01305, nu=36.1
 *   DAX bullish:  mu=+0.00126, sigma=0.00599, nu=5.4
 *   DAX LL: 17487.2   DAX wall time: ~2 s
 *
 * Reference
 * ---------
 * Oelschläger L, Adam T, Michels R (2024). fHMM: Hidden Markov Models for
 * Financial Time Series in R. J. Statistical Software, 109(9).
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
    std::vector<double> v;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line))
        if (!line.empty())
            v.push_back(std::stod(line));
    ObservationSet s(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        s(i) = v[i];
    return s;
}

static double ll(const Hmm &h, const ObservationLists &o) {
    double s = 0;
    for (const auto &q : o)
        s += ForwardBackwardCalculator(h, q).getLogProbability();
    return s;
}

static const StudentTDistribution &st(const Hmm &h, std::size_t s) {
    return static_cast<const StudentTDistribution &>(h.getDistribution(s));
}

int main(int argc, char *argv[]) {
    const std::string dir = (argc > 1) ? argv[1] : "/tmp";
    const std::string path = dir + "/sp500_logreturns.csv";

    std::cout << "S&P 500 Market Regime Detection — libhmm\n";
    std::cout << "=========================================\n\n";

    ObservationSet returns;
    try {
        returns = read_csv_column(path);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\nRun: Rscript scripts/prepare_dax_data.R\n";
        return 1;
    }

    ObservationLists obs = {returns};
    double mn = returns(0), mx = returns(0), sm = 0;
    for (std::size_t i = 0; i < returns.size(); ++i) {
        mn = std::min(mn, returns(i));
        mx = std::max(mx, returns(i));
        sm += returns(i);
    }
    std::cout << "S&P 500 log-returns: " << returns.size() << " daily observations (2000-2022)\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  min=" << mn << "  max=" << mx
              << "  mean=" << sm / static_cast<double>(returns.size()) << "\n\n";

    // 3-state StudentT HMM — same initial params as DAX
    Hmm hmm(3);
    Matrix A(3, 3);
    A(0, 0) = 0.95;
    A(0, 1) = 0.03;
    A(0, 2) = 0.02;
    A(1, 0) = 0.02;
    A(1, 1) = 0.95;
    A(1, 2) = 0.03;
    A(2, 0) = 0.02;
    A(2, 1) = 0.03;
    A(2, 2) = 0.95;
    hmm.setTrans(A);
    Vector pi(3);
    pi(0) = 0.333;
    pi(1) = 0.334;
    pi(2) = 0.333;
    hmm.setPi(pi);
    hmm.setDistribution(0, std::make_unique<StudentTDistribution>(10.0, -0.002, 0.025));
    hmm.setDistribution(1, std::make_unique<StudentTDistribution>(30.0, 0.000, 0.012));
    hmm.setDistribution(2, std::make_unique<StudentTDistribution>(5.0, 0.001, 0.006));

    std::cout << "Initial parameters (identical to DAX example):\n";
    std::cout << "  State 0 (bearish): nu=10  mu=-0.002  sigma=0.025\n";
    std::cout << "  State 1 (neutral): nu=30  mu= 0.000  sigma=0.012\n";
    std::cout << "  State 2 (bullish): nu= 5  mu= 0.001  sigma=0.006\n\n";

    std::cout << "Baum-Welch training:\n";
    std::cout << std::setw(8) << "iter" << std::setw(16) << "log-likelihood" << std::setw(12)
              << "delta\n";
    std::cout << std::string(36, '-') << "\n";

    BaumWelchTrainer trainer(&hmm, obs);
    const auto t0 = std::chrono::steady_clock::now();
    double prev = ll(hmm, obs);
    std::cout << std::setprecision(4);
    std::cout << std::setw(8) << 0 << std::setw(16) << prev << "  (initial)\n";

    int conv = -1;
    for (int i = 1; i <= 200; ++i) {
        trainer.train();
        const double cur = ll(hmm, obs);
        const double d = cur - prev;
        std::cout << std::setw(8) << i << std::setw(16) << cur << std::setw(12) << d;
        if (i > 1 && std::fabs(d) < 1e-4) {
            std::cout << "  <- converged";
            if (conv < 0)
                conv = i;
        }
        std::cout << "\n";
        if (conv > 0 && i >= conv + 2)
            break;
        prev = cur;
    }

    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double fll = ll(hmm, obs);

    // Sort by sigma desc: bearish, neutral, bullish
    int ord[3] = {0, 1, 2};
    std::sort(ord, ord + 3,
              [&](int a, int b) { return st(hmm, a).getScale() > st(hmm, b).getScale(); });
    const char *labs[3] = {"bearish", "neutral", "bullish"};

    std::cout << "\n=== S&P 500 results ===\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_s << " s\n\n";
    std::cout << std::setprecision(6);
    for (int i = 0; i < 3; ++i) {
        const int s = ord[i];
        const auto &d = st(hmm, s);
        std::cout << "State " << s << " (" << labs[i] << "):"
                  << "  nu=" << std::setw(8) << d.getDegreesOfFreedom() << "  mu=" << std::setw(10)
                  << d.getLocation() << "  sigma=" << std::setw(10) << d.getScale() << "\n";
    }

    std::cout << "\nTransition matrix:\n";
    const Matrix &T = hmm.getTrans();
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 3; ++j)
            std::cout << std::setw(9) << std::setprecision(5) << T(i, j);
        std::cout << " ]\n";
    }
    std::cout << "\nLog-likelihood: " << std::setprecision(2) << fll << "\n\n";

    ViterbiCalculator vc(hmm, returns);
    const StateSequence &vp = vc.getStateSequence();
    int cnt[3] = {0, 0, 0};
    for (std::size_t t = 0; t < vp.size(); ++t)
        ++cnt[vp(t)];
    std::cout << "Viterbi state occupancy:";
    for (int i = 0; i < 3; ++i)
        std::cout << "  " << labs[i] << "=" << cnt[ord[i]];
    std::cout << " days\n\n";

    // Cross-market comparison
    std::cout << "=== Cross-market comparison: S&P 500 vs DAX ===\n\n";
    std::cout << "(DAX results from same model, April 2026)\n\n";
    const struct {
        const char *lab;
        double mu_sp, mu_dax, sig_sp, sig_dax, nu_sp, nu_dax;
    } rows[3] = {
        {"bearish", st(hmm, ord[0]).getLocation(), -0.001793, st(hmm, ord[0]).getScale(), 0.026283,
         st(hmm, ord[0]).getDegreesOfFreedom(), 11.14},
        {"neutral", st(hmm, ord[1]).getLocation(), -0.000281, st(hmm, ord[1]).getScale(), 0.013049,
         st(hmm, ord[1]).getDegreesOfFreedom(), 36.09},
        {"bullish", st(hmm, ord[2]).getLocation(), 0.001258, st(hmm, ord[2]).getScale(), 0.005988,
         st(hmm, ord[2]).getDegreesOfFreedom(), 5.35},
    };
    std::cout << std::setw(10) << " " << std::setw(10) << "param" << std::setw(12) << "S&P 500"
              << std::setw(12) << "DAX\n";
    std::cout << std::string(44, '-') << "\n";
    std::cout << std::setprecision(6);
    for (const auto &r : rows) {
        std::cout << std::setw(10) << r.lab << std::setw(10) << "mu" << std::setw(12) << r.mu_sp
                  << std::setw(12) << r.mu_dax << "\n";
        std::cout << std::setw(10) << "" << std::setw(10) << "sigma" << std::setw(12) << r.sig_sp
                  << std::setw(12) << r.sig_dax << "\n";
        std::cout << std::setw(10) << "" << std::setw(10) << "nu" << std::setw(12) << r.nu_sp
                  << std::setw(12) << r.nu_dax << "\n";
    }
    std::cout << "\nBoth markets use the same model; parameter differences reflect\n";
    std::cout << "structural differences in US vs German equity risk and liquidity.\n";

    return 0;
}
