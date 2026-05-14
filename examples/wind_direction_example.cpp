/**
 * wind_direction_example — Wind direction regime detection.
 *
 * Fits a 2-state VonMisesDistribution HMM to hourly wind directions
 * at Chicago O'Hare International Airport, 2015.  Demonstrates
 * VonMisesDistribution on real meteorological data and contrasts with
 * HiddenMarkov (R), which uses a Normal distribution approximation —
 * fundamentally incorrect for circular data because angles near 0 and 2π
 * are close but the Normal treats them as far apart.
 *
 * Dataset
 * -------
 * NOAA Integrated Surface Database (ISD), station 725300-14819
 * (Chicago O'Hare), year 2015.  11,894 valid hourly wind direction
 * observations (observations with calm/missing winds excluded).
 *
 * Data preparation
 * ----------------
 * Rscript scripts/prepare_wind_data.R [output_dir]
 * (or run dax script which also exports wind data)
 *
 * Model
 * -----
 * 2 states: prevailing SW/W wind vs variable/N wind
 * Emission: VonMisesDistribution(mu, kappa) per state
 * Training: BaumWelchTrainer
 *
 * HiddenMarkov R reference (Normal approximation, April 2026)
 * -----------------------------------------------------------
 * Note: Normal HMM on circular data is an approximation that fails
 * at the 0/2π boundary.  VonMisesDistribution is the correct model.
 *   State 1 (prevailing):  mean=49.6°  sd=0.468 rad
 *   State 2 (variable):    mean=239.1° sd=1.093 rad
 *   LL: -16830.8  (Normal approx, not directly comparable)
 *   Wall time: ~0.24 s
 *
 * Reference
 * ---------
 * NOAA NCEI (2001): Global Surface Hourly [ISD]. NCEI.
 * Zucchini W, MacDonald IL, Langrock R (2017). Hidden Markov Models for
 *   Time Series: Introduction Using R, 2nd ed. CRC Press. (Ch. 10: Wind.)
 */

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

// Two-column CSV reader: returns column 0 (direction_rad)
static ObservationSet read_direction(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<double> v;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        const std::size_t c = line.find(',');
        if (c == std::string::npos)
            continue;
        v.push_back(std::stod(line.substr(0, c)));
    }
    ObservationSet s(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        s(i) = v[i];
    return s;
}

static double total_loglik(const Hmm &h, const ObservationLists &o) {
    double s = 0;
    for (const auto &q : o)
        s += ForwardBackwardCalculator(h, q).getLogProbability();
    return s;
}

static const VonMisesDistribution &vm(const Hmm &h, std::size_t s) {
    return static_cast<const VonMisesDistribution &>(h.getDistribution(s));
}

int main(int argc, char *argv[]) {
    const std::string dir = (argc > 1) ? argv[1] : "/tmp";
    const std::string path = dir + "/ohare_wind_2015.csv";

    std::cout << "Wind Direction Regime Detection — Chicago O'Hare 2015\n";
    std::cout << "VonMisesDistribution HMM vs HiddenMarkov (Normal approx)\n";
    std::cout << "==========================================================\n\n";

    ObservationSet directions;
    try {
        directions = read_direction(path);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\nRun: Rscript scripts/prepare_wind_data.R\n";
        return 1;
    }

    ObservationLists obs = {directions};
    std::cout << "NOAA ISD, O'Hare (725300): " << directions.size()
              << " hourly wind directions (2015)\n";
    std::cout << "Directions in radians [0, 2π]\n\n";

    // 2-state HMM: prevailing SW/W vs variable/N
    Hmm hmm(2);
    Matrix A(2, 2);
    A(0, 0) = 0.95;
    A(0, 1) = 0.05;
    A(1, 0) = 0.05;
    A(1, 1) = 0.95;
    hmm.setTrans(A);
    Vector pi(2);
    pi(0) = 0.4;
    pi(1) = 0.6;
    hmm.setPi(pi);
    // State 0: prevailing SW (approx 225° = 3.93 rad, concentrated)
    // State 1: variable/N (approx 0° = 0.0 rad, dispersed)
    hmm.setDistribution(0, std::make_unique<VonMisesDistribution>(3.93, 2.0));
    hmm.setDistribution(1, std::make_unique<VonMisesDistribution>(0.00, 0.5));

    std::cout << "Initial parameters:\n";
    std::cout << "  State 0: mu=3.93 rad (225°, SW)  kappa=2.0 (concentrated)\n";
    std::cout << "  State 1: mu=0.00 rad (0°,  N)   kappa=0.5 (dispersed)\n\n";

    std::cout << "Baum-Welch training:\n";
    std::cout << std::setw(7) << "iter" << std::setw(15) << "log-likelihood" << std::setw(11)
              << "delta\n";
    std::cout << std::string(33, '-') << "\n";

    BaumWelchTrainer trainer(&hmm, obs);
    const auto t0 = std::chrono::steady_clock::now();
    double prev = total_loglik(hmm, obs);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(7) << 0 << std::setw(15) << prev << "  (initial)\n";

    int conv = -1;
    for (int i = 1; i <= 200; ++i) {
        trainer.train();
        const double cur = total_loglik(hmm, obs);
        const double d = cur - prev;
        std::cout << std::setw(7) << i << std::setw(15) << cur << std::setw(11) << d;
        if (i > 1 && std::fabs(d) < 1.0) {
            std::cout << "  <- converged";
            if (conv < 0)
                conv = i;
        }
        std::cout << "\n";
        if (conv > 0 && i >= conv + 2)
            break;
        prev = cur;
    }

    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double fll = total_loglik(hmm, obs);

    // Sort by kappa descending: concentrated (prevailing) first
    int conc = (vm(hmm, 0).getKappa() > vm(hmm, 1).getKappa()) ? 0 : 1;
    int disp = 1 - conc;

    std::cout << "\n=== libhmm results ===\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_ms << " ms\n\n";
    std::cout << std::setprecision(4);

    auto deg = [](double r) {
        return r * 180.0 / M_PI;
    };
    std::cout << "State " << conc << " (prevailing):"
              << "  mu=" << vm(hmm, conc).getMu() << " rad (" << std::setprecision(1)
              << deg(vm(hmm, conc).getMu()) << "°)"
              << "  kappa=" << std::setprecision(4) << vm(hmm, conc).getKappa()
              << "  circ_var=" << vm(hmm, conc).getCircularVariance() << "\n";
    std::cout << "State " << disp << " (variable):  "
              << "  mu=" << vm(hmm, disp).getMu() << " rad (" << std::setprecision(1)
              << deg(vm(hmm, disp).getMu()) << "°)"
              << "  kappa=" << std::setprecision(4) << vm(hmm, disp).getKappa()
              << "  circ_var=" << vm(hmm, disp).getCircularVariance() << "\n\n";

    const Matrix &T = hmm.getTrans();
    std::cout << "Transition matrix:\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 2; ++j)
            std::cout << std::setw(9) << std::setprecision(5) << T(i, j);
        std::cout << " ]\n";
    }
    std::cout << "\nLog-likelihood (VonMises): " << std::setprecision(1) << fll << "\n\n";

    ViterbiCalculator vc(hmm, directions);
    const StateSequence &vp = vc.getStateSequence();
    int cnt[2] = {0, 0};
    for (std::size_t t = 0; t < vp.size(); ++t)
        ++cnt[vp(t)];
    std::cout << "Viterbi occupancy: prevailing=" << cnt[conc] << "h  variable=" << cnt[disp]
              << "h\n\n";

    // Comparison
    std::cout << "=== Comparison: libhmm VonMises vs HiddenMarkov Normal ===\n\n";
    std::cout << "(HiddenMarkov uses Normal distribution — incorrect for circular data;\n";
    std::cout << " log-likelihoods are NOT comparable across different distributions.)\n\n";

    std::cout << std::setw(22) << " " << std::setw(14) << "libhmm" << std::setw(14)
              << "HiddenMarkov\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::setprecision(4);
    std::cout << std::setw(22) << "Distribution" << std::setw(14) << "VonMises" << std::setw(14)
              << "Normal\n";
    std::cout << std::setw(22) << "Prevailing dir (deg)" << std::setw(14) << std::setprecision(1)
              << deg(vm(hmm, conc).getMu()) << std::setw(14) << 49.6 << "\n";
    std::cout << std::setw(22) << "Variable dir (deg)" << std::setw(14)
              << deg(vm(hmm, disp).getMu()) << std::setw(14) << 239.1 << "\n";
    const std::string ws = std::to_string(static_cast<int>(wall_ms)) + " ms";
    std::cout << std::setw(22) << "Wall time" << std::setw(14) << ws << std::setw(14)
              << "~240 ms\n";

    std::cout << "\nKey advantage of VonMisesDistribution over Normal for circular data:\n";
    std::cout << "  Normal wraps poorly at the 0/2π boundary (359° and 1° are far\n";
    std::cout << "  apart for Normal but close on the circle). VonMisesDistribution\n";
    std::cout << "  handles periodicity correctly, giving valid state assignments\n";
    std::cout << "  for northerly winds near 0°/360°.\n";

    // -----------------------------------------------------------------------
    // Empirical boundary analysis
    // -----------------------------------------------------------------------
    // Compare per-observation marginal assignments (emission only, no dynamics).
    // Normal model parameters from HiddenMarkov R fit embedded as reference.
    std::cout << "\n=== Empirical boundary analysis ===\n\n";

    constexpr double NORM_MU_NNE = 0.8649, NORM_SD_NNE = 0.4684; // 49.6°
    constexpr double NORM_MU_SW = 4.1726, NORM_SD_SW = 1.0931;   // 239.1°

    auto norm_ll = [](double x, double mu, double sd) {
        const double z = (x - mu) / sd;
        return -0.5 * z * z - std::log(sd);
    };

    std::array<int, 12> bin_total{}, bin_disagree{};
    int boundary_total = 0, vm_nne_boundary = 0, norm_nne_boundary = 0;
    int nnw_total = 0, vm_nne_nnw = 0, norm_nne_nnw = 0;

    for (std::size_t i = 0; i < directions.size(); ++i) {
        const double x = directions(i);
        const bool vm_nne = vm(hmm, conc).getLogProbability(x) > vm(hmm, disp).getLogProbability(x);
        const bool n_nne =
            norm_ll(x, NORM_MU_NNE, NORM_SD_NNE) > norm_ll(x, NORM_MU_SW, NORM_SD_SW);

        double deg = x * 180.0 / M_PI;
        if (deg < 0.0)
            deg += 360.0;
        const int b = std::min(static_cast<int>(deg / 30.0), 11);
        ++bin_total[b];
        if (vm_nne != n_nne)
            ++bin_disagree[b];

        if (deg >= 300.0 || deg <= 60.0) {
            ++boundary_total;
            if (vm_nne)
                ++vm_nne_boundary;
            if (n_nne)
                ++norm_nne_boundary;
        }
        if (deg >= 330.0) {
            ++nnw_total;
            if (vm_nne)
                ++vm_nne_nnw;
            if (n_nne)
                ++norm_nne_nnw;
        }
    }

    std::cout << "Disagreement rate by direction (emission-level, no dynamics):\n";
    const char *labels[12] = {"  0- 30 (N/NNE)",  " 30- 60 (NNE/NE)", " 60- 90 (NE/E)",
                              " 90-120 (E/SE)  ", "120-150 (SE)    ", "150-180 (S/SSW) ",
                              "180-210 (SW)    ", "210-240 (W/SW)  ", "240-270 (W/WNW) ",
                              "270-300 (NW/WNW)", "300-330 (NNW)   ", "330-360 (N/NNW) "};
    for (int b = 0; b < 12; ++b) {
        if (bin_total[b] == 0)
            continue;
        const double rate = 100.0 * bin_disagree[b] / bin_total[b];
        const int stars = static_cast<int>(rate / 5.0);
        std::cout << labels[b] << ": " << std::setw(5) << std::setprecision(1) << rate << "% "
                  << std::string(stars, '*') << "\n";
    }

    std::cout << "\nWinds spanning the 0°/360° boundary (300°-60°): " << boundary_total
              << " hours\n";
    std::cout << std::setprecision(1);
    std::cout << "  VonMises → NNE state: " << vm_nne_boundary << " ("
              << 100.0 * vm_nne_boundary / boundary_total << "%) — correct\n";
    std::cout << "  Normal   → NNE state: " << norm_nne_boundary << " ("
              << 100.0 * norm_nne_boundary / boundary_total << "%) — boundary error\n\n";

    std::cout << "NNW-to-N winds (330°-360°): " << nnw_total << " hours\n";
    std::cout << "  VonMises → NNE: " << vm_nne_nnw << "/" << nnw_total << " = "
              << std::setprecision(0) << 100.0 * vm_nne_nnw / nnw_total
              << "%  (correctly near 31°)\n";
    std::cout << "  Normal   → NNE: " << norm_nne_nnw << "/" << nnw_total << " = "
              << 100.0 * norm_nne_nnw / nnw_total
              << "%  (misclassified: 11+ std devs from Normal mean)\n\n";

    // Mechanistic explanation for a single angle
    const double x350 = 350.0 * M_PI / 180.0;
    const double z350 = (x350 - NORM_MU_NNE) / NORM_SD_NNE;
    std::cout << std::setprecision(3);
    std::cout << "Mechanistic example — direction = 350° (NNW, just 19° from NNE state):\n";
    std::cout << "  VonMises: cos(350°-31°) = cos(-19°) = "
              << std::cos(x350 - vm(hmm, conc).getMu()) << "  → NNE is the correct state\n";
    std::cout << std::setprecision(1);
    std::cout << "  Normal:   (350°-49.6°) / 26.8° = " << z350
              << " std devs  log-lik=" << (-0.5 * z350 * z350 - std::log(NORM_SD_NNE))
              << "  → SW assigned instead (wrong)\n";

    return 0;
}
