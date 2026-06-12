/**
 * mv_regime_example — Two-sector market regime detection via multivariate HMM (v4).
 *
 * Demonstrates DiagonalGaussianDistribution vs FullCovarianceGaussianDistribution
 * in a 3-state regime-switching model for correlated asset returns.
 *
 * Data sources (in priority order):
 *   1. Real: SPY (S&P 500) + QQQ (Nasdaq-100) monthly log-returns, 2000–2022.
 *      Download: Rscript scripts/prepare_mv_regime_data.R [output_dir]
 *      This writes /tmp/spy_qqq_monthly.csv (or output_dir/spy_qqq_monthly.csv).
 *      Reference comparison: scripts/verify_mv_regime.py (hmmlearn 0.3.3).
 *   2. Synthetic fallback: 240 embedded observations from a known 3-state DGP.
 *      Runs standalone with no data download required.
 *
 * Usage:
 *   ./mv_regime_example               # uses /tmp/spy_qqq_monthly.csv or synthetic
 *   ./mv_regime_example /path/to/dir  # looks for spy_qqq_monthly.csv in that dir
 *
 * Synthetic DGP ground truth:
 *   State 0 — Bull:   μ=(0.9, 0.8)%,   Σ=[[16, 9],[9, 14]]   ρ=0.60
 *   State 1 — Bear:   μ=(-1.3,-1.5)%,  Σ=[[30,22],[22,28]]   ρ=0.76
 *   State 2 — Crisis: μ=(-4.0,-5.0)%,  Σ=[[80,72],[72,90]]   ρ=0.85
 *
 * Models:
 *   Model A — DiagonalGaussian: independent return series within each state.
 *   Model B — FullCovarianceGaussian: captures cross-sector correlation.
 *
 * Both operate on the same observation space so BIC comparison is valid.
 * FullCovGaussian should win for both datasets: the DGP encodes ρ=0.60–0.85 and
 * SPY/QQQ exhibit within-state correlation of ~0.80–0.95 across all three regimes.
 */

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
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/training/basic_baum_welch_trainer.h"
#include "libhmm/training/model_selection.h"

using namespace libhmm;

// =============================================================================
// Embedded dataset — synthetic two-sector monthly returns (%), N=240
// Generated from a known 3-state regime HMM (seed 2024).
// Ground truth: bull ρ=0.60, bear ρ=0.76, crisis ρ=0.85.
// =============================================================================

// clang-format off
constexpr double kObs[][2] = {
    { -7.5359,  -1.0097},
    {  0.0169,   3.5014},
    { -1.1493,  -3.7200},
    {  4.8444,   4.2599},
    {  0.8304,   2.4758},
    {  8.2868,  12.4426},
    { -0.9071,  -1.4982},
    {  7.7229,  -0.4885},
    {  3.4360,   8.2393},
    { -0.6991,   3.2329},
    { -7.0012,  -6.9979},
    { -0.2567,   0.8960},
    {-15.9596,  -9.0426},
    { -5.6861,  -7.9995},
    {  1.4011,  -1.2523},
    {  3.8651,   2.2148},
    { -6.4686,  -1.5818},
    { -0.8289,  -4.6448},
    {  1.1281,   2.1997},
    { 11.0987,   1.4344},
    {  7.9813,   6.0025},
    { -8.8809,  -7.7717},
    {-12.0699,  -8.5972},
    {-10.6057, -12.5444},
    {  0.5827,  -4.4471},
    { -7.7838,  -8.6504},
    { -9.0807,  -7.9648},
    {  0.4100,  -0.1809},
    { -0.2968,  -3.3013},
    {  4.3352,   0.4773},
    { -6.7335,  -1.2771},
    { -2.0827,  -5.5423},
    { -1.4360,  -3.4267},
    {-11.8567,  -4.4674},
    {  3.3034,  -5.7268},
    {-11.5942, -15.7510},
    {  0.8493,  -0.8461},
    { -0.2967,  -2.6579},
    {  1.4586,   5.2577},
    { -4.7547,  -9.3781},
    { -2.7645,  -7.3039},
    {-16.0467,  -9.7706},
    {  1.4259,  -1.0119},
    {  1.6546,   1.6772},
    { -2.6427,  -1.5497},
    { 10.2354,   1.1554},
    {  1.1453,  -6.1951},
    {  6.1684,  -1.5721},
    {  3.9691,   2.4989},
    { -1.9250,  -6.2151},
    {  0.4034,   0.5574},
    {  5.3317,   0.9141},
    { -0.6231,   3.0518},
    { -2.7041,  -0.4820},
    {  2.5191,   0.4671},
    { -3.2063,  -0.8541},
    {  1.4258,   2.5963},
    { -3.6294,   3.5030},
    {  5.3404,   6.9337},
    {  4.4954,   2.0929},
    { -1.3303,  -3.7418},
    {  0.2000,   3.9259},
    {  0.7818,   1.2481},
    { -2.7935,  -5.1900},
    { -1.3468,  -4.6820},
    { -2.4848,  -1.6939},
    { -0.0411,   5.6567},
    {  2.9955,  -0.7891},
    {  7.4288,   3.3883},
    {  4.4392,   4.0586},
    { -3.8835,  -6.3793},
    { -4.4768,  -5.8008},
    { -0.8742,   1.3742},
    { -2.5520,  -3.1788},
    {  6.5313,   7.5039},
    {  0.6807,   0.3971},
    { -0.0246,  -1.1708},
    {  5.8773,   4.1369},
    {  2.7010,  -1.8823},
    {  1.7216,  -0.9909},
    { -2.5777,  -2.5681},
    { -2.6365,   2.2445},
    {  5.8482,   2.6011},
    { -3.5270,   4.1850},
    {  0.7632,   3.7243},
    { -5.2622,   1.5662},
    {  2.7412,   0.7486},
    { -0.8931,  -2.3319},
    { 14.2800,   9.4537},
    { -7.3038,  -9.3128},
    { -1.8154,  -4.9530},
    { -2.3391,  -4.6217},
    {  4.6276,   1.1384},
    { -1.6701,  -0.6739},
    { -6.9787,  -5.2702},
    { -1.2909,  -0.8391},
    {  7.7960,   3.4944},
    { -1.7611,  -1.6222},
    { -5.7981,  -3.4552},
    { -6.6948,  -6.6247},
    { -1.5715,   2.8404},
    { -4.4927,   0.2620},
    { -5.1245,  -4.3649},
    {  8.7098,   7.3429},
    { -5.6377,  -7.4680},
    { -4.3526,  -5.4922},
    { -4.6877,   1.2747},
    { -3.3236,   0.4850},
    { -3.2556,  -3.8299},
    {  1.8812,   2.1055},
    { -5.9455,  -6.9129},
    { -0.7230,  -0.7378},
    { -0.7671,   4.5562},
    {  1.3899,   0.6796},
    {  6.0184,  -0.1054},
    {  8.8484,   7.2363},
    {  2.8901,   6.9808},
    {  1.5687,   3.2108},
    { -3.3607,  -2.3787},
    {  4.2662,   6.3394},
    { -2.3693,   0.4397},
    {-14.3764, -10.1451},
    { -4.3031,  -5.8649},
    {  5.6035,   1.2653},
    {  6.4719,   5.9637},
    {  1.9407,   5.7929},
    {  0.1911,  -2.9922},
    {  0.3028,  -1.4062},
    { -5.0077,   0.2711},
    { -0.3501,  -8.7375},
    { -0.2843,  -2.7413},
    {  4.4290,  -3.7772},
    {  0.9102,   1.2897},
    {  2.5048,   8.8839},
    {  1.3708,   0.6990},
    { -3.2204,  -8.4480},
    {  4.8574,   5.4018},
    { -2.5825,   2.0012},
    { -0.5950,   0.6894},
    { -1.4114,   1.2313},
    {  2.4230,   5.1974},
    {  3.9304,   3.0756},
    { -0.0124,   2.9655},
    {  2.8155,   2.5720},
    {  0.8255,   0.6385},
    {  2.4893,   2.2157},
    { -8.5315, -13.0801},
    { -8.6591, -14.7629},
    { -1.0509,  -4.8928},
    {-10.4640, -12.1109},
    { -1.0158,  -3.4160},
    {  3.3782,   4.3928},
    { -7.6494,  -6.5058},
    {  4.1329,   4.9630},
    { -2.6988,  -3.2565},
    { -4.9036,  -9.1036},
    {  1.0943,  -0.4253},
    { -0.4977,   0.1974},
    {-11.6560,  -8.5969},
    { -3.6298,  -1.9638},
    { -1.0134,  -4.6755},
    { -2.6234,  -2.4587},
    { -4.3897,  -4.6967},
    {  1.8288,  -3.3671},
    {  1.9853,   2.4506},
    { -2.6665,  -2.5035},
    { -0.1622,  -2.2876},
    { -1.0320,  -4.1290},
    {  9.6224,  10.9534},
    { -2.9691,   2.2236},
    { 10.9262,   5.7208},
    {  4.0202,   3.9306},
    { -6.6483,  -6.9889},
    { 11.5520,   9.5041},
    {  3.9430,   4.4889},
    { -3.7748,  -0.7525},
    {  0.6017,   2.0211},
    {  0.6268,   1.8594},
    { -5.0153,   2.5148},
    { -2.9429,  -0.8960},
    {  2.1419,  -6.1420},
    { -7.4393,  -8.2184},
    { -3.9470,   2.6371},
    { -6.0926,  -8.6509},
    { -1.5585,   0.1051},
    { -3.5391,  -6.6521},
    {  0.2977,  -1.5490},
    {-18.8174, -17.5245},
    { -0.2180,   1.0261},
    { -4.8837,  -3.0430},
    {  0.3387,  -2.1554},
    {  2.8505,  -0.1947},
    {  2.0807,  -0.0840},
    { -4.5647,  -7.2901},
    { -0.6140,   1.8330},
    {  3.0159,  -1.0801},
    { -6.0359,  -8.7719},
    {  2.3829,  -3.6410},
    { -1.7242,  -2.9584},
    { -3.3922,   0.0153},
    {  6.0980,   2.0704},
    {  2.9792,   1.8439},
    { -5.3839,  -9.2372},
    { -5.8000, -10.4187},
    {  5.5720,   2.5265},
    { -2.7255,  -5.6131},
    {  4.5944,   1.2870},
    {  7.8335,   0.8581},
    { -8.2387,  -6.8321},
    { -7.8562,  -4.6356},
    { -3.8735,  -7.3602},
    {  0.0113,  -2.2913},
    { -4.6940,  -4.2359},
    { -1.2035,  -2.5947},
    {  2.0188,   6.6188},
    {  3.4784,  -1.4715},
    { -2.0686,   5.6154},
    { -1.1274,   0.2468},
    {  3.4485,   7.3812},
    {  2.2646,   2.6807},
    {  1.5850,   0.6789},
    {  4.5824,   1.1177},
    { -0.7831,   3.5086},
    { -7.8472,  -1.7356},
    { -5.1079,  -0.6965},
    { -1.2303,  -2.0593},
    { -3.5292,  -5.0374},
    {  0.6957,  -3.9070},
    {  6.2523,  -1.2372},
    { -5.4522,  -4.0722},
    { -2.0924,  -1.4329},
    {  2.6594,  -4.2996},
    { -3.3712,  -3.3186},
    {  3.3007,   2.1521},
    {  1.3469,  -2.1184},
    { -1.9433,   0.6875},
    {  1.4653,   2.7789},
    { -6.3383,   1.4806},
    {  0.3379,   1.3149},
    {  6.6546,   6.2406}
};
constexpr std::size_t kNObs = 240;
// clang-format on

// =============================================================================
// Helpers
// =============================================================================

/// Build the single-sequence ObservationMatrix from the embedded synthetic array.
static MultiObservationLists make_synthetic_data() {
    ObservationMatrix mat(kNObs, 2);
    for (std::size_t t = 0; t < kNObs; ++t) {
        mat(t, 0) = kObs[t][0];
        mat(t, 1) = kObs[t][1];
    }
    return {std::move(mat)};
}

/// Try to load spy_qqq_monthly.csv from @p path.
/// Returns an empty list on any error (missing file, bad format, etc.).
static MultiObservationLists try_load_csv(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open())
        return {};

    std::string line;
    std::getline(f, line); // skip header: date,spy_logret,qqq_logret

    std::vector<double> col0, col1;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        // Format: "date",spy,qqq  (date may be quoted)
        // Find second and third comma-separated fields.
        const std::size_t c1 = line.find(',');
        if (c1 == std::string::npos)
            continue;
        const std::size_t c2 = line.find(',', c1 + 1);
        if (c2 == std::string::npos)
            continue;
        try {
            col0.push_back(std::stod(line.substr(c1 + 1, c2 - c1 - 1)));
            col1.push_back(std::stod(line.substr(c2 + 1)));
        } catch (...) {
            continue;
        }
    }
    if (col0.empty())
        return {};

    ObservationMatrix mat(col0.size(), 2);
    for (std::size_t t = 0; t < col0.size(); ++t) {
        mat(t, 0) = col0[t];
        mat(t, 1) = col1[t];
    }
    return {std::move(mat)};
}

/// Total log-probability over all sequences.
static double total_logp(HmmMV &hmm, const MultiObservationLists &lists) {
    double lp = 0.0;
    for (const auto &seq : lists) {
        BasicForwardBackwardCalculator<ObservationVectorView> fbc(hmm, seq);
        lp += fbc.getLogProbability();
    }
    return lp;
}

/// Run Baum-Welch to convergence; return final log-likelihood.
static double run_bwt(HmmMV &hmm, const MultiObservationLists &lists, int max_iter = 300,
                      double tol = 1e-4) {
    BasicBaumWelchTrainer<ObservationVectorView> trainer(hmm, lists);
    double prev = total_logp(hmm, lists);
    int conv = -1;
    for (int k = 0; k < max_iter; ++k) {
        trainer.train();
        const double cur = total_logp(hmm, lists);
        const double delta = cur - prev;
        if (k > 0 && delta >= -1e-8 && delta < tol) {
            if (conv < 0)
                conv = k;
        }
        if (conv >= 0 && k >= conv + 2)
            break;
        prev = cur;
    }
    return total_logp(hmm, lists);
}

/// Build a 3-state HmmMV with uniform-ish transitions and equal initial probs.
static HmmMV make_base_hmm() {
    HmmMV hmm(3);
    Matrix A(3, 3);
    A(0, 0) = 0.92;
    A(0, 1) = 0.06;
    A(0, 2) = 0.02;
    A(1, 0) = 0.05;
    A(1, 1) = 0.90;
    A(1, 2) = 0.05;
    A(2, 0) = 0.04;
    A(2, 1) = 0.10;
    A(2, 2) = 0.86;
    hmm.setTrans(A);
    Vector pi(3);
    pi(0) = 0.5;
    pi(1) = 0.4;
    pi(2) = 0.1;
    hmm.setPi(pi);
    return hmm;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char *argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "/tmp";
    const std::string csv_path = data_dir + "/spy_qqq_monthly.csv";

    // -----------------------------------------------------------------
    // Load data: prefer real SPY+QQQ, fall back to embedded synthetic.
    // -----------------------------------------------------------------
    MultiObservationLists lists = try_load_csv(csv_path);
    const bool using_real = !lists.empty();
    if (!using_real) {
        lists = make_synthetic_data();
        std::cerr << "Note: " << csv_path << " not found; using embedded synthetic data.\n"
                  << "Run: Rscript scripts/prepare_mv_regime_data.R to use real SPY+QQQ data.\n\n";
    }

    const auto &seq = lists[0];
    const std::size_t N = seq.size1();
    const char *col1_name = using_real ? "SPY" : "Sector1";
    const char *col2_name = using_real ? "QQQ" : "Sector2";

    std::cout << "Market Regime Detection — Multivariate HMM (v4 API)\n";
    if (using_real) {
        std::cout << "SPY (S&P 500) + QQQ (Nasdaq-100) monthly log-returns, 2000–2022\n";
    } else {
        std::cout << "[synthetic fallback data — see prepare_mv_regime_data.R for real data]\n";
    }
    std::cout << "Diagonal vs Full-Covariance Gaussian, 3 states\n";
    std::cout << "====================================================\n\n";

    // Summary statistics
    double sum0 = 0.0, sum1 = 0.0;
    for (std::size_t t = 0; t < N; ++t) {
        sum0 += seq(t, 0);
        sum1 += seq(t, 1);
    }
    const double mean0 = sum0 / static_cast<double>(N);
    const double mean1 = sum1 / static_cast<double>(N);
    double cov01 = 0.0, var0 = 0.0, var1 = 0.0;
    for (std::size_t t = 0; t < N; ++t) {
        const double r0 = seq(t, 0) - mean0;
        const double r1 = seq(t, 1) - mean1;
        cov01 += r0 * r1;
        var0 += r0 * r0;
        var1 += r1 * r1;
    }
    const double marginal_rho = cov01 / std::sqrt(var0 * var1);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Observations: " << N << "\n";
    std::cout << col1_name << ": mean=" << mean0 << "%\n";
    std::cout << col2_name << ": mean=" << mean1 << "%\n";
    std::cout << "Marginal correlation: " << marginal_rho << "\n";
    if (!using_real)
        std::cout << "Ground truth (DGP): bull \u03c1=0.60  bear \u03c1=0.76  crisis \u03c1=0.85\n";
    std::cout << "\n";

    // =========================================================================
    // Model A — DiagonalGaussian (independence assumed within each state)
    // =========================================================================
    std::cout << "--- Model A: DiagonalGaussianDistribution ---\n\n";
    std::cout << "Assumes Tech and Finance returns are independent within each state.\n\n";

    // ----------------------------------------------------------------
    // Shared initialisation — reasonable for both real and synthetic.
    // Means and variances span the expected range for both datasets;
    // Baum-Welch refines from these starting points.
    // ----------------------------------------------------------------
    HmmMV hmm_a = make_base_hmm();
    {
        auto a0 = std::make_unique<DiagonalGaussianDistribution>(2);
        a0->setParameters({0.8, 1.0}, {9.0, 16.0}); // bull
        hmm_a.setDistribution(0, std::move(a0));

        auto a1 = std::make_unique<DiagonalGaussianDistribution>(2);
        a1->setParameters({-1.5, -2.0}, {25.0, 50.0}); // bear
        hmm_a.setDistribution(1, std::move(a1));

        auto a2 = std::make_unique<DiagonalGaussianDistribution>(2);
        a2->setParameters({-5.0, -7.0}, {64.0, 100.0}); // crisis
        hmm_a.setDistribution(2, std::move(a2));
    }

    const auto t_a0 = std::chrono::steady_clock::now();
    const double ll_a = run_bwt(hmm_a, lists);
    const double wall_a =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_a0).count();

    std::cout << std::fixed << std::setprecision(4);
    const char *state_names[] = {"Bull  ", "Bear  ", "Crisis"};
    for (int s = 0; s < 3; ++s) {
        const auto &dg =
            static_cast<const DiagonalGaussianDistribution &>(hmm_a.getDistribution(s));
        std::cout << "State " << s << " (" << state_names[s] << "): "
                  << "mu=[" << dg.getMean()[0] << ", " << dg.getMean()[1] << "]  "
                  << "sd=[" << std::sqrt(dg.getVariance()[0]) << ", "
                  << std::sqrt(dg.getVariance()[1]) << "]\n";
    }
    std::cout << "Log-likelihood: " << ll_a << "\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_a << " ms\n\n";

    // =========================================================================
    // Model B — FullCovarianceGaussian (fits cross-sector correlation)
    // =========================================================================
    std::cout << "--- Model B: FullCovarianceGaussianDistribution ---\n\n";
    std::cout << "Fits a full 2x2 covariance per state, capturing cross-sector correlation.\n\n";

    HmmMV hmm_b = make_base_hmm();
    {
        auto b0 = std::make_unique<FullCovarianceGaussianDistribution>(2);
        {
            BasicMatrix<double> S(2, 2, 0.0);
            S(0, 0) = 9.0;
            S(0, 1) = 5.0;
            S(1, 0) = 5.0;
            S(1, 1) = 16.0; // bull rho~0.42
            b0->setParameters({0.8, 1.0}, std::move(S));
        }
        hmm_b.setDistribution(0, std::move(b0));

        auto b1 = std::make_unique<FullCovarianceGaussianDistribution>(2);
        {
            BasicMatrix<double> S(2, 2, 0.0);
            S(0, 0) = 25.0;
            S(0, 1) = 18.0;
            S(1, 0) = 18.0;
            S(1, 1) = 50.0; // bear rho~0.51
            b1->setParameters({-1.5, -2.0}, std::move(S));
        }
        hmm_b.setDistribution(1, std::move(b1));

        auto b2 = std::make_unique<FullCovarianceGaussianDistribution>(2);
        {
            BasicMatrix<double> S(2, 2, 0.0);
            S(0, 0) = 64.0;
            S(0, 1) = 55.0;
            S(1, 0) = 55.0;
            S(1, 1) = 100.0; // crisis rho~0.69
            b2->setParameters({-5.0, -7.0}, std::move(S));
        }
        hmm_b.setDistribution(2, std::move(b2));
    }

    const auto t_b0 = std::chrono::steady_clock::now();
    const double ll_b = run_bwt(hmm_b, lists);
    const double wall_b =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_b0).count();

    std::cout << std::fixed << std::setprecision(4);
    for (int s = 0; s < 3; ++s) {
        const auto &fc =
            static_cast<const FullCovarianceGaussianDistribution &>(hmm_b.getDistribution(s));
        const auto &C = fc.getCovariance();
        const double rho = C(0, 1) / std::sqrt(C(0, 0) * C(1, 1));
        std::cout << "State " << s << " (" << state_names[s] << "): "
                  << "mu=[" << fc.getMean()[0] << ", " << fc.getMean()[1] << "]  "
                  << "sd=[" << std::sqrt(C(0, 0)) << ", " << std::sqrt(C(1, 1)) << "]  "
                  << "rho=" << rho << "\n";
    }
    std::cout << "Log-likelihood: " << ll_b << "\n";
    std::cout << "Wall time: " << std::setprecision(1) << wall_b << " ms\n\n";

    // =========================================================================
    // BIC comparison
    // =========================================================================
    const std::size_t k_a = count_free_parameters(hmm_a);
    const std::size_t k_b = count_free_parameters(hmm_b);
    const double bic_a = compute_bic(ll_a, k_a, kNObs);
    const double bic_b = compute_bic(ll_b, k_b, kNObs);

    std::cout << "=== Model comparison ===\n\n";
    std::cout << std::setw(26) << " " << std::setw(14) << "Model A (Diag)" << std::setw(14)
              << "Model B (Full)\n";
    std::cout << std::string(54, '-') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(26) << "Log-likelihood" << std::setw(14) << ll_a << std::setw(14) << ll_b
              << "\n";
    std::cout << std::setw(26) << "Free parameters k" << std::setw(14) << k_a << std::setw(14)
              << k_b << "\n";
    std::cout << std::setw(26) << "BIC (lower = better)" << std::setw(14) << bic_a << std::setw(14)
              << bic_b << "\n";
    std::cout << std::setw(26) << "Wall time (ms)" << std::setw(14) << static_cast<int>(wall_a)
              << std::setw(14) << static_cast<int>(wall_b) << "\n";

    std::cout << "\nNotes:\n";
    std::cout << "  Model B has " << (k_b - k_a)
              << " extra parameters (one off-diagonal covariance per state).\n";
    if (using_real) {
        std::cout << "  " << col1_name << " and " << col2_name
                  << " are large-cap US equity ETFs; within-state correlation\n";
        std::cout << "  is typically 0.80\u20130.95, strongly motivating full covariance.\n";
    } else {
        std::cout << "  Synthetic DGP: bull \u03c1=0.60, bear \u03c1=0.76, crisis \u03c1=0.85.\n";
    }
    if (bic_b < bic_a)
        std::cout << "  -> Model B wins on BIC: cross-sector correlation is informative.\n"
                     "     FullCovarianceGaussian is the correct model for this data.\n";
    else
        std::cout << "  -> Model A wins on BIC: diagonal approximation is sufficient here.\n";

    if (using_real) {
        std::cout << "\nCompare against hmmlearn reference (20 random restarts):\n";
        std::cout << "  /tmp/libhmm_hmmlearn_venv/bin/python3 scripts/verify_mv_regime.py "
                  << data_dir << "\n";
        std::cout << "Model B (full) LLs should agree to < 0.1 nat; Model A (diag)\n";
        std::cout << "may differ by a few nats since hmmlearn uses random restarts.\n";
    }

    return 0;
}
