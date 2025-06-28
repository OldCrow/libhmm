#include <iostream>
#include <vector>
#include <iomanip>
#include <memory>
#include <cmath>
#include <random>

// libhmm includes
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/calculators/calculators.h"

using namespace std;

/**
 * DIAGNOSTIC TEST FOR NUMERICAL ACCURACY DISCREPANCIES
 * 
 * This test isolates potential issues in:
 * 1. Distribution implementations (PDF/log-PDF calculations)
 * 2. HMM setup (parameter setting)
 * 3. Algorithm implementations (Forward-Backward, Viterbi)
 * 4. API usage patterns
 */

void testGaussianDistributionBasics() {
    cout << "=== GAUSSIAN DISTRIBUTION BASIC ACCURACY TEST ===" << endl;
    cout << "Testing standard normal (μ=0, σ=1) against known values" << endl;
    
    // Create standard normal distribution
    libhmm::GaussianDistribution std_normal(0.0, 1.0);
    
    // Test values and their expected PDF values
    vector<pair<double, double>> test_cases = {
        {-3.0, 0.004431848},  // Expected PDF at x=-3
        {-1.0, 0.241970725},  // Expected PDF at x=-1  
        { 0.0, 0.398942280},  // Expected PDF at x=0 (peak)
        { 1.0, 0.241970725},  // Expected PDF at x=1
        { 3.0, 0.004431848}   // Expected PDF at x=3
    };
    
    cout << left << setw(10) << "x" << setw(18) << "libhmm PDF" 
         << setw(18) << "Expected PDF" << setw(15) << "Difference" 
         << setw(18) << "libhmm LogPDF" << setw(18) << "Expected LogPDF" 
         << setw(15) << "Log Diff" << endl;
    cout << string(110, '-') << endl;
    
    for (const auto& test : test_cases) {
        double x = test.first;
        double expected_pdf = test.second;
        double expected_log_pdf = log(expected_pdf);
        
        double libhmm_pdf = std_normal.getProbability(x);
        double libhmm_log_pdf = std_normal.getLogProbability(x);
        
        double pdf_diff = abs(libhmm_pdf - expected_pdf);
        double log_pdf_diff = abs(libhmm_log_pdf - expected_log_pdf);
        
        cout << left << setw(10) << fixed << setprecision(1) << x
             << setw(18) << scientific << setprecision(9) << libhmm_pdf
             << setw(18) << scientific << setprecision(9) << expected_pdf
             << setw(15) << scientific << setprecision(3) << pdf_diff
             << setw(18) << scientific << setprecision(9) << libhmm_log_pdf
             << setw(18) << scientific << setprecision(9) << expected_log_pdf
             << setw(15) << scientific << setprecision(3) << log_pdf_diff << endl;
    }
    
    cout << endl;
}

void testGaussianParameterInterpretation() {
    cout << "=== GAUSSIAN PARAMETER INTERPRETATION TEST ===" << endl;
    cout << "Testing if our constructor uses std_dev or variance" << endl;
    
    // Test case: Create distribution with variance=4 (std_dev=2)
    double mean = 1.0;
    double variance = 4.0;
    double std_dev = 2.0;
    
    // Test both interpretations
    libhmm::GaussianDistribution gauss_with_variance(mean, variance);  
    libhmm::GaussianDistribution gauss_with_stddev(mean, std_dev);     
    
    // Calculate PDF at x=3 (1 std_dev away from mean)
    double x = 3.0;  // mean + std_dev = 1 + 2 = 3
    
    double pdf_variance_interp = gauss_with_variance.getProbability(x);
    double pdf_stddev_interp = gauss_with_stddev.getProbability(x);
    
    cout << "At x=3.0 (mean + 2.0):" << endl;
    cout << "  If constructor uses variance (4.0): PDF = " << scientific << setprecision(6) << pdf_variance_interp << endl;
    cout << "  If constructor uses std_dev (2.0):  PDF = " << scientific << setprecision(6) << pdf_stddev_interp << endl;
    
    // Calculate expected value manually for std_dev=2
    double expected_pdf = (1.0 / (2.0 * sqrt(2 * M_PI))) * exp(-0.5 * pow((3.0 - 1.0) / 2.0, 2));
    cout << "  Expected (manual std_dev=2):        PDF = " << scientific << setprecision(6) << expected_pdf << endl;
    
    // Determine which interpretation we're using
    double diff_variance = abs(pdf_variance_interp - expected_pdf);
    double diff_stddev = abs(pdf_stddev_interp - expected_pdf);
    
    cout << "  Difference (variance interpretation): " << scientific << setprecision(3) << diff_variance << endl;
    cout << "  Difference (std_dev interpretation):  " << scientific << setprecision(3) << diff_stddev << endl;
    
    if (diff_stddev < diff_variance) {
        cout << "  CONCLUSION: Constructor uses std_dev parameter" << endl;
    } else {
        cout << "  CONCLUSION: Constructor uses variance parameter" << endl;
    }
    cout << endl;
}

void testDiscreteDistributionLogProbability() {
    cout << "=== DISCRETE DISTRIBUTION LOG PROBABILITY TEST ===" << endl;
    
    // Create discrete distribution with known probabilities
    libhmm::DiscreteDistribution discrete(4);
    discrete.setProbability(0, 0.4);  // P(0) = 0.4
    discrete.setProbability(1, 0.3);  // P(1) = 0.3
    discrete.setProbability(2, 0.2);  // P(2) = 0.2
    discrete.setProbability(3, 0.1);  // P(3) = 0.1
    
    cout << left << setw(12) << "Observation" << setw(15) << "PDF" 
         << setw(18) << "Log PDF" << setw(18) << "Expected Log" 
         << setw(15) << "Difference" << endl;
    cout << string(78, '-') << endl;
    
    for (int i = 0; i < 4; ++i) {
        double pdf = discrete.getProbability(i);
        double log_pdf = discrete.getLogProbability(i);
        double expected_log = log(pdf);
        double diff = abs(log_pdf - expected_log);
        
        cout << left << setw(12) << i
             << setw(15) << fixed << setprecision(6) << pdf
             << setw(18) << scientific << setprecision(6) << log_pdf
             << setw(18) << scientific << setprecision(6) << expected_log
             << setw(15) << scientific << setprecision(3) << diff << endl;
    }
    cout << endl;
}

void testHMMSetupConsistency() {
    cout << "=== HMM SETUP CONSISTENCY TEST ===" << endl;
    cout << "Testing if HMM parameters are set correctly" << endl;
    
    // Create simple 2-state HMM
    auto hmm = make_unique<libhmm::Hmm>(2);
    
    // Set initial probabilities
    libhmm::Vector pi(2);
    pi(0) = 0.6;
    pi(1) = 0.4;
    hmm->setPi(pi);
    
    // Set transition matrix
    libhmm::Matrix trans(2, 2);
    trans(0, 0) = 0.7; trans(0, 1) = 0.3;
    trans(1, 0) = 0.4; trans(1, 1) = 0.6;
    hmm->setTrans(trans);
    
    // Set emission distributions (discrete with 3 symbols)
    auto dist0 = make_unique<libhmm::DiscreteDistribution>(3);
    dist0->setProbability(0, 0.5);
    dist0->setProbability(1, 0.3);
    dist0->setProbability(2, 0.2);
    
    auto dist1 = make_unique<libhmm::DiscreteDistribution>(3);
    dist1->setProbability(0, 0.2);
    dist1->setProbability(1, 0.3);
    dist1->setProbability(2, 0.5);
    
    hmm->setProbabilityDistribution(0, move(dist0));
    hmm->setProbabilityDistribution(1, move(dist1));
    
    // Verify parameters are set correctly
    cout << "Initial probabilities:" << endl;
    const auto& retrieved_pi = hmm->getPi();
    for (int i = 0; i < 2; ++i) {
        cout << "  π[" << i << "] = " << fixed << setprecision(6) << retrieved_pi(i) << endl;
    }
    
    cout << "Transition matrix:" << endl;
    const auto& retrieved_trans = hmm->getTrans();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << "  T[" << i << "," << j << "] = " << fixed << setprecision(6) << retrieved_trans(i, j) << endl;
        }
    }
    
    cout << "Emission probabilities:" << endl;
    for (int state = 0; state < 2; ++state) {
        for (int symbol = 0; symbol < 3; ++symbol) {
            double prob = hmm->getProbabilityDistribution(state)->getProbability(symbol);
            cout << "  E[" << state << "," << symbol << "] = " << fixed << setprecision(6) << prob << endl;
        }
    }
    cout << endl;
}

void testCalculatorConsistency() {
    cout << "=== CALCULATOR CONSISTENCY TEST ===" << endl;
    cout << "Testing if different calculators give same results" << endl;
    
    // Create simple HMM (same as above)
    auto hmm = make_unique<libhmm::Hmm>(2);
    
    libhmm::Vector pi(2);
    pi(0) = 0.6; pi(1) = 0.4;
    hmm->setPi(pi);
    
    libhmm::Matrix trans(2, 2);
    trans(0, 0) = 0.7; trans(0, 1) = 0.3;
    trans(1, 0) = 0.4; trans(1, 1) = 0.6;
    hmm->setTrans(trans);
    
    auto dist0 = make_unique<libhmm::DiscreteDistribution>(3);
    dist0->setProbability(0, 0.5); dist0->setProbability(1, 0.3); dist0->setProbability(2, 0.2);
    auto dist1 = make_unique<libhmm::DiscreteDistribution>(3);
    dist1->setProbability(0, 0.2); dist1->setProbability(1, 0.3); dist1->setProbability(2, 0.5);
    
    hmm->setProbabilityDistribution(0, move(dist0));
    hmm->setProbabilityDistribution(1, move(dist1));
    
    // Create test observation sequence
    libhmm::ObservationSet obs(5);
    obs(0) = 0; obs(1) = 1; obs(2) = 2; obs(3) = 0; obs(4) = 1;
    
    cout << "Test sequence: 0 1 2 0 1" << endl;
    
    // Test different calculators
    try {
        // AutoCalculator
        libhmm::forwardbackward::AutoCalculator auto_calc(hmm.get(), obs);
        double auto_result = auto_calc.getLogProbability();
        cout << "AutoCalculator:           " << scientific << setprecision(10) << auto_result << endl;
        cout << "  Selected: " << auto_calc.getSelectionRationale() << endl;
        
        // Scaled SIMD
        libhmm::ScaledSIMDForwardBackwardCalculator scaled_calc(hmm.get(), obs);
        scaled_calc.compute();
        double scaled_result = scaled_calc.getLogProbability();
        cout << "ScaledSIMD Calculator:    " << scientific << setprecision(10) << scaled_result << endl;
        
        // Log SIMD
        libhmm::LogSIMDForwardBackwardCalculator log_calc(hmm.get(), obs);
        log_calc.compute();
        double log_result = log_calc.getLogProbability();
        cout << "LogSIMD Calculator:       " << scientific << setprecision(10) << log_result << endl;
        
        // Unscaled (if possible)
        try {
            libhmm::ForwardBackwardCalculator unscaled_calc(hmm.get(), obs);
            double unscaled_prob = unscaled_calc.probability();
            double unscaled_result = (unscaled_prob > 0) ? log(unscaled_prob) : -INFINITY;
            cout << "Unscaled Calculator:      " << scientific << setprecision(10) << unscaled_result << endl;
            cout << "  (raw probability: " << scientific << setprecision(10) << unscaled_prob << ")" << endl;
        } catch (const exception& e) {
            cout << "Unscaled Calculator:      FAILED (" << e.what() << ")" << endl;
        }
        
        // Check consistency
        double auto_scaled_diff = abs(auto_result - scaled_result);
        double auto_log_diff = abs(auto_result - log_result);
        double scaled_log_diff = abs(scaled_result - log_result);
        
        cout << "Differences:" << endl;
        cout << "  Auto vs Scaled:         " << scientific << setprecision(3) << auto_scaled_diff << endl;
        cout << "  Auto vs Log:            " << scientific << setprecision(3) << auto_log_diff << endl;
        cout << "  Scaled vs Log:          " << scientific << setprecision(3) << scaled_log_diff << endl;
        
        if (max({auto_scaled_diff, auto_log_diff, scaled_log_diff}) < 1e-10) {
            cout << "  STATUS: All calculators AGREE" << endl;
        } else {
            cout << "  STATUS: DISCREPANCY detected" << endl;
        }
        
    } catch (const exception& e) {
        cout << "Calculator test failed: " << e.what() << endl;
    }
    cout << endl;
}

void testManualProbabilityCalculation() {
    cout << "=== MANUAL PROBABILITY CALCULATION TEST ===" << endl;
    cout << "Computing Forward probability manually for verification" << endl;
    
    // Simple 2x2 system, 3 observations
    vector<double> pi = {0.6, 0.4};
    vector<vector<double>> A = {{0.7, 0.3}, {0.4, 0.6}};
    vector<vector<double>> B = {{0.5, 0.3, 0.2}, {0.2, 0.3, 0.5}};
    vector<int> obs = {0, 1, 2};
    
    cout << "Model parameters:" << endl;
    cout << "  π = [" << pi[0] << ", " << pi[1] << "]" << endl;
    cout << "  A = [[" << A[0][0] << ", " << A[0][1] << "], [" << A[1][0] << ", " << A[1][1] << "]]" << endl;
    cout << "  B = [[" << B[0][0] << ", " << B[0][1] << ", " << B[0][2] << "], [" << B[1][0] << ", " << B[1][1] << ", " << B[1][2] << "]]" << endl;
    cout << "  O = [" << obs[0] << ", " << obs[1] << ", " << obs[2] << "]" << endl;
    
    // Manual Forward algorithm
    int T = obs.size();
    int N = 2;
    vector<vector<double>> alpha(T, vector<double>(N, 0.0));
    
    // Initialize
    for (int i = 0; i < N; ++i) {
        alpha[0][i] = pi[i] * B[i][obs[0]];
    }
    cout << "α[0] = [" << alpha[0][0] << ", " << alpha[0][1] << "]" << endl;
    
    // Forward steps
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < N; ++j) {
            alpha[t][j] = 0.0;
            for (int i = 0; i < N; ++i) {
                alpha[t][j] += alpha[t-1][i] * A[i][j];
            }
            alpha[t][j] *= B[j][obs[t]];
        }
        cout << "α[" << t << "] = [" << alpha[t][0] << ", " << alpha[t][1] << "]" << endl;
    }
    
    // Final probability
    double manual_prob = alpha[T-1][0] + alpha[T-1][1];
    double manual_log_prob = log(manual_prob);
    
    cout << "Manual calculation:" << endl;
    cout << "  P(O|λ) = " << scientific << setprecision(10) << manual_prob << endl;
    cout << "  log P(O|λ) = " << scientific << setprecision(10) << manual_log_prob << endl;
    
    // Compare with libhmm
    try {
        auto hmm = make_unique<libhmm::Hmm>(2);
        
        libhmm::Vector libhmm_pi(2);
        libhmm_pi(0) = pi[0]; libhmm_pi(1) = pi[1];
        hmm->setPi(libhmm_pi);
        
        libhmm::Matrix libhmm_trans(2, 2);
        libhmm_trans(0, 0) = A[0][0]; libhmm_trans(0, 1) = A[0][1];
        libhmm_trans(1, 0) = A[1][0]; libhmm_trans(1, 1) = A[1][1];
        hmm->setTrans(libhmm_trans);
        
        auto dist0 = make_unique<libhmm::DiscreteDistribution>(3);
        dist0->setProbability(0, B[0][0]); dist0->setProbability(1, B[0][1]); dist0->setProbability(2, B[0][2]);
        auto dist1 = make_unique<libhmm::DiscreteDistribution>(3);
        dist1->setProbability(0, B[1][0]); dist1->setProbability(1, B[1][1]); dist1->setProbability(2, B[1][2]);
        
        hmm->setProbabilityDistribution(0, move(dist0));
        hmm->setProbabilityDistribution(1, move(dist1));
        
        libhmm::ObservationSet libhmm_obs(3);
        libhmm_obs(0) = obs[0]; libhmm_obs(1) = obs[1]; libhmm_obs(2) = obs[2];
        
        libhmm::ForwardBackwardCalculator calc(hmm.get(), libhmm_obs);
        double libhmm_prob = calc.probability();
        double libhmm_log_prob = log(libhmm_prob);
        
        cout << "libhmm calculation:" << endl;
        cout << "  P(O|λ) = " << scientific << setprecision(10) << libhmm_prob << endl;
        cout << "  log P(O|λ) = " << scientific << setprecision(10) << libhmm_log_prob << endl;
        
        double prob_diff = abs(manual_prob - libhmm_prob);
        double log_prob_diff = abs(manual_log_prob - libhmm_log_prob);
        
        cout << "Differences:" << endl;
        cout << "  Probability: " << scientific << setprecision(3) << prob_diff << endl;
        cout << "  Log probability: " << scientific << setprecision(3) << log_prob_diff << endl;
        
        if (prob_diff < 1e-12 && log_prob_diff < 1e-12) {
            cout << "  STATUS: Manual and libhmm calculations MATCH" << endl;
        } else {
            cout << "  STATUS: DISCREPANCY detected" << endl;
        }
        
    } catch (const exception& e) {
        cout << "libhmm comparison failed: " << e.what() << endl;
    }
    cout << endl;
}

int main() {
    cout << "COMPREHENSIVE NUMERICAL ACCURACY DIAGNOSTIC" << endl;
    cout << "===========================================" << endl;
    cout << "This test isolates potential sources of numerical discrepancies" << endl;
    cout << "between libhmm and other HMM libraries." << endl << endl;
    
    testGaussianDistributionBasics();
    testGaussianParameterInterpretation();
    testDiscreteDistributionLogProbability();
    testHMMSetupConsistency();
    testCalculatorConsistency();
    testManualProbabilityCalculation();
    
    cout << "=== DIAGNOSTIC COMPLETE ===" << endl;
    cout << "Review the results above to identify:" << endl;
    cout << "1. Distribution calculation accuracy" << endl;
    cout << "2. Parameter interpretation issues" << endl;
    cout << "3. HMM setup consistency" << endl;
    cout << "4. Calculator algorithm differences" << endl;
    cout << "5. Basic probability computation accuracy" << endl;
    
    return 0;
}
