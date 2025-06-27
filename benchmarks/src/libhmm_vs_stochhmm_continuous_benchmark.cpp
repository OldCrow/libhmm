#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <memory>

// libhmm includes
#include "libhmm/libhmm.h"
#include "libhmm/distributions/gaussian_distribution.h"

// StochHMM includes
#include "../StochHMM/source/src/StochHMMlib.h"

using namespace std;
using namespace libhmm;

struct BenchmarkResult {
    string library;
    string problem;
    int sequence_length;
    double forward_time_ms;
    double viterbi_time_ms;
    double throughput_obs_per_ms;
    bool success;
    double log_likelihood;
};

class ContinuousGaussianProblems {
private:
    mt19937 rng;
    
public:
    ContinuousGaussianProblems() : rng(42) {} // Fixed seed for reproducibility
    
    // Common model parameters for both libraries
    struct GaussianSignalModel {
        int num_states = 2;
        vector<double> initial_probs = {0.6, 0.4};
        vector<vector<double>> transition_matrix = {
            {0.7, 0.3},
            {0.4, 0.6}
        };
        // State 0: Low signal (mean=0, var=1), State 1: High signal (mean=3, var=2)
        vector<double> means = {0.0, 3.0};
        vector<double> variances = {1.0, 2.0};
    };
    
    // Generate identical observation sequence for both libraries
    vector<double> generateGaussianSequence(int length) {
        vector<double> observations;
        observations.reserve(length);
        
        // Generate realistic Gaussian observations that could come from either state
        normal_distribution<double> obs_dist(1.5, 2.0); // Intermediate between both states
        
        for (int i = 0; i < length; ++i) {
            observations.push_back(obs_dist(rng));
        }
        
        return observations;
    }
    
    GaussianSignalModel getModelParameters() {
        return GaussianSignalModel();
    }
};

// libhmm implementation
BenchmarkResult runLibhmmGaussian(const ContinuousGaussianProblems::GaussianSignalModel& model, 
                                 const vector<double>& observations) {
    BenchmarkResult result;
    result.library = "libhmm";
    result.problem = "Gaussian Signal Detection";
    result.sequence_length = observations.size();
    result.success = false;
    
    try {
        // Create HMM with Gaussian distributions
        auto hmm = make_unique<Hmm>(model.num_states);
        
        // Set initial probabilities
        Vector pi(model.num_states);
        for (int i = 0; i < model.num_states; ++i) {
            pi(i) = model.initial_probs[i];
        }
        hmm->setPi(pi);
        
        // Set transition matrix
        Matrix transition_matrix(model.num_states, model.num_states);
        for (int i = 0; i < model.num_states; ++i) {
            for (int j = 0; j < model.num_states; ++j) {
                transition_matrix(i, j) = model.transition_matrix[i][j];
            }
        }
        hmm->setTrans(transition_matrix);
        
        // Set Gaussian emission distributions
        for (int state = 0; state < model.num_states; ++state) {
            auto gaussian = make_unique<GaussianDistribution>(
                model.means[state], 
                model.variances[state]
            );
            hmm->setProbabilityDistribution(state, move(gaussian));
        }
        
        // Convert observations to ObservationSet format
        ObservationSet obs(observations.size());
        for (size_t i = 0; i < observations.size(); ++i) {
            obs(i) = observations[i];
        }
        
        // Benchmark Forward-Backward using AutoCalculator for optimal performance
        auto start = chrono::high_resolution_clock::now();
        libhmm::forwardbackward::AutoCalculator fb_calc(hmm.get(), obs);
        double forward_likelihood = fb_calc.getLogProbability();
        auto end = chrono::high_resolution_clock::now();
        
        result.forward_time_ms = chrono::duration<double, milli>(end - start).count();
        result.log_likelihood = forward_likelihood;
        
        // Benchmark Viterbi using AutoCalculator for optimal performance
        start = chrono::high_resolution_clock::now();
        libhmm::viterbi::AutoCalculator viterbi_calc(hmm.get(), obs);
        auto viterbi_path = viterbi_calc.decode();
        end = chrono::high_resolution_clock::now();
        
        result.viterbi_time_ms = chrono::duration<double, milli>(end - start).count();
        result.throughput_obs_per_ms = observations.size() / result.forward_time_ms;
        result.success = true;
        
        cout << "  libhmm - Forward: " << fixed << setprecision(3) << result.forward_time_ms 
             << "ms, Viterbi: " << result.viterbi_time_ms << "ms\n";
        cout << "  libhmm - Log-likelihood: " << scientific << setprecision(3) << result.log_likelihood << "\n";
        
    } catch (const exception& e) {
        cerr << "libhmm error: " << e.what() << endl;
        result.forward_time_ms = -1;
        result.viterbi_time_ms = -1;
        result.throughput_obs_per_ms = 0;
        result.log_likelihood = -INFINITY;
    }
    
    return result;
}

// StochHMM implementation
BenchmarkResult runStochHMMGaussian(const ContinuousGaussianProblems::GaussianSignalModel& model,
                                   const vector<double>& observations) {
    BenchmarkResult result;
    result.library = "StochHMM";
    result.problem = "Gaussian Signal Detection";
    result.sequence_length = observations.size();
    result.success = false;
    
    try {
        cout << "  StochHMM continuous distributions not supported in this benchmark version\n";
        cout << "  (Segfault detected in model parsing - needs further investigation)\n";
        
        result.forward_time_ms = -1;
        result.viterbi_time_ms = -1;
        result.throughput_obs_per_ms = 0;
        result.log_likelihood = -INFINITY;
        result.success = false;
        
    } catch (const exception& e) {
        cerr << "StochHMM error: " << e.what() << endl;
        result.forward_time_ms = -1;
        result.viterbi_time_ms = -1;
        result.throughput_obs_per_ms = 0;
        result.log_likelihood = -INFINITY;
    }
    
    return result;
}

void printResults(const vector<BenchmarkResult>& results) {
    cout << "\n========================================================================================================================\n";
    cout << "CONTINUOUS GAUSSIAN HMM BENCHMARK RESULTS\n";
    cout << "========================================================================================================================\n";
    cout << left << setw(10) << "Library" 
         << setw(25) << "Problem" 
         << setw(8) << "Length"
         << setw(15) << "Forward-B (ms)"
         << setw(15) << "Viterbi (ms)"
         << setw(20) << "Throughput (obs/ms)"
         << setw(10) << "Success"
         << setw(20) << "Log-likelihood" << endl;
    cout << "------------------------------------------------------------------------------------------------------------------------\n";
    
    for (const auto& result : results) {
        cout << left << setw(10) << result.library
             << setw(25) << result.problem
             << setw(8) << result.sequence_length
             << setw(15) << fixed << setprecision(3) << result.forward_time_ms
             << setw(15) << fixed << setprecision(3) << result.viterbi_time_ms
             << setw(20) << fixed << setprecision(1) << result.throughput_obs_per_ms
             << setw(10) << (result.success ? "YES" : "NO")
             << setw(20) << scientific << setprecision(3) << result.log_likelihood << endl;
    }
    cout << "========================================================================================================================\n";
}

void analyzeNumericalAccuracy(const vector<BenchmarkResult>& results) {
    cout << "\nNUMERICAL ACCURACY ANALYSIS\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << left << setw(25) << "Problem" 
         << setw(10) << "Length"
         << setw(18) << "libhmm LogLik"
         << setw(18) << "StochHMM LogLik"
         << setw(18) << "Difference" << endl;
    cout << "--------------------------------------------------------------------------------\n";
    
    for (size_t i = 0; i < results.size(); i += 2) {
        if (i + 1 < results.size() && results[i].success && results[i+1].success) {
            double diff = abs(results[i].log_likelihood - results[i+1].log_likelihood);
            cout << left << setw(25) << results[i].problem
                 << setw(10) << results[i].sequence_length
                 << setw(18) << scientific << setprecision(3) << results[i].log_likelihood
                 << setw(18) << scientific << setprecision(3) << results[i+1].log_likelihood
                 << setw(18) << scientific << setprecision(3) << diff << endl;
            
            if (diff > 1e-6) {
                cout << "    *** Potential numerical discrepancy detected ***\n";
            }
        }
    }
    cout << "================================================================================\n";
}

int main() {
    cout << "Continuous Gaussian HMM Benchmark\n";
    cout << "==================================\n";
    cout << "Comparing libhmm vs StochHMM performance on Gaussian emissions\n";
    cout << "Fixed random seed (42) for reproducibility\n\n";
    
    ContinuousGaussianProblems problems;
    auto model = problems.getModelParameters();
    vector<BenchmarkResult> results;
    vector<int> test_lengths = {100, 500, 1000, 2000, 5000};
    
    cout << "Model Parameters:\n";
    cout << "  States: " << model.num_states << "\n";
    cout << "  Initial probs: [" << model.initial_probs[0] << ", " << model.initial_probs[1] << "]\n";
    cout << "  Means: [" << model.means[0] << ", " << model.means[1] << "]\n";
    cout << "  Variances: [" << model.variances[0] << ", " << model.variances[1] << "]\n\n";
    
    for (int length : test_lengths) {
        cout << "=== Gaussian Signal Detection (length: " << length << ") ===\n";
        
        // Generate identical observation sequence for both libraries
        auto observations = problems.generateGaussianSequence(length);
        cout << "Generated sequence: first 10 observations: ";
        for (int i = 0; i < min(10, length); ++i) {
            cout << fixed << setprecision(1) << observations[i] << " ";
        }
        cout << "...\n";
        
        cout << "Running libhmm...\n";
        auto libhmm_result = runLibhmmGaussian(model, observations);
        results.push_back(libhmm_result);
        
        cout << "Running StochHMM...\n";
        auto stochhmm_result = runStochHMMGaussian(model, observations);
        results.push_back(stochhmm_result);
        
        // Compare results immediately
        if (libhmm_result.success && stochhmm_result.success) {
            double likelihood_diff = abs(libhmm_result.log_likelihood - stochhmm_result.log_likelihood);
            cout << "Log-likelihood difference: " << scientific << setprecision(3) << likelihood_diff << "\n";
            cout << "Numerical match: " << (likelihood_diff < 1e-6 ? "YES" : "NO") << "\n";
        }
        cout << "\n";
    }
    
    printResults(results);
    analyzeNumericalAccuracy(results);
    
    // Performance comparison
    cout << "\nPERFORMANCE COMPARISON ANALYSIS\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << left << setw(25) << "Problem" 
         << setw(10) << "Length"
         << setw(15) << "libhmm (ms)"
         << setw(15) << "StochHMM (ms)"
         << setw(15) << "Speedup" << endl;
    cout << "--------------------------------------------------------------------------------\n";
    
    double total_libhmm_throughput = 0, total_stochhmm_throughput = 0;
    int valid_comparisons = 0;
    
    for (size_t i = 0; i < results.size(); i += 2) {
        if (i + 1 < results.size() && results[i].success && results[i+1].success) {
            double speedup = results[i].forward_time_ms / results[i+1].forward_time_ms;
            cout << left << setw(25) << results[i].problem
                 << setw(10) << results[i].sequence_length
                 << setw(15) << fixed << setprecision(3) << results[i].forward_time_ms
                 << setw(15) << fixed << setprecision(3) << results[i+1].forward_time_ms
                 << setw(15) << fixed << setprecision(2) << speedup << "x" << endl;
            
            total_libhmm_throughput += results[i].throughput_obs_per_ms;
            total_stochhmm_throughput += results[i+1].throughput_obs_per_ms;
            valid_comparisons++;
        }
    }
    cout << "================================================================================\n";
    
    if (valid_comparisons > 0) {
        cout << "\nSUMMARY\n";
        cout << "-------\n";
        cout << "libhmm average throughput: " << fixed << setprecision(1) 
             << (total_libhmm_throughput / valid_comparisons) << " observations/ms\n";
        cout << "StochHMM average throughput: " << fixed << setprecision(1) 
             << (total_stochhmm_throughput / valid_comparisons) << " observations/ms\n";
        cout << "Overall performance ratio: " << fixed << setprecision(2) 
             << (total_stochhmm_throughput / total_libhmm_throughput) << "x (StochHMM/libhmm)\n";
    }
    
    return 0;
}
