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

// GHMM includes
extern "C" {
#include "ghmm/ghmm.h"
#include "ghmm/smodel.h"
#include "ghmm/sfoba.h"
#include "ghmm/sviterbi.h"
}

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
                sqrt(model.variances[state])  // Convert variance to std_dev
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
        
        // Debug output for first test to show which calculator was selected
        if (observations.size() == 100) {
            cout << "  [DEBUG] libhmm selected FB calculator: " << fb_calc.getSelectionRationale() << "\n";
        }
        
        // Benchmark Viterbi using AutoCalculator for optimal performance
        start = chrono::high_resolution_clock::now();
        libhmm::viterbi::AutoCalculator viterbi_calc(hmm.get(), obs);
        auto viterbi_path = viterbi_calc.decode();
        end = chrono::high_resolution_clock::now();
        
        // Debug output for first test to show which calculator was selected
        if (observations.size() == 100) {
            cout << "  [DEBUG] libhmm selected Viterbi calculator: " << viterbi_calc.getSelectionRationale() << "\n";
        }
        
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

// GHMM implementation
BenchmarkResult runGhmmGaussian(const ContinuousGaussianProblems::GaussianSignalModel& model,
                               const vector<double>& observations) {
    BenchmarkResult result;
    result.library = "GHMM";
    result.problem = "Gaussian Signal Detection";
    result.sequence_length = observations.size();
    result.success = false;
    
    try {
        // Create GHMM continuous model
        ghmm_cmodel* ghmm_model = (ghmm_cmodel*)calloc(1, sizeof(ghmm_cmodel));
        if (!ghmm_model) {
            throw runtime_error("Failed to allocate GHMM model");
        }
        
        ghmm_model->N = model.num_states;
        ghmm_model->M = 1; // One component per state
        ghmm_model->dim = 1; // Univariate
        ghmm_model->cos = 1; // Single transition matrix
        ghmm_model->model_type = GHMM_kContinuousHMM;
        
        // Allocate states
        ghmm_model->s = (ghmm_cstate*)calloc(ghmm_model->N, sizeof(ghmm_cstate));
        if (!ghmm_model->s) {
            free(ghmm_model);
            throw runtime_error("Failed to allocate GHMM states");
        }
        
        // Configure states with identical parameters to libhmm
        for (int i = 0; i < ghmm_model->N; ++i) {
            ghmm_cstate* state = &ghmm_model->s[i];
            state->M = 1;
            state->pi = model.initial_probs[i];
            
            // Allocate transition arrays
            state->out_states = model.num_states;
            state->in_states = model.num_states;
            state->out_id = (int*)calloc(state->out_states, sizeof(int));
            state->in_id = (int*)calloc(state->in_states, sizeof(int));
            state->out_a = (double**)calloc(1, sizeof(double*));
            state->out_a[0] = (double*)calloc(state->out_states, sizeof(double));
            state->in_a = (double**)calloc(1, sizeof(double*));
            state->in_a[0] = (double*)calloc(state->in_states, sizeof(double));
            
            // Set identical transition probabilities
            for (int j = 0; j < model.num_states; ++j) {
                state->out_id[j] = j;
                state->in_id[j] = j;
                state->out_a[0][j] = model.transition_matrix[i][j];
                state->in_a[0][j] = model.transition_matrix[j][i];
            }
            
            // Allocate emission parameters
            state->c = (double*)calloc(1, sizeof(double));
            state->c[0] = 1.0;
            state->e = (ghmm_c_emission*)calloc(1, sizeof(ghmm_c_emission));
            
            // Configure Gaussian emission with identical parameters
            state->e[0].type = normal;
            state->e[0].dimension = 1;
            state->e[0].mean.val = model.means[i];
            state->e[0].variance.val = model.variances[i];
            state->e[0].fixed = 0;
        }
        
        // Convert observations to GHMM format (same sequence as libhmm)
        double* obs_array = (double*)malloc(observations.size() * sizeof(double));
        for (size_t i = 0; i < observations.size(); ++i) {
            obs_array[i] = observations[i];
        }
        
        // Benchmark Forward-Backward
        auto start = chrono::high_resolution_clock::now();
        double forward_likelihood;
        int result_code = ghmm_cmodel_logp(ghmm_model, obs_array, observations.size(), &forward_likelihood);
        auto end = chrono::high_resolution_clock::now();
        
        result.forward_time_ms = chrono::duration<double, milli>(end - start).count();
        result.log_likelihood = (result_code == 0) ? forward_likelihood : -INFINITY;
        
        // Benchmark Viterbi
        double viterbi_logp;
        start = chrono::high_resolution_clock::now();
        int* viterbi_path = ghmm_cmodel_viterbi(ghmm_model, obs_array, observations.size(), &viterbi_logp);
        end = chrono::high_resolution_clock::now();
        
        result.viterbi_time_ms = chrono::duration<double, milli>(end - start).count();
        result.throughput_obs_per_ms = observations.size() / result.forward_time_ms;
        result.success = true;
        
        cout << "  GHMM - Forward: " << fixed << setprecision(3) << result.forward_time_ms 
             << "ms, Viterbi: " << result.viterbi_time_ms << "ms\n";
        cout << "  GHMM - Log-likelihood: " << scientific << setprecision(3) << result.log_likelihood << "\n";
        
        // Cleanup
        free(obs_array);
        free(viterbi_path);
        
        // Free GHMM model (simplified cleanup)
        for (int i = 0; i < ghmm_model->N; ++i) {
            free(ghmm_model->s[i].out_id);
            free(ghmm_model->s[i].in_id);
            if (ghmm_model->s[i].out_a) {
                free(ghmm_model->s[i].out_a[0]);
                free(ghmm_model->s[i].out_a);
            }
            if (ghmm_model->s[i].in_a) {
                free(ghmm_model->s[i].in_a[0]);
                free(ghmm_model->s[i].in_a);
            }
            free(ghmm_model->s[i].c);
            free(ghmm_model->s[i].e);
        }
        free(ghmm_model->s);
        free(ghmm_model);
        
    } catch (const exception& e) {
        cerr << "GHMM error: " << e.what() << endl;
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
         << setw(18) << "GHMM LogLik"
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
    cout << "Comparing libhmm vs GHMM performance on Gaussian emissions\n";
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
        
        cout << "Running GHMM...\n";
        auto ghmm_result = runGhmmGaussian(model, observations);
        results.push_back(ghmm_result);
        
        // Compare results immediately
        if (libhmm_result.success && ghmm_result.success) {
            double likelihood_diff = abs(libhmm_result.log_likelihood - ghmm_result.log_likelihood);
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
         << setw(15) << "GHMM (ms)"
         << setw(15) << "Speedup" << endl;
    cout << "--------------------------------------------------------------------------------\n";
    
    double total_libhmm_throughput = 0, total_ghmm_throughput = 0;
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
            total_ghmm_throughput += results[i+1].throughput_obs_per_ms;
            valid_comparisons++;
        }
    }
    cout << "================================================================================\n";
    
    if (valid_comparisons > 0) {
        cout << "\nSUMMARY\n";
        cout << "-------\n";
        cout << "libhmm average throughput: " << fixed << setprecision(1) 
             << (total_libhmm_throughput / valid_comparisons) << " observations/ms\n";
        cout << "GHMM average throughput: " << fixed << setprecision(1) 
             << (total_ghmm_throughput / valid_comparisons) << " observations/ms\n";
        cout << "Overall performance ratio: " << fixed << setprecision(2) 
             << (total_ghmm_throughput / total_libhmm_throughput) << "x (GHMM/libhmm)\n";
    }
    
    return 0;
}
