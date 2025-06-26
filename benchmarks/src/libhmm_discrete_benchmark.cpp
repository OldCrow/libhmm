#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <memory>
#include <cstring>

// libhmm includes
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/calculators/calculator.h"
#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/calculators/viterbi_traits.h"

using namespace std;
using namespace std::chrono;

// Random number generation with fixed seed for reproducibility
random_device rd;
mt19937 gen(42);

struct BenchmarkResults {
    string problem_name;
    int sequence_length;
    
    double forward_time;
    double viterbi_time;
    double likelihood;
    
    bool success;
};

class ClassicHMMProblems {
public:
    // Classic "Occasionally Dishonest Casino" problem
    struct CasinoProblem {
        vector<double> initial_probs = {0.5, 0.5};  // Fair, Loaded
        vector<vector<double>> transition_matrix = {
            {0.95, 0.05},  // Fair -> {Fair, Loaded}
            {0.10, 0.90}   // Loaded -> {Fair, Loaded}
        };
        vector<vector<double>> emission_matrix = {
            // Emissions: 1, 2, 3, 4, 5, 6 (dice faces)
            {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6},  // Fair die
            {0.10, 0.10, 0.10, 0.10, 0.10, 0.50}          // Loaded die (6 is more likely)
        };
        int num_states = 2;
        int alphabet_size = 6;
        string name = "Dishonest Casino";
        
        vector<unsigned int> generateSequence(int length) {
            uniform_int_distribution<unsigned int> dist(0, alphabet_size - 1);
            vector<unsigned int> sequence(length);
            for (int i = 0; i < length; ++i) {
                sequence[i] = dist(gen);
            }
            return sequence;
        }
    };
    
    // Simple Weather Model (Sunny/Rainy with Hot/Cold observations)
    struct WeatherProblem {
        vector<double> initial_probs = {0.6, 0.4};  // Sunny, Rainy
        vector<vector<double>> transition_matrix = {
            {0.7, 0.3},  // Sunny -> {Sunny, Rainy}
            {0.4, 0.6}   // Rainy -> {Sunny, Rainy}
        };
        vector<vector<double>> emission_matrix = {
            // Emissions: Hot, Cold
            {0.8, 0.2},  // Sunny -> {Hot, Cold}
            {0.3, 0.7}   // Rainy -> {Hot, Cold}
        };
        int num_states = 2;
        int alphabet_size = 2;
        string name = "Weather Model";
        
        vector<unsigned int> generateSequence(int length) {
            uniform_int_distribution<unsigned int> dist(0, alphabet_size - 1);
            vector<unsigned int> sequence(length);
            for (int i = 0; i < length; ++i) {
                sequence[i] = dist(gen);
            }
            return sequence;
        }
    };
    
    // CpG Island Detection (simplified 3-state model)
    struct CpGProblem {
        vector<double> initial_probs = {0.5, 0.4, 0.1};  // A+T rich, Normal, CpG Island
        vector<vector<double>> transition_matrix = {
            {0.8, 0.15, 0.05},  // A+T rich transitions
            {0.1, 0.8, 0.1},    // Normal transitions  
            {0.05, 0.15, 0.8}   // CpG Island transitions
        };
        vector<vector<double>> emission_matrix = {
            // Emissions: A, C, G, T
            {0.4, 0.1, 0.1, 0.4},   // A+T rich region
            {0.25, 0.25, 0.25, 0.25}, // Normal region
            {0.15, 0.35, 0.35, 0.15}  // CpG Island (C,G rich)
        };
        int num_states = 3;
        int alphabet_size = 4;
        string name = "CpG Island Detection";
        
        vector<unsigned int> generateSequence(int length) {
            uniform_int_distribution<unsigned int> dist(0, alphabet_size - 1);
            vector<unsigned int> sequence(length);
            for (int i = 0; i < length; ++i) {
                sequence[i] = dist(gen);
            }
            return sequence;
        }
    };
    
    // Speech Recognition (simplified 4-state phoneme model)
    struct SpeechProblem {
        vector<double> initial_probs = {0.4, 0.3, 0.2, 0.1};  // 4 phoneme states
        vector<vector<double>> transition_matrix = {
            {0.6, 0.2, 0.1, 0.1},
            {0.1, 0.6, 0.2, 0.1},
            {0.1, 0.1, 0.6, 0.2},
            {0.2, 0.1, 0.1, 0.6}
        };
        vector<vector<double>> emission_matrix = {
            // 8 acoustic features
            {0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1},  // Phoneme 1
            {0.1, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1},  // Phoneme 2
            {0.1, 0.1, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1},  // Phoneme 3
            {0.1, 0.1, 0.1, 0.2, 0.15, 0.15, 0.1, 0.1}   // Phoneme 4
        };
        int num_states = 4;
        int alphabet_size = 8;
        string name = "Speech Recognition";
        
        vector<unsigned int> generateSequence(int length) {
            uniform_int_distribution<unsigned int> dist(0, alphabet_size - 1);
            vector<unsigned int> sequence(length);
            for (int i = 0; i < length; ++i) {
                sequence[i] = dist(gen);
            }
            return sequence;
        }
    };
};

class DiscreteBenchmark {
public:
    template<typename ProblemType>
    BenchmarkResults runBenchmark(ProblemType& problem, int sequence_length) {
        cout << "\n=== " << problem.name << " (length: " << sequence_length << ") ===" << endl;
        
        BenchmarkResults results;
        results.problem_name = problem.name;
        results.sequence_length = sequence_length;
        results.success = false;
        
        // Generate test sequence
        auto obs_sequence = problem.generateSequence(sequence_length);
        cout << "Generated sequence: first 10 observations: ";
        for (int i = 0; i < min(10, (int)obs_sequence.size()); ++i) {
            cout << obs_sequence[i] << " ";
        }
        cout << (obs_sequence.size() > 10 ? "..." : "") << endl;
        
        try {
            // Create libhmm HMM
            auto hmm = make_unique<libhmm::Hmm>(problem.num_states);
            
            // Set up transition matrix
            libhmm::Matrix trans_matrix(problem.num_states, problem.num_states);
            for (int i = 0; i < problem.num_states; ++i) {
                for (int j = 0; j < problem.num_states; ++j) {
                    trans_matrix(i, j) = problem.transition_matrix[i][j];
                }
            }
            hmm->setTrans(trans_matrix);
            
            // Set up initial probabilities
            libhmm::Vector pi_vector(problem.num_states);
            for (int i = 0; i < problem.num_states; ++i) {
                pi_vector(i) = problem.initial_probs[i];
            }
            hmm->setPi(pi_vector);
            
            // Set discrete emission distributions
            for (int i = 0; i < problem.num_states; ++i) {
                auto discrete_dist = make_unique<libhmm::DiscreteDistribution>(problem.alphabet_size);
                for (int j = 0; j < problem.alphabet_size; ++j) {
                    discrete_dist->setProbability(static_cast<libhmm::Observation>(j), problem.emission_matrix[i][j]);
                }
                hmm->setProbabilityDistribution(i, discrete_dist.release());
            }
            
            // Convert observation sequence to libhmm format
            libhmm::ObservationSet libhmm_obs(obs_sequence.size());
            for (size_t i = 0; i < obs_sequence.size(); ++i) {
                libhmm_obs(i) = static_cast<libhmm::Observation>(obs_sequence[i]);
            }
            
            // Use AutoCalculator for optimal Forward-Backward performance
            auto start = high_resolution_clock::now();
            libhmm::forwardbackward::AutoCalculator fb_calc(hmm.get(), libhmm_obs);
            
            // Debug output to show calculator selection
            cout << "  Selected Forward-Backward calculator: " << fb_calc.getSelectionRationale() << endl;
            
            double forward_backward_log_likelihood = fb_calc.getLogProbability();
            auto end = high_resolution_clock::now();
            results.forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.likelihood = forward_backward_log_likelihood;  // Numerically stable log-likelihood
            
            // Use AutoCalculator for optimal Viterbi performance
            start = high_resolution_clock::now();
            libhmm::viterbi::AutoCalculator viterbi_calc(hmm.get(), libhmm_obs);
            
            cout << "  Selected Viterbi calculator: " << viterbi_calc.getSelectionRationale() << endl;
            
            auto states = viterbi_calc.decode();
            end = high_resolution_clock::now();
            results.viterbi_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            
            results.success = true;
            
            cout << "  Forward-Backward: " << fixed << setprecision(3) << results.forward_time 
                 << "ms, Viterbi: " << results.viterbi_time << "ms" << endl;
            cout << "  Log-likelihood: " << scientific << results.likelihood << endl;
            
        } catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
            results.forward_time = -1;
            results.viterbi_time = -1;
            results.likelihood = 0;
        }
        
        return results;
    }
};

void printResultsTable(const vector<BenchmarkResults>& results) {
    cout << "\n" << string(90, '=') << endl;
    cout << "LIBHMM DISCRETE HMM BENCHMARK RESULTS" << endl;
    cout << string(90, '=') << endl;
    
    cout << left << setw(25) << "Problem"
         << setw(8) << "Length"
         << setw(15) << "Forward-B (ms)"
         << setw(15) << "Viterbi (ms)"
         << setw(15) << "Throughput (obs/ms)"
         << setw(12) << "Success" << endl;
    cout << string(90, '-') << endl;
    
    for (const auto& result : results) {
        if (!result.success) continue;
        
        double throughput = result.sequence_length / max(result.forward_time, 0.001);
        
        cout << left << setw(25) << result.problem_name
             << setw(8) << result.sequence_length
             << setw(15) << fixed << setprecision(3) << result.forward_time
             << setw(15) << result.viterbi_time
             << setw(15) << setprecision(1) << throughput
             << setw(12) << (result.success ? "YES" : "NO") << endl;
    }
    
    cout << string(90, '=') << endl;
}

void printScalingAnalysis(const vector<BenchmarkResults>& results) {
    cout << "\nSCALING ANALYSIS" << endl;
    cout << string(70, '-') << endl;
    cout << left << setw(25) << "Problem"
         << setw(8) << "Length"
         << setw(20) << "Log-likelihood"
         << setw(15) << "FB Time/Obs (μs)" << endl;
    cout << string(70, '-') << endl;
    
    for (const auto& result : results) {
        if (!result.success) continue;
        
        double time_per_obs = (result.forward_time * 1000.0) / result.sequence_length;
        
        cout << left << setw(25) << result.problem_name
             << setw(8) << result.sequence_length
             << setw(20) << scientific << setprecision(6) << result.likelihood
             << setw(15) << fixed << setprecision(2) << time_per_obs << endl;
    }
    cout << string(70, '=') << endl;
}

int main() {
    cout << "libhmm Discrete HMM Performance Benchmark" << endl;
    cout << "=========================================" << endl;
    cout << "Testing classic discrete HMM problems" << endl;
    cout << "Fixed random seed (42) for reproducibility" << endl;
    
    DiscreteBenchmark benchmark;
    vector<BenchmarkResults> results;
    
    // Test different sequence lengths for each problem
    vector<int> test_lengths = {100, 500, 1000, 2000, 5000};
    
    // Casino Problem
    cout << "\n--- TESTING CASINO PROBLEM ---" << endl;
    for (int length : test_lengths) {
        ClassicHMMProblems::CasinoProblem casino;
        auto result = benchmark.runBenchmark(casino, length);
        results.push_back(result);
    }
    
    // Weather Problem
    cout << "\n--- TESTING WEATHER PROBLEM ---" << endl;
    for (int length : test_lengths) {
        ClassicHMMProblems::WeatherProblem weather;
        auto result = benchmark.runBenchmark(weather, length);
        results.push_back(result);
    }
    
    // CpG Island Problem
    cout << "\n--- TESTING CpG ISLAND PROBLEM ---" << endl;
    for (int length : test_lengths) {
        ClassicHMMProblems::CpGProblem cpg;
        auto result = benchmark.runBenchmark(cpg, length);
        results.push_back(result);
    }
    
    // Speech Recognition Problem
    cout << "\n--- TESTING SPEECH RECOGNITION PROBLEM ---" << endl;
    for (int length : test_lengths) {
        ClassicHMMProblems::SpeechProblem speech;
        auto result = benchmark.runBenchmark(speech, length);
        results.push_back(result);
    }
    
    // Print comprehensive results
    printResultsTable(results);
    printScalingAnalysis(results);
    
    // Summary statistics
    int successful_tests = 0;
    double total_throughput = 0.0;
    double best_throughput = 0.0;
    string best_problem;
    
    for (const auto& result : results) {
        if (result.success) {
            successful_tests++;
            double throughput = result.sequence_length / max(result.forward_time, 0.001);
            total_throughput += throughput;
            
            if (throughput > best_throughput) {
                best_throughput = throughput;
                best_problem = result.problem_name + " (" + to_string(result.sequence_length) + ")";
            }
        }
    }
    
    if (successful_tests > 0) {
        cout << "\nSUMMARY" << endl;
        cout << "-------" << endl;
        cout << "Successful tests: " << successful_tests << "/" << results.size() << endl;
        cout << "Average throughput: " << fixed << setprecision(1) 
             << total_throughput / successful_tests << " observations/ms" << endl;
        cout << "Best performance: " << setprecision(1) << best_throughput 
             << " obs/ms (" << best_problem << ")" << endl;
             
        // Analyze scaling behavior
        cout << "\nSCALING BEHAVIOR:" << endl;
        for (const string& problem_name : {"Dishonest Casino", "Weather Model", "CpG Island Detection", "Speech Recognition"}) {
            vector<pair<int, double>> problem_results;
            for (const auto& result : results) {
                if (result.success && result.problem_name == problem_name) {
                    double time_per_obs = (result.forward_time * 1000.0) / result.sequence_length;
                    problem_results.push_back({result.sequence_length, time_per_obs});
                }
            }
            
            if (problem_results.size() >= 2) {
                double first_time = problem_results.front().second;
                double last_time = problem_results.back().second;
                double scaling_factor = last_time / first_time;
                
                cout << "  " << problem_name << ": " << setprecision(2) << scaling_factor 
                     << "x time increase from " << problem_results.front().first
                     << " to " << problem_results.back().first << " observations" << endl;
            }
        }
    }
    
    return 0;
}
