#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <memory>
#include <cstring>
#include <sstream>
#include <map>

// libhmm includes
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/calculators/calculator.h"
#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/calculators/viterbi_traits.h"

// HMMLib includes  
#include "HMMlib/hmm_table.hpp"
#include "HMMlib/hmm_vector.hpp"
#include "HMMlib/hmm_matrix.hpp"
#include "HMMlib/hmm.hpp"

using namespace std;
using namespace std::chrono;

// Random number generation with fixed seed for reproducibility
random_device rd;
mt19937 gen(42);

struct BenchmarkResults {
    string library_name;
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
};

class LibHMMBenchmark {
public:
    // Original method that generates sequence internally
    template<typename ProblemType>
    BenchmarkResults runBenchmark(ProblemType& problem, int sequence_length) {
        // Generate test sequence
        auto obs_sequence = problem.generateSequence(sequence_length);
        return runBenchmark(problem, obs_sequence, sequence_length);
    }
    
    // New method that accepts pre-generated sequence
    template<typename ProblemType>
    BenchmarkResults runBenchmark(ProblemType& problem, const vector<unsigned int>& full_obs_sequence, int sequence_length) {
        BenchmarkResults results;
        results.library_name = "libhmm";
        results.problem_name = problem.name;
        results.sequence_length = sequence_length;
        results.success = false;
        
        // Use the first 'sequence_length' observations from the shared sequence
        vector<unsigned int> obs_sequence(full_obs_sequence.begin(), 
                                         full_obs_sequence.begin() + sequence_length);
        
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
            double forward_backward_log_likelihood = fb_calc.getLogProbability();
            auto end = high_resolution_clock::now();
            results.forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.likelihood = forward_backward_log_likelihood;
            
            // Debug output for first test
            if (sequence_length == 100) {
                cout << "[DEBUG] libhmm selected FB calculator: " << fb_calc.getSelectionRationale() << endl;
            }
            
            // Use AutoCalculator for optimal Viterbi performance
            start = high_resolution_clock::now();
            libhmm::viterbi::AutoCalculator viterbi_calc(hmm.get(), libhmm_obs);
            auto states = viterbi_calc.decode();
            end = high_resolution_clock::now();
            results.viterbi_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            
            // Debug output for first test
            if (sequence_length == 100) {
                cout << "[DEBUG] libhmm selected Viterbi calculator: " << viterbi_calc.getSelectionRationale() << endl;
            }
            
            results.success = true;
            
        } catch (const exception& e) {
            cout << "LibHMM Error: " << e.what() << endl;
            results.forward_time = -1;
            results.viterbi_time = -1;
            results.likelihood = 0;
        }
        
        return results;
    }
};

class HMMLibBenchmark {
public:
    // Original method that generates sequence internally
    template<typename ProblemType>
    BenchmarkResults runBenchmark(ProblemType& problem, int sequence_length) {
        // Generate test sequence
        auto obs_sequence = problem.generateSequence(sequence_length);
        return runBenchmark(problem, obs_sequence, sequence_length);
    }
    
    // New method that accepts pre-generated sequence
    template<typename ProblemType>
    BenchmarkResults runBenchmark(ProblemType& problem, const vector<unsigned int>& full_obs_sequence, int sequence_length) {
        BenchmarkResults results;
        results.library_name = "HMMLib";
        results.problem_name = problem.name;
        results.sequence_length = sequence_length;
        results.success = false;
        
        // Use the first 'sequence_length' observations from the shared sequence
        vector<unsigned int> obs_sequence(full_obs_sequence.begin(), 
                                         full_obs_sequence.begin() + sequence_length);
        
        try {
            // Create HMMLib matrices and vectors
            auto pi_ptr = boost::shared_ptr<hmmlib::HMMVector<double>>(
                new hmmlib::HMMVector<double>(problem.num_states));
            auto T_ptr = boost::shared_ptr<hmmlib::HMMMatrix<double>>(
                new hmmlib::HMMMatrix<double>(problem.num_states, problem.num_states));
            auto E_ptr = boost::shared_ptr<hmmlib::HMMMatrix<double>>(
                new hmmlib::HMMMatrix<double>(problem.alphabet_size, problem.num_states));
            
            // Set initial probabilities
            for (int i = 0; i < problem.num_states; ++i) {
                (*pi_ptr)(i) = problem.initial_probs[i];
            }
            
            // Set transition matrix
            for (int i = 0; i < problem.num_states; ++i) {
                for (int j = 0; j < problem.num_states; ++j) {
                    (*T_ptr)(i, j) = problem.transition_matrix[i][j];
                }
            }
            
            // Set emission matrix (note: HMMLib uses E(symbol, state) indexing)
            for (int symbol = 0; symbol < problem.alphabet_size; ++symbol) {
                for (int state = 0; state < problem.num_states; ++state) {
                    (*E_ptr)(symbol, state) = problem.emission_matrix[state][symbol];
                }
            }
            
            // Create HMMLib HMM
            hmmlib::HMM<double> hmm(pi_ptr, T_ptr, E_ptr);
            
            // Prepare data structures for algorithms
            hmmlib::HMMMatrix<double> F(obs_sequence.size(), problem.num_states);
            hmmlib::HMMMatrix<double> B(obs_sequence.size(), problem.num_states);
            hmmlib::HMMVector<double> scales(obs_sequence.size());
            
            // Benchmark forward-backward
            auto start = high_resolution_clock::now();
            hmm.forward(obs_sequence, scales, F);
            hmm.backward(obs_sequence, scales, B);
            double forward_backward_likelihood = hmm.likelihood(scales);
            auto end = high_resolution_clock::now();
            results.forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.likelihood = forward_backward_likelihood;
            
            // Debug output for first test
            if (sequence_length == 100) {
                cout << "[DEBUG] HMMLib using forward-backward with scaling" << endl;
            }
            
            // Benchmark Viterbi
            vector<unsigned int> hidden_sequence(obs_sequence.size());
            start = high_resolution_clock::now();
            double viterbi_likelihood = hmm.viterbi(obs_sequence, hidden_sequence);
            end = high_resolution_clock::now();
            results.viterbi_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            
            results.success = true;
            
        } catch (const exception& e) {
            cout << "HMMLib Error: " << e.what() << endl;
            results.forward_time = -1;
            results.viterbi_time = -1;
            results.likelihood = 0;
        }
        
        return results;
    }
};

void printComparisonResults(const vector<BenchmarkResults>& results) {
    cout << "\n" << string(120, '=') << endl;
    cout << "LIBRARY COMPARISON BENCHMARK RESULTS" << endl;
    cout << string(120, '=') << endl;
    
    cout << left << setw(12) << "Library"
         << setw(25) << "Problem"
         << setw(10) << "Length"
         << setw(15) << "Forward-B (ms)"
         << setw(15) << "Viterbi (ms)"
         << setw(20) << "Throughput (obs/ms)"
         << setw(15) << "Success"
         << setw(20) << "Log-likelihood" << endl;
    cout << string(120, '-') << endl;
    
    for (const auto& result : results) {
        if (!result.success) continue;
        
        double throughput = result.sequence_length / max(result.forward_time, 0.001);
        
        cout << left << setw(12) << result.library_name
             << setw(25) << result.problem_name
             << setw(10) << result.sequence_length
             << setw(15) << fixed << setprecision(3) << result.forward_time
             << setw(15) << result.viterbi_time
             << setw(20) << setprecision(1) << throughput
             << setw(15) << (result.success ? "YES" : "NO")
             << setw(20) << scientific << setprecision(3) << result.likelihood << endl;
    }
    
    cout << string(120, '=') << endl;
}

void printPerformanceComparison(const vector<BenchmarkResults>& results) {
    cout << "\nPERFORMANCE COMPARISON ANALYSIS" << endl;
    cout << string(80, '-') << endl;
    
    // Group results by problem and sequence length
    map<pair<string, int>, vector<BenchmarkResults>> grouped_results;
    for (const auto& result : results) {
        if (result.success) {
            grouped_results[{result.problem_name, result.sequence_length}].push_back(result);
        }
    }
    
    cout << left << setw(25) << "Problem"
         << setw(10) << "Length"
         << setw(16) << "libhmm (ms)"
         << setw(16) << "HMMLib (ms)"
         << setw(12) << "Speedup" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& group : grouped_results) {
        if (group.second.size() >= 2) {
            const string& problem_name = group.first.first;
            int length = group.first.second;
            
            double libhmm_time = -1, hmmlib_time = -1;
            for (const auto& result : group.second) {
                if (result.library_name == "libhmm") {
                    libhmm_time = result.forward_time;
                } else if (result.library_name == "HMMLib") {
                    hmmlib_time = result.forward_time;
                }
            }
            
            if (libhmm_time > 0 && hmmlib_time > 0) {
                double speedup = hmmlib_time / libhmm_time;
                cout << left << setw(25) << problem_name
                     << setw(10) << length
                     << setw(16) << fixed << setprecision(3) << libhmm_time
                     << setw(16) << hmmlib_time
                     << setprecision(2) << speedup << "x" << endl;
            }
        }
    }
    cout << string(80, '=') << endl;
}

int main() {
    cout << "HMM Library Comparison Benchmark" << endl;
    cout << "================================" << endl;
    cout << "Comparing libhmm vs HMMLib performance" << endl;
    cout << "Fixed random seed (42) for reproducibility" << endl;
    
    LibHMMBenchmark libhmm_benchmark;
    HMMLibBenchmark hmmlib_benchmark;
    vector<BenchmarkResults> results;
    
    // Test different sequence lengths for each problem
    // Include longer sequences as used in StochHMM's own benchmarking
    vector<int> test_lengths = {100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000};
    
    // Casino Problem
    cout << "\n--- TESTING CASINO PROBLEM ---" << endl;
    // Generate a single, large observation sequence to share between tests
    ClassicHMMProblems::CasinoProblem casino;
    auto full_obs_sequence = casino.generateSequence(1000000);

    for (int length : test_lengths) {
        cout << "  libhmm (length: " << length << "): ";
        auto libhmm_result = libhmm_benchmark.runBenchmark(casino, full_obs_sequence, length);
        cout << (libhmm_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(libhmm_result);

        cout << "  HMMLib (length: " << length << "): ";
        auto hmmlib_result = hmmlib_benchmark.runBenchmark(casino, full_obs_sequence, length);
        cout << (hmmlib_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(hmmlib_result);
    }
    
    // Weather Problem
    cout << "\n--- TESTING WEATHER PROBLEM ---" << endl;
    // Generate a single, large observation sequence to share between tests
    ClassicHMMProblems::WeatherProblem weather;
    auto weather_full_obs_sequence = weather.generateSequence(1000000);
    
    for (int length : test_lengths) {
        cout << "  libhmm (length: " << length << "): ";
        auto libhmm_result = libhmm_benchmark.runBenchmark(weather, weather_full_obs_sequence, length);
        cout << (libhmm_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(libhmm_result);
        
        cout << "  HMMLib (length: " << length << "): ";
        auto hmmlib_result = hmmlib_benchmark.runBenchmark(weather, weather_full_obs_sequence, length);
        cout << (hmmlib_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(hmmlib_result);
    }
    
    // Print comprehensive results
    printComparisonResults(results);
    printPerformanceComparison(results);
    
    // Summary statistics
    int libhmm_successes = 0, hmmlib_successes = 0;
    double libhmm_total_throughput = 0.0, hmmlib_total_throughput = 0.0;
    
    for (const auto& result : results) {
        if (result.success) {
            double throughput = result.sequence_length / max(result.forward_time, 0.001);
            if (result.library_name == "libhmm") {
                libhmm_successes++;
                libhmm_total_throughput += throughput;
            } else if (result.library_name == "HMMLib") {
                hmmlib_successes++;
                hmmlib_total_throughput += throughput;
            }
        }
    }
    
    cout << "\nSUMMARY" << endl;
    cout << "-------" << endl;
    if (libhmm_successes > 0) {
        cout << "libhmm average throughput: " << fixed << setprecision(1) 
             << libhmm_total_throughput / libhmm_successes << " observations/ms" << endl;
    }
    if (hmmlib_successes > 0) {
        cout << "HMMLib average throughput: " << fixed << setprecision(1) 
             << hmmlib_total_throughput / hmmlib_successes << " observations/ms" << endl;
    }
    
    if (libhmm_successes > 0 && hmmlib_successes > 0) {
        double avg_speedup = (hmmlib_total_throughput / hmmlib_successes) / 
                           (libhmm_total_throughput / libhmm_successes);
        cout << "Overall performance ratio: " << setprecision(2) << avg_speedup 
             << "x (HMMLib/libhmm)" << endl;
    }
    
    return 0;
}
