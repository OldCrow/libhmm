#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <memory>
#include <string>
#include <sstream>

// libhmm includes
#include "libhmm/libhmm.h"
#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/calculators/viterbi_traits.h"

// HMMLib includes  
#include "HMMlib/hmm_table.hpp"
#include "HMMlib/hmm_vector.hpp"
#include "HMMlib/hmm_matrix.hpp"
#include "HMMlib/hmm.hpp"

using namespace std;
using namespace std::chrono;

// Precision tolerance for numerical comparisons
const double TOLERANCE = 1e-6;

// Random number generation with fixed seed for reproducibility
random_device rd;
mt19937 gen(42);

struct BenchmarkResults {
    string problem_name;
    int sequence_length;
    
    double libhmm_forward_time;
    double libhmm_viterbi_time;
    double libhmm_forward_likelihood;
    double libhmm_viterbi_likelihood;
    
    double hmmlib_forward_time;
    double hmmlib_viterbi_time;
    double hmmlib_forward_likelihood;
    double hmmlib_viterbi_likelihood;
    
    bool numerical_match;
    double likelihood_difference;
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

class ComparativeBenchmark {
private:
    ClassicHMMProblems problems;
    
public:
    template<typename ProblemType>
    BenchmarkResults runProblemComparison(ProblemType& problem, int sequence_length) {
        cout << "\n=== " << problem.name << " (length: " << sequence_length << ") ===" << endl;
        
        BenchmarkResults results;
        results.problem_name = problem.name;
        results.sequence_length = sequence_length;
        
        // Generate test sequence
        auto obs_sequence = problem.generateSequence(sequence_length);
        cout << "Generated sequence: first 10 observations: ";
        for (int i = 0; i < min(10, (int)obs_sequence.size()); ++i) {
            cout << obs_sequence[i] << " ";
        }
        cout << (obs_sequence.size() > 10 ? "..." : "") << endl;
        
        // Run libhmm benchmark
        cout << "Running libhmm..." << endl;
        runLibHmmBenchmark(problem, obs_sequence, results);
        
        // Run HMMLib benchmark
        cout << "Running HMMLib..." << endl;
        runHmmLibBenchmark(problem, obs_sequence, results);
        
        // Compare numerical results - compare Viterbi with Viterbi
        results.likelihood_difference = abs(results.libhmm_viterbi_likelihood - results.hmmlib_viterbi_likelihood);
        results.numerical_match = results.likelihood_difference < TOLERANCE;
        
        cout << "Viterbi Likelihood difference: " << scientific << results.likelihood_difference << endl;
        cout << "Viterbi Numerical match: " << (results.numerical_match ? "YES" : "NO") << endl;
        
        return results;
    }
    
private:
    template<typename ProblemType>
    void runLibHmmBenchmark(const ProblemType& problem,
                           const vector<unsigned int>& obs_sequence,
                           BenchmarkResults& results) {
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
                    discrete_dist->setProbability(j, problem.emission_matrix[i][j]);
                }
                hmm->setProbabilityDistribution(i, discrete_dist.release());
            }
            
            // Convert observation sequence to libhmm format
            libhmm::ObservationSet libhmm_obs(obs_sequence.size());
            for (size_t i = 0; i < obs_sequence.size(); ++i) {
                libhmm_obs(i) = obs_sequence[i];
            }
            
            // Benchmark forward-backward using optimal calculator selection
            auto start = high_resolution_clock::now();
            
            // Use ForwardBackward AutoCalculator for optimal performance
            libhmm::forwardbackward::AutoCalculator fb_calc(hmm.get(), libhmm_obs);
            
            // Debug output to show what calculator was selected
            cout << "  Selected Forward-Backward calculator: " << fb_calc.getSelectionRationale() << endl;
            
            // Get the probability (or log-likelihood)
            double forward_backward_likelihood = fb_calc.probability();
            
            auto end = high_resolution_clock::now();
            results.libhmm_forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.libhmm_forward_likelihood = forward_backward_likelihood; // Store Forward-Backward result
            
            // Benchmark Viterbi using AutoCalculator for optimal performance
            start = high_resolution_clock::now();
            
            // Use Viterbi AutoCalculator for optimal performance  
            libhmm::viterbi::AutoCalculator viterbi_calc(hmm.get(), libhmm_obs);
            auto states = viterbi_calc.decode();
            double viterbi_log_prob = viterbi_calc.getLogProbability();
            
            cout << "  Selected Viterbi calculator: " << viterbi_calc.getSelectionRationale() << endl;
            
            end = high_resolution_clock::now();
            results.libhmm_viterbi_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.libhmm_viterbi_likelihood = viterbi_log_prob; // Store Viterbi result
            
            cout << "  libhmm - Forward: " << fixed << setprecision(3) << results.libhmm_forward_time 
                 << "ms, Viterbi: " << results.libhmm_viterbi_time << "ms" << endl;
            cout << "  libhmm - Forward-Backward Log-likelihood: " << scientific << results.libhmm_forward_likelihood << endl;
            cout << "  libhmm - Viterbi Log-likelihood: " << scientific << results.libhmm_viterbi_likelihood << endl;
            
        } catch (const exception& e) {
            cout << "libhmm error: " << e.what() << endl;
            results.libhmm_forward_time = -1;
            results.libhmm_viterbi_time = -1;
            results.libhmm_forward_likelihood = 0;
            results.libhmm_viterbi_likelihood = 0;
        }
    }
    
    template<typename ProblemType>
    void runHmmLibBenchmark(const ProblemType& problem,
                           const vector<unsigned int>& obs_sequence,
                           BenchmarkResults& results) {
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
            results.hmmlib_forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.hmmlib_forward_likelihood = forward_backward_likelihood; // Store Forward-Backward result
            
            // Benchmark Viterbi
            vector<unsigned int> hidden_sequence(obs_sequence.size());
            
            start = high_resolution_clock::now();
            
            double viterbi_likelihood = hmm.viterbi(obs_sequence, hidden_sequence);
            
            end = high_resolution_clock::now();
            results.hmmlib_viterbi_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.hmmlib_viterbi_likelihood = viterbi_likelihood; // Store Viterbi result
            
            cout << "  HMMLib - Forward: " << fixed << setprecision(3) << results.hmmlib_forward_time 
                 << "ms, Viterbi: " << results.hmmlib_viterbi_time << "ms" << endl;
            cout << "  HMMLib - Forward-Backward Log-likelihood: " << scientific << results.hmmlib_forward_likelihood << endl;
            cout << "  HMMLib - Viterbi Log-likelihood: " << scientific << results.hmmlib_viterbi_likelihood << endl;
            
        } catch (const exception& e) {
            cout << "HMMLib error: " << e.what() << endl;
            results.hmmlib_forward_time = -1;
            results.hmmlib_viterbi_time = -1;
            results.hmmlib_forward_likelihood = 0;
            results.hmmlib_viterbi_likelihood = 0;
        }
    }
};

void printResultsTable(const vector<BenchmarkResults>& results) {
    cout << "\n" << string(130, '=') << endl;
    cout << "COMPARATIVE BENCHMARK RESULTS" << endl;
    cout << string(130, '=') << endl;
    cout << "Performance comparison: Ratios > 1.0 indicate HMMLib is faster by that factor" << endl;
    cout << string(130, '-') << endl;
    
    cout << left << setw(22) << "Problem"
         << right << setw(8) << "Length"
         << right << setw(13) << "libhmm F-B"
         << right << setw(13) << "HMMLib F-B" 
         << right << setw(14) << "HMMLib/libhmm"
         << right << setw(13) << "libhmm Vit"
         << right << setw(13) << "HMMLib Vit"
         << right << setw(14) << "HMMLib/libhmm"
         << right << setw(10) << "Match" << endl;
    cout << left << setw(22) << ""
         << right << setw(8) << ""
         << right << setw(13) << "(ms)"
         << right << setw(13) << "(ms)"
         << right << setw(14) << "F-B Ratio"
         << right << setw(13) << "(ms)"
         << right << setw(13) << "(ms)"
         << right << setw(14) << "Vit Ratio"
         << right << setw(10) << "Numeric" << endl;
    cout << string(130, '-') << endl;
    
    for (const auto& result : results) {
        if (result.libhmm_forward_time <= 0 || result.hmmlib_forward_time <= 0) continue;
        
        // Calculate speedup as HMMLib_time / libhmm_time
        // Values < 1.0 mean HMMLib is faster, values > 1.0 mean libhmm is faster
        double fb_speedup = result.libhmm_forward_time / result.hmmlib_forward_time;
        double vit_speedup = result.libhmm_viterbi_time / result.hmmlib_viterbi_time;
        
        // Format speedup values with 'x' suffix included in the string
        stringstream fb_speedup_str, vit_speedup_str;
        fb_speedup_str << fixed << setprecision(2) << fb_speedup << "x";
        vit_speedup_str << fixed << setprecision(2) << vit_speedup << "x";
        
        cout << left << setw(22) << result.problem_name
             << right << setw(8) << result.sequence_length
             << right << setw(13) << fixed << setprecision(2) << result.libhmm_forward_time
             << right << setw(13) << setprecision(2) << result.hmmlib_forward_time
             << right << setw(14) << fb_speedup_str.str()
             << right << setw(13) << setprecision(2) << result.libhmm_viterbi_time
             << right << setw(13) << setprecision(2) << result.hmmlib_viterbi_time
             << right << setw(14) << vit_speedup_str.str()
             << right << setw(10) << (result.numerical_match ? "YES" : "NO") << endl;
    }
    
    cout << string(130, '=') << endl;
    cout << "Note: Ratio values > 1.0 indicate HMMLib is faster by that factor" << endl;
    cout << "      Ratio values < 1.0 indicate libhmm is faster by 1/ratio factor" << endl;
}

void printLikelihoodComparison(const vector<BenchmarkResults>& results) {
    cout << "\nLIKELIHOOD VALIDATION" << endl;
    cout << string(90, '-') << endl;
    cout << left << setw(25) << "Problem"
         << setw(8) << "Length"
         << setw(20) << "libhmm Likelihood"
         << setw(20) << "HMMLib Likelihood"
         << setw(15) << "Difference" << endl;
    cout << string(90, '-') << endl;
    
    for (const auto& result : results) {
        if (result.libhmm_forward_time <= 0 || result.hmmlib_forward_time <= 0) continue;
        
        cout << left << setw(25) << result.problem_name
             << setw(8) << result.sequence_length
             << setw(20) << scientific << setprecision(6) << result.libhmm_viterbi_likelihood
             << setw(20) << result.hmmlib_viterbi_likelihood
             << setw(15) << result.likelihood_difference << endl;
    }
    cout << string(90, '=') << endl;
}

int main() {
    cout << "libhmm vs HMMLib: Classic Discrete HMM Problems Benchmark" << endl;
    cout << "=========================================================" << endl;
    cout << "Using well-known discrete HMM problems for comparison" << endl;
    cout << "Testing both scalar and SIMD Viterbi calculators where applicable" << endl;
    cout << "Fixed random seed (42) for reproducibility" << endl;
    
    ComparativeBenchmark benchmark;
    ClassicHMMProblems problems;
    vector<BenchmarkResults> results;
    
    // Test different sequence lengths for each problem
    vector<int> test_lengths = {100, 500, 1000, 2000};
    
    // Casino Problem
    for (int length : test_lengths) {
        ClassicHMMProblems::CasinoProblem casino;
        auto result = benchmark.runProblemComparison(casino, length);
        results.push_back(result);
    }
    
    // Weather Problem
    for (int length : test_lengths) {
        ClassicHMMProblems::WeatherProblem weather;
        auto result = benchmark.runProblemComparison(weather, length);
        results.push_back(result);
    }
    
    // CpG Island Problem
    for (int length : test_lengths) {
        ClassicHMMProblems::CpGProblem cpg;
        auto result = benchmark.runProblemComparison(cpg, length);
        results.push_back(result);
    }
    
    // Speech Recognition Problem
    for (int length : test_lengths) {
        ClassicHMMProblems::SpeechProblem speech;
        auto result = benchmark.runProblemComparison(speech, length);
        results.push_back(result);
    }
    
    // Print comprehensive results
    printResultsTable(results);
    printLikelihoodComparison(results);
    
    // Summary statistics
    int successful_comparisons = 0;
    int numerical_matches = 0;
    double total_fb_speedup = 0.0;
    double total_vit_speedup = 0.0;
    
    for (const auto& result : results) {
        if (result.libhmm_forward_time > 0 && result.hmmlib_forward_time > 0) {
            successful_comparisons++;
            total_fb_speedup += result.libhmm_forward_time / result.hmmlib_forward_time;
            total_vit_speedup += result.libhmm_viterbi_time / result.hmmlib_viterbi_time;
            
            if (result.numerical_match) {
                numerical_matches++;
            }
        }
    }
    
    if (successful_comparisons > 0) {
        cout << "\nSUMMARY" << endl;
        cout << "-------" << endl;
        cout << "Successful comparisons: " << successful_comparisons << "/" << results.size() << endl;
        cout << "Numerical matches: " << numerical_matches << "/" << successful_comparisons 
             << " (" << fixed << setprecision(1) << (100.0 * numerical_matches / successful_comparisons) << "%)" << endl;
        cout << "Average Forward-Backward speedup: " << setprecision(2) 
             << total_fb_speedup / successful_comparisons << "x" << endl;
        cout << "Average Viterbi speedup: " << total_vit_speedup / successful_comparisons << "x" << endl;
        
        double avg_fb_speedup = total_fb_speedup / successful_comparisons;
        double avg_vit_speedup = total_vit_speedup / successful_comparisons;
        
        if (avg_fb_speedup > 1.0) {
            cout << "\nOverall: HMMLib is " << setprecision(1) << avg_fb_speedup << "x faster on average for Forward-Backward" << endl;
        } else {
            cout << "\nOverall: libhmm is " << setprecision(1) << (1.0/avg_fb_speedup) << "x faster on average for Forward-Backward" << endl;
        }
        
        if (avg_vit_speedup > 1.0) {
            cout << "Overall: HMMLib is " << setprecision(1) << avg_vit_speedup << "x faster on average for Viterbi" << endl;
        } else {
            cout << "Overall: libhmm is " << setprecision(1) << (1.0/avg_vit_speedup) << "x faster on average for Viterbi" << endl;
        }
    }
    
    return 0;
}
