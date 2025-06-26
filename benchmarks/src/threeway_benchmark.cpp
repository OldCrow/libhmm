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
#include "HMMlib/hmm.h"
#include "HMMlib/nStateHMM.h"
#include "HMMlib/forwardAlg.h"
#include "HMMlib/viterbiAlg.h"

// StochHMM includes
#include "StochHMM/src/stochMath.h"
#include "StochHMM/src/hmm.h"
#include "StochHMM/src/seqTracks.h"

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
            
            // Use AutoCalculator for optimal Viterbi performance
            start = high_resolution_clock::now();
            libhmm::viterbi::AutoCalculator viterbi_calc(hmm.get(), libhmm_obs);
            auto states = viterbi_calc.decode();
            end = high_resolution_clock::now();
            results.viterbi_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            
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
            // Create HMMLib model
            HMMLib::nStateHMM* hmm = new HMMLib::nStateHMM();
            
            // Set up the model structure
            hmm->setStates(problem.num_states);
            hmm->setEmissions(problem.alphabet_size);
            
            // Set initial probabilities
            for (int i = 0; i < problem.num_states; ++i) {
                hmm->setInitial(i, problem.initial_probs[i]);
            }
            
            // Set transition probabilities
            for (int i = 0; i < problem.num_states; ++i) {
                for (int j = 0; j < problem.num_states; ++j) {
                    hmm->setTransition(i, j, problem.transition_matrix[i][j]);
                }
            }
            
            // Set emission probabilities
            for (int i = 0; i < problem.num_states; ++i) {
                for (int j = 0; j < problem.alphabet_size; ++j) {
                    hmm->setEmission(i, j, problem.emission_matrix[i][j]);
                }
            }
            
            // Convert observation sequence to HMMLib format
            vector<short> hmmlib_sequence(obs_sequence.size());
            for (size_t i = 0; i < obs_sequence.size(); ++i) {
                hmmlib_sequence[i] = static_cast<short>(obs_sequence[i]);
            }
            
            // Forward-Backward benchmark
            auto start = high_resolution_clock::now();
            HMMLib::forwardAlg forward_alg(hmm);
            forward_alg.runForward(hmmlib_sequence);
            double forward_prob = forward_alg.getLogProbability();
            auto end = high_resolution_clock::now();
            results.forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.likelihood = forward_prob;
            
            // Viterbi benchmark
            start = high_resolution_clock::now();
            HMMLib::viterbiAlg viterbi_alg(hmm);
            viterbi_alg.runViterbi(hmmlib_sequence);
            auto viterbi_states = viterbi_alg.getPath();
            end = high_resolution_clock::now();
            results.viterbi_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            
            results.success = true;
            
            delete hmm;
            
        } catch (const exception& e) {
            cout << "HMMLib Error: " << e.what() << endl;
            results.forward_time = -1;
            results.viterbi_time = -1;
            results.likelihood = 0;
        }
        
        return results;
    }
};

class StochHMMBenchmark {
public:
    template<typename ProblemType>
    BenchmarkResults runBenchmark(ProblemType& problem, const vector<unsigned int>& full_obs_sequence, int sequence_length) {
        BenchmarkResults results;
        results.library_name = "StochHMM";
        results.problem_name = problem.name;
        results.sequence_length = sequence_length;
        results.success = false;
        
        // Use the first 'sequence_length' observations from the shared sequence
        vector<unsigned int> obs_sequence(full_obs_sequence.begin(), 
                                         full_obs_sequence.begin() + sequence_length);
        
        try {
            // Create StochHMM model
            StochHMM::model hmm;
            
            // Set up alphabet
            StochHMM::track* tr = new StochHMM::track("observations");
            for (int i = 0; i < problem.alphabet_size; ++i) {
                tr->addAlphaName(to_string(i));
            }
            hmm.addTrack(tr);
            
            // Add states
            for (int i = 0; i < problem.num_states; ++i) {
                StochHMM::state* st = new StochHMM::state();
                st->setName("State" + to_string(i));
                
                // Set emission probabilities
                for (int j = 0; j < problem.alphabet_size; ++j) {
                    st->setEmission(tr, j, log(problem.emission_matrix[i][j]));
                }
                
                // Set initial probability
                st->setInitial(log(problem.initial_probs[i]));
                
                hmm.addState(st);
            }
            
            // Set transition probabilities
            for (int i = 0; i < problem.num_states; ++i) {
                for (int j = 0; j < problem.num_states; ++j) {
                    hmm.setTransition(i, j, log(problem.transition_matrix[i][j]));
                }
            }
            
            // Convert observation sequence to StochHMM format
            StochHMM::sequences seqs(&hmm);
            StochHMM::sequence seq("test_seq");
            for (size_t i = 0; i < obs_sequence.size(); ++i) {
                seq.addObservation(to_string(obs_sequence[i]));
            }
            seqs.addSeq(seq);
            
            // Forward-Backward benchmark
            auto start = high_resolution_clock::now();
            double forward_prob = hmm.forward(&seq);
            auto end = high_resolution_clock::now();
            results.forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.likelihood = forward_prob;
            
            // Viterbi benchmark
            start = high_resolution_clock::now();
            StochHMM::traceback_path viterbi_path = hmm.viterbi(&seq);
            end = high_resolution_clock::now();
            results.viterbi_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            
            results.success = true;
            
        } catch (const exception& e) {
            cout << "StochHMM Error: " << e.what() << endl;
            results.forward_time = -1;
            results.viterbi_time = -1;
            results.likelihood = 0;
        }
        
        return results;
    }
};

void printComparisonResults(const vector<BenchmarkResults>& results) {
    cout << "\n" << string(140, '=') << endl;
    cout << "THREE-WAY LIBRARY COMPARISON BENCHMARK RESULTS" << endl;
    cout << string(140, '=') << endl;
    
    cout << left << setw(12) << "Library"
         << setw(25) << "Problem"
         << setw(10) << "Length"
         << setw(15) << "Forward-B (ms)"
         << setw(15) << "Viterbi (ms)"
         << setw(20) << "Throughput (obs/ms)"
         << setw(15) << "Success"
         << setw(25) << "Log-likelihood" << endl;
    cout << string(140, '-') << endl;
    
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
             << setw(25) << scientific << setprecision(6) << result.likelihood << endl;
    }
    
    cout << string(140, '=') << endl;
}

void printNumericalAccuracyAnalysis(const vector<BenchmarkResults>& results) {
    cout << "\nNUMERICAL ACCURACY ANALYSIS" << endl;
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
         << setw(18) << "libhmm LogLik"
         << setw(18) << "HMMLib LogLik"
         << setw(18) << "StochHMM LogLik" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& group : grouped_results) {
        if (group.second.size() >= 3) {
            const string& problem_name = group.first.first;
            int length = group.first.second;
            
            double libhmm_loglik = 0, hmmlib_loglik = 0, stochhmm_loglik = 0;
            for (const auto& result : group.second) {
                if (result.library_name == "libhmm") {
                    libhmm_loglik = result.likelihood;
                } else if (result.library_name == "HMMLib") {
                    hmmlib_loglik = result.likelihood;
                } else if (result.library_name == "StochHMM") {
                    stochhmm_loglik = result.likelihood;
                }
            }
            
            cout << left << setw(25) << problem_name
                 << setw(10) << length
                 << setw(18) << scientific << setprecision(6) << libhmm_loglik
                 << setw(18) << hmmlib_loglik
                 << setw(18) << stochhmm_loglik << endl;
                 
            // Calculate differences
            double diff_hmmlib = abs(libhmm_loglik - hmmlib_loglik);
            double diff_stochhmm = abs(libhmm_loglik - stochhmm_loglik);
            
            if (diff_hmmlib > 1e-6 || diff_stochhmm > 1e-6) {
                cout << "    *** Potential numerical discrepancy detected ***" << endl;
                cout << "        libhmm-HMMLib diff: " << scientific << diff_hmmlib << endl;
                cout << "        libhmm-StochHMM diff: " << scientific << diff_stochhmm << endl;
            }
        }
    }
    cout << string(80, '=') << endl;
}

int main() {
    cout << "Three-Way HMM Library Comparison Benchmark" << endl;
    cout << "===========================================" << endl;
    cout << "Comparing libhmm vs HMMLib vs StochHMM performance" << endl;
    cout << "Fixed random seed (42) for reproducibility" << endl;
    
    LibHMMBenchmark libhmm_benchmark;
    HMMLibBenchmark hmmlib_benchmark;
    StochHMMBenchmark stochhmm_benchmark;
    vector<BenchmarkResults> results;
    
    // Test different sequence lengths for each problem
    vector<int> test_lengths = {100, 500, 1000, 2000, 5000, 10000};
    
    // Casino Problem
    cout << "\n--- TESTING CASINO PROBLEM ---" << endl;
    ClassicHMMProblems::CasinoProblem casino;
    auto casino_full_obs_sequence = casino.generateSequence(10000);

    for (int length : test_lengths) {
        cout << "  Testing length: " << length << endl;
        
        cout << "    libhmm: ";
        auto libhmm_result = libhmm_benchmark.runBenchmark(casino, casino_full_obs_sequence, length);
        cout << (libhmm_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(libhmm_result);

        cout << "    HMMLib: ";
        auto hmmlib_result = hmmlib_benchmark.runBenchmark(casino, casino_full_obs_sequence, length);
        cout << (hmmlib_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(hmmlib_result);
        
        cout << "    StochHMM: ";
        auto stochhmm_result = stochhmm_benchmark.runBenchmark(casino, casino_full_obs_sequence, length);
        cout << (stochhmm_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(stochhmm_result);
    }
    
    // Weather Problem
    cout << "\n--- TESTING WEATHER PROBLEM ---" << endl;
    ClassicHMMProblems::WeatherProblem weather;
    auto weather_full_obs_sequence = weather.generateSequence(10000);
    
    for (int length : test_lengths) {
        cout << "  Testing length: " << length << endl;
        
        cout << "    libhmm: ";
        auto libhmm_result = libhmm_benchmark.runBenchmark(weather, weather_full_obs_sequence, length);
        cout << (libhmm_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(libhmm_result);
        
        cout << "    HMMLib: ";
        auto hmmlib_result = hmmlib_benchmark.runBenchmark(weather, weather_full_obs_sequence, length);
        cout << (hmmlib_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(hmmlib_result);
        
        cout << "    StochHMM: ";
        auto stochhmm_result = stochhmm_benchmark.runBenchmark(weather, weather_full_obs_sequence, length);
        cout << (stochhmm_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(stochhmm_result);
    }
    
    // Print comprehensive results
    printComparisonResults(results);
    printNumericalAccuracyAnalysis(results);
    
    return 0;
}
