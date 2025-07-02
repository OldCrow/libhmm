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
#include "HMMlib/hmm.hpp"

// StochHMM includes
#include "StochHMMlib.h"

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
            // Create StochHMM model string
            string model_str = createModelString(problem);
            
            // Create sequence string
            string seq_str = createSequenceString(obs_sequence);
            
            // Parse model
            StochHMM::model hmm;
            hmm.importFromString(model_str);
            hmm.finalize();
            
            // Create sequence object and load from string
            StochHMM::sequences seqs;
            StochHMM::sequence* seq = new StochHMM::sequence();
            
            // Set up sequence data (simplified approach)
            stringstream seq_stream(seq_str);
            string line;
            getline(seq_stream, line); // Skip header
            getline(seq_stream, line); // Get sequence data
            
            // Set sequence using the setSeq method
            seq->setSeq(line, hmm.getTrack(0));
            seqs.addSeq(seq);
            
            // Create trellis
            StochHMM::trellis trel(&hmm, &seqs);
            
            // Forward-Backward benchmark
            auto start = high_resolution_clock::now();
            trel.simple_forward();
            trel.simple_backward();
            auto end = high_resolution_clock::now();
            results.forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            
            // Get likelihood
            results.likelihood = trel.getForwardProbability();
            
            // Viterbi benchmark
            start = high_resolution_clock::now();
            trel.simple_viterbi();
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

private:
    template<typename ProblemType>
    string createModelString(ProblemType& problem) {
        stringstream ss;
        
        // Model header
        ss << "#STOCHHMM MODEL FILE\n";
        ss << "MODEL INFORMATION\n";
        ss << "======================================================\n";
        ss << "MODEL_NAME:\t" << problem.name << "\n";
        ss << "MODEL_DESCRIPTION:\tAuto-generated for benchmarking\n";
        ss << "\n";
        
        // Track symbol definitions
        ss << "TRACK SYMBOL DEFINITIONS\n";
        ss << "======================================================\n";
        string track_name = "TRACK";
        if (problem.name == "Dishonest Casino") {
            track_name = "DICE";
        } else if (problem.name == "Weather Model") {
            track_name = "WEATHER";
        }
        ss << track_name << ":\t";
        for (int i = 0; i < problem.alphabet_size; ++i) {
            ss << i;
            if (i < problem.alphabet_size - 1) ss << ",";
        }
        ss << "\n\n";
        
        // State definitions header
        ss << "STATE DEFINITIONS\n";
        ss << "#############################################\n";
        
        // Initial state (if we have initial probabilities)
        if (problem.initial_probs.size() > 0) {
            ss << "STATE:\n";
            ss << "\tNAME:\tINIT\n";
            ss << "TRANSITION:\tSTANDARD: P(X)\n";
            for (int j = 0; j < problem.num_states; ++j) {
                ss << "\tState" << j << ":\t" << problem.initial_probs[j] << "\n";
            }
            ss << "#############################################\n";
        }
        
        // States section
        for (int i = 0; i < problem.num_states; ++i) {
            ss << "STATE:\n";
            ss << "\tNAME:\tState" << i << "\n";
            ss << "\tPATH_LABEL:\tS" << i << "\n";
            
            // Transition probabilities
            ss << "TRANSITION:\tSTANDARD: P(X)\n";
            for (int j = 0; j < problem.num_states; ++j) {
                ss << "\tState" << j << ":\t" << problem.transition_matrix[i][j] << "\n";
            }
            ss << "\tEND:\t1\n";
            
            // Emission probabilities
            string track_name = "TRACK";
            if (problem.name == "Dishonest Casino") {
                track_name = "DICE";
            } else if (problem.name == "Weather Model") {
                track_name = "WEATHER";
            }
            ss << "EMISSION:\t" << track_name << ": P(X)\n";
            ss << "\tORDER:\t0\n";
            ss << "@";
            for (int k = 0; k < problem.alphabet_size; ++k) {
                ss << k;
                if (k < problem.alphabet_size - 1) ss << "\t\t";
            }
            ss << "\n";
            for (int k = 0; k < problem.alphabet_size; ++k) {
                ss << problem.emission_matrix[i][k];
                if (k < problem.alphabet_size - 1) ss << "\t";
            }
            ss << "\n";
            ss << "#############################################\n";
        }
        
        ss << "//END\n";
        
        return ss.str();
    }
    
    string createSequenceString(const vector<unsigned int>& sequence) {
        stringstream ss;
        ss << ">test_sequence\n";
        for (size_t i = 0; i < sequence.size(); ++i) {
            ss << sequence[i];
        }
        ss << "\n";
        return ss.str();
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
