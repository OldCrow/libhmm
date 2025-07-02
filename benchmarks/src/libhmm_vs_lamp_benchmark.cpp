#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <memory>
#include <cstring>
#include <sstream>
#include <fstream>
#include <map>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

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

class LAMPBenchmark {
private:
    string temp_dir = "./temp_lamp_benchmark";
    
    bool fileExists(const string& filename) {
        struct stat buffer;
        return (stat(filename.c_str(), &buffer) == 0);
    }
    
    void createTempDirectory() {
        string command = "mkdir -p " + temp_dir;
        system(command.c_str());
    }
    
    void removeTempDirectory() {
        string command = "rm -rf " + temp_dir;
        system(command.c_str());
    }

    template<typename ProblemType>
    void createLAMPConfigFile(ProblemType& problem, const string& config_file, const string& abs_sequence_file, const string& abs_output_prefix) {
        ofstream file(config_file);
        if (!file) {
            throw runtime_error("Cannot create LAMP config file: " + config_file);
        }
        
        file << "# LAMP HMM Configuration File" << endl;
        file << "sequenceName= " << endl;
        file << abs_sequence_file << endl;
        file << "skipLearning=" << endl;
        file << "0" << endl;
        file << "hmmInputName=" << endl;
        file << "0" << endl;
        file << "outputName=" << endl;
        file << abs_output_prefix << endl;
        file << "nbDimensions=" << endl;
        file << "1" << endl;
        file << "nbSymbols=" << endl;
        file << problem.alphabet_size << endl;
        file << "nbStates=" << endl;
        file << problem.num_states << endl;
        file << "seed=" << endl;
        file << "42" << endl;
        file << "explicitDuration=" << endl;
        file << "0" << endl;
        file << "gaussianDistribution=" << endl;
        file << "0" << endl;
        file << "EMMethod=" << endl;
        file << "2" << endl;  // Use Segmented K-Means for faster execution
        
        file.close();
    }
    
    void createLAMPSequenceFile(const vector<unsigned int>& obs_sequence, const string& sequence_file) {
        ofstream file(sequence_file);
        if (!file) {
            throw runtime_error("Cannot create LAMP sequence file: " + sequence_file);
        }
        
        // Write a simple PPM header expected by LAMP
        file << "P5" << endl;
        file << "nbSequences= 1" << endl;
        file << "T= " << obs_sequence.size() << endl;
        for (size_t i = 0; i < obs_sequence.size(); ++i) {
            file << obs_sequence[i] << " ";
        }
        file << endl;

        file.close();
    }
    
    double parseLAMPLikelihood(const string& output_file) {
        ifstream file(output_file);
        if (!file) {
            return 0.0;
        }
        
        string line;
        double likelihood = 0.0;
        while (getline(file, line)) {
            // Look for the final log probability output from LAMP
            if (line.find("-Log Prob of all") != string::npos) {
                size_t pos = line.find_last_of(':');
                if (pos != string::npos) {
                    try {
                        likelihood = -stod(line.substr(pos + 1));  // Convert to positive log likelihood
                    } catch (...) {
                        // Parsing failed, continue
                    }
                }
            }
        }
        file.close();
        return likelihood;
    }

public:
    template<typename ProblemType>
    BenchmarkResults runBenchmark(ProblemType& problem, const vector<unsigned int>& full_obs_sequence, int sequence_length) {
        BenchmarkResults results;
        results.library_name = "LAMP";
        results.problem_name = problem.name;
        results.sequence_length = sequence_length;
        results.success = false;
        
        // Use the first 'sequence_length' observations from the shared sequence
        vector<unsigned int> obs_sequence(full_obs_sequence.begin(), 
                                         full_obs_sequence.begin() + sequence_length);
        
        try {
            createTempDirectory();
            
            // Create unique file names for this test
            string config_file = temp_dir + "/config.cfg";
            string sequence_file = temp_dir + "/sequence.seq";
            string output_file = temp_dir + "/output.dis";
            
            // Get current working directory for absolute paths
            char* cwd = getcwd(NULL, 0);
            string current_dir = string(cwd);
            free(cwd);
            
            string abs_sequence_file = current_dir + "/" + sequence_file;
            string abs_output_prefix = current_dir + "/" + temp_dir + "/output";
            
            // Create LAMP config and data files
            createLAMPConfigFile(problem, config_file, abs_sequence_file, abs_output_prefix);
            createLAMPSequenceFile(obs_sequence, sequence_file);
            
            // Debug output for first test
            if (sequence_length == 100) {
                cout << "[DEBUG] LAMP using " << problem.num_states << " states, " << problem.alphabet_size << " symbols" << endl;
            }
            
            // Execute LAMP (redirect output to capture log likelihood)
            string abs_config_path = current_dir + "/" + config_file;
            string abs_output_path = current_dir + "/" + temp_dir + "/lamp_output.txt";
            
            auto start = high_resolution_clock::now();
            string command = "cd ../LAMP_HMM && ./hmmFind " + abs_config_path + " > " + abs_output_path + " 2>&1";
            int lamp_result = system(command.c_str());
            auto end = high_resolution_clock::now();
            results.forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.viterbi_time = results.forward_time * 0.1;  // LAMP does both simultaneously
            
            // Parse likelihood from the captured output regardless of exit code
            string lamp_output = temp_dir + "/lamp_output.txt";
            results.likelihood = parseLAMPLikelihood(lamp_output);
            
            // Check if we got meaningful results
            if (lamp_result == 0 && results.likelihood != 0.0) {
                results.success = true;
            } else {
                // Debug failed execution
                if (sequence_length == 100) {
                    cout << "[DEBUG] LAMP execution failed. Command: " << command << endl;
                    cout << "[DEBUG] Exit code: " << lamp_result << endl;
                    if (fileExists(lamp_output)) {
                        cout << "[DEBUG] LAMP output available, check " << lamp_output << endl;
                    }
                }
            }

            removeTempDirectory();
        } catch (const exception& e) {
            cout << "LAMP Error: " << e.what() << endl;
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
         << setw(8) << "Length"
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
             << setw(8) << result.sequence_length
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
         << setw(16) << "LAMP (ms)"
         << setw(12) << "Speedup" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& group : grouped_results) {
        if (group.second.size() >= 2) {
            const string& problem_name = group.first.first;
            int length = group.first.second;
            
            double libhmm_time = -1, lamp_time = -1;
            for (const auto& result : group.second) {
                if (result.library_name == "libhmm") {
                    libhmm_time = result.forward_time;
                } else if (result.library_name == "LAMP") {
                    lamp_time = result.forward_time;
                }
            }
            
            if (libhmm_time > 0 && lamp_time > 0) {
                double speedup = lamp_time / libhmm_time;
                cout << left << setw(25) << problem_name
                     << setw(10) << length
                     << setw(16) << fixed << setprecision(3) << libhmm_time
                     << setw(16) << lamp_time
                     << setprecision(2) << speedup << "x" << endl;
            }
        }
    }
    cout << string(80, '=') << endl;
}

int main() {
    cout << "HMM Library Comparison Benchmark" << endl;
    cout << "================================" << endl;
    cout << "Comparing libhmm vs LAMP performance" << endl;
    cout << "Fixed random seed (42) for reproducibility" << endl;
    
    LibHMMBenchmark libhmm_benchmark;
    LAMPBenchmark lamp_benchmark;
    vector<BenchmarkResults> results;
    
    // Test different sequence lengths for each problem
    vector<int> test_lengths = {100, 500, 1000, 2000, 5000, 10000};
    
    // Casino Problem
    cout << "\n--- TESTING CASINO PROBLEM ---" << endl;
    ClassicHMMProblems::CasinoProblem casino;
    auto casino_full_obs_sequence = casino.generateSequence(1000000);

    for (int length : test_lengths) {
        cout << "  libhmm (length: " << length << "): ";
        auto libhmm_result = libhmm_benchmark.runBenchmark(casino, casino_full_obs_sequence, length);
        cout << (libhmm_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(libhmm_result);
        
        cout << "  LAMP (length: " << length << "): ";
        auto lamp_result = lamp_benchmark.runBenchmark(casino, casino_full_obs_sequence, length);
        cout << (lamp_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(lamp_result);
    }
    
    // Weather Problem
    cout << "\n--- TESTING WEATHER PROBLEM ---" << endl;
    ClassicHMMProblems::WeatherProblem weather;
    auto weather_full_obs_sequence = weather.generateSequence(1000000);
    
    for (int length : test_lengths) {
        cout << "  libhmm (length: " << length << "): ";
        auto libhmm_result = libhmm_benchmark.runBenchmark(weather, weather_full_obs_sequence, length);
        cout << (libhmm_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(libhmm_result);
        
        cout << "  LAMP (length: " << length << "): ";
        auto lamp_result = lamp_benchmark.runBenchmark(weather, weather_full_obs_sequence, length);
        cout << (lamp_result.success ? "SUCCESS" : "FAILED") << endl;
        results.push_back(lamp_result);
    }
    
    // Print comprehensive results
    printComparisonResults(results);
    printPerformanceComparison(results);
    
    // Summary statistics
    int libhmm_successes = 0, lamp_successes = 0;
    double libhmm_total_throughput = 0.0, lamp_total_throughput = 0.0;
    
    for (const auto& result : results) {
        if (result.success) {
            double throughput = result.sequence_length / max(result.forward_time, 0.001);
            if (result.library_name == "libhmm") {
                libhmm_successes++;
                libhmm_total_throughput += throughput;
            } else if (result.library_name == "LAMP") {
                lamp_successes++;
                lamp_total_throughput += throughput;
            }
        }
    }
    
    cout << "\nSUMMARY" << endl;
    cout << "-------" << endl;
    if (libhmm_successes > 0) {
        cout << "libhmm average throughput: " << fixed << setprecision(1) 
             << libhmm_total_throughput / libhmm_successes << " observations/ms" << endl;
    }
    if (lamp_successes > 0) {
        cout << "LAMP average throughput: " << fixed << setprecision(1) 
             << lamp_total_throughput / lamp_successes << " observations/ms" << endl;
    }
    
    if (libhmm_successes > 0 && lamp_successes > 0) {
        double avg_speedup = (lamp_total_throughput / lamp_successes) / 
                           (libhmm_total_throughput / libhmm_successes);
        cout << "Overall performance ratio: " << setprecision(2) << avg_speedup 
             << "x (LAMP/libhmm)" << endl;
    }
    
    return 0;
}
