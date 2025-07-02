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
#include <fstream>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>
#include <climits>

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
            // Emissions: 0, 1, 2, 3, 4, 5 (dice faces, 0-indexed)
            {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6},  // Fair die
            {0.10, 0.10, 0.10, 0.10, 0.10, 0.50}          // Loaded die (5 is more likely)
        };
        int num_states = 2;
        int alphabet_size = 6;
        string name = "Dishonest Casino";
        
        vector<unsigned int> generateSequence(int length) {
            uniform_int_distribution<unsigned int> dist(0, alphabet_size - 1);  // 0-indexed
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
            uniform_int_distribution<unsigned int> dist(0, alphabet_size - 1);  // 0-indexed
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
            // Emissions: A, C, G, T (0, 1, 2, 3)
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
            
            // Convert observation sequence to libhmm format (already 0-indexed)
            libhmm::ObservationSet libhmm_obs(obs_sequence.size());
            for (size_t i = 0; i < obs_sequence.size(); ++i) {
                libhmm_obs(i) = static_cast<libhmm::Observation>(obs_sequence[i]);
            }
            
            // Use Forward-Backward calculator with auto selection
            auto start = high_resolution_clock::now();
            libhmm::forwardbackward::AutoCalculator fb_calc(hmm.get(), libhmm_obs, true); // require stability
            double forward_backward_log_likelihood = fb_calc.getLogProbability();
            auto end = high_resolution_clock::now();
            results.forward_time = duration_cast<microseconds>(end - start).count() / 1000.0;
            results.likelihood = forward_backward_log_likelihood;
            
            // Use Viterbi calculator with auto selection
            start = high_resolution_clock::now();
            libhmm::viterbi::AutoCalculator viterbi_calc(hmm.get(), libhmm_obs, true); // require stability
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

class JAHMMBenchmark {
private:
    string jahmm_path;
    string temp_dir;
    
    // Write HMM configuration to Java properties file
    template<typename ProblemType>
    void writeHMMConfig(const ProblemType& problem, const string& config_file) {
        ofstream out(config_file);
        
        // Write number of states and alphabet size
        out << "num_states=" << problem.num_states << "\n";
        out << "alphabet_size=" << problem.alphabet_size << "\n";
        
        // Write initial probabilities
        out << "initial_probs=";
        for (int i = 0; i < problem.num_states; ++i) {
            if (i > 0) out << ",";
            out << problem.initial_probs[i];
        }
        out << "\n";
        
        // Write transition matrix
        for (int i = 0; i < problem.num_states; ++i) {
            out << "transition_" << i << "=";
            for (int j = 0; j < problem.num_states; ++j) {
                if (j > 0) out << ",";
                out << problem.transition_matrix[i][j];
            }
            out << "\n";
        }
        
        // Write emission matrix
        for (int i = 0; i < problem.num_states; ++i) {
            out << "emission_" << i << "=";
            for (int j = 0; j < problem.alphabet_size; ++j) {
                if (j > 0) out << ",";
                out << problem.emission_matrix[i][j];
            }
            out << "\n";
        }
        
        out.close();
    }
    
    // Write observation sequence to file
    void writeObservationSequence(const vector<unsigned int>& sequence, const string& obs_file) {
        ofstream out(obs_file);
        for (size_t i = 0; i < sequence.size(); ++i) {
            if (i > 0) out << " ";
            out << sequence[i];
        }
        out << "\n";
        out.close();
    }
    
    // Execute Java process and measure timing
    pair<double, string> executeJavaCommand(const string& command) {
        auto start = high_resolution_clock::now();
        
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            return {-1.0, "Failed to execute command"};
        }
        
        string result;
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        
        int return_code = pclose(pipe);
        auto end = high_resolution_clock::now();
        
        double execution_time = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        if (return_code != 0) {
            return {-1.0, "Command failed with return code " + to_string(return_code)};
        }
        
        return {execution_time, result};
    }
    
public:
    JAHMMBenchmark(const string& jahmm_jar_path) : jahmm_path(jahmm_jar_path) {
        temp_dir = "/tmp/jahmm_benchmark_" + to_string(getpid());
        system(("mkdir -p " + temp_dir).c_str());
    }
    
    ~JAHMMBenchmark() {
        system(("rm -rf " + temp_dir).c_str());
    }
    
    template<typename ProblemType>
    BenchmarkResults runBenchmark(ProblemType& problem, const vector<unsigned int>& full_obs_sequence, int sequence_length) {
        BenchmarkResults results;
        results.library_name = "JAHMM";
        results.problem_name = problem.name;
        results.sequence_length = sequence_length;
        results.success = false;
        
        // Use the first 'sequence_length' observations from the shared sequence
        vector<unsigned int> obs_sequence(full_obs_sequence.begin(), 
                                         full_obs_sequence.begin() + sequence_length);
        
        try {
            // Write configuration files
            string config_file = temp_dir + "/hmm_config.properties";
            string obs_file = temp_dir + "/observations.txt";
            string result_file = temp_dir + "/results.txt";
            
            writeHMMConfig(problem, config_file);
            writeObservationSequence(obs_sequence, obs_file);
            
            // Create Java benchmark runner command
            string java_command = "cd " + temp_dir + " && java -cp " + jahmm_path + " JAHMMBenchmarkRunner " 
                                + config_file + " " + obs_file + " " + result_file;
            
            // Execute the Java benchmark
            auto [exec_time, output] = executeJavaCommand(java_command);
            
            if (exec_time < 0) {
                cout << "JAHMM execution failed: " << output << endl;
                return results;
            }
            
            // Parse results from the output file
            ifstream result_stream(result_file);
            if (result_stream.is_open()) {
                string line;
                while (getline(result_stream, line)) {
                    if (line.find("FORWARD_TIME:") == 0) {
                        results.forward_time = stod(line.substr(13));
                    } else if (line.find("VITERBI_TIME:") == 0) {
                        results.viterbi_time = stod(line.substr(13));
                    } else if (line.find("LIKELIHOOD:") == 0) {
                        results.likelihood = stod(line.substr(11));
                    }
                }
                result_stream.close();
                results.success = true;
            } else {
                cout << "Failed to read JAHMM results file" << endl;
            }
            
        } catch (const exception& e) {
            cout << "JAHMM Error: " << e.what() << endl;
            results.forward_time = -1;
            results.viterbi_time = -1;
            results.likelihood = 0;
        }
        
        return results;
    }
};

// Java benchmark runner that we'll create alongside this C++ file
void createJavaRunner() {
    string java_code = R"(
import java.io.*;
import java.util.*;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

import be.ac.ulg.montefiore.run.jahmm.*;
import be.ac.ulg.montefiore.run.jahmm.learn.*;

public class JAHMMBenchmarkRunner {
    
    // Enum to represent discrete observations
    public enum ObsSymbol {
        SYM_0, SYM_1, SYM_2, SYM_3, SYM_4, SYM_5, SYM_6, SYM_7;
        
        public static ObsSymbol fromInteger(int value) {
            switch(value) {
                case 0: return SYM_0;
                case 1: return SYM_1;
                case 2: return SYM_2;
                case 3: return SYM_3;
                case 4: return SYM_4;
                case 5: return SYM_5;
                case 6: return SYM_6;
                case 7: return SYM_7;
                default: throw new IllegalArgumentException("Invalid observation value: " + value);
            }
        }
    }
    
    private static class HMMConfig {
        int numStates;
        int alphabetSize;
        double[] initialProbs;
        double[][] transitionMatrix;
        double[][] emissionMatrix;
        
        public static HMMConfig loadFromFile(String configFile) throws IOException {
            Properties props = new Properties();
            props.load(new FileInputStream(configFile));
            
            HMMConfig config = new HMMConfig();
            config.numStates = Integer.parseInt(props.getProperty("num_states"));
            config.alphabetSize = Integer.parseInt(props.getProperty("alphabet_size"));
            
            // Parse initial probabilities
            String[] initialProbsStr = props.getProperty("initial_probs").split(",");
            config.initialProbs = new double[config.numStates];
            for (int i = 0; i < config.numStates; i++) {
                config.initialProbs[i] = Double.parseDouble(initialProbsStr[i]);
            }
            
            // Parse transition matrix
            config.transitionMatrix = new double[config.numStates][config.numStates];
            for (int i = 0; i < config.numStates; i++) {
                String[] transitionStr = props.getProperty("transition_" + i).split(",");
                for (int j = 0; j < config.numStates; j++) {
                    config.transitionMatrix[i][j] = Double.parseDouble(transitionStr[j]);
                }
            }
            
            // Parse emission matrix
            config.emissionMatrix = new double[config.numStates][config.alphabetSize];
            for (int i = 0; i < config.numStates; i++) {
                String[] emissionStr = props.getProperty("emission_" + i).split(",");
                for (int j = 0; j < config.alphabetSize; j++) {
                    config.emissionMatrix[i][j] = Double.parseDouble(emissionStr[j]);
                }
            }
            
            return config;
        }
    }
    
    private static class TimingResult {
        double forwardTime;
        double viterbiTime;
        double likelihood;
    }
    
    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: JAHMMBenchmarkRunner <config_file> <obs_file> <result_file>");
            System.exit(1);
        }
        
        try {
            String configFile = args[0];
            String obsFile = args[1];
            String resultFile = args[2];
            
            // Load configuration
            HMMConfig config = HMMConfig.loadFromFile(configFile);
            
            // Load observations
            List<Integer> observations = new ArrayList<>();
            Scanner scanner = new Scanner(new File(obsFile));
            while (scanner.hasNextInt()) {
                observations.add(scanner.nextInt());
            }
            scanner.close();
            
            // Build HMM
            Hmm<ObservationDiscrete<ObsSymbol>> hmm = buildHMM(config);
            
            // Convert observations to JAHMM format
            List<ObservationDiscrete<ObsSymbol>> observationSequence = new ArrayList<>();
            for (Integer obs : observations) {
                observationSequence.add(new ObservationDiscrete<ObsSymbol>(ObsSymbol.fromInteger(obs)));
            }
            
            // Run benchmarks
            TimingResult result = runBenchmarks(hmm, observationSequence);
            
            // Write results
            PrintWriter writer = new PrintWriter(new FileWriter(resultFile));
            writer.println("FORWARD_TIME:" + result.forwardTime);
            writer.println("VITERBI_TIME:" + result.viterbiTime);
            writer.println("LIKELIHOOD:" + result.likelihood);
            writer.close();
            
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static Hmm<ObservationDiscrete<ObsSymbol>> buildHMM(HMMConfig config) {
        // Create HMM with discrete observations
        Hmm<ObservationDiscrete<ObsSymbol>> hmm = 
            new Hmm<ObservationDiscrete<ObsSymbol>>(config.numStates, 
                new OpdfDiscreteFactory<ObsSymbol>(ObsSymbol.class));
        
        // Set initial probabilities
        for (int i = 0; i < config.numStates; i++) {
            hmm.setPi(i, config.initialProbs[i]);
        }
        
        // Set transition probabilities
        for (int i = 0; i < config.numStates; i++) {
            for (int j = 0; j < config.numStates; j++) {
                hmm.setAij(i, j, config.transitionMatrix[i][j]);
            }
        }
        
        // Set emission probabilities
        for (int i = 0; i < config.numStates; i++) {
            // Create emission probability array padded to enum size (8)
            double[] paddedEmissions = new double[8];
            System.arraycopy(config.emissionMatrix[i], 0, paddedEmissions, 0, config.alphabetSize);
            // Fill remaining with zeros (they won't be used anyway)
            for (int j = config.alphabetSize; j < 8; j++) {
                paddedEmissions[j] = 0.0;
            }
            OpdfDiscrete<ObsSymbol> opdf = new OpdfDiscrete<ObsSymbol>(ObsSymbol.class, paddedEmissions);
            hmm.setOpdf(i, opdf);
        }
        
        return hmm;
    }
    
    private static TimingResult runBenchmarks(Hmm<ObservationDiscrete<ObsSymbol>> hmm, 
                                            List<ObservationDiscrete<ObsSymbol>> observations) {
        TimingResult result = new TimingResult();
        
        // JVM warmup - run a few iterations to warm up the JIT compiler
        for (int warmup = 0; warmup < 3; warmup++) {
            hmm.probability(observations);
            hmm.mostLikelyStateSequence(observations);
        }
        
        // Benchmark Forward-Backward algorithm
        long startTime = System.nanoTime();
        double probability = hmm.probability(observations);
        // Handle very small probabilities that might cause underflow
        if (probability <= 0.0 || Double.isNaN(probability) || Double.isInfinite(probability)) {
            result.likelihood = Double.NEGATIVE_INFINITY;
        } else {
            result.likelihood = Math.log(probability); // Convert to log probability like libhmm
        }
        long endTime = System.nanoTime();
        result.forwardTime = (endTime - startTime) / 1000000.0; // Convert to milliseconds
        
        // Benchmark Viterbi algorithm
        startTime = System.nanoTime();
        int[] viterbiPath = hmm.mostLikelyStateSequence(observations);
        endTime = System.nanoTime();
        result.viterbiTime = (endTime - startTime) / 1000000.0; // Convert to milliseconds
        
        return result;
    }
}
)";

    // Write Java file to the JAHMM directory (using relative path from benchmarks directory)
    string java_file = "../JAHMM/source/jahmm-0.6.2/src/JAHMMBenchmarkRunner.java";
    ofstream out(java_file);
    out << java_code;
    out.close();
    
    // Compile the Java file (using relative path with portable Java detection)
    string compile_cmd = "cd ../JAHMM/source/jahmm-0.6.2 && "
                        "(export PATH=/opt/homebrew/opt/openjdk/bin:$PATH 2>/dev/null || "
                        "export PATH=/usr/local/opt/openjdk/bin:$PATH 2>/dev/null || true) && "
                        "javac -cp build src/JAHMMBenchmarkRunner.java -d build";
    
    int result = system(compile_cmd.c_str());
    if (result != 0) {
        cout << "Failed to compile Java benchmark runner" << endl;
        exit(1);
    }
}

void displayResults(const vector<BenchmarkResults>& all_results) {
    cout << "\n\n=== LIBHMM vs JAHMM BENCHMARK RESULTS ===\n" << endl;
    
    cout << setw(20) << "Library" 
         << setw(25) << "Problem" 
         << setw(10) << "Length" 
         << setw(15) << "Forward(ms)" 
         << setw(15) << "Viterbi(ms)"
         << setw(18) << "Log Likelihood" << endl;
    cout << string(103, '-') << endl;
    
    map<string, map<int, pair<BenchmarkResults, BenchmarkResults>>> comparison_data;
    
    for (const auto& result : all_results) {
        if (result.success) {
            cout << setw(20) << result.library_name
                 << setw(25) << result.problem_name
                 << setw(10) << result.sequence_length
                 << setw(15) << fixed << setprecision(3) << result.forward_time
                 << setw(15) << fixed << setprecision(3) << result.viterbi_time
                 << setw(18) << fixed << setprecision(6) << result.likelihood << endl;
            
            // Store for comparison
            string key = result.problem_name;
            if (result.library_name == "libhmm") {
                comparison_data[key][result.sequence_length].first = result;
            } else {
                comparison_data[key][result.sequence_length].second = result;
            }
        } else {
            cout << setw(20) << result.library_name
                 << setw(25) << result.problem_name
                 << setw(10) << result.sequence_length
                 << setw(15) << "FAILED"
                 << setw(15) << "FAILED"
                 << setw(18) << "FAILED" << endl;
        }
    }
    
    // Performance comparison analysis
    cout << "\n\n=== PERFORMANCE COMPARISON ANALYSIS ===\n" << endl;
    cout << setw(25) << "Problem" 
         << setw(10) << "Length" 
         << setw(20) << "Forward Speedup" 
         << setw(20) << "Viterbi Speedup"
         << setw(25) << "Likelihood Difference" << endl;
    cout << string(100, '-') << endl;
    
    for (const auto& problem_data : comparison_data) {
        for (const auto& length_data : problem_data.second) {
            const auto& libhmm_result = length_data.second.first;
            const auto& jahmm_result = length_data.second.second;
            
            if (libhmm_result.success && jahmm_result.success) {
                double forward_speedup = jahmm_result.forward_time / libhmm_result.forward_time;
                double viterbi_speedup = jahmm_result.viterbi_time / libhmm_result.viterbi_time;
                double likelihood_diff = abs(libhmm_result.likelihood - jahmm_result.likelihood);
                
                cout << setw(25) << problem_data.first
                     << setw(10) << length_data.first
                     << setw(20) << fixed << setprecision(2) << forward_speedup << "x"
                     << setw(20) << fixed << setprecision(2) << viterbi_speedup << "x"
                     << setw(25) << scientific << setprecision(3) << likelihood_diff << endl;
            }
        }
    }
    
    cout << "\nNote: Speedup > 1.0 means libhmm is faster than JAHMM" << endl;
    cout << "Note: Likelihood differences should be minimal for correct implementations" << endl;
}

int main() {
    cout << "=== LibHMM vs JAHMM Discrete HMM Benchmark ===" << endl;
    cout << "Comparing performance of libhmm (C++) vs JAHMM (Java)" << endl;
    cout << "Testing with classic HMM problems and various sequence lengths" << endl << endl;
    
    // Create Java benchmark runner
    cout << "Creating and compiling Java benchmark runner..." << endl;
    createJavaRunner();
    
    // Set up JAHMM path (construct absolute path)
    char* cwd = getcwd(NULL, 0);
    string current_dir(cwd);
    free(cwd);
    string jahmm_classpath = current_dir + "/../JAHMM/source/jahmm-0.6.2/build";
    
    // Create benchmark instances
    LibHMMBenchmark libhmm_bench;
    JAHMMBenchmark jahmm_bench(jahmm_classpath);
    
    // Define test problems
    ClassicHMMProblems::CasinoProblem casino_problem;
    ClassicHMMProblems::WeatherProblem weather_problem;
    ClassicHMMProblems::CpGProblem cpg_problem;
    
    // Test sequence lengths
    vector<int> sequence_lengths = {100, 500, 1000, 2000, 5000};
    
    vector<BenchmarkResults> all_results;
    
    // Generate shared sequences for each problem to ensure fair comparison
    map<string, vector<unsigned int>> shared_sequences;
    
    cout << "Generating test sequences..." << endl;
    shared_sequences["Dishonest Casino"] = casino_problem.generateSequence(5000);
    shared_sequences["Weather Model"] = weather_problem.generateSequence(5000);
    shared_sequences["CpG Island Detection"] = cpg_problem.generateSequence(5000);
    
    // Run benchmarks for each problem and sequence length
    vector<pair<string, void*>> test_problems = {
        {"Dishonest Casino", &casino_problem},
        {"Weather Model", &weather_problem},
        {"CpG Island Detection", &cpg_problem}
    };
    
    for (auto& [problem_name, problem_ptr] : test_problems) {
        cout << "\n=== Testing " << problem_name << " ===" << endl;
        
        for (int length : sequence_lengths) {
            cout << "Testing sequence length: " << length << endl;
            
            // Test libhmm
            BenchmarkResults libhmm_result;
            if (problem_name == "Dishonest Casino") {
                libhmm_result = libhmm_bench.runBenchmark(
                    *static_cast<ClassicHMMProblems::CasinoProblem*>(problem_ptr),
                    shared_sequences[problem_name], length);
            } else if (problem_name == "Weather Model") {
                libhmm_result = libhmm_bench.runBenchmark(
                    *static_cast<ClassicHMMProblems::WeatherProblem*>(problem_ptr),
                    shared_sequences[problem_name], length);
            } else if (problem_name == "CpG Island Detection") {
                libhmm_result = libhmm_bench.runBenchmark(
                    *static_cast<ClassicHMMProblems::CpGProblem*>(problem_ptr),
                    shared_sequences[problem_name], length);
            }
            all_results.push_back(libhmm_result);
            
            // Test JAHMM
            BenchmarkResults jahmm_result;
            if (problem_name == "Dishonest Casino") {
                jahmm_result = jahmm_bench.runBenchmark(
                    *static_cast<ClassicHMMProblems::CasinoProblem*>(problem_ptr),
                    shared_sequences[problem_name], length);
            } else if (problem_name == "Weather Model") {
                jahmm_result = jahmm_bench.runBenchmark(
                    *static_cast<ClassicHMMProblems::WeatherProblem*>(problem_ptr),
                    shared_sequences[problem_name], length);
            } else if (problem_name == "CpG Island Detection") {
                jahmm_result = jahmm_bench.runBenchmark(
                    *static_cast<ClassicHMMProblems::CpGProblem*>(problem_ptr),
                    shared_sequences[problem_name], length);
            }
            all_results.push_back(jahmm_result);
            
            cout << "  libhmm: Forward=" << libhmm_result.forward_time << "ms, "
                 << "Viterbi=" << libhmm_result.viterbi_time << "ms" << endl;
            cout << "  JAHMM:  Forward=" << jahmm_result.forward_time << "ms, "
                 << "Viterbi=" << jahmm_result.viterbi_time << "ms" << endl;
        }
    }
    
    // Display comprehensive results
    displayResults(all_results);
    
    return 0;
}
