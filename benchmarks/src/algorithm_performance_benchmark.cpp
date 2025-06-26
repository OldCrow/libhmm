#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <random>
#include <iomanip>
#include <fstream>

// libhmm includes
#include "libhmm/libhmm.h"
#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/calculators/viterbi_traits.h"

// HMMLib includes 
#include "HMMlib/hmm.hpp"
using namespace hmmlib;

using namespace std::chrono;

class BenchmarkTimer {
private:
    high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

struct BenchmarkResult {
    std::string test_name;
    std::string library;
    double time_ms;
    double likelihood;
    size_t num_states;
    size_t seq_length;
    size_t num_sequences;
    
    void print() const {
        std::cout << std::setw(25) << test_name 
                  << std::setw(12) << library
                  << std::setw(12) << std::fixed << std::setprecision(3) << time_ms << "ms"
                  << std::setw(15) << std::scientific << std::setprecision(3) << likelihood
                  << std::setw(8) << num_states
                  << std::setw(12) << seq_length
                  << std::setw(12) << num_sequences << std::endl;
    }
};

class HMMBenchmarkSuite {
private:
    std::vector<BenchmarkResult> results;
    std::mt19937 gen;
    
public:
    HMMBenchmarkSuite() : gen(42) {} // Fixed seed for reproducibility
    
    // Generate test observation sequences
    std::vector<std::vector<unsigned int>> generateObservationSequences(
        size_t num_sequences, size_t seq_length, size_t alphabet_size) {
        
        std::uniform_int_distribution<unsigned int> dist(0, alphabet_size - 1);
        std::vector<std::vector<unsigned int>> sequences;
        
        for (size_t i = 0; i < num_sequences; ++i) {
            std::vector<unsigned int> seq(seq_length);
            for (size_t j = 0; j < seq_length; ++j) {
                seq[j] = dist(gen);
            }
            sequences.push_back(seq);
        }
        return sequences;
    }
    
    // Benchmark Forward-Backward algorithm with libhmm
    BenchmarkResult benchmarkLibhmmForwardBackward(
        size_t num_states, size_t alphabet_size, size_t seq_length) {
        
        BenchmarkTimer timer;
        BenchmarkResult result;
        result.test_name = "Forward-Backward";
        result.library = "libhmm";
        result.num_states = num_states;
        result.seq_length = seq_length;
        result.num_sequences = 1;
        
        // Create HMM
        auto hmm = std::make_unique<libhmm::Hmm>(num_states);
        
        // Initialize with random but valid probabilities
        libhmm::Vector pi(num_states);
        libhmm::Matrix trans(num_states, num_states);
        
        // Random initial probabilities (normalized)
        double sum = 0.0;
        for (size_t i = 0; i < num_states; ++i) {
            pi(i) = std::uniform_real_distribution<double>(0.1, 1.0)(gen);
            sum += pi(i);
        }
        for (size_t i = 0; i < num_states; ++i) {
            pi(i) /= sum;
        }
        
        // Random transition matrix (row-normalized)
        for (size_t i = 0; i < num_states; ++i) {
            sum = 0.0;
            for (size_t j = 0; j < num_states; ++j) {
                trans(i, j) = std::uniform_real_distribution<double>(0.1, 1.0)(gen);
                sum += trans(i, j);
            }
            for (size_t j = 0; j < num_states; ++j) {
                trans(i, j) /= sum;
            }
        }
        
        hmm->setPi(pi);
        hmm->setTrans(trans);
        
        // Set discrete emission distributions
        for (size_t i = 0; i < num_states; ++i) {
            auto dist = std::make_unique<libhmm::DiscreteDistribution>(alphabet_size);
            // Random emission probabilities (normalized)
            libhmm::Vector probs(alphabet_size);
            sum = 0.0;
            for (size_t j = 0; j < alphabet_size; ++j) {
                probs(j) = std::uniform_real_distribution<double>(0.1, 1.0)(gen);
                sum += probs(j);
            }
            for (size_t j = 0; j < alphabet_size; ++j) {
                probs(j) /= sum;
            }
            // Set individual probabilities
            for (size_t j = 0; j < alphabet_size; ++j) {
                dist->setProbability(j, probs(j));
            }
            hmm->setProbabilityDistribution(i, std::move(dist));
        }
        
        // Generate test sequence
        auto sequences = generateObservationSequences(1, seq_length, alphabet_size);
        libhmm::ObservationSet obs(seq_length);
        for (size_t i = 0; i < seq_length; ++i) {
            obs(i) = sequences[0][i];
        }
        
        // Benchmark forward-backward with automatic calculator selection
        timer.start();
        libhmm::forwardbackward::AutoCalculator calc(hmm.get(), obs);
        result.likelihood = calc.probability();
        result.time_ms = timer.stop();
        
        // Debug: print which calculator was selected
        std::cout << "  [DEBUG] Selected calculator: " << calc.getSelectionRationale() << std::endl;
        
        return result;
    }
    
    // Benchmark Forward-Backward algorithm with HMMLib
    BenchmarkResult benchmarkHMMLibForwardBackward(
        size_t num_states, size_t alphabet_size, size_t seq_length) {
        
        BenchmarkTimer timer;
        BenchmarkResult result;
        result.test_name = "Forward-Backward";
        result.library = "HMMLib";
        result.num_states = num_states;
        result.seq_length = seq_length;
        result.num_sequences = 1;
        
        using float_type = double;
        using sse_float_type = double;
        
        // Create HMM components
        auto pi_ptr = boost::shared_ptr<hmmlib::HMMVector<float_type, sse_float_type>>(
            new hmmlib::HMMVector<float_type, sse_float_type>(num_states));
        auto T_ptr = boost::shared_ptr<hmmlib::HMMMatrix<float_type, sse_float_type>>(
            new hmmlib::HMMMatrix<float_type, sse_float_type>(num_states, num_states));
        auto E_ptr = boost::shared_ptr<hmmlib::HMMMatrix<float_type, sse_float_type>>(
            new hmmlib::HMMMatrix<float_type, sse_float_type>(alphabet_size, num_states));
        
        auto& pi = *pi_ptr;
        auto& T = *T_ptr;
        auto& E = *E_ptr;
        
        // Initialize with same random seed for fair comparison
        std::mt19937 local_gen(42);
        
        // Random initial probabilities (normalized)
        double sum = 0.0;
        for (size_t i = 0; i < num_states; ++i) {
            pi(i) = std::uniform_real_distribution<double>(0.1, 1.0)(local_gen);
            sum += pi(i);
        }
        for (size_t i = 0; i < num_states; ++i) {
            pi(i) /= sum;
        }
        
        // Random transition matrix (row-normalized)
        for (size_t i = 0; i < num_states; ++i) {
            sum = 0.0;
            for (size_t j = 0; j < num_states; ++j) {
                T(i, j) = std::uniform_real_distribution<double>(0.1, 1.0)(local_gen);
                sum += T(i, j);
            }
            for (size_t j = 0; j < num_states; ++j) {
                T(i, j) /= sum;
            }
        }
        
        // Random emission matrix (column-normalized)
        for (size_t j = 0; j < num_states; ++j) {
            sum = 0.0;
            for (size_t i = 0; i < alphabet_size; ++i) {
                E(i, j) = std::uniform_real_distribution<double>(0.1, 1.0)(local_gen);
                sum += E(i, j);
            }
            for (size_t i = 0; i < alphabet_size; ++i) {
                E(i, j) /= sum;
            }
        }
        
        // Create HMM
        hmmlib::HMM<float_type, sse_float_type> hmm(pi_ptr, T_ptr, E_ptr);
        
        // Generate test sequence (same as libhmm)
        auto sequences = generateObservationSequences(1, seq_length, alphabet_size);
        sequence obs(sequences[0].begin(), sequences[0].end());
        
        // Benchmark forward algorithm
        timer.start();
        hmmlib::HMMMatrix<float_type, sse_float_type> F(seq_length, num_states);
        hmmlib::HMMVector<float_type, sse_float_type> scales(seq_length);
        hmm.forward(obs, scales, F);
        result.likelihood = std::exp(hmm.likelihood(scales)); // Convert from log
        result.time_ms = timer.stop();
        
        return result;
    }
    
    // Benchmark Viterbi algorithm with libhmm (both scalar and SIMD)
    BenchmarkResult benchmarkLibhmmViterbi(
        size_t num_states, size_t alphabet_size, size_t seq_length, bool use_simd = false) {
        
        BenchmarkTimer timer;
        BenchmarkResult result;
        result.test_name = use_simd ? "Viterbi-SIMD" : "Viterbi-Scalar";
        result.library = "libhmm";
        result.num_states = num_states;
        result.seq_length = seq_length;
        result.num_sequences = 1;
        
        // Create HMM (similar to forward-backward setup)
        auto hmm = std::make_unique<libhmm::Hmm>(num_states);
        
        // Initialize with random but valid probabilities
        libhmm::Vector pi(num_states);
        libhmm::Matrix trans(num_states, num_states);
        
        // Random initial probabilities (normalized)
        double sum = 0.0;
        for (size_t i = 0; i < num_states; ++i) {
            pi(i) = std::uniform_real_distribution<double>(0.1, 1.0)(gen);
            sum += pi(i);
        }
        for (size_t i = 0; i < num_states; ++i) {
            pi(i) /= sum;
        }
        
        // Random transition matrix (row-normalized)
        for (size_t i = 0; i < num_states; ++i) {
            sum = 0.0;
            for (size_t j = 0; j < num_states; ++j) {
                trans(i, j) = std::uniform_real_distribution<double>(0.1, 1.0)(gen);
                sum += trans(i, j);
            }
            for (size_t j = 0; j < num_states; ++j) {
                trans(i, j) /= sum;
            }
        }
        
        hmm->setPi(pi);
        hmm->setTrans(trans);
        
        // Set discrete emission distributions
        for (size_t i = 0; i < num_states; ++i) {
            auto dist = std::make_unique<libhmm::DiscreteDistribution>(alphabet_size);
            libhmm::Vector probs(alphabet_size);
            sum = 0.0;
            for (size_t j = 0; j < alphabet_size; ++j) {
                probs(j) = std::uniform_real_distribution<double>(0.1, 1.0)(gen);
                sum += probs(j);
            }
            for (size_t j = 0; j < alphabet_size; ++j) {
                probs(j) /= sum;
            }
            // Set individual probabilities
            for (size_t j = 0; j < alphabet_size; ++j) {
                dist->setProbability(j, probs(j));
            }
            hmm->setProbabilityDistribution(i, std::move(dist));
        }
        
        // Generate test sequence
        auto sequences = generateObservationSequences(1, seq_length, alphabet_size);
        libhmm::ObservationSet obs(seq_length);
        for (size_t i = 0; i < seq_length; ++i) {
            obs(i) = sequences[0][i];
        }
        
        // Benchmark Viterbi with automatic calculator selection  
        timer.start();
        
        // Use AutoCalculator to get optimal Viterbi algorithm
        libhmm::viterbi::AutoCalculator viterbi(hmm.get(), obs);
        auto states = viterbi.decode();
        result.likelihood = viterbi.getLogProbability();
        
        result.time_ms = timer.stop();
        
        // Debug: print which calculator was actually used
        std::cout << "  [DEBUG] Selected Viterbi calculator: " << viterbi.getSelectionRationale() << std::endl;
        
        return result;
    }
    
    // Benchmark Viterbi algorithm with HMMLib
    BenchmarkResult benchmarkHMMLibViterbi(
        size_t num_states, size_t alphabet_size, size_t seq_length) {
        
        BenchmarkTimer timer;
        BenchmarkResult result;
        result.test_name = "Viterbi";
        result.library = "HMMLib";
        result.num_states = num_states;
        result.seq_length = seq_length;
        result.num_sequences = 1;
        
        using float_type = double;
        using sse_float_type = double;
        
        // Create HMM components (same setup as forward-backward)
        auto pi_ptr = boost::shared_ptr<hmmlib::HMMVector<float_type, sse_float_type>>(
            new hmmlib::HMMVector<float_type, sse_float_type>(num_states));
        auto T_ptr = boost::shared_ptr<hmmlib::HMMMatrix<float_type, sse_float_type>>(
            new hmmlib::HMMMatrix<float_type, sse_float_type>(num_states, num_states));
        auto E_ptr = boost::shared_ptr<hmmlib::HMMMatrix<float_type, sse_float_type>>(
            new hmmlib::HMMMatrix<float_type, sse_float_type>(alphabet_size, num_states));
        
        auto& pi = *pi_ptr;
        auto& T = *T_ptr;
        auto& E = *E_ptr;
        
        // Initialize with same random seed for fair comparison
        std::mt19937 local_gen(42);
        
        // Random initial probabilities (normalized)
        double sum = 0.0;
        for (size_t i = 0; i < num_states; ++i) {
            pi(i) = std::uniform_real_distribution<double>(0.1, 1.0)(local_gen);
            sum += pi(i);
        }
        for (size_t i = 0; i < num_states; ++i) {
            pi(i) /= sum;
        }
        
        // Random transition matrix (row-normalized)  
        for (size_t i = 0; i < num_states; ++i) {
            sum = 0.0;
            for (size_t j = 0; j < num_states; ++j) {
                T(i, j) = std::uniform_real_distribution<double>(0.1, 1.0)(local_gen);
                sum += T(i, j);
            }
            for (size_t j = 0; j < num_states; ++j) {
                T(i, j) /= sum;
            }
        }
        
        // Random emission matrix (column-normalized)
        for (size_t j = 0; j < num_states; ++j) {
            sum = 0.0;
            for (size_t i = 0; i < alphabet_size; ++i) {
                E(i, j) = std::uniform_real_distribution<double>(0.1, 1.0)(local_gen);
                sum += E(i, j);
            }
            for (size_t i = 0; i < alphabet_size; ++i) {
                E(i, j) /= sum;
            }
        }
        
        // Create HMM
        hmmlib::HMM<float_type, sse_float_type> hmm(pi_ptr, T_ptr, E_ptr);
        
        // Generate test sequence (same as libhmm)
        auto sequences = generateObservationSequences(1, seq_length, alphabet_size);
        sequence obs(sequences[0].begin(), sequences[0].end());
        sequence hidden_states(seq_length);
        
        // Benchmark Viterbi
        timer.start();
        result.likelihood = hmm.viterbi(obs, hidden_states);
        result.time_ms = timer.stop();
        
        return result;
    }
    
    void runComprehensiveBenchmarks() {
        std::cout << "\n=== HMM Library Comparison Benchmarks ===\n\n";
        
        std::cout << std::setw(25) << "Test" 
                  << std::setw(12) << "Library"
                  << std::setw(12) << "Time"
                  << std::setw(15) << "Likelihood"
                  << std::setw(8) << "States"
                  << std::setw(12) << "Seq Length"
                  << std::setw(12) << "Sequences" << std::endl;
        std::cout << std::string(96, '-') << std::endl;
        
        // Test configurations: {num_states, alphabet_size, seq_length}
        std::vector<std::tuple<size_t, size_t, size_t>> configs = {
            {3, 4, 50},       // Small
            {5, 6, 100},      // Medium  
            {10, 8, 200},     // Large
            {20, 10, 500},    // Very Large
            {50, 20, 1000}    // Extreme
        };
        
        for (const auto& config : configs) {
            size_t num_states, alphabet_size, seq_length;
            std::tie(num_states, alphabet_size, seq_length) = config;
            
            std::cout << "\n--- Configuration: " << num_states << " states, " 
                      << alphabet_size << " symbols, " << seq_length << " length ---\n";
            
            // Forward-Backward comparison
            try {
                auto libhmm_fb = benchmarkLibhmmForwardBackward(num_states, alphabet_size, seq_length);
                auto hmmlib_fb = benchmarkHMMLibForwardBackward(num_states, alphabet_size, seq_length);
                
                libhmm_fb.print();
                hmmlib_fb.print();
                
                results.push_back(libhmm_fb);
                results.push_back(hmmlib_fb);
                
                // Performance comparison
                double speedup = hmmlib_fb.time_ms / libhmm_fb.time_ms;
                std::cout << "    Forward-Backward Speedup: " << std::fixed << std::setprecision(2) 
                          << speedup << "x " << (speedup > 1 ? "(libhmm faster)" : "(HMMLib faster)") << std::endl;
            } catch (const std::exception& e) {
                std::cout << "    Forward-Backward benchmark failed: " << e.what() << std::endl;
            }
            
            // Viterbi comparison (both scalar and SIMD)
            try {
                // Test scalar Viterbi
                auto libhmm_vit_scalar = benchmarkLibhmmViterbi(num_states, alphabet_size, seq_length, false);
                auto hmmlib_vit = benchmarkHMMLibViterbi(num_states, alphabet_size, seq_length);
                
                libhmm_vit_scalar.print();
                hmmlib_vit.print();
                
                results.push_back(libhmm_vit_scalar);
                results.push_back(hmmlib_vit);
                
                // Performance comparison for scalar
                double speedup_scalar = hmmlib_vit.time_ms / libhmm_vit_scalar.time_ms;
                std::cout << "    Viterbi Scalar Speedup: " << std::fixed << std::setprecision(2) 
                          << speedup_scalar << "x " << (speedup_scalar > 1 ? "(libhmm faster)" : "(HMMLib faster)") << std::endl;
                
                // Test SIMD Viterbi if applicable
                if (num_states >= 4) {
                    auto libhmm_vit_simd = benchmarkLibhmmViterbi(num_states, alphabet_size, seq_length, true);
                    libhmm_vit_simd.print();
                    results.push_back(libhmm_vit_simd);
                    
                    // Performance comparison for SIMD
                    double speedup_simd = hmmlib_vit.time_ms / libhmm_vit_simd.time_ms;
                    std::cout << "    Viterbi SIMD Speedup: " << std::fixed << std::setprecision(2) 
                              << speedup_simd << "x " << (speedup_simd > 1 ? "(libhmm faster)" : "(HMMLib faster)") << std::endl;
                    
                    // SIMD vs Scalar comparison
                    double simd_vs_scalar = libhmm_vit_scalar.time_ms / libhmm_vit_simd.time_ms;
                    std::cout << "    SIMD vs Scalar: " << std::fixed << std::setprecision(2) 
                              << simd_vs_scalar << "x " << (simd_vs_scalar > 1 ? "(SIMD faster)" : "(Scalar faster)") << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "    Viterbi benchmark failed: " << e.what() << std::endl;
            }
        }
    }
    
    void saveResults(const std::string& filename) {
        std::ofstream file(filename);
        file << "test_name,library,time_ms,likelihood,num_states,seq_length,num_sequences\n";
        for (const auto& result : results) {
            file << result.test_name << ","
                 << result.library << ","
                 << result.time_ms << ","
                 << result.likelihood << ","
                 << result.num_states << ","
                 << result.seq_length << ","
                 << result.num_sequences << "\n";
        }
    }
};

int main() {
    std::cout << "HMM Library Benchmarking Suite\n";
    std::cout << "Comparing libhmm vs HMMLib performance and accuracy\n";
    
    HMMBenchmarkSuite suite;
    suite.runComprehensiveBenchmarks();
    
    // Save results
    suite.saveResults("benchmark_results.csv");
    std::cout << "\nResults saved to benchmark_results.csv\n";
    
    return 0;
}
