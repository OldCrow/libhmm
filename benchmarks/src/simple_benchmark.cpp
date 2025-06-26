#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <random>
#include <iomanip>

// libhmm includes
#include "../include/libhmm/libhmm.h"

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

void benchmarkLibhmm() {
    std::cout << "\n=== libhmm Performance Benchmarks ===\n\n";
    
    std::vector<std::tuple<size_t, size_t, size_t>> configs = {
        {3, 4, 50},       // Small: 3 states, 4 symbols, 50 length
        {5, 6, 100},      // Medium: 5 states, 6 symbols, 100 length  
        {10, 8, 200},     // Large: 10 states, 8 symbols, 200 length
        {20, 10, 500},    // Very Large: 20 states, 10 symbols, 500 length
    };
    
    std::mt19937 gen(42); // Fixed seed for reproducibility
    BenchmarkTimer timer;
    
    std::cout << std::setw(15) << "Configuration" 
              << std::setw(20) << "Forward-Backward (ms)"
              << std::setw(15) << "Viterbi (ms)"
              << std::setw(15) << "Likelihood" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    for (const auto& config : configs) {
        size_t num_states, alphabet_size, seq_length;
        std::tie(num_states, alphabet_size, seq_length) = config;
        
        try {
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
                std::vector<double> probs(alphabet_size);
                sum = 0.0;
                for (size_t j = 0; j < alphabet_size; ++j) {
                    probs[j] = std::uniform_real_distribution<double>(0.1, 1.0)(gen);
                    sum += probs[j];
                }
                // Normalize and set probabilities
                for (size_t j = 0; j < alphabet_size; ++j) {
                    dist->setProbability(j, probs[j] / sum);
                }
                hmm->setProbabilityDistribution(i, std::move(dist));
            }
            
            // Generate test sequence
            std::uniform_int_distribution<unsigned int> obs_dist(0, alphabet_size - 1);
            libhmm::ObservationSet obs(seq_length);
            for (size_t i = 0; i < seq_length; ++i) {
                obs(i) = obs_dist(gen);
            }
            
            // Benchmark Forward-Backward
            timer.start();
            libhmm::ForwardBackwardCalculator calc(hmm.get(), obs);
            double likelihood = calc.probability();
            double fb_time = timer.stop();
            
            // Benchmark Viterbi
            timer.start();
            libhmm::ViterbiCalculator viterbi(hmm.get(), obs);
            auto states = viterbi.decode();
            double viterbi_time = timer.stop();
            
            // Display results
            std::cout << std::setw(15) << (std::to_string(num_states) + "x" + std::to_string(alphabet_size) + "x" + std::to_string(seq_length))
                      << std::setw(20) << std::fixed << std::setprecision(3) << fb_time
                      << std::setw(15) << std::fixed << std::setprecision(3) << viterbi_time
                      << std::setw(15) << std::scientific << std::setprecision(2) << likelihood << std::endl;
                      
        } catch (const std::exception& e) {
            std::cout << "Error with configuration " << num_states << "x" << alphabet_size << "x" << seq_length 
                      << ": " << e.what() << std::endl;
        }
    }
}

void benchmarkDistributions() {
    std::cout << "\n=== Distribution Performance Benchmarks ===\n\n";
    
    std::mt19937 gen(42);
    BenchmarkTimer timer;
    const size_t num_evaluations = 100000;
    
    std::cout << std::setw(25) << "Distribution" 
              << std::setw(20) << "Time per 100k evals (ms)"
              << std::setw(15) << "Throughput (M/s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // Test different distributions
    std::vector<std::unique_ptr<libhmm::ProbabilityDistribution>> distributions;
    std::vector<std::string> names;
    
    distributions.push_back(std::make_unique<libhmm::GaussianDistribution>(0.0, 1.0));
    names.push_back("Gaussian");
    
    distributions.push_back(std::make_unique<libhmm::ExponentialDistribution>(1.0));
    names.push_back("Exponential");
    
    distributions.push_back(std::make_unique<libhmm::PoissonDistribution>(5.0));
    names.push_back("Poisson");
    
    distributions.push_back(std::make_unique<libhmm::GammaDistribution>(2.0, 1.0));
    names.push_back("Gamma");
    
    auto discrete = std::make_unique<libhmm::DiscreteDistribution>(10);
    for (int i = 0; i < 10; ++i) {
        discrete->setProbability(i, 0.1);
    }
    distributions.push_back(std::move(discrete));
    names.push_back("Discrete");
    
    for (size_t i = 0; i < distributions.size(); ++i) {
        timer.start();
        
        for (size_t j = 0; j < num_evaluations; ++j) {
            double x = std::uniform_real_distribution<double>(-5.0, 5.0)(gen);
            if (names[i] == "Discrete" || names[i] == "Poisson") {
                x = std::abs(x);
                if (x > 20) x = 20; // Limit for discrete distributions
            }
            distributions[i]->getProbability(x);
        }
        
        double time_ms = timer.stop();
        double throughput = num_evaluations / 1000.0 / time_ms; // Millions per second
        
        std::cout << std::setw(25) << names[i]
                  << std::setw(20) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << throughput << std::endl;
    }
}

void benchmarkTraining() {
    std::cout << "\n=== Training Performance Benchmarks ===\n\n";
    
    std::mt19937 gen(42);
    BenchmarkTimer timer;
    
    std::cout << std::setw(20) << "Configuration" 
              << std::setw(20) << "Viterbi Training (ms)"
              << std::setw(15) << "Sequences" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> configs = {
        {3, 4, 20, 10},   // 3 states, 4 symbols, 20 length, 10 sequences
        {5, 6, 30, 15},   // 5 states, 6 symbols, 30 length, 15 sequences
        {10, 8, 50, 20},  // 10 states, 8 symbols, 50 length, 20 sequences
    };
    
    for (const auto& config : configs) {
        size_t num_states, alphabet_size, seq_length, num_sequences;
        std::tie(num_states, alphabet_size, seq_length, num_sequences) = config;
        
        try {
            // Create HMM for training
            auto hmm = std::make_unique<libhmm::Hmm>(num_states);
            
            // Set up discrete distributions
            for (size_t i = 0; i < num_states; ++i) {
                auto dist = std::make_unique<libhmm::DiscreteDistribution>(alphabet_size);
                for (size_t j = 0; j < alphabet_size; ++j) {
                    dist->setProbability(j, 1.0 / alphabet_size); // Uniform
                }
                hmm->setProbabilityDistribution(i, std::move(dist));
            }
            
            // Generate training data
            libhmm::ObservationLists training_data;
            std::uniform_int_distribution<unsigned int> obs_dist(0, alphabet_size - 1);
            
            for (size_t s = 0; s < num_sequences; ++s) {
                libhmm::ObservationSet seq(seq_length);
                for (size_t i = 0; i < seq_length; ++i) {
                    seq(i) = obs_dist(gen);
                }
                training_data.push_back(seq);
            }
            
            // Benchmark Viterbi training
            timer.start();
            libhmm::ViterbiTrainer trainer(hmm.get(), training_data);
            trainer.train();
            double training_time = timer.stop();
            
            std::cout << std::setw(20) << (std::to_string(num_states) + "x" + std::to_string(alphabet_size) + "x" + std::to_string(seq_length))
                      << std::setw(20) << std::fixed << std::setprecision(3) << training_time
                      << std::setw(15) << num_sequences << std::endl;
                      
        } catch (const std::exception& e) {
            std::cout << "Training error: " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "libhmm Performance Benchmarking Suite\n";
    std::cout << "=====================================\n";
    
    benchmarkLibhmm();
    benchmarkDistributions();
    benchmarkTraining();
    
    std::cout << "\nBenchmarking complete!\n";
    return 0;
}
