#include "libhmm/performance/thread_pool.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace libhmm;

int main() {
    std::cout << "Thread Pool Overhead Analysis\n";
    std::cout << "============================\n\n";
    
    // Test overhead of parallel vs sequential operations
    std::vector<int> problemSizes = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    std::cout << "Size\tSequential(μs)\tParallel(μs)\tOverhead\tSpeedup\n";
    std::cout << "----\t--------------\t------------\t--------\t-------\n";
    
    for (int size : problemSizes) {
        // Create test data
        std::vector<double> data(size);
        std::vector<double> result(size);
        
        // Sequential version
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 1000; ++iter) {
            for (int i = 0; i < size; ++i) {
                result[i] = data[i] * 2.0 + 1.0; // Simple operation
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto seqTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
        
        // Parallel version
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 1000; ++iter) {
            performance::ParallelUtils::parallelFor(0, size, [&](std::size_t i) {
                result[i] = data[i] * 2.0 + 1.0;
            }, 1);
        }
        end = std::chrono::high_resolution_clock::now();
        auto parTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
        
        double overhead = parTime - seqTime;
        double speedup = seqTime / parTime;
        
        std::cout << size << "\t" << std::fixed << std::setprecision(1)
                  << seqTime << "\t\t" << parTime << "\t\t" 
                  << overhead << "\t\t" << speedup << "x\n";
    }
    
    std::cout << "\nThread count: " << performance::ThreadPool::getOptimalThreadCount() << "\n";
    std::cout << "\nRecommendations:\n";
    std::cout << "- Use parallel processing only when speedup > 1.0x\n";
    std::cout << "- Consider overhead when setting parallel thresholds\n";
    
    return 0;
}
