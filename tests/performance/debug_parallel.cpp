#include "libhmm/performance/thread_pool.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace libhmm;

int main() {
    std::cout << "Debug Parallel Path Selection\n";
    std::cout << "============================\n\n";
    
    // Test with different sizes to see when parallel vs sequential is chosen
    std::vector<int> problemSizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    
    std::cout << "Size\tThreads\tGrainSize\tActualGrain\tRange<=Grain\tPath\n";
    std::cout << "----\t-------\t---------\t-----------\t-----------\t----\n";
    
    for (int size : problemSizes) {
        const std::size_t numThreads = performance::ThreadPool::getOptimalThreadCount();
        const std::size_t grainSize = 1;
        const std::size_t range = size;
        const std::size_t minWorkPerThread = std::max(grainSize, std::size_t(50));
        const std::size_t actualGrainSize = std::max(minWorkPerThread, range / numThreads);
        const bool sequential = (range < minWorkPerThread * numThreads || range <= actualGrainSize);
        
        std::cout << size << "\t" << numThreads << "\t" << grainSize 
                  << "\t\t" << actualGrainSize << "\t\t" 
                  << (sequential ? "true" : "false") << "\t\t"
                  << (sequential ? "SEQ" : "PAR") << "\n";
    }
    
    std::cout << "\nTesting actual execution paths:\n";
    std::cout << "===============================\n";
    
    // Test with instrumented version including parallel sizes
    for (int size : {4, 64, 256, 1024}) {
        std::cout << "\nTesting size " << size << ":\n";
        std::vector<double> data(size);
        std::vector<double> result(size);
        
        // Time a single call to see overhead
        auto start = std::chrono::high_resolution_clock::now();
        
        performance::ParallelUtils::parallelFor(0, size, [&](std::size_t i) {
            result[i] = data[i] * 2.0 + 1.0;
        }, 1);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "  Single call time: " << time << " μs\n";
        
        // Compare to direct sequential
        start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < static_cast<std::size_t>(size); ++i) {
            result[i] = data[i] * 2.0 + 1.0;
        }
        end = std::chrono::high_resolution_clock::now();
        auto seqTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "  Direct sequential: " << seqTime << " μs\n";
        std::cout << "  Overhead: " << (time - seqTime) << " μs\n";
    }
    
    return 0;
}
