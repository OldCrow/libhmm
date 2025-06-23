#ifndef LIBHMM_THREAD_POOL_H_
#define LIBHMM_THREAD_POOL_H_

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <memory>

namespace libhmm {
namespace performance {

/// High-performance thread pool for parallel HMM computations
/// Designed for CPU-intensive tasks with minimal synchronization overhead
class ThreadPool {
public:
    /// Task function type
    using Task = std::function<void()>;
    
    /// Constructor with specified number of threads
    /// @param numThreads Number of worker threads (0 = auto-detect)
    explicit ThreadPool(std::size_t numThreads = 0);
    
    /// Destructor - waits for all tasks to complete
    ~ThreadPool();
    
    /// Submit a task for execution
    /// @param task Task to execute
    /// @return Future that will contain the result
    template<typename F, typename... Args>
    auto submit(F&& task, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        
        using ReturnType = typename std::result_of<F(Args...)>::type;
        
        auto taskPtr = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(task), std::forward<Args>(args)...)
        );
        
        std::future<ReturnType> result = taskPtr->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            
            // Don't allow enqueueing after stopping the pool
            if (stop_) {
                throw std::runtime_error("Cannot submit task to stopped ThreadPool");
            }
            
            tasks_.emplace([taskPtr]() { (*taskPtr)(); });
        }
        
        condition_.notify_one();
        return result;
    }
    
    /// Submit a task without return value
    /// @param task Task to execute
    void submitVoid(Task task);
    
    /// Get number of worker threads
    /// @return Number of threads in the pool
    std::size_t getNumThreads() const noexcept {
        return workers_.size();
    }
    
    /// Get number of pending tasks
    /// @return Number of tasks waiting to be executed
    std::size_t getPendingTasks() const;
    
    /// Wait for all current tasks to complete
    void waitForAll();
    
    /// Get optimal number of threads for this system
    /// @return Recommended thread count
    static std::size_t getOptimalThreadCount() noexcept;

private:
    /// Worker threads
    std::vector<std::thread> workers_;
    
    /// Task queue
    std::queue<Task> tasks_;
    
    /// Synchronization primitives
    mutable std::mutex queueMutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;
    
    /// Stop flag and active task counter
    std::atomic<bool> stop_;
    std::atomic<std::size_t> activeTasks_;
    
    /// Worker thread function
    void workerLoop();
};

/// Parallel execution utilities for HMM algorithms
class ParallelUtils {
public:
    /// Parallel for loop execution
    /// @param start Start index (inclusive)
    /// @param end End index (exclusive)
    /// @param task Function to execute for each index
    /// @param grainSize Minimum work per thread
    template<typename Func>
    static void parallelFor(std::size_t start, std::size_t end, Func&& task, 
                           std::size_t grainSize = 1) {
        
        const std::size_t range = end - start;
        if (range == 0) return;
        
        const std::size_t numThreads = ThreadPool::getOptimalThreadCount();
        const std::size_t actualGrainSize = std::max(grainSize, range / (numThreads * 4));
        
        if (range <= actualGrainSize) {
            // Execute sequentially for small ranges
            for (std::size_t i = start; i < end; ++i) {
                task(i);
            }
            return;
        }
        
        // Parallel execution
        ThreadPool& pool = getGlobalThreadPool();
        std::vector<std::future<void>> futures;
        
        for (std::size_t i = start; i < end; i += actualGrainSize) {
            const std::size_t chunkEnd = std::min(i + actualGrainSize, end);
            
            auto future = pool.submit([&task, i, chunkEnd]() {
                for (std::size_t j = i; j < chunkEnd; ++j) {
                    task(j);
                }
            });
            
            futures.push_back(std::move(future));
        }
        
        // Wait for all chunks to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    /// Parallel reduction operation
    /// @param start Start index (inclusive)
    /// @param end End index (exclusive)
    /// @param init Initial value for reduction
    /// @param task Function to compute value for each index
    /// @param reduce Function to combine two values
    /// @param grainSize Minimum work per thread
    /// @return Reduced result
    template<typename T, typename TaskFunc, typename ReduceFunc>
    static T parallelReduce(std::size_t start, std::size_t end, T init,
                           TaskFunc&& task, ReduceFunc&& reduce,
                           std::size_t grainSize = 1) {
        
        const std::size_t range = end - start;
        if (range == 0) return init;
        
        const std::size_t numThreads = ThreadPool::getOptimalThreadCount();
        const std::size_t actualGrainSize = std::max(grainSize, range / (numThreads * 4));
        
        if (range <= actualGrainSize) {
            // Execute sequentially for small ranges
            T result = init;
            for (std::size_t i = start; i < end; ++i) {
                result = reduce(result, task(i));
            }
            return result;
        }
        
        // Parallel execution
        ThreadPool& pool = getGlobalThreadPool();
        std::vector<std::future<T>> futures;
        
        for (std::size_t i = start; i < end; i += actualGrainSize) {
            const std::size_t chunkEnd = std::min(i + actualGrainSize, end);
            
            auto future = pool.submit([&task, &reduce, i, chunkEnd, init]() {
                T localResult = init;
                for (std::size_t j = i; j < chunkEnd; ++j) {
                    localResult = reduce(localResult, task(j));
                }
                return localResult;
            });
            
            futures.push_back(std::move(future));
        }
        
        // Combine partial results
        T finalResult = init;
        for (auto& future : futures) {
            finalResult = reduce(finalResult, future.get());
        }
        
        return finalResult;
    }
    
    /// Get the global thread pool instance
    /// @return Reference to singleton thread pool
    static ThreadPool& getGlobalThreadPool();
};

/// RAII helper for thread affinity setting
class ThreadAffinityGuard {
private:
    bool affinitySet_;
    std::thread::id threadId_;
    
public:
    /// Constructor that optionally sets thread affinity
    /// @param cpuId CPU core to bind to (-1 for no binding)
    explicit ThreadAffinityGuard(int cpuId = -1);
    
    /// Destructor restores original affinity
    ~ThreadAffinityGuard();
    
    /// Set thread affinity to specific CPU core
    /// @param cpuId CPU core ID
    /// @return True if affinity was successfully set
    bool setAffinity(int cpuId);
    
    /// Get current CPU core (if available)
    /// @return Current CPU core ID or -1 if unknown
    int getCurrentCpu() const noexcept;
};

/// CPU information and optimization hints
class CpuInfo {
public:
    /// CPU feature flags
    struct Features {
        bool hasSSE2 = false;
        bool hasSSE4_1 = false;
        bool hasAVX = false;
        bool hasAVX2 = false;
        bool hasAVX512 = false;
        bool hasNEON = false;
        bool hasHyperthreading = false;
    };
    
    /// Get CPU feature information
    /// @return CPU features detected at runtime
    static Features getCpuFeatures() noexcept;
    
    /// Get number of physical CPU cores
    /// @return Number of physical cores
    static std::size_t getPhysicalCores() noexcept;
    
    /// Get number of logical CPU cores (including hyperthreads)
    /// @return Number of logical cores
    static std::size_t getLogicalCores() noexcept;
    
    /// Get L1 cache size per core
    /// @return L1 cache size in bytes
    static std::size_t getL1CacheSize() noexcept;
    
    /// Get L2 cache size per core
    /// @return L2 cache size in bytes
    static std::size_t getL2CacheSize() noexcept;
    
    /// Get L3 cache size (shared)
    /// @return L3 cache size in bytes
    static std::size_t getL3CacheSize() noexcept;
    
    /// Get cache line size
    /// @return Cache line size in bytes
    static std::size_t getCacheLineSize() noexcept;
    
    /// Get CPU information summary
    /// @return String describing CPU capabilities
    static std::string getCpuInfoString();
};

} // namespace performance
} // namespace libhmm

#endif // LIBHMM_THREAD_POOL_H_
