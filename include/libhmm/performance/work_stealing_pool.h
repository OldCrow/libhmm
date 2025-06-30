#ifndef LIBHMM_PERFORMANCE_WORK_STEALING_POOL_H_
#define LIBHMM_PERFORMANCE_WORK_STEALING_POOL_H_

#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <random>

namespace libhmm {
namespace performance {

/**
 * @brief Work-stealing thread pool for high-performance parallel computing
 * 
 * This implementation provides better load balancing than traditional thread pools
 * by allowing idle threads to "steal" work from busy threads. Key features:
 * 
 * - Per-thread work queues to reduce contention
 * - Work stealing algorithm for automatic load balancing
 * - NUMA-aware thread affinity (where supported)
 * - Lock-free operations where possible
 * - Persistent threads to avoid creation/destruction overhead
 */
class WorkStealingPool {
public:
    using Task = std::function<void()>;
    
    /**
     * @brief Construct work-stealing thread pool
     * @param numThreads Number of worker threads (0 = auto-detect)
     * @param enableAffinity Enable CPU affinity for threads
     */
    explicit WorkStealingPool(std::size_t numThreads = 0, bool enableAffinity = true);
    
    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~WorkStealingPool();
    
    /**
     * @brief Submit a task for execution
     * @param task Function to execute
     */
    void submit(Task task);
    
    /**
     * @brief Submit a range-based parallel task with automatic work distribution
     * 
     * @param start Start of range (inclusive)
     * @param end End of range (exclusive)
     * @param func Function to call for each index: func(std::size_t index)
     * @param grainSize Minimum work per thread (0 = auto-detect)
     */
    template<typename Func>
    void parallelFor(std::size_t start, std::size_t end, Func func, std::size_t grainSize = 0);
    
    /**
     * @brief Wait for all currently submitted tasks to complete
     */
    void waitForAll();
    
    /**
     * @brief Get number of worker threads
     */
    std::size_t getThreadCount() const noexcept { return workers_.size(); }
    
    /**
     * @brief Get number of pending tasks across all queues
     */
    std::size_t getPendingTasks() const;
    
    /**
     * @brief Get statistics about work stealing efficiency
     */
    struct Statistics {
        std::size_t tasksExecuted;
        std::size_t workSteals;
        std::size_t failedSteals;
        double stealSuccessRate;
    };
    
    Statistics getStatistics() const;
    
    /**
     * @brief Reset statistics counters
     */
    void resetStatistics();

private:
    struct alignas(64) WorkerData {  // Cache line alignment
        std::deque<Task> localQueue;
        mutable std::mutex queueMutex;
        std::atomic<std::size_t> tasksExecuted{0};
        std::atomic<std::size_t> workSteals{0};
        std::atomic<std::size_t> failedSteals{0};
        std::thread worker;
        std::mt19937 rng;  // For random victim selection
        
        WorkerData() : rng(std::random_device{}()) {}
    };
    
    std::vector<std::unique_ptr<WorkerData>> workers_;
    std::atomic<bool> shutdown_{false};
    std::atomic<std::size_t> activeTasks_{0};
    
    // Global synchronization for waitForAll()
    mutable std::mutex globalMutex_;
    std::condition_variable allTasksComplete_;
    
    // Thread-local storage for current worker ID
    static thread_local int currentWorkerId_;
    
    /**
     * @brief Main worker loop
     * @param workerId ID of this worker thread
     */
    void workerLoop(int workerId);
    
    /**
     * @brief Try to steal work from another worker
     * @param thiefId ID of the stealing thread
     * @return Task if successful, nullptr otherwise
     */
    Task tryStealWork(int thiefId);
    
    /**
     * @brief Execute a task and handle exceptions
     * @param task Task to execute
     * @param workerId ID of executing worker
     */
    void executeTask(Task&& task, int workerId);
    
    /**
     * @brief Get optimal number of threads for this system
     */
    static std::size_t getOptimalThreadCount() noexcept;
    
    /**
     * @brief Set CPU affinity for a thread
     * @param threadId Thread to set affinity for
     * @param cpuId CPU core to bind to
     */
    static void setThreadAffinity(std::thread& thread, int cpuId);
};

/**
 * @brief Optimized parallel range implementation
 * 
 * Provides efficient work distribution for parallel loops with
 * automatic grain size calculation and work stealing support.
 */
template<typename Func>
void WorkStealingPool::parallelFor(std::size_t start, std::size_t end, Func func, std::size_t grainSize) {
    if (start >= end) return;
    
    const std::size_t totalWork = end - start;
    const std::size_t numWorkers = getThreadCount();
    
    // Auto-calculate grain size if not specified
    if (grainSize == 0) {
        // Aim for ~4x more tasks than threads for good load balancing
        const std::size_t targetTasks = numWorkers * 4;
        grainSize = std::max(std::size_t{1}, totalWork / targetTasks);
        
        // Ensure grain size is reasonable (not too small or too large)
        grainSize = std::max(std::size_t{8}, std::min(grainSize, std::size_t{1024}));
    }
    
    // Create tasks for each chunk
    std::atomic<std::size_t> completedTasks{0};
    const std::size_t numTasks = (totalWork + grainSize - 1) / grainSize;
    
    for (std::size_t taskId = 0; taskId < numTasks; ++taskId) {
        const std::size_t taskStart = start + taskId * grainSize;
        const std::size_t taskEnd = std::min(start + (taskId + 1) * grainSize, end);
        
        submit([=, &func, &completedTasks]() {
            // Execute function for this range
            for (std::size_t i = taskStart; i < taskEnd; ++i) {
                func(i);
            }
            completedTasks.fetch_add(1, std::memory_order_relaxed);
        });
    }
    
    // Wait for all tasks to complete
    while (completedTasks.load(std::memory_order_relaxed) < numTasks) {
        std::this_thread::yield();
    }
}

/**
 * @brief Global work-stealing thread pool instance
 * 
 * Provides a singleton work-stealing pool for use throughout the library.
 * This avoids the overhead of creating multiple thread pools.
 */
class GlobalWorkStealingPool {
public:
    static WorkStealingPool& getInstance() {
        static WorkStealingPool instance;
        return instance;
    }
    
private:
    GlobalWorkStealingPool() = default;
};

/**
 * @brief Utility functions for work-stealing parallel operations
 */
namespace WorkStealingUtils {
    
    /**
     * @brief Execute a parallel for loop using the global work-stealing pool
     */
    template<typename Func>
    inline void parallelFor(std::size_t start, std::size_t end, Func func, std::size_t grainSize = 0) {
        GlobalWorkStealingPool::getInstance().parallelFor(start, end, func, grainSize);
    }
    
    /**
     * @brief Submit a task to the global work-stealing pool
     */
    inline void submit(WorkStealingPool::Task task) {
        GlobalWorkStealingPool::getInstance().submit(std::move(task));
    }
    
    /**
     * @brief Wait for all tasks in the global pool to complete
     */
    inline void waitForAll() {
        GlobalWorkStealingPool::getInstance().waitForAll();
    }
    
    /**
     * @brief Check if work-stealing should be used for a given problem size
     * 
     * @param problemSize Size of the computational problem
     * @param threshold Minimum size to enable work-stealing
     * @return True if work-stealing pool should be used
     */
    inline bool shouldUseWorkStealing(std::size_t problemSize, std::size_t threshold = 1024) {
        return problemSize >= threshold && GlobalWorkStealingPool::getInstance().getThreadCount() > 1;
    }
}

} // namespace performance
} // namespace libhmm

#endif // LIBHMM_PERFORMANCE_WORK_STEALING_POOL_H_
