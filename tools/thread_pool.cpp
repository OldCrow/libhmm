#include "thread_pool.h"
#include <algorithm>
#include <iostream>
#include <set>

// Platform-specific includes
#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <fstream>
#elif defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#endif

namespace libhmm {
namespace performance {

//========== ThreadPool Implementation ==========

ThreadPool::ThreadPool(std::size_t numThreads) : stop_(false), activeTasks_(0) {

    if (numThreads == 0) {
        numThreads = getOptimalThreadCount();
    }

    workers_.reserve(numThreads);
    for (std::size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        stop_ = true;
    }

    condition_.notify_all();

    for (std::thread &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::submitVoid(Task task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);

        if (stop_) {
            throw std::runtime_error("Cannot submit task to stopped ThreadPool");
        }

        tasks_.emplace(std::move(task));
    }

    condition_.notify_one();
}

std::size_t ThreadPool::getPendingTasks() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    return tasks_.size();
}

void ThreadPool::waitForAll() {
    std::unique_lock<std::mutex> lock(queueMutex_);
    finished_.wait(lock, [this] { return tasks_.empty() && activeTasks_.load() == 0; });
}

void ThreadPool::workerLoop() {
    while (true) {
        Task task;

        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
            activeTasks_++;
        }

        try {
            task();
        } catch (const std::exception &e) {
            std::cerr << "Exception in thread pool task: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in thread pool task" << std::endl;
        }

        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            activeTasks_--;
            if (tasks_.empty() && activeTasks_.load() == 0) {
                finished_.notify_all();
            }
        }
    }
}

std::size_t ThreadPool::getOptimalThreadCount() noexcept {
    const std::size_t hardwareThreads = std::thread::hardware_concurrency();
    if (hardwareThreads == 0) {
        return 4; // Fallback value
    }

    // For CPU-intensive tasks, use physical cores if hyperthreading is available
    const auto features = CpuInfo::getCpuFeatures();
    if (features.hasHyperthreading) {
        const std::size_t physicalCores = CpuInfo::getPhysicalCores();
        if (physicalCores > 0) {
            return physicalCores;
        }
    }

    return hardwareThreads;
}

//========== ParallelUtils Implementation ==========

ThreadPool &ParallelUtils::getGlobalThreadPool() {
    static ThreadPool globalPool;
    return globalPool;
}

//========== CpuInfo Implementation ==========

CpuInfo::Features CpuInfo::getCpuFeatures() noexcept {
    Features features;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    // x86/x64 CPUID detection

    // Check for SSE2 (should be available on all modern x64 CPUs)
#ifdef __SSE2__
    features.hasSSE2 = true;
#endif

    // Check for SSE4.1
#ifdef __SSE4_1__
    features.hasSSE4_1 = true;
#endif

    // Check for AVX
#ifdef __AVX__
    features.hasAVX = true;
#endif

    // Check for AVX2
#ifdef __AVX2__
    features.hasAVX2 = true;
#endif

    // Check for AVX-512
#ifdef __AVX512F__
    features.hasAVX512 = true;
#endif

#elif defined(__ARM_NEON) || defined(__aarch64__)
    // ARM NEON detection
    features.hasNEON = true;
#endif

    // Check for hyperthreading (simplified heuristic)
    const std::size_t logicalCores = getLogicalCores();
    const std::size_t physicalCores = getPhysicalCores();
    features.hasHyperthreading = (logicalCores > physicalCores);

    return features;
}

std::size_t CpuInfo::getPhysicalCores() noexcept {
#ifdef __APPLE__
    std::size_t cores = 0;
    std::size_t size = sizeof(cores);
    if (sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0) == 0) {
        return cores;
    }

#elif defined(__linux__)
    // Read from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::set<int> physicalIds;

    while (std::getline(cpuinfo, line)) {
        if (line.starts_with("physical id")) {
            const std::size_t colonPos = line.find(':');
            if (colonPos != std::string::npos) {
                const int physicalId = std::stoi(line.substr(colonPos + 1));
                physicalIds.insert(physicalId);
            }
        }
    }

    if (!physicalIds.empty()) {
        return physicalIds.size();
    }

#elif defined(_WIN32)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return static_cast<std::size_t>(sysInfo.dwNumberOfProcessors);
#endif

    // Fallback to logical cores
    return getLogicalCores();
}

std::size_t CpuInfo::getLogicalCores() noexcept {
    const unsigned int cores = std::thread::hardware_concurrency();
    return cores > 0 ? cores : 1;
}

} // namespace performance
} // namespace libhmm
