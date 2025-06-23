#include "libhmm/performance/thread_pool.h"
#include <algorithm>
#include <iostream>
#include <sstream>
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

ThreadPool::ThreadPool(std::size_t numThreads) 
    : stop_(false), activeTasks_(0) {
    
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
    
    for (std::thread& worker : workers_) {
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
    finished_.wait(lock, [this] { 
        return tasks_.empty() && activeTasks_.load() == 0; 
    });
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
        } catch (const std::exception& e) {
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

ThreadPool& ParallelUtils::getGlobalThreadPool() {
    static ThreadPool globalPool;
    return globalPool;
}

//========== ThreadAffinityGuard Implementation ==========

ThreadAffinityGuard::ThreadAffinityGuard(int cpuId) 
    : affinitySet_(false), threadId_(std::this_thread::get_id()) {
    
    if (cpuId >= 0) {
        setAffinity(cpuId);
    }
}

ThreadAffinityGuard::~ThreadAffinityGuard() {
    // On most systems, we don't need to explicitly restore affinity
    // as it's typically inherited and managed by the OS scheduler
}

bool ThreadAffinityGuard::setAffinity(int cpuId) {
    if (cpuId < 0) {
        return false;
    }
    
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuId, &cpuset);
    
    const int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    affinitySet_ = (result == 0);
    return affinitySet_;
    
#elif defined(__APPLE__)
    thread_affinity_policy_data_t policy = { cpuId };
    const kern_return_t result = thread_policy_set(
        mach_thread_self(),
        THREAD_AFFINITY_POLICY,
        (thread_policy_t)&policy,
        THREAD_AFFINITY_POLICY_COUNT
    );
    affinitySet_ = (result == KERN_SUCCESS);
    return affinitySet_;
    
#elif defined(_WIN32)
    const DWORD_PTR mask = 1ULL << cpuId;
    const DWORD_PTR result = SetThreadAffinityMask(GetCurrentThread(), mask);
    affinitySet_ = (result != 0);
    return affinitySet_;
    
#else
    // Unsupported platform
    return false;
#endif
}

int ThreadAffinityGuard::getCurrentCpu() const noexcept {
#ifdef __linux__
    return sched_getcpu();
#elif defined(_WIN32)
    return GetCurrentProcessorNumber();
#else
    return -1; // Unsupported platform
#endif
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
        if (line.find("physical id") == 0) {
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

std::size_t CpuInfo::getL1CacheSize() noexcept {
#ifdef __APPLE__
    std::size_t cacheSize = 0;
    std::size_t size = sizeof(cacheSize);
    if (sysctlbyname("hw.l1dcachesize", &cacheSize, &size, nullptr, 0) == 0) {
        return cacheSize;
    }
    
#elif defined(__linux__)
    std::ifstream cacheInfo("/sys/devices/system/cpu/cpu0/cache/index0/size");
    if (cacheInfo.is_open()) {
        std::string sizeStr;
        std::getline(cacheInfo, sizeStr);
        if (!sizeStr.empty()) {
            // Parse size (e.g., "32K")
            const char lastChar = sizeStr.back();
            const int multiplier = (lastChar == 'K' || lastChar == 'k') ? 1024 : 
                                  (lastChar == 'M' || lastChar == 'm') ? 1024 * 1024 : 1;
            if (multiplier > 1) {
                sizeStr.pop_back();
            }
            return std::stoul(sizeStr) * multiplier;
        }
    }
#endif
    
    return 32 * 1024; // 32KB default
}

std::size_t CpuInfo::getL2CacheSize() noexcept {
#ifdef __APPLE__
    std::size_t cacheSize = 0;
    std::size_t size = sizeof(cacheSize);
    if (sysctlbyname("hw.l2cachesize", &cacheSize, &size, nullptr, 0) == 0) {
        return cacheSize;
    }
    
#elif defined(__linux__)
    std::ifstream cacheInfo("/sys/devices/system/cpu/cpu0/cache/index2/size");
    if (cacheInfo.is_open()) {
        std::string sizeStr;
        std::getline(cacheInfo, sizeStr);
        if (!sizeStr.empty()) {
            const char lastChar = sizeStr.back();
            const int multiplier = (lastChar == 'K' || lastChar == 'k') ? 1024 : 
                                  (lastChar == 'M' || lastChar == 'm') ? 1024 * 1024 : 1;
            if (multiplier > 1) {
                sizeStr.pop_back();
            }
            return std::stoul(sizeStr) * multiplier;
        }
    }
#endif
    
    return 256 * 1024; // 256KB default
}

std::size_t CpuInfo::getL3CacheSize() noexcept {
#ifdef __APPLE__
    std::size_t cacheSize = 0;
    std::size_t size = sizeof(cacheSize);
    if (sysctlbyname("hw.l3cachesize", &cacheSize, &size, nullptr, 0) == 0) {
        return cacheSize;
    }
    
#elif defined(__linux__)
    std::ifstream cacheInfo("/sys/devices/system/cpu/cpu0/cache/index3/size");
    if (cacheInfo.is_open()) {
        std::string sizeStr;
        std::getline(cacheInfo, sizeStr);
        if (!sizeStr.empty()) {
            const char lastChar = sizeStr.back();
            const int multiplier = (lastChar == 'K' || lastChar == 'k') ? 1024 : 
                                  (lastChar == 'M' || lastChar == 'm') ? 1024 * 1024 : 1;
            if (multiplier > 1) {
                sizeStr.pop_back();
            }
            return std::stoul(sizeStr) * multiplier;
        }
    }
#endif
    
    return 8 * 1024 * 1024; // 8MB default
}

std::size_t CpuInfo::getCacheLineSize() noexcept {
#ifdef __APPLE__
    std::size_t lineSize = 0;
    std::size_t size = sizeof(lineSize);
    if (sysctlbyname("hw.cachelinesize", &lineSize, &size, nullptr, 0) == 0) {
        return lineSize;
    }
    
#elif defined(__linux__)
    std::ifstream cacheInfo("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    if (cacheInfo.is_open()) {
        std::size_t lineSize;
        cacheInfo >> lineSize;
        if (lineSize > 0) {
            return lineSize;
        }
    }
#endif
    
    return 64; // 64 bytes default
}

std::string CpuInfo::getCpuInfoString() {
    std::ostringstream oss;
    const Features features = getCpuFeatures();
    
    oss << "CPU Info: ";
    oss << getLogicalCores() << " logical cores, ";
    oss << getPhysicalCores() << " physical cores";
    
    if (features.hasHyperthreading) {
        oss << " (hyperthreading)";
    }
    
    oss << "\nSIMD: ";
    if (features.hasAVX512) {
        oss << "AVX-512";
    } else if (features.hasAVX2) {
        oss << "AVX2";
    } else if (features.hasAVX) {
        oss << "AVX";
    } else if (features.hasSSE4_1) {
        oss << "SSE4.1";
    } else if (features.hasSSE2) {
        oss << "SSE2";
    } else if (features.hasNEON) {
        oss << "ARM NEON";
    } else {
        oss << "None";
    }
    
    oss << "\nCache: L1=" << (getL1CacheSize() / 1024) << "KB, ";
    oss << "L2=" << (getL2CacheSize() / 1024) << "KB, ";
    oss << "L3=" << (getL3CacheSize() / 1024 / 1024) << "MB, ";
    oss << "Line=" << getCacheLineSize() << "B";
    
    return oss.str();
}

} // namespace performance
} // namespace libhmm
