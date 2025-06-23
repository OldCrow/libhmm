#ifndef LIBHMM_BENCHMARK_H_
#define LIBHMM_BENCHMARK_H_

#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <map>
#include <iostream>

namespace libhmm {

// Forward declarations
class Hmm;
class ObservationSet;
using ObservationLists = std::vector<ObservationSet>;

namespace performance {

/// High-resolution timer for performance measurements
class Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;
    
    TimePoint startTime_;
    bool running_;
    
public:
    Timer() : running_(false) {}
    
    /// Start the timer
    void start() {
        startTime_ = Clock::now();
        running_ = true;
    }
    
    /// Stop the timer and return elapsed time in seconds
    /// @return Elapsed time in seconds
    double stop() {
        if (!running_) {
            return 0.0;
        }
        
        const auto endTime = Clock::now();
        const Duration elapsed = endTime - startTime_;
        running_ = false;
        return elapsed.count();
    }
    
    /// Get elapsed time without stopping (if running)
    /// @return Elapsed time in seconds
    double elapsed() const {
        if (!running_) {
            return 0.0;
        }
        
        const auto currentTime = Clock::now();
        const Duration elapsed = currentTime - startTime_;
        return elapsed.count();
    }
    
    /// Check if timer is running
    /// @return True if timer is running
    bool isRunning() const noexcept {
        return running_;
    }
};

/// Statistics for benchmark results
struct BenchmarkStats {
    double mean = 0.0;           ///< Mean execution time
    double median = 0.0;         ///< Median execution time
    double stddev = 0.0;         ///< Standard deviation
    double min = 0.0;            ///< Minimum execution time
    double max = 0.0;            ///< Maximum execution time
    std::size_t samples = 0;     ///< Number of samples
    double throughput = 0.0;     ///< Operations per second (if applicable)
    
    /// Format statistics as string
    /// @return Formatted statistics string
    std::string toString() const;
};

/// Benchmark result for a single test
struct BenchmarkResult {
    std::string name;                    ///< Test name
    BenchmarkStats stats;                ///< Statistical results
    std::vector<double> rawTimes;        ///< Raw timing measurements
    std::map<std::string, double> metrics; ///< Custom metrics
    
    /// Add a custom metric
    /// @param key Metric name
    /// @param value Metric value
    void addMetric(const std::string& key, double value) {
        metrics[key] = value;
    }
    
    /// Get formatted result string
    /// @return Formatted result
    std::string toString() const;
};

/// Benchmark suite for HMM performance testing
class Benchmark {
public:
    /// Test function type
    using TestFunction = std::function<void()>;
    
    /// Setup function type (called before each test iteration)
    using SetupFunction = std::function<void()>;
    
    /// Teardown function type (called after each test iteration)
    using TeardownFunction = std::function<void()>;
    
private:
    struct TestCase {
        std::string name;
        TestFunction testFunc;
        SetupFunction setupFunc;
        TeardownFunction teardownFunc;
        std::size_t iterations;
        std::size_t warmupRuns;
        double operationCount; ///< For throughput calculation
    };
    
    std::vector<TestCase> testCases_;
    std::vector<BenchmarkResult> results_;
    bool enableWarmup_;
    std::size_t defaultIterations_;
    std::size_t defaultWarmupRuns_;
    
public:
    /// Constructor
    /// @param enableWarmup Enable warmup runs to stabilize CPU state
    /// @param defaultIterations Default number of iterations per test
    /// @param defaultWarmupRuns Default number of warmup runs
    explicit Benchmark(bool enableWarmup = true, 
                      std::size_t defaultIterations = 100,
                      std::size_t defaultWarmupRuns = 10);
    
    /// Add a test case
    /// @param name Test name
    /// @param testFunc Function to benchmark
    /// @param iterations Number of iterations (0 = use default)
    /// @param operationCount Number of operations per test (for throughput)
    /// @param setupFunc Optional setup function
    /// @param teardownFunc Optional teardown function
    /// @param warmupRuns Number of warmup runs (0 = use default)
    void addTest(const std::string& name, 
                 TestFunction testFunc,
                 std::size_t iterations = 0,
                 double operationCount = 1.0,
                 SetupFunction setupFunc = nullptr,
                 TeardownFunction teardownFunc = nullptr,
                 std::size_t warmupRuns = 0);
    
    /// Run all registered benchmarks
    /// @return Vector of benchmark results
    std::vector<BenchmarkResult> runAll();
    
    /// Run a specific benchmark by name
    /// @param testName Name of test to run
    /// @return Benchmark result
    BenchmarkResult run(const std::string& testName);
    
    /// Clear all test cases and results
    void clear();
    
    /// Get results from last run
    /// @return Vector of results
    const std::vector<BenchmarkResult>& getResults() const {
        return results_;
    }
    
    /// Print results to output stream
    /// @param os Output stream
    /// @param showRawTimes Include raw timing data
    void printResults(std::ostream& os = std::cout, bool showRawTimes = false) const;
    
    /// Compare two benchmark results
    /// @param baseline Baseline results
    /// @param comparison Results to compare against baseline
    /// @param os Output stream for comparison
    static void compareResults(const std::vector<BenchmarkResult>& baseline,
                              const std::vector<BenchmarkResult>& comparison,
                              std::ostream& os = std::cout);

private:
    /// Calculate statistics from raw timing data
    /// @param times Vector of timing measurements
    /// @param operationCount Number of operations (for throughput)
    /// @return Calculated statistics
    BenchmarkStats calculateStats(const std::vector<double>& times, double operationCount) const;
    
    /// Run warmup iterations
    /// @param testCase Test case to warm up
    void runWarmup(const TestCase& testCase);
    
    /// Execute a single test case
    /// @param testCase Test case to execute
    /// @return Benchmark result
    BenchmarkResult executeTest(const TestCase& testCase);
};

/// Convenience macros for benchmarking
#define LIBHMM_BENCHMARK_SIMPLE(name, func) \
    benchmark.addTest(name, func)

#define LIBHMM_BENCHMARK_WITH_SETUP(name, func, setup, teardown) \
    benchmark.addTest(name, func, 0, 1.0, setup, teardown)

#define LIBHMM_BENCHMARK_ITERATIONS(name, func, iters) \
    benchmark.addTest(name, func, iters)

/// HMM-specific benchmark utilities
class HmmBenchmarkUtils {
public:
    /// Create test observation sequences of varying lengths
    /// @param minLength Minimum sequence length
    /// @param maxLength Maximum sequence length
    /// @param numSequences Number of sequences to generate
    /// @param numSymbols Number of possible symbols
    /// @return Vector of observation sequences
    static std::vector<ObservationSet> createTestSequences(
        std::size_t minLength, std::size_t maxLength, 
        std::size_t numSequences, std::size_t numSymbols);
    
    /// Create test HMMs of varying sizes
    /// @param minStates Minimum number of states
    /// @param maxStates Maximum number of states
    /// @param numSymbols Number of observation symbols
    /// @return Vector of test HMMs
    static std::vector<std::unique_ptr<Hmm>> createTestHmms(
        std::size_t minStates, std::size_t maxStates, 
        std::size_t numSymbols);
    
    /// Benchmark forward-backward algorithms
    /// @param hmm HMM to test
    /// @param observations Test observations
    /// @param benchmark Benchmark object to add tests to
    static void benchmarkForwardBackward(
        Hmm* hmm, const ObservationSet& observations, 
        Benchmark& benchmark);
    
    /// Benchmark Viterbi algorithm
    /// @param hmm HMM to test
    /// @param observations Test observations
    /// @param benchmark Benchmark object to add tests to
    static void benchmarkViterbi(
        Hmm* hmm, const ObservationSet& observations,
        Benchmark& benchmark);
    
    /// Benchmark training algorithms
    /// @param hmm HMM to test
    /// @param observationLists Training data
    /// @param benchmark Benchmark object to add tests to
    static void benchmarkTraining(
        Hmm* hmm, const ObservationLists& observationLists,
        Benchmark& benchmark);
};

/// Performance regression testing
class RegressionTester {
private:
    std::string baselineFile_;
    std::vector<BenchmarkResult> baselineResults_;
    double tolerancePercent_;
    
public:
    /// Constructor
    /// @param baselineFile File containing baseline results
    /// @param tolerancePercent Allowed performance degradation percentage
    explicit RegressionTester(const std::string& baselineFile, 
                             double tolerancePercent = 5.0);
    
    /// Load baseline results from file
    /// @return True if loaded successfully
    bool loadBaseline();
    
    /// Save results as new baseline
    /// @param results Results to save
    /// @return True if saved successfully
    bool saveBaseline(const std::vector<BenchmarkResult>& results);
    
    /// Test for performance regressions
    /// @param currentResults Current benchmark results
    /// @return True if no significant regressions detected
    bool checkRegressions(const std::vector<BenchmarkResult>& currentResults,
                         std::ostream& os = std::cout);
    
    /// Set tolerance for regression detection
    /// @param percent Allowed degradation percentage
    void setTolerance(double percent) {
        tolerancePercent_ = percent;
    }
};

} // namespace performance
} // namespace libhmm

#endif // LIBHMM_BENCHMARK_H_
