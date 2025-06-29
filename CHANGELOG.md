# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.7.2] - 2025-06-29

### Complete Gold Standard Distribution Optimizations Release

This release delivers comprehensive performance optimizations across all probability distributions while maintaining mathematical correctness and adding new functionality. Major improvements include the addition of the Rayleigh distribution, significant performance gains through standard library integration, and enhanced numerical accuracy.

### Added

#### New Distribution
- **Rayleigh Distribution**: Complete implementation with Gold Standard compliance
  - Specialized case of Weibull distribution (k=2) for modeling vector magnitudes
  - Applications in communications, wind modeling, and signal processing
  - Full test suite with performance benchmarks and edge case coverage
  - Optimized PDF/CDF calculations with caching mechanisms

#### Enhanced Performance Features
- **Vectorized Batch Processing**: Added batch computation methods for Beta distribution
  - `getProbabilityBatch()` and `getLogProbabilityBatch()` for efficient bulk operations
  - Optimized for processing multiple values with enhanced cache reuse
  - Foundation for future SIMD vectorization across all distributions

#### Mathematical Correctness Improvements
- **Proper Beta CDF Implementation**: Fixed critical mathematical error
  - Replaced incorrect incomplete gamma function with proper incomplete beta function
  - Implemented continued fraction expansion for numerical accuracy
  - Ensures mathematically correct cumulative probability calculations

### Enhanced

#### Distribution Performance Optimizations
- **Beta Distribution**: 24% PDF improvement, 62% log PDF improvement
  - Enhanced caching system with `invBeta_`, `alphaMinus1_`, `betaMinus1_`
  - Binary exponentiation for fast integer power calculations
  - Optimized boundary case handling and numerical stability
- **Log-Normal Distribution**: 61% PDF improvement, 53% fitting improvement
  - Welford's algorithm integration for numerically stable fitting
  - Enhanced caching mechanisms and optimized log-space calculations
- **Gaussian Distribution**: Enhanced CDF using `std::erf` for improved accuracy
  - 31% performance improvement in error function calculations
  - Better numerical stability and hardware optimization

#### Standard Library Integration
- **Mathematical Function Optimization**: Replaced custom implementations with standard library
  - `std::lgamma` replaces custom `loggamma` (31% faster)
  - `std::erf` replaces custom `errorf` implementation
  - Leverages hardware-optimized mathematical functions
  - Maintains compatibility while improving performance and maintainability

#### Numerical Enhancements
- **Binary Exponentiation**: Fast integer power calculations across distributions
  - Optimized small case handling (powers 0-4) with direct computation
  - Logarithmic complexity for larger integer exponents
  - Significant speedup for distributions with integer shape parameters
- **Enhanced Caching Systems**: Comprehensive pre-computation strategies
  - Distribution-specific cached values for frequently used calculations
  - Automatic cache invalidation on parameter changes
  - Reduced redundant mathematical operations

### Fixed

#### Mathematical Correctness
- **Beta Distribution CDF**: Corrected from incorrect gamma-based to proper beta function implementation
- **Boundary Value Handling**: Improved edge case processing across all distributions
- **Numerical Stability**: Enhanced precision in extreme value scenarios

#### Code Quality and Maintainability
- **Dead Code Removal**: Eliminated unused custom mathematical implementations
- **Function Declarations**: Updated headers to reflect standard library usage
- **Compilation Optimization**: Resolved build warnings and enhanced compiler optimization

### Performance Results

#### Current Performance Benchmarks
```
Beta Distribution:     0.097μs PDF, 0.047μs log PDF, 0.029μs fitting
Log-Normal:           0.079μs PDF, 0.045μs log PDF, 0.037μs fitting  
Gaussian:             0.045μs PDF, 0.027μs log PDF, 0.017μs fitting
Rayleigh:             Sub-microsecond performance across all operations
```

#### Optimization Impact Summary
- **Beta PDF**: 24% improvement (118ns → 90ns per call)
- **Beta Log PDF**: 62% improvement (93ns → 35ns per call)
- **Standard Library Functions**: 31% improvement in mathematical operations
- **Overall**: Maintained sub-microsecond performance while adding functionality

### Technical Implementation

#### Binary Exponentiation Algorithm
```cpp
auto fastPower = [](double base, int exp) -> double {
    if (exp <= 4) return directComputation(base, exp);  // Optimized small cases
    // Binary exponentiation for larger powers
    double result = 1.0;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
};
```

#### Enhanced Caching Strategy
```cpp
void updateCache() const noexcept {
    logBeta_ = std::lgamma(alpha_) + std::lgamma(beta_) - std::lgamma(alpha_ + beta_);
    invBeta_ = std::exp(-logBeta_);  // Direct computation cache
    alphaMinus1_ = alpha_ - 1.0;      // Frequent calculation cache
    betaMinus1_ = beta_ - 1.0;        // Frequent calculation cache
    cacheValid_ = true;
}
```

### Quality Assurance

#### Test Coverage
```
Distribution Tests: 15/15 distributions with complete Gold Standard coverage
Performance Tests: Comprehensive benchmarking across all optimized functions
Numerical Tests: Edge cases, boundary values, and extreme parameter validation
Batch Processing: Vectorized operations testing for future SIMD integration
```

#### Compatibility
- **API Compatibility**: All changes maintain full backward compatibility
- **Performance**: Enhanced speed while maintaining mathematical correctness
- **Reliability**: Improved numerical stability and error handling

### Breaking Changes

**None** - All optimizations are internal improvements maintaining full API compatibility.

### Migration Notes

Existing code continues to work unchanged with enhanced performance and accuracy:
- Beta distribution CDF calculations now return mathematically correct results
- All distributions benefit from faster mathematical function evaluation
- New batch processing methods available for performance-critical applications

### Future Roadmap

This release establishes the foundation for:
- **SIMD Vectorization**: Batch processing infrastructure ready for vector instructions
- **Calculator Optimizations**: Next phase focusing on algorithm-level improvements  
- **Advanced Features**: Enhanced parallel processing and specialized distributions

---

## [2.7.1] - 2025-06-28

### Discrete Distributions Gold Standard & HTK Benchmark Enhancement Release

This release elevates all 4 discrete distributions (Discrete, Binomial, Negative Binomial, Poisson) to the gold standard of exception safety, input validation, and naming consistency. Additionally, the benchmarking suite now distinctly separates discrete from continuous benchmarks and includes extended performance scaling analysis.

### Added

#### Discrete Distribution Enhancements
- **Gold Standard Upgrades**: All discrete distributions updated to gold standard
  - Comprehensive exception handling in stream input operators
  - Input validation for edge cases, e.g., k=0 for Poisson, negative probabilities
  - Consistent variable naming following the project guidelines

#### Continuous Benchmarking Improvements
- **HTK Benchmark Separation**: Clean discrete and continuous benchmark separation
- **Performance Scaling Analysis**: Extended sequences ranging 100 to 1,000,000 observations
- **Gaussian Distribution Fixes**: Proper handling of mean/variance pairs

### Enhanced

#### Benchmarking and Accuracy
- **Numerical Validation**: Accurate log-likelihood computation for continuous models
- **Performance Benchmarking**: Updated benchmarks showcase distinct advantages of libhmm and HTK under different scenarios

#### Code Reliability and Consistency
- **Error Messages**: Improved clarity and consistency in error reporting
- **Documentation**: Updated all affected components to align with enhanced functionality

### Fixed

#### Stream I/O and Naming Conventions
- **Variable Naming**: Consistent patterns across all discrete distributions
- **Stream Input Resilience**: No crashes on malformed input for discrete distributions
- **Numerical Accuracy**: Precise computation ensures 100% correctness in tests

### Breaking Changes

**None** - All modifications maintain full backward compatibility and primarily elevate internal robustness.

### Migration Notes

No actions required from prior versions. Code continues to operate correctly with improved internal robustness. Enhanced error safety in all four discrete distributions offers better reliability for production systems.

### Quality Improvements

This release attains the next level of quality, highlighting our ongoing commitment to continuous improvement and excellence in HMM library development:
- **Systematic Upgrades**: Completing goals outlined in the Q4 roadmap of 2024
- **Continuous Refactoring**: Focus on internal, non-breaking refactoring for long-term maintainability
- **Code Quality Enforcements**: Automatized checks via clang-tidy ensure adherence to standards

---

## [2.7.0] - 2024-06-27

### Stream I/O Robustness & Code Quality Release

This release focuses on implementing robust exception handling in distribution stream input operators and establishing consistent coding standards across the library.

### Added

#### Exception Handling Infrastructure
- **Robust Stream Input Operators**: All distribution `operator>>` implementations now include comprehensive exception handling
  - Try-catch blocks around `std::stod()` calls to handle malformed input gracefully
  - Stream failbit setting on parsing errors instead of throwing exceptions
  - Consistent error recovery patterns across all 10 distributions
- **Input Validation**: Enhanced validation for boundary values and special cases
  - Proper handling of negative values, NaN, and infinity in stream parsing
  - Consistent use of library constants instead of magic numbers

#### Code Quality Standards
- **Gold Standard Checklist**: Comprehensive quality standards document for distribution implementations
  - 43-point checklist covering robustness, consistency, and maintainability
  - Guidelines for exception handling, variable naming, and documentation
  - Progressive upgrade path for bringing all distributions to gold standard
- **Variable Naming Consistency**: Standardized variable naming in stream input operators
  - Discarded tokens consistently named "token" across all distributions
  - Value tokens use meaningful names ("alpha_str", "lambda_str", etc.)
  - Enhanced code readability and maintainability

### Enhanced

#### Distribution Robustness
- **Beta Distribution**: Fixed boundary value handling to return exact 0.0 for mathematical correctness
- **Stream Parsing**: All distributions now handle malformed input gracefully without crashes
- **Error Messages**: Improved error handling with consistent stream state management
- **Constants Usage**: Proper use of `math::ZERO_DOUBLE` vs `precision::ZERO` for exact comparisons

#### Code Consistency
- **Uniform Patterns**: All stream input operators follow identical exception handling patterns
- **Naming Standards**: Consistent variable naming improves code readability across distributions
- **Documentation**: Clear standards established for future distribution implementations

### Fixed

#### Stream I/O Reliability
- **Exception Safety**: Eliminated potential crashes from malformed stream input across all distributions
- **Boundary Cases**: Fixed Beta distribution boundary value handling for Google Test compatibility
- **Input Validation**: Robust handling of edge cases in all distribution stream operators

#### Code Quality
- **Magic Numbers**: Replaced hardcoded values with proper library constants
- **Variable Naming**: Standardized naming conventions across all distribution implementations
- **Error Handling**: Consistent error recovery patterns prevent undefined behavior

### Technical Implementation

#### Exception Handling Pattern
```cpp
// Before (prone to crashes):
std::string s;
is >> s;
double value = std::stod(s);  // Could throw exception

// After (robust):
std::string value_str;
try {
    is >> token >> token >> value_str;
    double value = std::stod(value_str);
    if (is.good()) {
        distribution.setParameter(value);
    }
} catch (const std::exception& e) {
    is.setstate(std::ios::failbit);
}
```

#### Distributions Enhanced (10 Total)
- **Beta Distribution** - Added robust exception handling and fixed boundary values
- **Log-Normal Distribution** - Enhanced stream parsing with proper error handling
- **Pareto Distribution** - Added comprehensive exception handling
- **Poisson Distribution** - Implemented robust stream input parsing
- **Weibull Distribution** - Added exception safety to stream operators
- **Uniform Distribution** - Enhanced with robust error handling
- **Chi-squared Distribution** - Implemented consistent exception handling
- **Gaussian Distribution** - Added robust stream input parsing
- **Exponential Distribution** - Enhanced with exception safety
- **Gamma Distribution** - Implemented comprehensive error handling

### Test Results

#### Quality Assurance
```
Test Suite Results: 40/40 tests passing (100%)
Distribution Tests: 10/10 distributions with robust stream I/O
Code Quality: All distributions follow consistent patterns
Exception Handling: 100% coverage across stream input operators
```

#### Validation Coverage
- **Stream I/O Testing**: All distributions tested with malformed input
- **Edge Cases**: Boundary values, NaN, and infinity handling verified
- **Error Recovery**: Consistent failbit setting confirmed across all operators
- **Variable Naming**: Consistent patterns verified across all implementations

### Breaking Changes

**None** - All changes are internal implementation improvements that maintain full API compatibility.

### Migration Notes

Existing code continues to work unchanged. The improvements provide:
- More robust stream input parsing with graceful error handling
- Better error reporting through proper stream state management
- Enhanced reliability for applications using stream I/O operations

### Quality Standards

This release establishes the foundation for systematic quality improvements:
- **Gold Standard Checklist**: Clear criteria for distribution quality
- **Progressive Enhancement**: Roadmap for upgrading remaining distributions
- **Consistent Patterns**: Established templates for robust implementations
- **Maintainability**: Simplified code patterns for easier maintenance

---

## [2.6.0] - 2024-06-26

### Major Release - Boost Elimination & Benchmarking Framework

This version removes all Boost library dependencies and introduces a comprehensive benchmarking suite for validating libhmm against other HMM implementations.

### Key Accomplishments

#### Complete Boost Dependency Removal (Phase 8.3)
- **Self-Contained Library**: Now requires only C++17 standard library
- **Custom Matrix/Vector Classes**: Replaced `boost::numeric::ublas` with efficient custom implementations
  - Contiguous memory layout optimized for cache performance
  - Template-based design supporting extensible numeric types
  - SIMD-friendly memory alignment for vectorization
- **Custom XML Serialization**: Replaced `boost::serialization` with lightweight implementation
  - Compact code footprint with clear, readable output
  - Full support for all 17 distribution types and model structures
- **Build System Modernization**: Simplified CMake configuration
  - Reduced compilation time and binary size
  - Enhanced cross-platform compatibility

#### Comprehensive Benchmarking Framework
- **Multi-Library Integration**: Successfully integrated 5 HMM libraries (libhmm, HMMLib, GHMM, StochHMM, HTK)
- **Numerical Validation**: Achieved 100% numerical agreement across libraries at machine precision
- **Performance Characterization**: Established baseline performance metrics across sequence lengths from 1,000 to 1,000,000 observations
- **Compatibility Documentation**: Complete integration guides with fixes for each library

### Added

#### Core Infrastructure
- **Custom Matrix/Vector Classes** (`BasicMatrix<T>`, `BasicVector<T>`)
  - Template-based design with type aliases for clean API
  - Standard mathematical operators and efficient memory management
  - Zero external dependencies with move semantics support

- **XML Serialization System**
  - Direct XML generation with proper formatting
  - Support for all distribution types and model components
  - Human-readable output format

#### Benchmarking Suite
- **22 Benchmark Programs**: Comprehensive testing across multiple libraries and scenarios
- **7 Documentation Files**: Detailed analysis, compatibility guides, and methodology
- **Library Integration Solutions**:
  - HMMLib: Fixed C++17 template compatibility issues
  - GHMM: Resolved indexing assumptions and Python environment setup
  - StochHMM: Dynamic model file generation and format conversion
  - HTK: File I/O wrappers for speech recognition toolkit integration

### Enhanced

#### Performance & Quality
- **Memory Layout**: Optimized for better cache locality and SIMD operations
- **Compilation Speed**: Significant improvement without Boost template instantiation
- **Code Maintainability**: Clean separation of concerns and modern C++17 practices

#### Numerical Validation Results
```
Library Performance vs libhmm:
├─ GHMM:      23x faster (100% numerical agreement)
├─ HMMLib:    17-20x faster (100% numerical agreement)  
├─ HTK:       Variable performance (intentionally rounded results)
└─ StochHMM:  2x faster (100% numerical agreement)

Test Coverage: 32 test cases across 4 classic HMM problems
Numerical Accuracy: Machine precision agreement (≤1e-14)
```

### Fixed

#### Library Compatibility
- **Template Dependencies**: Fixed modern C++ template inheritance issues in HMMLib
- **API Integration**: Corrected indexing assumptions and format handling across libraries
- **Build Conflicts**: Clean separation of internal vs external dependencies

#### Repository Organization  
- **Git Configuration**: Proper `.gitignore` setup for benchmarks and build artifacts
- **Directory Structure**: Organized source code vs third-party library separation
- **CMake Integration**: Removed generated `Testing/` directory from version control

### Performance Analysis

#### Key Insights
- **Numerical Correctness**: libhmm maintains perfect accuracy across all test scenarios
- **Dependency Independence**: Unique among tested libraries for complete self-containment
- **Modern Architecture**: Contemporary C++17 codebase with extensible design
- **Performance Position**: Establishes baseline for future optimization work

### Technical Implementation

```cpp
// Migration from Boost to custom implementation
// Before:
#include <boost/numeric/ublas/matrix.hpp>
using Matrix = boost::numeric::ublas::matrix<double>;

// After:
#include "libhmm/common/common.h"
using Matrix = libhmm::BasicMatrix<double>;
```

### Breaking Changes

**None** - Full API compatibility maintained while removing dependencies.

### Migration Notes

Existing code works unchanged:
```cpp
Matrix transition_matrix(2, 2);
transition_matrix(0, 1) = 0.3;
auto hmm = std::make_unique<Hmm>(num_states);
```

Benefits are automatic:
- Faster compilation without Boost dependencies
- Smaller binaries and easier deployment
- Enhanced performance through optimized memory layout

### Dependencies

**Before (v2.5.0)**: C++17, CMake 3.15+, Boost Libraries  
**After (v2.6.0)**: C++17, CMake 3.15+ only

### Future Development

This release establishes:
- Foundation for advanced SIMD optimization
- Benchmarking framework for measuring improvements
- Clean architecture for extending distributions and algorithms
- Validation infrastructure for continuous development

---

## [2.5.0] - 2024-06-25

### 🎯 Calculator Modernization & Benchmark Validation Release

This release focuses on AutoCalculator system validation, benchmark suite modernization, and significant performance improvements through validated SIMD optimizations.

### Added

#### AutoCalculator System Validation
- **Complete API Modernization**: All calculator code migrated to current `libhmm::forwardbackward::AutoCalculator` and `libhmm::viterbi::AutoCalculator` APIs
- **Intelligent Algorithm Selection**: Enhanced calculator selection with detailed performance rationale
  - Automatic Scaled-SIMD selection for appropriate problem sizes
  - Smart fallback strategies based on problem characteristics
  - Numerical stability prioritization for long sequences (≥1000 observations)
- **Performance Transparency**: Calculator selection rationale now visible in debug output

#### Benchmark Suite Modernization
- **Algorithm Performance Benchmark**: Updated to use AutoCalculator APIs with enhanced debug output
- **Classic Problems Benchmark**: Comprehensive 16-test validation suite
  - 4 Classic HMM Problems: Dishonest Casino, Weather Model, CpG Island Detection, Speech Recognition
  - 4 Sequence Lengths: 100, 500, 1000, 2000 observations per problem
  - Both Forward-Backward and Viterbi algorithm validation
- **API Compatibility**: All benchmark code uses current libhmm API patterns

#### Performance Validation Infrastructure
- **Numerical Accuracy Verification**: 100% accuracy maintained across all 16 benchmark comparisons
- **Performance Measurement**: Reliable timing infrastructure for ongoing optimization work
- **Calculator Selection Validation**: Verified AutoCalculator system works correctly across all problem sizes

### Enhanced

#### Performance Improvements
- **Major Performance Gains**: ~17x improvement from previous performance gaps
  - Forward-Backward: Reduced from ~540x to 31.3x gap vs HMMLib (average)
  - Viterbi: Improved to 20.9x gap vs HMMLib (average)
  - Range: 18x-47x depending on problem size and algorithm type
- **SIMD Effectiveness**: Clear evidence that SIMD optimizations provide substantial performance benefits
- **Algorithm Maturity**: AutoCalculator system selecting appropriate algorithms effectively

#### API Modernization
- **Namespace Consolidation**: Migrated from deprecated `libhmm::calculators` to current `libhmm::forwardbackward` and `libhmm::viterbi` namespaces
- **Simplified Calculator Usage**: Removed manual SIMD vs scalar selection logic - now handled automatically
- **Enhanced Debug Information**: Detailed calculator selection explanations with performance predictions

#### Code Quality
- **Future-Ready Infrastructure**: Benchmark system ready for ongoing optimization work
- **Maintainable Codebase**: Simplified calculator instantiation while maintaining full functionality
- **Development Workflow**: Reliable performance measurement tools for continuous improvement

### Fixed

#### Benchmark Compatibility
- **Include Path Updates**: Updated from old `calculator_traits.h` to new `forward_backward_traits.h` and `viterbi_traits.h`
- **API Deprecation**: Removed usage of deprecated calculator classes and selection patterns
- **Build System**: All benchmarks compile successfully with current API

#### Performance Measurement
- **Accurate Timing**: Validated benchmark timing infrastructure provides consistent results
- **Numerical Stability**: Perfect agreement between libhmm and HMMLib across all test cases (≤ 2e-10 precision)
- **Calculator Selection**: Verified AutoCalculator chooses optimal algorithms based on problem characteristics

### Performance Analysis

#### Detailed Performance Breakdown
- **Dishonest Casino**: 15-25x gap (Forward-Backward), 10-19x gap (Viterbi)
- **Weather Model**: 20-35x gap (Forward-Backward), 19-20x gap (Viterbi)
- **CpG Island Detection**: 18-37x gap (Forward-Backward), 18-24x gap (Viterbi)
- **Speech Recognition**: 39-47x gap (Forward-Backward), 18-34x gap (Viterbi)

#### Key Performance Insights
- **Trend Analysis**: Performance gap decreases with larger, more complex problems
- **SIMD Impact**: ScaledSIMD calculator correctly selected for largest problems showing optimization benefits
- **Remaining Opportunity**: 20-31x gap suggests room for further architectural improvements
- **Optimization Evidence**: Clear demonstration that SIMD work is providing substantial real-world gains

### Validation Results

#### Numerical Accuracy
```
Successful comparisons: 16/16
Numerical matches: 16/16 (100.0%)
Viterbi likelihood differences: ≤ 2e-10 (machine precision)
```

#### Calculator Selection Examples
- Small problems (100 obs): "Predicted performance: 1.65x baseline"
- Medium problems (500 obs): "Predicted performance: 3.125x baseline"
- Large problems (1000+ obs): "Provides numerical stability for long sequences"

### Technical Specifications

#### Updated Files
- **algorithm_performance_benchmark.cpp**: Modernized to AutoCalculator API
- **classic_problems_benchmark.cpp**: Comprehensive test suite with current API
- **Phase 8 Documentation**: Updated to reflect current performance improvements and validation status

#### Infrastructure Improvements
- **API Consistency**: All calculator usage follows current best practices
- **Debug Visibility**: Calculator selection rationale available for development and optimization
- **Measurement Reliability**: Validated benchmark infrastructure for ongoing performance work

### Breaking Changes

None - this release maintains full backward compatibility while significantly improving performance measurement and validation capabilities.

### Migration Notes

Existing code continues to work unchanged. The improvements provide:
- Better performance through validated SIMD optimizations
- More intelligent algorithm selection with transparency
- Reliable benchmark infrastructure for measuring future improvements

### Future Work Foundation

This release establishes a solid foundation for:
1. **Algorithm Optimization**: Using reliable benchmark system to measure improvements
2. **Performance Profiling**: Identifying specific bottlenecks with confidence in measurement accuracy
3. **SIMD Enhancement**: Expanding vectorization with validated numerical stability checks

---

## [2.4.0] - 2024-06-24

### 🧹 Include Consolidation & Numerical Stability Release

This release focuses on code maintainability, numerical robustness, and developer experience improvements through header consolidation and comprehensive stability infrastructure.

### Added

#### Numerical Stability Infrastructure
- **NumericalSafety Class**: Comprehensive finite value validation and safe mathematical operations
  - `safeLog()` and `safeExp()` with underflow/overflow protection
  - Probability range validation and automatic normalization
  - Container validation for matrices and vectors
- **ConvergenceDetector**: Adaptive convergence detection with oscillation and stagnation detection
- **AdaptivePrecision**: Dynamic tolerance adjustment based on problem characteristics
- **ErrorRecovery**: Multiple recovery strategies (STRICT, GRACEFUL, ROBUST, ADAPTIVE)
- **NumericalDiagnostics**: Real-time health monitoring with actionable recommendations

#### Trainer Traits System
- **Compile-time Type Safety**: Distribution compatibility checking at build time
- **Template Metaprogramming**: Modern C++17 type traits with `constexpr` and SFINAE
- **Trainer Selection**: Automatic algorithm selection based on distribution capabilities
- **Zero Runtime Overhead**: All type checking resolved during compilation

#### Development Infrastructure
- **Internal Documentation**: Organized development docs in `.dev-docs/` (gitignored)
- **Future Work Parking Lot**: Comprehensive roadmap for v2.x and v3.x development
- **Phase Documentation**: Complete modernization history and rationale

### Enhanced

#### Include Structure Modernization
- **Umbrella Header Consolidation**: Replaced 70+ individual distribution includes with single `distributions.h`
- **Consistent Architecture**: Unified include pattern across 11 core files
- **Reduced Maintenance Overhead**: Single point of distribution header management
- **Build Efficiency**: Improved incremental compilation performance

#### Code Quality
- **Maintainability**: Significantly easier to add new distributions
- **Readability**: Cleaner, more professional header structure
- **Standards Compliance**: Aligned with C++ umbrella header best practices
- **Documentation**: Clear dependency relationships and API organization

#### Testing Infrastructure
- **Extended Test Suite**: 31 test suites (up from 28)
- **Numerical Stability Tests**: 24 new tests for edge case handling
- **Trainer Traits Tests**: 12 new tests for compile-time type safety
- **Comprehensive Coverage**: 100% pass rate with zero regressions

### Fixed

#### Robustness Improvements
- **Edge Case Handling**: Comprehensive protection against NaN, infinity, and underflow
- **Training Stability**: Enhanced convergence detection prevents infinite loops
- **Error Recovery**: Graceful handling of degenerate data and numerical issues
- **Memory Safety**: Continued adherence to RAII principles

### Performance Improvements

- **Build Times**: Potential improvement through optimized include structure
- **Runtime Stability**: Proactive numerical issue detection and correction
- **Zero Overhead**: Type safety checking with no runtime cost
- **Adaptive Precision**: Dynamic adjustment based on problem characteristics

### Technical Specifications

#### Files Modified
- **Header Files**: 4 core library headers consolidated
- **Test Files**: 7 test files with simplified includes
- **Lines Reduced**: 70+ redundant include lines eliminated
- **Maintainability**: Single point of distribution header management

#### New Infrastructure
- **Numerical Constants**: Carefully tuned for different scenarios
- **Error Recovery Strategies**: 4 different approaches based on requirements
- **Diagnostic Capabilities**: Real-time numerical health assessment
- **Future-Ready**: Foundation for advanced trainer selection (Phase 6)

### Breaking Changes

None - this release maintains full backward compatibility while significantly improving maintainability.

### Migration Notes

Existing code works unchanged. The improvements are transparent:

```cpp
// Existing includes still work
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/poisson_distribution.h"

// But now you can simply use:
#include "libhmm/distributions/distributions.h"  // All distributions available
```

### Dependencies

- **C++17 Compatible Compiler**: GCC 7+, Clang 6+, MSVC 2017+
- **CMake**: 3.15 or later
- **Boost Libraries**: For matrix operations
- **Platform**: macOS, Linux, Unix-like systems

---

## [2.3.0] - 2024-06-23

### 🚀 Major Feature Release - Advanced Statistical Distributions & Performance Optimization

This release adds powerful new statistical distributions and comprehensive performance optimizations, making libhmm suitable for advanced statistical modeling and high-performance applications.

### Added

#### New Statistical Distributions
- **Student's t-distribution**: Complete implementation for robust statistical modeling
  - Location (μ), scale (σ), and degrees of freedom (ν) parameters
  - Heavy-tailed distribution perfect for financial modeling and outlier-robust analysis
  - MLE parameter fitting and comprehensive validation
- **Chi-squared distribution**: Essential for goodness-of-fit testing and statistical analysis
  - Degrees of freedom parameter
  - Used in hypothesis testing and categorical data analysis
  - Efficient implementation with numerical stability

#### Performance & Optimization Framework
- **SIMD Support**: Platform-specific vectorized operations
  - AVX support for Intel/AMD processors
  - SSE2 fallback for older x86 systems
  - ARM NEON support for Apple Silicon and ARM processors
  - Automatic CPU feature detection and optimization selection
- **Thread Pool**: Modern C++17 concurrent processing
  - Work-stealing algorithm for optimal load balancing
  - Thread affinity support for NUMA systems
  - Configurable thread count with automatic detection
- **Optimized Forward-Backward Calculator**: 
  - SIMD-accelerated matrix-vector operations
  - Cache-optimized memory layouts
  - Blocked algorithms for large matrices
  - Up to 3x performance improvement on compatible hardware
- **Calculator Traits System**: 
  - Automatic algorithm selection based on problem size
  - Runtime optimization based on CPU capabilities
  - Performance profiling and reporting

#### Advanced Examples
- **Robust Financial HMM**: Demonstrates Student's t-distribution for modeling heavy-tailed financial returns
- **Statistical Process Control HMM**: Quality control monitoring using comprehensive statistical methods

#### Infrastructure Improvements
- **Distribution Traits**: Compile-time distribution analysis for type safety
- **Convenience Headers**: `distributions.h` umbrella header for easy inclusion
- **Memory Management**: Aligned allocators for SIMD operations
- **CPU Detection**: Runtime CPU feature detection and optimization

### Enhanced

#### Build System
- **CMake Policy Compliance**: Fixed FindBoost deprecation warnings
- **Cross-Platform Optimization**: Platform-specific SIMD compilation flags
- **Zero-Warning Builds**: Eliminated all compiler warnings

#### Testing Framework
- **Comprehensive Unit Tests**: Full coverage for new distributions
- **Performance Testing**: Benchmarking and optimization validation
- **Edge Case Validation**: Robust handling of boundary conditions
- **Integration Testing**: Cross-distribution compatibility verification

#### Parser & I/O
- **Multi-line Format Support**: Enhanced HMM stream parser for complex distribution outputs
- **Token-based Parsing**: Robust parsing for all distribution types
- **Serialization Consistency**: Reliable round-trip serialization for all distributions

### Fixed

#### Critical Issues
- **Stream Parser**: Fixed "stod: no conversion" errors in HMM I/O
- **Gaussian Distribution Parser**: Corrected multi-line format token consumption
- **Memory Safety**: Enhanced validation for edge cases and invalid inputs

#### Code Quality
- **Unused Variables**: Eliminated all unused variable warnings
- **Deprecated Functions**: Replaced deprecated API calls with modern equivalents
- **Exception Handling**: Improved error messages and exception safety

### Performance Improvements

- **Matrix Operations**: Up to 3x speedup with SIMD vectorization
- **Memory Access**: Cache-optimized layouts reduce memory latency
- **Parallel Processing**: Multi-core training algorithms for large datasets
- **Algorithm Selection**: Automatic optimization based on problem characteristics

### Technical Specifications

#### Supported Distributions (17 total)
**Discrete**: Discrete, Poisson, Binomial, Negative Binomial  
**Continuous**: Gaussian, Gamma, Exponential, Log-Normal, Pareto, Beta, Weibull, Uniform, **Student's t**, **Chi-squared**

#### SIMD Support
- **Intel/AMD**: AVX, SSE2 instruction sets
- **ARM**: NEON instruction set (Apple Silicon, ARM processors)
- **Automatic Detection**: Runtime CPU feature detection
- **Fallback**: Scalar implementations for unsupported hardware

#### Threading
- **Work-Stealing Thread Pool**: Optimal load distribution
- **NUMA Awareness**: Thread affinity for multi-socket systems
- **Scalable Design**: Efficient scaling from 1 to 64+ cores

### Breaking Changes

None - this release maintains full backward compatibility while adding new features.

### Migration Notes

All existing code continues to work unchanged. New features are opt-in:

```cpp
// New distributions
auto studentT = std::make_unique<StudentTDistribution>(3.0, 0.0, 1.0);
auto chiSquared = std::make_unique<ChiSquaredDistribution>(5.0);

// Performance optimization (automatic)
OptimizedForwardBackwardCalculator calc(hmm.get(), observations);
```

### Dependencies

- **C++17 Compatible Compiler**: GCC 7+, Clang 5+, MSVC 2017+
- **CMake**: 3.15 or later
- **Boost Libraries**: For matrix operations
- **Platform**: macOS, Linux, Unix-like systems

---

## [2.0.0] - 2024-06-21

### 🎉 Major Release - C++17 Modernization

This release represents a complete modernization of the libhmm library with critical bug fixes and enhanced memory safety.

### Added
- **C++17 Standard Compliance**: Full modernization to C++17 standards
- **Smart Pointer Memory Management**: Replaced all raw pointers with `std::unique_ptr` and `std::shared_ptr`
- **Modern CMake Build System**: Enhanced CMake configuration with proper target management
- **Comprehensive Test Suite**: Expanded unit tests with better coverage
- **Enhanced Type Safety**: Explicit type casting and bounds checking throughout
- **Memory Safety**: RAII principles implemented consistently
- **Modern Loop Constructs**: Range-based for loops and auto type deduction

### Fixed
- **CRITICAL**: Fixed segmentation fault in `ViterbiTrainer::train()` 
  - **Root Cause**: Double ownership of `ProbabilityDistribution*` objects in `viterbi_trainer.cpp:124`
  - **Solution**: Modified distributions in place rather than reassigning ownership
  - **Impact**: ViterbiTrainer now runs successfully without crashes
- **Memory Leaks**: Eliminated all raw pointer memory leaks through smart pointer adoption
- **Deprecated Functions**: Updated `prepare_hmm()` to `prepareTwoStateHmm()` calls
- **Build Warnings**: Resolved all compilation warnings in C++17 mode

### Changed
- **API Modernization**: 
  - `int main(void)` → `int main()`
  - Replaced global `using namespace` with selective imports
  - Modern function parameter styles
- **Memory Management**: 
  - `new`/`delete` → `std::make_unique<>()`
  - Raw pointers → Smart pointers throughout
- **Loop Syntax**: 
  - C-style loops → Modern C++17 range-based loops
  - Manual indexing → Iterator-based approaches where appropriate
- **Error Handling**: Enhanced exception safety and error reporting

### Removed
- **Raw Pointer Usage**: Eliminated unsafe manual memory management
- **C-Style Constructs**: Removed outdated C-style function signatures
- **Memory Unsafe Patterns**: Cleaned up potential double-free scenarios

### Performance
- **Memory Efficiency**: Smart pointers provide better memory locality and automatic cleanup
- **Compilation Speed**: Modern C++17 features enable better compiler optimizations
- **Runtime Safety**: Bounds checking and type safety prevent runtime errors

### Technical Details

#### ViterbiTrainer Bug Fix
```cpp
// BEFORE (Buggy - caused segfault):
ProbabilityDistribution* pdist = hmm_->getProbabilityDistribution(i);
pdist->fit(clusterObservations);
hmm_->setProbabilityDistribution(i, pdist);  // Double ownership!

// AFTER (Fixed):
ProbabilityDistribution* pdist = hmm_->getProbabilityDistribution(i);
pdist->fit(clusterObservations);
// No reassignment needed - HMM already owns the distribution
```

#### Smart Pointer Migration
```cpp
// BEFORE:
Hmm* hmm = new Hmm(2);
// ... use hmm
delete hmm;

// AFTER:
auto hmm = std::make_unique<Hmm>(2);
// Automatic cleanup when out of scope
```

### Migration Guide

For users upgrading from v1.x:

1. **Update Compiler**: Ensure C++17 compatible compiler (GCC 7+, Clang 6+, MSVC 2017+)
2. **Memory Management**: Replace any direct `new`/`delete` with smart pointers
3. **Function Calls**: Update `prepare_hmm()` to `prepareTwoStateHmm()`
4. **Build System**: Use CMake for modern builds (legacy Makefile still supported)

### Compatibility
- **Backwards Compatible**: API remains largely unchanged
- **ABI Breaking**: Memory management changes require recompilation
- **C++17 Required**: No longer compatible with pre-C++17 compilers

---

## [1.0.0] - Previous Version

### Features
- Basic HMM implementation
- Viterbi training and decoding
- Multiple probability distributions
- Forward-Backward algorithms
- File I/O support

### Known Issues (Fixed in 2.0.0)
- Segmentation faults in ViterbiTrainer
- Memory leaks from raw pointer usage
- Non-standard C++ constructs
- Build warnings in modern compilers
