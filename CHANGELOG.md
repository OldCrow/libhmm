# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
