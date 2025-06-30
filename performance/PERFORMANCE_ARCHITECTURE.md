# Performance Architecture

This document outlines the architecture for performance optimization within the libhmm project. Our goal is to leverage platform-specific SIMD instructions and parallel execution to optimize performance across a variety of hardware while maintaining clean, maintainable code.

## Overview

The performance architecture was designed to solve the circular dependency and platform detection issues that arose during development. It provides a clear hierarchy that enables advanced optimizations without compromising code maintainability.

## Hierarchical Design

The design is structured into four levels to ensure clear separation of responsibilities and to prevent circular dependencies.

### Level 1: Platform Detection (Foundation)
- **Files**: 
  - `performance/simd_platform.h` - SIMD instruction set detection
  - `performance/parallel_execution.h` - C++17 parallel execution policy detection
- **Purpose**: Pure platform detection with zero dependencies
- **Features**:
  - Compile-time detection of AVX, SSE2, ARM NEON
  - Platform-specific alignment requirements
  - Thread pool availability detection
- **Example Constants**:
  ```cpp
  #define LIBHMM_HAS_AVX 1
  constexpr std::size_t SIMD_ALIGNMENT = 32;  // AVX alignment
  constexpr std::size_t DOUBLE_SIMD_WIDTH = 4; // AVX doubles per register
  ```

### Level 2: Performance Infrastructure (Core)
- **Files**: 
  - `performance/simd_support.h` - SIMD operations and aligned allocators
  - `performance/parallel_constants.h` - Parallel processing thresholds
  - `common/common.h` - Basic mathematical constants
- **Purpose**: Provides centralized performance constants and utilities
- **Dependency**: Only depends on Level 1
- **Key Features**:
  - Platform-adaptive constants based on detected capabilities
  - SIMD operation wrappers with fallbacks
  - Parallel processing thresholds optimized for different workload types
- **Example Constants**:
  ```cpp
  namespace performance::parallel {
      constexpr std::size_t MIN_STATES_FOR_CALCULATOR_PARALLEL = 512;
      constexpr std::size_t CALCULATOR_GRAIN_SIZE = 64;
  }
  namespace constants::simd {
      constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;
      constexpr std::size_t MAX_BLOCK_SIZE = 32;
  }
  ```

### Level 3: Optimized Classes (Implementation)
- **Files**: 
  - `common/optimized_vector.h` - High-performance vector operations
  - `common/optimized_matrix.h` - High-performance matrix operations  
  - `common/optimized_matrix3d.h` - High-performance 3D matrix operations
- **Purpose**: High-performance data structures utilizing SIMD and parallel operations
- **Dependency**: Uses Level 2 for constants and Level 1 detection macros
- **Key Features**:
  - Automatic SIMD vs serial selection based on data size
  - Parallel execution for large datasets
  - Cache-friendly memory layouts
  - Backward compatibility with basic classes
- **Example Usage**:
  ```cpp
  // In OptimizedVector.h
  static constexpr std::size_t SIMD_BLOCK_SIZE = constants::simd::DEFAULT_BLOCK_SIZE;
  static constexpr std::size_t PARALLEL_THRESHOLD = performance::parallel::MIN_WORK_PER_THREAD;
  
  T sum() const {
      if (data_.size() > PARALLEL_THRESHOLD) {
          return sum_parallel();
      } else {
          return sum_serial();
      }
  }
  ```

### Level 4: Application Classes (Integration)
- **Usage**: 
  - Calculator classes (Viterbi, Forward-Backward)
  - Training algorithms (Baum-Welch)
  - Other HMM components
- **Purpose**: Integrates optimized classes into higher-level HMM functionality
- **Dependency**: Uses Level 3 for high-performance operations
- **Benefits**:
  - Transparent performance improvements
  - No API changes required for existing code
  - Automatic platform optimization
- **Example Integration**:
  ```cpp
  class ScaledSIMDForwardBackwardCalculator {
      OptimizedMatrix<double> alpha_;    // Automatically optimized
      OptimizedVector<double> scaling_;  // SIMD operations
      // Uses performance::parallel::MIN_STATES_FOR_CALCULATOR_PARALLEL
  };
  ```

## Core Principles

1. **Single Source of Truth**: All performance constants are defined in the performance module.
2. **No Upward Dependencies**: Lower levels never include higher levels.
3. **Consistent Constants Access**: Classes use `performance::` and `constants::` consistently.
4. **Automatic Platform Adaptation**: Determines optimal settings at compile-time for any platform.

## Benefits

- **Modular Design**: Easy to test and maintain.
- **Performance Scaling**: Automatically chooses SIMD/parallel based on data size.
- **Platform Portability**: Adapts to use native instructions on macOS, Intel, etc.
- **Maintainability**: Isolated changes at each level.
- **Testing**: Independent testing at each hierarchy level.

## Next Steps

1. Implement full SIMD infrastructure in `simd_support.cpp`.
2. Replace serial implementations with SIMD-optimized versions.
3. Integrate into calculators and trainers.
4. Benchmark improvements across platforms.

This architecture ensures that libhmm can harness the full power of modern hardware for performance-critical operations efficiently and maintainably.
