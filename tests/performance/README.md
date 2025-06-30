# Development Performance Tests

This directory contains ad-hoc development tests and benchmarks for performance analysis and debugging. These are not part of the main test suite but are useful for development and optimization work.

## Naming Convention

- **Test suite tests**: Begin with `test_` (e.g., `test_performance.cpp` in the main tests)
- **Development tools**: Use informal descriptive names (e.g., `benchmark_parallel`, `analyze_overhead`)

## Available Tools

### 1. `benchmark_parallel`
**Purpose**: Comprehensive performance benchmark comparing different calculator implementations
**Features**:
- Tests reference, scaled-SIMD, and log-SIMD calculators
- Multiple problem sizes from small (sequential) to large (parallel)
- Reports speedup ratios and performance metrics
- Currently excludes 1024-state test for faster development iterations

**Usage**:
```bash
cd build
DYLD_LIBRARY_PATH=. ./tests/benchmark_parallel
```

### 2. `debug_parallel`
**Purpose**: Debug tool to understand parallel vs sequential execution paths
**Features**:
- Shows which path (SEQ/PAR) is chosen for different problem sizes
- Measures overhead of thread pool parallelization
- Useful for tuning parallel thresholds

**Usage**:
```bash
cd build
DYLD_LIBRARY_PATH=. ./tests/debug_parallel
```

### 3. `analyze_overhead`
**Purpose**: Microbenchmark to analyze thread pool overhead
**Features**:
- Compares sequential vs parallel execution times
- Measures overhead in microseconds
- Helps determine optimal grain sizes and thresholds

**Usage**:
```bash
cd build
DYLD_LIBRARY_PATH=. ./tests/analyze_overhead
```

### 4. `parallel_calculator_assessment`
**Purpose**: Assess the automatic calculator selection with parallel support
**Features**:
- Tests automatic selection of optimal calculator
- Shows which optimizations are being used
- Verifies parallel calculator integration

**Usage**:
```bash
cd build
DYLD_LIBRARY_PATH=. ./tests/parallel_calculator_assessment
```

### 5. `parallel_constants_tuning`
**Purpose**: Empirical testing of parallel constants for optimization
**Features**:
- Tests various parallelization thresholds and grain sizes
- Compares base-2 vs non-base-2 constants for performance
- Measures speedup across different problem sizes (128-1024 states)
- Analyzes cache alignment and SIMD efficiency impact

**Usage**:
```bash
cd build
DYLD_LIBRARY_PATH=. ./tests/parallel_constants_tuning
```

## CMake Integration

These development tools are built automatically but are not included in the main test suites. They are defined in the main `CMakeLists.txt` under the "Development Benchmarking and Debug Tests" section:

```cmake
# Development Benchmarking and Debug Tests
add_executable(analyze_overhead tests/performance/analyze_overhead.cpp)
target_link_libraries(analyze_overhead hmm)

add_executable(benchmark_parallel tests/performance/benchmark_parallel.cpp)
target_link_libraries(benchmark_parallel hmm)

add_executable(debug_parallel tests/performance/debug_parallel.cpp)
target_link_libraries(debug_parallel hmm)

add_executable(parallel_calculator_assessment tests/performance/parallel_calculator_assessment.cpp)
target_link_libraries(parallel_calculator_assessment hmm)

add_executable(parallel_constants_tuning tests/performance/parallel_constants_tuning.cpp)
target_link_libraries(parallel_constants_tuning hmm)
```

## Current Performance Results

As of the latest optimizations:

**Scaled-SIMD Calculator**:
- Small problems (≤128 states): ~3.3x speedup (sequential execution)
- Large problems (512+ states): ~7-10x speedup (parallel execution)

**Log-SIMD Calculator**:
- Small problems: ~0.6-0.8x (slower due to log-space overhead)
- Large problems (512 states): ~2.0x speedup (parallel execution)

## Development Notes

1. **Parallel Thresholds**: Optimized to 512 states minimum for parallel execution (base-2 multiple)
2. **Grain Sizes**: Optimized to 64 for calculator operations, 32 for simple operations (base-2 multiples)
3. **Thread Count**: Automatically detected (typically 4 on development machines)
4. **Base-2 Optimization**: Constants tuned to base-2 multiples for better cache alignment and SIMD efficiency

## Adding New Development Tools

To add a new development tool:

1. Create a `.cpp` file in `tests/performance/` with a descriptive name (avoid `test_` prefix)
2. Add the executable and link to the main `CMakeLists.txt`
3. Document it in this README
4. Ensure it includes proper includes for the libhmm library

## Relationship to Main Test Suite

The main test suite includes formal tests like `test_performance.cpp` which are part of the automated testing infrastructure. These development tools are separate utilities for:

- Performance benchmarking during optimization work
- Debugging parallel execution paths
- Analyzing threading overhead
- Assessing calculator selection algorithms

They complement but do not replace the formal test suite.

## Parallel Constants Optimization (June 2025)

We conducted empirical testing of parallelization constants to optimize performance. Key findings:

**Methodology**:
- Tested various threshold and grain size combinations
- Compared base-2 multiples vs original constants
- Measured performance across problem sizes: 128, 256, 512, 1024 states
- Focused on Scaled-SIMD and Log-SIMD calculator performance

**Results**:
- **Base-2 multiples showed 10-12% performance improvement** for large problems (1024 states)
- **Optimal configuration**: 512 threshold, 64 grain size for calculator operations
- **Cache alignment benefits**: Base-2 multiples improve SIMD register efficiency
- **Load balancing**: Proper grain sizes reduce thread pool overhead

**Updated Constants** (in `include/libhmm/performance/parallel_constants.h`):
- `MIN_STATES_FOR_CALCULATOR_PARALLEL`: 500 → 512
- `CALCULATOR_GRAIN_SIZE`: 50 → 64  
- `MIN_STATES_FOR_EMISSION_PARALLEL`: 200 → 256
- `SIMPLE_OPERATION_GRAIN_SIZE`: 25 → 32

**Future Work**:
We may investigate a **dynamic tuning mechanism** in the future that could:
- Auto-detect optimal constants based on CPU architecture
- Adapt thresholds based on actual thread pool performance
- Provide runtime calibration for different workload patterns
- Support heterogeneous computing environments

This would allow the library to automatically optimize for different hardware configurations without requiring manual tuning.
