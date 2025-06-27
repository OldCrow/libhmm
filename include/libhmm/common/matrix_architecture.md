# libhmm Matrix Architecture Design

## Overview

The libhmm library provides a comprehensive, high-performance matrix and vector system designed for numerical computing in Hidden Markov Model algorithms. The architecture provides both basic and optimized implementations for 1D (Vector), 2D (Matrix), and 3D (Matrix3D) data structures.

## Design Philosophy

### 1. **Performance Hierarchy**
- **Basic Classes**: Simple, safe, API-compatible implementations
- **Optimized Classes**: High-performance SIMD and parallel implementations
- **Automatic Selection**: Smart selection based on problem size and characteristics

### 2. **Memory Layout Optimization**
- **Contiguous Memory**: All optimized classes use flat, contiguous storage
- **Cache-Friendly**: Row-major ordering optimized for CPU cache utilization
- **SIMD-Aligned**: Memory alignment for efficient vectorized operations

### 3. **API Consistency**
- **Backward Compatibility**: Drop-in replacements for Basic classes
- **Consistent Naming**: Uniform method names across all dimensions
- **Type Safety**: Template-based with proper type aliases

## Class Hierarchy

### 1D: Vector Classes

#### `BasicVector<T>` (basic implementation)
- **Purpose**: Simple, safe vector operations
- **Memory**: std::vector-based, contiguous
- **Features**: Basic arithmetic, bounds checking, iterator support
- **Use Cases**: Small vectors, development, debugging

#### `OptimizedVector<T>` (high-performance)
- **Purpose**: SIMD-optimized numerical operations
- **Memory**: Contiguous with SIMD alignment considerations
- **Features**: 
  - SIMD-optimized arithmetic (add, subtract, multiply, dot product)
  - Parallel execution for large vectors (>1000 elements)
  - Automatic fallback to scalar operations
  - Advanced mathematical functions (norm, normalize, Hadamard product)
- **Use Cases**: Large-scale numerical computations, training algorithms

### 2D: Matrix Classes

#### `BasicMatrix<T>` (basic implementation)
- **Purpose**: General-purpose 2D matrix operations
- **Memory**: Contiguous row-major storage
- **Features**: Element access, basic arithmetic, row/column operations
- **Use Cases**: Small matrices, API compatibility, simple operations

#### `OptimizedMatrix<T>` (high-performance)
- **Purpose**: SIMD and cache-optimized matrix operations
- **Memory**: Contiguous row-major with cache-friendly blocking
- **Features**:
  - SIMD-optimized element-wise operations
  - Cache-blocked matrix multiplication and transpose
  - Parallel execution for large matrices (>10k elements)
  - Advanced linear algebra (matrix-vector multiply, Hadamard operations)
  - Row/column extraction as OptimizedVector objects
- **Use Cases**: Large matrices, performance-critical algorithms, linear algebra

### 3D: Matrix3D Classes

#### `Matrix3D<T>` (basic implementation)
- **Purpose**: Simple 3D tensor operations
- **Memory**: vector<vector<vector<T>>> - NON-CONTIGUOUS (performance issue)
- **Features**: Bounds-checked access, basic operations
- **Use Cases**: Small 3D data, debugging, API compatibility
- **⚠️ Performance Warning**: Should be avoided for large data due to memory fragmentation

#### `OptimizedMatrix3D<T>` (high-performance)
- **Purpose**: High-performance 3D tensor operations
- **Memory**: Flat contiguous storage with row-major indexing
- **Features**:
  - Optimal cache performance with flat memory layout
  - SIMD-optimized operations where applicable
  - Parallel execution for large tensors
  - 2D slicing support for efficient submatrix operations
  - Overflow protection and bounds checking
- **Use Cases**: Large 3D tensors, training algorithms (xi/gamma matrices), performance-critical code

## Performance Characteristics

### Automatic Optimization Selection

```cpp
// Vector operations automatically choose implementation based on size
OptimizedVector<double> vec(1000);  // Uses parallel/SIMD for large operations
vec += other_vec;  // Automatically SIMD-optimized

// Matrix operations automatically choose blocking strategy
OptimizedMatrix<double> mat(500, 500);  // 250k elements
auto result = mat.multiply(other_mat);  // Uses cache-blocked multiplication
```

### SIMD Thresholds
- **Vector SIMD**: >8 elements
- **Matrix SIMD**: >8 elements  
- **Parallel execution**: >1000 elements (vectors), >10k elements (matrices)

### Cache Optimization
- **Block sizes**: 64 elements (configurable via constants)
- **Memory alignment**: 32-byte aligned for AVX operations
- **Prefetching**: Automatic with modern compilers

## Usage Guidelines

### When to Use Basic Classes
- **Small data structures** (<100 elements)
- **Development and debugging** (bounds checking always enabled)
- **API compatibility** with existing code
- **Simple operations** without performance requirements

### When to Use Optimized Classes
- **Large data structures** (>100 elements)
- **Performance-critical code paths**
- **Training algorithms** that process large datasets
- **Mathematical computations** requiring high throughput

### Type Selection Guide

```cpp
// Small vectors/matrices - use Basic classes
BasicVector<double> small_vec(10);
BasicMatrix<double> small_mat(5, 5);

// Large vectors/matrices - use Optimized classes  
OptimizedVector<double> large_vec(10000);
OptimizedMatrix<double> large_mat(500, 500);

// 3D tensors - ALWAYS use Optimized for performance
OptimizedMatrix3D<double> tensor(100, 100, 50);  // ✅ Good
Matrix3D<double> bad_tensor(100, 100, 50);       // ❌ Poor performance
```

## Memory Layout Details

### 2D Matrix Indexing (Row-Major)
```
Matrix[i][j] → data[i * cols + j]

For 3x4 matrix:
[0,0] [0,1] [0,2] [0,3]   →   [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11]
[1,0] [1,1] [1,2] [1,3]
[2,0] [2,1] [2,2] [2,3]
```

### 3D Matrix Indexing (Row-Major)
```
Matrix3D[i][j][k] → data[i * (y*z) + j * z + k]

Pre-computed stride: yz_stride = y * z
Optimized indexing: data[i * yz_stride + j * z + k]
```

## Integration with libhmm

### Type Aliases in common.h
```cpp
// Current aliases (basic implementations)
using Matrix = BasicMatrix<Observation>;
using Vector = BasicVector<Observation>;

// Proposed optimized aliases
using OptMatrix = OptimizedMatrix<Observation>;
using OptVector = OptimizedVector<Observation>;
using OptMatrix3D = OptimizedMatrix3D<double>;  // For training algorithms
```

### Gradual Migration Strategy
1. **Phase 1**: Keep existing Basic classes for compatibility
2. **Phase 2**: Introduce Optimized classes for new performance-critical code
3. **Phase 3**: Gradually migrate calculators and trainers to Optimized classes
4. **Phase 4**: Benchmark and validate performance improvements

## Performance Benchmarks

### Expected Performance Improvements
- **Vector operations**: 2-4x speedup with SIMD (float/double arithmetic)
- **Matrix multiplication**: 3-10x speedup with cache blocking + SIMD
- **3D operations**: 5-50x speedup (contiguous vs fragmented memory)
- **Large data parallel ops**: 2-8x speedup (depending on core count)

### Memory Usage
- **Basic classes**: Standard std::vector overhead
- **Optimized classes**: ~Same memory usage, better locality
- **3D comparison**: OptimizedMatrix3D uses ~50% less memory than Matrix3D for large tensors

## Future Enhancements

### SIMD Implementation Status
- ✅ **Architecture**: SIMD method signatures defined
- ⏳ **Implementation**: Platform-specific SIMD code (AVX2, SSE2, ARM NEON)
- ⏳ **Auto-vectorization**: Compiler hints for optimal vectorization

### Advanced Features
- **GPU acceleration**: CUDA/OpenCL kernels for very large problems
- **Sparse matrix support**: Specialized implementations for sparse data
- **Custom allocators**: Memory pools for frequent allocations
- **Expression templates**: Lazy evaluation for complex expressions

## Migration Checklist

### For New Code ✅
- Use `OptimizedVector<T>` for vectors >100 elements
- Use `OptimizedMatrix<T>` for matrices >10x10
- Use `OptimizedMatrix3D<T>` for all 3D tensors
- Leverage automatic SIMD/parallel selection

### For Existing Code 🔄
- Profile performance bottlenecks
- Identify matrix-heavy operations
- Replace Basic with Optimized classes incrementally
- Validate numerical accuracy after migration
- Benchmark performance improvements

This architecture provides a solid foundation for high-performance numerical computing while maintaining API compatibility and ease of use.
