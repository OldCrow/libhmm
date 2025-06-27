# libhmm C++ Style Guide

## Table of Contents
1. [Overview](#overview)
2. [General Principles](#general-principles)
3. [Code Organization](#code-organization)
4. [Naming Conventions](#naming-conventions)
5. [Formatting and Layout](#formatting-and-layout)
6. [Language Features](#language-features)
7. [Error Handling](#error-handling)
8. [Documentation](#documentation)
9. [Testing](#testing)
10. [Static Analysis](#static-analysis)
11. [Performance Guidelines](#performance-guidelines)
12. [Distribution Parameter Validation](#distribution-parameter-validation)

## Overview

This style guide defines coding standards for the libhmm C++17 Hidden Markov Model library. It ensures consistency, maintainability, and high code quality across the codebase.

**Key Goals:**
- **Consistency**: Uniform code style across all modules
- **Readability**: Self-documenting code with clear intent
- **Maintainability**: Easy to modify, extend, and debug
- **Performance**: Efficient use of modern C++17 features
- **Safety**: Strong type safety and error handling

## General Principles

### 1. Modern C++17 First
- Use C++17 features and idioms
- Prefer standard library over custom implementations
- Use RAII (Resource Acquisition Is Initialization)
- Embrace move semantics and perfect forwarding

### 2. Zero Dependencies Policy
- Rely only on C++17 standard library
- No external dependencies beyond compiler and build tools
- Custom implementations for specialized needs

### 3. Performance-Oriented Design
- SIMD-friendly memory layouts
- Cache-aware algorithms
- Minimize allocations in hot paths
- Template-based optimizations

## Code Organization

### File Structure
```
include/libhmm/
├── distributions/        # Probability distributions
├── calculators/         # HMM algorithms
├── training/           # Training algorithms  
├── io/                 # File I/O operations
├── common/             # Utilities and common types
└── performance/        # Performance optimizations
```

### Header Organization
```cpp
// 1. System headers (alphabetical)
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

// 2. Project headers (alphabetical)
#include "libhmm/common/common.h"
#include "libhmm/distributions/probability_distribution.h"
```

### Include Guards
Use `#ifndef` include guards with consistent naming:
```cpp
#ifndef LIBHMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define LIBHMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
// ... content ...
#endif // LIBHMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
```

## Naming Conventions

### 1. Classes and Types
- **PascalCase** for class names: `GaussianDistribution`, `ViterbiCalculator`
- **PascalCase** for type aliases: `StateSequence`, `ObservationList`
- **PascalCase** for enums: `CalculatorType`, `TrainingMethod`

### 2. Functions and Methods
- **camelCase** for functions and methods: `getProbability()`, `updateCache()`
- **camelCase** for public methods: `setLambda()`, `getMean()`
- Descriptive names that indicate action: `validateParameters()`, `computeLogLikelihood()`

### 3. Variables
- **camelCase** for local variables: `observationIndex`, `logProbability`
- **snake_case with trailing underscore** for private members: `lambda_`, `mean_`, `cacheValid_`
- **camelCase** for parameters: `lambda`, `standardDeviation`, `observationSequence`

### 4. Constants
- **SCREAMING_SNAKE_CASE** for compile-time constants: `MAX_ITERATIONS`, `DEFAULT_TOLERANCE`
- **camelCase** for `const` variables: `defaultLambda`, `minObservations`

### 5. Namespaces
- **lowercase** with underscores: `libhmm`, `libhmm::detail`

### Examples
```cpp
namespace libhmm {

class GaussianDistribution : public ProbabilityDistribution {
private:
    double mean_{0.0};                    // Private member
    double standardDeviation_{1.0};      // Private member
    mutable bool cacheValid_{false};     // Private member
    
    static constexpr double DEFAULT_MEAN = 0.0;  // Constant
    
public:
    void setMean(double mean);           // Public method
    double getMean() const noexcept;    // Public method
    
private:
    void validateParameters(double mean, double stdDev) const;  // Private method
    void updateCache() const noexcept;                          // Private method
};

} // namespace libhmm
```

## Formatting and Layout

### 1. Indentation and Spacing
- **4 spaces** for indentation (no tabs)
- **No trailing whitespace**
- **Single space** around binary operators: `x + y`, `a == b`
- **No space** around unary operators: `++i`, `*ptr`

### 2. Line Length
- **Maximum 100 characters** per line
- Break long function signatures and calls logically
- Prefer breaking after commas and operators

### 3. Braces and Brackets
- **Allman style** for class/function definitions:
```cpp
class GaussianDistribution 
{
public:
    void setMean(double mean) 
    {
        validateParameters(mean, standardDeviation_);
        mean_ = mean;
    }
};
```

- **Same line** for control structures:
```cpp
if (condition) {
    doSomething();
} else {
    doSomethingElse();
}

for (const auto& item : container) {
    processItem(item);
}
```

### 4. Function Signatures
```cpp
// Short signatures (single line)
double getMean() const noexcept;

// Long signatures (multi-line with parameters aligned)
void setParameters(double mean, 
                  double standardDeviation, 
                  bool validateInputs = true);

// Constructor initialization lists
ExponentialDistribution(double lambda = 1.0)
    : lambda_{lambda}, 
      logLambda_{0.0}, 
      cacheValid_{false} {
    validateParameters(lambda);
    updateCache();
}
```

### 5. Pointer and Reference Alignment
```cpp
double* ptr;          // Preferred
double& ref;          // Preferred
std::vector<double>   // No space before template args
```

## Language Features

### 1. Modern C++17 Features
**Prefer:**
- `auto` for type deduction when type is obvious
- Range-based for loops
- `constexpr` for compile-time constants
- `noexcept` for functions that don't throw
- Smart pointers (`std::unique_ptr`, `std::shared_ptr`)
- `std::optional` for optional values
- Structured bindings where appropriate

**Examples:**
```cpp
// Auto with obvious types
auto distribution = std::make_unique<GaussianDistribution>();
auto lambda = 2.5;  // Clear from context

// Range-based loops
for (const auto& observation : observations) {
    processObservation(observation);
}

// constexpr and noexcept
constexpr double PI = 3.14159265358979323846;
double getMean() const noexcept { return mean_; }

// Optional return values
std::optional<double> safeDivide(double numerator, double denominator) {
    if (denominator == 0.0) {
        return std::nullopt;
    }
    return numerator / denominator;
}
```

### 2. Class Design
- **Rule of Zero**: Prefer default special member functions
- **Rule of Five**: If you define one, define all five (destructor, copy constructor, copy assignment, move constructor, move assignment)
- Use `= default` and `= delete` explicitly
- Mark single-argument constructors as `explicit`

```cpp
class GaussianDistribution {
public:
    // Explicit single-argument constructor
    explicit GaussianDistribution(double mean = 0.0, double stdDev = 1.0);
    
    // Default special members when possible
    ~GaussianDistribution() = default;
    GaussianDistribution(const GaussianDistribution&) = default;
    GaussianDistribution& operator=(const GaussianDistribution&) = default;
    GaussianDistribution(GaussianDistribution&&) = default;
    GaussianDistribution& operator=(GaussianDistribution&&) = default;
};
```

### 3. Templates
- Use `template<typename T>` (not `template<class T>`)
- Prefer concepts when available (C++20) or SFINAE for constraints
- Use `auto` return type deduction for complex template returns

### 4. Memory Management
- **Prefer stack allocation** over heap allocation
- Use **smart pointers** for dynamic allocation
- Use **containers** instead of raw arrays
- **RAII** for all resource management

```cpp
// Good
auto distribution = std::make_unique<GaussianDistribution>(0.0, 1.0);
std::vector<double> observations(1000);

// Avoid
GaussianDistribution* distribution = new GaussianDistribution(0.0, 1.0);
double* observations = new double[1000];
```

## Error Handling

### 1. Exception Policy
- Use **exceptions for exceptional conditions**
- **`std::invalid_argument`** for parameter validation
- **`std::runtime_error`** for runtime failures
- **`std::out_of_range`** for index bounds violations
- **`noexcept`** for functions that guarantee no exceptions

### 2. Parameter Validation Pattern
**All distributions must follow the separate validation method pattern:**

```cpp
class ExampleDistribution {
private:
    /**
     * Validates parameters for the distribution
     * @param param1 First parameter with constraints
     * @param param2 Second parameter with constraints  
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double param1, double param2) const {
        if (std::isnan(param1) || std::isinf(param1) || param1 <= 0.0) {
            throw std::invalid_argument("param1 must be positive and finite");
        }
        if (std::isnan(param2) || std::isinf(param2) || param2 <= 0.0) {
            throw std::invalid_argument("param2 must be positive and finite");  
        }
    }

public:
    ExampleDistribution(double param1, double param2) 
        : param1_{param1}, param2_{param2} {
        validateParameters(param1, param2);  // Validate in constructor
    }
    
    void setParam1(double param1) {
        validateParameters(param1, param2_);  // Validate in setter
        param1_ = param1;
    }
};
```

### 3. Error Messages
- **Clear and specific** error messages
- **Include parameter names** and constraints
- **Consistent formatting** across the library

```cpp
// Good
throw std::invalid_argument("Lambda (rate parameter) must be a positive finite number");

// Avoid
throw std::invalid_argument("Invalid parameter");
```

## Documentation

### 1. Doxygen Documentation
Use **Doxygen-style comments** for all public interfaces:

```cpp
/**
 * Computes the probability density function for the Gaussian distribution.
 * 
 * The PDF is computed using the formula:
 * f(x) = (1/σ√(2π)) * exp(-0.5*((x-μ)/σ)²)
 * 
 * @param value The value at which to evaluate the PDF
 * @return Probability density at the given value
 * @throws std::invalid_argument if value is NaN or infinite
 * 
 * @note This method is thread-safe and uses cached normalization constants
 * @complexity O(1) - constant time computation
 * 
 * @example
 * @code
 * GaussianDistribution dist(0.0, 1.0);  // Standard normal
 * double prob = dist.getProbability(1.0);  // Returns ~0.242
 * @endcode
 */
double getProbability(double value) override;
```

### 2. Class Documentation
```cpp
/**
 * Modern C++17 Gaussian distribution for modeling continuous symmetric data.
 * 
 * The Gaussian (Normal) distribution is a continuous probability distribution
 * characterized by its bell-shaped curve. It's fundamental in statistics and
 * is used extensively in machine learning and data analysis.
 * 
 * PDF: f(x) = (1/σ√(2π)) * exp(-0.5*((x-μ)/σ)²)
 * where μ is the mean and σ is the standard deviation (σ > 0)
 * 
 * Properties:
 * - Mean: μ  
 * - Variance: σ²
 * - Support: x ∈ (-∞, ∞)
 * - Symmetry: Symmetric around μ
 * 
 * @note Thread-safe for read operations, not thread-safe for modifications
 * @note Uses efficient caching for repeated probability calculations
 * 
 * @example Basic usage:
 * @code
 * GaussianDistribution normal(0.0, 1.0);  // Standard normal distribution
 * double prob = normal.getProbability(0.0);  // Peak probability
 * normal.setMean(5.0);  // Shift distribution
 * @endcode
 */
class GaussianDistribution : public ProbabilityDistribution {
    // ...
};
```

### 3. Code Comments
- **Explain why**, not what
- **Complex algorithms** should have step-by-step explanation
- **Performance-critical sections** should be documented
- **Mathematical formulas** should be clearly stated

```cpp
// Cache normalization constant for efficiency in repeated calculations
// This avoids recomputing 1/(σ√(2π)) for every getProbability() call
if (!cacheValid_) {
    normalizationConstant_ = 1.0 / (standardDeviation_ * std::sqrt(2.0 * M_PI));
    negHalfSigmaSquaredInv_ = -0.5 / (standardDeviation_ * standardDeviation_);
    cacheValid_ = true;
}
```

## Testing

### 1. Test Organization
- **One test file per class**: `test_gaussian_distribution.cpp`
- **Comprehensive coverage**: Constructor, methods, edge cases
- **Descriptive test names**: `testParameterValidation()`, `testStatisticalMoments()`

### 2. Test Structure
```cpp
/**
 * Test parameter validation for invalid inputs
 */
void testParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    // Test invalid constructor parameters
    try {
        GaussianDistribution dist(0.0, 0.0);  // Invalid stddev
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    // Test NaN and infinity
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    try {
        GaussianDistribution dist(nan_val, 1.0);
        assert(false);  // Should not reach here  
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "✓ Parameter validation tests passed" << std::endl;
}
```

### 3. Test Categories
- **Basic Functionality**: Constructor, getters, setters
- **Parameter Validation**: Invalid inputs, edge cases
- **Mathematical Correctness**: Statistical properties, known values
- **Performance**: Efficiency of critical paths
- **Edge Cases**: Boundary conditions, extreme values

## Static Analysis

### 1. Required Tools
- **clang-tidy**: Static analysis and code quality
- **cppcheck**: Additional static analysis
- **Address Sanitizer**: Memory error detection  
- **Undefined Behavior Sanitizer**: UB detection

### 2. Enabled Checks
See the static analysis configuration in `.clang-tidy` and CMake integration.

### 3. CI Integration
All static analysis tools run automatically on:
- Pull requests
- Main branch commits
- Release builds

## Performance Guidelines

### 1. Memory Layout
- **Contiguous memory** for data structures
- **Aligned allocators** for SIMD operations
- **Cache-friendly** access patterns
- **Minimize allocations** in hot paths

### 2. SIMD Optimization
- **SIMD-friendly** data layouts
- **Automatic fallback** to scalar implementations
- **Runtime CPU detection** for optimal algorithms

### 3. Caching Strategy
- **Cache expensive computations** (normalization constants)
- **Lazy evaluation** for optional calculations
- **Cache invalidation** on parameter changes

```cpp
class GaussianDistribution {
private:
    mutable double normalizationConstant_{0.0};
    mutable bool cacheValid_{false};
    
    void updateCache() const noexcept {
        normalizationConstant_ = 1.0 / (standardDeviation_ * std::sqrt(2.0 * M_PI));
        cacheValid_ = true;
    }
    
public:
    double getProbability(double value) override {
        if (!cacheValid_) {
            updateCache();
        }
        // Use cached values...
    }
};
```

## Distribution Parameter Validation

### Mandatory Pattern
**All probability distributions MUST implement parameter validation using the separate validation method pattern:**

```cpp
class DistributionName : public ProbabilityDistribution {
private:
    /**
     * Validates parameters for the distribution
     * @param param1 Description and constraints
     * @param param2 Description and constraints
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(ParamType1 param1, ParamType2 param2) const {
        // Comprehensive validation logic
        if (std::isnan(param1) || std::isinf(param1) || /* constraint */) {
            throw std::invalid_argument("Detailed error message");
        }
        // Additional validations...
    }
    
public:
    // Constructor MUST call validateParameters
    DistributionName(ParamType1 param1, ParamType2 param2) 
        : param1_{param1}, param2_{param2} {
        validateParameters(param1, param2);
    }
    
    // Setters MUST call validateParameters  
    void setParam1(ParamType1 param1) {
        validateParameters(param1, param2_);
        param1_ = param1;
    }
};
```

### Benefits of This Pattern:
1. **Single Responsibility**: Validation logic is centralized
2. **Code Reuse**: Same validation for constructor and setters
3. **Maintainability**: Easy to modify validation rules
4. **Testability**: Validation can be tested independently
5. **Exception Safety**: Validation happens before state changes

### Validation Requirements:
- **Check for NaN and infinity** using `std::isnan()` and `std::isinf()`
- **Validate parameter ranges** according to mathematical constraints
- **Provide clear error messages** with parameter names and expected ranges
- **Use `std::invalid_argument`** for parameter validation failures
- **Validate all parameters together** to catch interdependent constraints

## Enforcement

### Automated Checks
- **Static analysis** enforces most style rules
- **Unit tests** verify validation patterns
- **CI/CD pipeline** prevents non-compliant code
- **Code reviews** catch remaining issues

### Style Violations
Style violations will be caught by:
1. **clang-tidy** configuration
2. **Automated formatting** tools
3. **Code review** process
4. **CI build failures**

This style guide is enforced through automated tooling to maintain consistency across the libhmm codebase.
