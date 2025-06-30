# Calculator Architecture Migration Guide

## Overview

The `Calculator` base class has been modernized to use C++17 type-safe patterns while maintaining full backward compatibility. This guide shows how to migrate derived calculators to the new interface.

## What Changed

### Before (Legacy - Still Works):
```cpp
class MyCalculator : public Calculator {
public:
    MyCalculator(Hmm* hmm, const ObservationSet& observations)
        : Calculator(hmm, observations) {}
    
    double compute() {
        // Uses hmm_ pointer directly
        return hmm_->getEmissionProbability(0, observations_(0));
    }
};
```

### After (Modern - Recommended):
```cpp
class MyCalculator : public Calculator {
public:
    // Preferred: const reference constructor
    MyCalculator(const Hmm& hmm, const ObservationSet& observations)
        : Calculator(hmm, observations) {}
        
    // Legacy constructor for backward compatibility
    [[deprecated("Use const reference constructor")]]
    MyCalculator(Hmm* hmm, const ObservationSet& observations)
        : Calculator(hmm, observations) {}
    
    double compute() {
        // Modern: use const reference
        return getHmmRef().getEmissionProbability(0, observations_(0));
        
        // Legacy: still works but deprecated
        // return hmm_->getEmissionProbability(0, observations_(0));
    }
};
```

## Migration Steps

### Step 1: Add Modern Constructor (Non-Breaking)
```cpp
// Add this to all calculator classes
MyCalculator(const Hmm& hmm, const ObservationSet& observations)
    : Calculator(hmm, observations) {}
```

### Step 2: Update Internal Code (Gradual)
Replace `hmm_->method()` with `getHmmRef().method()`:

```cpp
// Old
double prob = hmm_->getEmissionProbability(state, obs);
const Matrix& trans = hmm_->getTrans();
int numStates = hmm_->getNumStates();

// New
double prob = getHmmRef().getEmissionProbability(state, obs);
const Matrix& trans = getHmmRef().getTrans();
int numStates = getHmmRef().getNumStates();
```

### Step 3: Deprecate Legacy Constructor (Optional)
```cpp
[[deprecated("Use const reference constructor for better type safety")]]
MyCalculator(Hmm* hmm, const ObservationSet& observations)
    : Calculator(hmm, observations) {}
```

## Benefits of Migration

### 1. Type Safety
- **No null pointer risk**: References cannot be null
- **Const correctness**: HMM is immutable during calculation
- **Lifetime safety**: Clear ownership semantics

### 2. Modern C++17
- **RAII compliance**: No manual lifetime management
- **Exception safety**: Guaranteed initialization
- **API clarity**: Intent is explicit

### 3. Backward Compatibility
- **Existing code works**: No breaking changes
- **Gradual migration**: Update at your own pace
- **Clear deprecation path**: Compiler warnings guide migration

## Advanced Patterns

### Smart Pointer Integration
```cpp
// Works seamlessly with smart pointers
std::unique_ptr<Hmm> hmm = createHmm();
MyCalculator calc(*hmm, observations);  // Type-safe

// Or with shared ownership
std::shared_ptr<const Hmm> hmm = getSharedHmm();
MyCalculator calc(*hmm, observations);
```

### Factory Functions
```cpp
// Type-safe factory
template<typename CalculatorType>
std::unique_ptr<CalculatorType> createCalculator(const Hmm& hmm, const ObservationSet& obs) {
    return std::make_unique<CalculatorType>(hmm, obs);
}

// Usage
auto calc = createCalculator<MyCalculator>(*hmm, observations);
```

## Timeline

- **v2.x**: Both interfaces available, legacy deprecated
- **v3.0**: Legacy interface removed, modern only

## Migration Checklist

- [ ] Add const reference constructor to derived classes
- [ ] Replace `hmm_->` with `getHmmRef().` in implementation
- [ ] Update factory functions to use references
- [ ] Add deprecation warnings to legacy constructors
- [ ] Update unit tests to use modern interface
- [ ] Update documentation and examples

## Examples

See the `AdvancedLogSIMDForwardBackwardCalculator` for a complete example of the modern pattern:

```cpp
// Modern constructors
AdvancedLogSIMDForwardBackwardCalculator(const Hmm& hmm, const ObservationSet& observations, ...);

// Legacy constructor with deprecation
[[deprecated("Use const reference constructor for better type safety")]]
AdvancedLogSIMDForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations, ...);

// Modern implementation
const Matrix& trans = getHmmRef().getTrans();  // Type-safe
```

This pattern provides maximum type safety while maintaining full backward compatibility.
