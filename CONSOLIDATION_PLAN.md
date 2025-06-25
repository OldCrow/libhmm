# Forward-Backward Calculator Consolidation Plan

## Current Status Analysis

### Existing Forward-Backward Calculators:
1. `ForwardBackwardCalculator` - Standard implementation
2. `ScaledForwardBackwardCalculator` - Scaling for numerical stability
3. `LogForwardBackwardCalculator` - Log-space for numerical stability
4. `OptimizedForwardBackwardCalculator` - Complex SIMD with memory pooling
5. `LogSIMDForwardBackwardCalculator` - SIMD log-space
6. `ScaledSIMDForwardBackwardCalculator` - SIMD scaling

### Existing Viterbi Calculators (Clean Pattern):
1. `ViterbiCalculator` - Standard implementation
2. `ScaledSIMDViterbiCalculator` - SIMD scaling with fallback
3. `LogSIMDViterbiCalculator` - SIMD log-space with fallback

## Consolidation Goals

### Target Architecture (Match Viterbi Pattern):
1. **`ForwardBackwardCalculator`** - Standard implementation (KEEP)
2. **`ScaledSIMDForwardBackwardCalculator`** - SIMD scaling with automatic scalar fallback (CONSOLIDATE)
3. **`LogSIMDForwardBackwardCalculator`** - SIMD log-space with automatic scalar fallback (CONSOLIDATE)

### Files to Remove/Consolidate:
- **REMOVE:** `ScaledForwardBackwardCalculator` → functionality integrated into ScaledSIMDForwardBackwardCalculator fallback
- **REMOVE:** `LogForwardBackwardCalculator` → functionality integrated into LogSIMDForwardBackwardCalculator fallback  
- **REMOVE:** `OptimizedForwardBackwardCalculator` → overly complex, replace with clean SIMD implementations

## Implementation Steps

### Phase 1: Update SIMD Forward-Backward Calculators
1. **Update `ScaledSIMDForwardBackwardCalculator`:**
   - Add automatic fallback to scalar implementation
   - Include internal scalar methods (like Viterbi pattern)
   - Simplify architecture, remove complex memory pooling

2. **Update `LogSIMDForwardBackwardCalculator`:**
   - Add automatic fallback to scalar implementation
   - Include internal scalar methods (like Viterbi pattern)
   - Ensure consistency with ScaledSIMD pattern

### Phase 2: Update Calculator Selection System
1. **Update `CalculatorTraits`:**
   - Remove references to removed calculators
   - Update selection logic to use new consolidated calculators
   - Ensure proper fallback selection

2. **Update Auto-selection:**
   - Standard → ForwardBackwardCalculator
   - Scaled → ScaledSIMDForwardBackwardCalculator (with fallback)
   - Log → LogSIMDForwardBackwardCalculator (with fallback)
   - Optimized → ScaledSIMDForwardBackwardCalculator

### Phase 3: Update Tests
1. **Update `test_calculators.cpp`:**
   - Remove tests for deleted calculators
   - Update tests to use new calculator names
   - Add SIMD fallback testing

2. **Create `test_simd_forward_backward_calculators.cpp`:**
   - Mirror the SIMD Viterbi test structure
   - Test both SIMD and scalar fallback paths
   - Test consistency between SIMD and standard implementations

### Phase 4: Update Examples and Documentation
1. **Update benchmark files** to use new calculator names
2. **Update documentation** to reflect simplified architecture
3. **Update CMake files** to remove deleted source files

## Benefits of Consolidation

### 🎯 **Consistency:**
- Forward-Backward calculators match Viterbi calculator pattern
- Easier to understand and maintain
- Consistent SIMD fallback behavior

### 🚀 **Performance:**
- SIMD optimizations available by default
- Automatic fallback ensures compatibility
- Cleaner implementations without complex memory pooling

### 🧪 **Testing:**
- Better test coverage for SIMD implementations
- Consistent testing patterns across calculator types
- Easier to verify numerical accuracy

### 📚 **Maintainability:**
- Fewer calculator variants to maintain
- Clear separation of concerns
- Simplified selection logic

## Migration Path

### For Users:
- `ScaledForwardBackwardCalculator` → `ScaledSIMDForwardBackwardCalculator` 
- `LogForwardBackwardCalculator` → `LogSIMDForwardBackwardCalculator`
- `OptimizedForwardBackwardCalculator` → `ScaledSIMDForwardBackwardCalculator`
- Automatic selection continues to work seamlessly

### For Developers:
- Cleaner codebase with fewer variants
- Consistent patterns for adding new optimizations
- Better separation between algorithm logic and optimization details
