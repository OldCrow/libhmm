# libhmm Testing Strategy (v3.0)

All tests use GoogleTest and are organised into 8 architectural levels
matching the library layered dependency graph.

## Directory Structure

```
tests/
|├── common/                     # Level 1: Math, basic types
|│   ├── test_modern_constants.cpp
|│   └── test_common.cpp
├── distributions/              # Level 3: All 15 distributions
│   ├── test_distributions.cpp
│   ├── test_distribution_type_safety.cpp
│   ├── test_distribution_traits.cpp
│   ├── test_distributions_header.cpp
│   └── test_<name>_distribution.cpp   (one per distribution)
├── test_hmm_core.cpp           # Level 4: Core HMM
├── calculators/                # Level 5: Inference
│   ├── test_canonical_calculators.cpp
│   ├── test_calculator_continuous.cpp
│   └── test_calculator_edge_cases.cpp
├── training/                   # Level 6: Training algorithms
│   ├── test_canonical_training.cpp
│   ├── test_training.cpp
│   ├── test_training_edge_cases.cpp
│   └── test_baum_welch_convergence.cpp
├── io/                         # Level 7: IO
│   ├── test_xml_file_io.cpp
│   ├── test_hmm_stream_io.cpp
│   └── test_hmm_json.cpp
├── integration/                # Level 7: End-to-end
│   └── test_end_to_end.cpp
├── CMakeLists.txt
└── TESTING_STRATEGY.md
```

## Running Tests

```bash
# Standard run -- all 40 tests (mirrors CI)
ctest --test-dir build -C Release --output-on-failure

# cmake custom targets
cmake --build build --target check           # parallel, correctness
cmake --build build --target check_timing    # serial, for timing accuracy

# Build and run a single test
cmake --build build --config Release --target test_canonical_calculators
./build/tests/Release/test_canonical_calculators

# GTest filter within a binary
./build/tests/Release/test_distributions --gtest_filter="*Discrete*"
```

## Architectural Levels

| Level | Content |
|-------|---------|
| 1 | Math & Numerics |
| 3 | Distributions (15 individual + 4 shared) |
| 4 | Core HMM |
| 5 | Calculators |
| 6 | Trainers |
| 7 | IO + Integration |

(Level 2 'Linear Algebra' tests were removed alongside the dead `Optimized*` class family.)

## Warning Policy

Tests compile at the same warning level as the library
(MSVC /W4 /permissive-, GCC/Clang -Wall -Wextra -Wpedantic).

## Performance Tools

Performance tools live in tools/, not tests/:

  build/tools/Release/simd_inspection      -- SIMD ISA report + smoke tests
  build/tools/Release/batch_performance    -- FB + Viterbi throughput
  build/tools/Release/hmm_validator        -- Load, validate, infer

---

40/40 tests pass on all platforms (Linux/GCC, Linux/Clang, macOS, Windows/MSVC).
