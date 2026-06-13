# libhmm Testing Strategy (v4.0)

All tests use GoogleTest and are organised into 8 architectural levels
matching the library layered dependency graph.

## Directory Structure

```
tests/
|├── common/                     # Level 1: Math, basic types
|│   ├── test_modern_constants.cpp
|│   └── test_common.cpp
├── distributions/              # Level 3: All 19 distributions (16 scalar + 3 MV)
│   ├── test_distributions.cpp
│   ├── test_distribution_type_safety.cpp
│   ├── test_distribution_traits.cpp
│   ├── test_distributions_header.cpp
│   └── test_<name>_distribution.cpp   (one per distribution)
├── linalg/                     # Level 2: Cholesky + linear algebra helpers
│   └── test_cholesky.cpp
├── test_hmm_core.cpp           # Level 4: Core HMM (BasicHmm<Obs>)
├── calculators/                # Level 5: Inference
│   ├── test_canonical_calculators.cpp
│   ├── test_calculator_continuous.cpp
│   ├── test_calculator_edge_cases.cpp
│   └── test_mv_calculator.cpp      # v4: BasicFBC<OVV> + BasicViterbi<OVV>
├── training/                   # Level 6: Training algorithms
│   ├── test_canonical_training.cpp
│   ├── test_training.cpp
│   ├── test_training_edge_cases.cpp
│   ├── test_baum_welch_convergence.cpp
│   └── test_mv_training.cpp        # v4: BasicBWT<OVV>, kmeans_init
├── io/                         # Level 7: IO
│   ├── test_xml_file_io.cpp
│   ├── test_hmm_stream_io.cpp
│   ├── test_hmm_json.cpp
│   └── test_hmm_json_mv.cpp        # v4: to_json(HmmMV) + from_json_mv
├── integration/                # Level 7: End-to-end
│   └── test_end_to_end.cpp
├── CMakeLists.txt
└── TESTING_STRATEGY.md
```

## Running Tests

```bash
# Standard run -- all 47 tests (mirrors CI)
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
| 2 | Linalg (Cholesky) |
| 3 | Distributions (19: 16 scalar individual + 3 MV + shared suites) |
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

47/47 tests pass on all platforms (Linux/GCC, Linux/Clang, macOS, Windows/MSVC).
