# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

C++20 Hidden Markov Model library. Zero external dependencies (C++20 standard library only). GTest is fetched via `FetchContent` only for the test suite. Produces both a shared (`hmm`) and static (`hmm_static`) library from a single OBJECT target.

Current development branch: `feature/v4-multivariate-emissions` (v4.0.0). v4 adds `BasicHmm<Obs>` and `BasicEmissionDistribution<Obs>` templates; `using Hmm = BasicHmm<double>` and `using EmissionDistribution = BasicEmissionDistribution<double>` preserve v3 source compatibility.

## Build commands

CMake presets map to fixed binary directories:

```bash
# Release (default; output in build/)
cmake --preset release && cmake --build build

# Debug (output in build-debug/)
cmake --preset debug && cmake --build build-debug

# RelWithDebInfo — preferred for profiling (output in build-relwithdebinfo/)
cmake --preset rel-with-debug && cmake --build build-relwithdebinfo

# Manual configure (no preset)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Build options: `BUILD_EXAMPLES`, `BUILD_TESTS`, `BUILD_TOOLS` (all `ON` by default), `BUILD_BENCHMARKS` (`OFF`), `ENABLE_CLANG_TIDY` (`OFF`).

## Test commands

```bash
# Run all tests (mirrors CI)
ctest --test-dir build -C Release --output-on-failure

# Parallel correctness suite — excludes known_broken and benchmark labels
cmake --build build --target check

# Build and run a single test executable
cmake --build build --target test_gaussian_distribution
./build/tests/test_gaussian_distribution

# GTest filter within a binary
./build/tests/test_distributions --gtest_filter="*Discrete*"

# Serial run (for timing-sensitive tests)
cmake --build build --target check_timing
```

Tests use the `known_broken` label for pre-existing failures and `benchmark` for on-demand-only tests; both are excluded from the standard CI run via `-LE "known_broken|benchmark"`.

## Linting and formatting

```bash
# clang-format (enforced by pre-commit and CI)
clang-format --style=file -i <file>

# cmake-format
cmake-format -i CMakeLists.txt

# cppcheck (as run in CI)
cppcheck --enable=warning,style,performance --error-exitcode=1 \
  --suppress=missingIncludeSystem --suppress=useStlAlgorithm \
  --suppress=shadowFunction --suppress=virtualCallInConstructor \
  --suppress=constParameterReference --suppress=noExplicitConstructor \
  --suppress=toomanyconfigs --std=c++20 -I include src/

# Install pre-commit hooks
bash scripts/setup-pre-commit.sh
```

Active pre-commit hooks: trailing whitespace, end-of-file newline, LF line endings, YAML/JSON syntax, large-file guard, clang-format, cmake-format, and a project-specific `#pragma once` checker.

## Architecture

### Header layer model (`include/libhmm/`)

Dependencies flow strictly downward:

| Layer | Path | Contents |
|-------|------|----------|
| 0 | `platform/` | SIMD CPU detection (`simd_platform.h`) |
| 1 | `math/` | Constants (`constants.h`), Bessel, digamma/polygamma (`psi_functions.h`), weighted stats |
| 2 | `linalg/` | `BasicMatrix<T>`, `BasicVector<T>`, `BasicMatrix3D<T>`; `linalg_types.h` defines `Matrix`, `Vector`, `ObservationList`, etc. |
| 3 | `distributions/` | `BasicEmissionDistribution<Obs>` abstract base (in `basic_emission_distribution.h`); 16 concrete distributions; `distribution_traits.h`, `emission_concepts.h` |
| 4a | `calculators/` | `ForwardBackwardCalculator`, `ViterbiCalculator` |
| 4b | `training/` | `BaumWelchTrainer`, `MapBaumWelchTrainer`, `ViterbiTrainer`, `SegmentalKMeansTrainer` |
| — | `io/` | JSON (`hmm_json.h`, recommended), legacy XML, `FileIOManager` |
| — | `performance/` | `TranscendentalKernels`, `simd_kernels_internal.h` (SIMD helpers shared by distribution TUs), `fb_recurrence_policy.h` |

`libhmm.h` is the single umbrella include.

### v4 template parameterization

`BasicHmm<Obs>` and `BasicEmissionDistribution<Obs>` are the new v4 base types:

- `using Hmm = BasicHmm<double>` — scalar HMM (v3 API preserved)
- `using HmmMV = BasicHmm<ObservationVectorView>` — multivariate HMM (v4 addition); `ObservationVectorView = std::span<const double>`; emission slots start null and must be set explicitly
- `using EmissionDistribution = BasicEmissionDistribution<double>`

`Hmm` is non-copyable but movable. Default construction creates a 4-state model with `GaussianDistribution` emissions on the scalar path.

### SIMD strategy

SIMD compile flags (`LIBHMM_BEST_SIMD_FLAGS` = `-march=native` on GCC/Clang, CPU-probed `/arch:AVX512|AVX2|AVX` on MSVC) are applied **per-TU** to `LIBHMM_SIMD_SOURCES`—not globally—so non-SIMD code compiles at the platform baseline ISA.

There are two tiers of SIMD implementation:

- **Tier 2 (explicit intrinsics)**: `GaussianDistribution` and `ExponentialDistribution`. Dispatch chain: AVX-512 → AVX/AVX2 → SSE2 → NEON → scalar. `ForwardBackwardCalculator` and `BaumWelchTrainer` also have explicit recurrence kernels via `TranscendentalKernels`.
- **Tier 1 (compiler auto-vectorization)**: The other 14 distributions implement concrete non-virtual `getBatchLogProbabilities` loops. Whether the compiler emits vector instructions depends on the loop body and compiler; tier 1 is reliable "well-shaped scalar code" rather than guaranteed SIMD.

`getBatchLogProbabilities(std::span<const double> obs, std::span<double> out)` is the SIMD interface: calculators call it once per state per `compute()` and consume a flat row-major buffer of log-emission values.

Threading is **not used** in the production path. `ThreadPool` exists in `platform/thread_pool.h` but is consumed only by two diagnostic tools in `tools/`.

### Distribution fit quality

The weighted `fit(data, weights)` method is the Baum-Welch M-step. Fit quality varies by distribution:

- **Tier A — exact weighted MLE**: Gaussian, Exponential, Poisson, Discrete, LogNormal, Pareto, Rayleigh, VonMises, Binomial
- **Tier B — MOM ≈ MLE**: ChiSquared
- **Tier C — MOM (gap can be material)**: Gamma, Weibull, NegativeBinomial, Uniform, Beta, StudentT

Priority M-step improvements are documented in `docs/GOLD_STANDARD_CHECKLIST.md`.

All `fit(data, weights)` implementations guard against near-zero weight by preserving current parameters (not calling `reset()`):
```cpp
if (sumW < precision::ZERO || std::isnan(sumW)) return;
```
`reset()` is called only for genuinely degenerate *data*.

### Model selection

`count_free_parameters(hmm)`, `compute_aic()`, `compute_bic()`, `compute_aicc()`, and `evaluate_model()` are declared in `include/libhmm/training/model_selection.h`.

### I/O

JSON (`save_json` / `load_json`) is the recommended format—exact IEEE 754 round-trip, no external dependencies. Legacy XML support (`XMLFileReader` / `XMLFileWriter`) is retained for reading existing `.xml` files but deprecated for new code. Reference HMM files in both formats live in `samples/`.

## Coding conventions

See `docs/STYLE_GUIDE.md` for the full guide. Key points:

- `#pragma once` always (no `#ifndef` guards); enforced by pre-commit hook
- Classes: `PascalCase`; methods and local variables: `camelCase`; private members: `camelCase_` (trailing underscore); constants: `SCREAMING_SNAKE_CASE`
- All distributions must implement the separate `validateParameters()` pattern: called in constructors and setters, throws `std::invalid_argument`
- Each distribution `.cpp` uses `using namespace constants;` inside `namespace libhmm` — no magic numbers
- Expensive per-call values (normalization constants, log parameters) are cached via `mutable` members invalidated by setters; the cache flag is `std::atomic<bool>` for read-thread safety
- K&R brace style, 4-space indentation, 100-character line limit (enforced by `.clang-format`)
- Use `[[nodiscard]]`, `noexcept`, `std::span`, `std::optional` where appropriate (C++20)

## Adding a new distribution

Use an existing distribution (e.g. `src/distributions/rayleigh_distribution.cpp` for single-parameter, `src/distributions/gamma_distribution.cpp` for two-parameter) as a template. Required checklist (`docs/GOLD_STANDARD_CHECKLIST.md`):

1. Concrete non-virtual `getBatchLogProbabilities` override (tier 1 minimum)
2. Weighted `fit(data, weights)` with near-zero weight guard
3. `reset()`, `clone()`, `sample()`, `to_json()` / `from_json()` (registered in `src/io/hmm_json.cpp`), `getNumParameters()`
4. `std::atomic<bool> cacheValid_` thread-safe cache
5. Add source to `LIBHMM_SIMD_SOURCES` and `LIBHMM_SOURCES` in `CMakeLists.txt`
6. Add a test file under `tests/distributions/` and register it in `tests/CMakeLists.txt`

## CI matrix

Four parallel jobs: Linux/GCC, Linux/Clang, macOS/AppleClang, Windows/MSVC 2022. Two additional jobs: pre-commit (ubuntu) and cppcheck (ubuntu). Tests run with `-LE "known_broken|benchmark"`. `clang-tidy` is available but disabled in CI (`ENABLE_CLANG_TIDY=OFF`); enable locally when needed.
