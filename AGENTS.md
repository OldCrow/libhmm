# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project overview

C++20 Hidden Markov Model library. Zero external dependencies (C++20 standard library only). GTest is fetched via `FetchContent` only for the test suite. Produces both a shared (`hmm`) and static (`hmm_static`) library from a single OBJECT target.

`main` is the stable v4 branch (current release: v4.2.4). Multivariate HMM support is provided via `BasicHmm<Obs>` and `BasicEmissionDistribution<Obs>` templates. `using Hmm = BasicHmm<double>` and `using EmissionDistribution = BasicEmissionDistribution<double>` preserve v3 source compatibility; users consuming only the v3 API can build from `main` unchanged.

## Session start

**Compiler prerequisites:**
- **macOS:** Xcode Command Line Tools (`xcode-select --install`) provides AppleClang. Full Xcode is not required for the library build.
- **Linux:** GCC ≥ 12 (`apt install g++-12`) or Clang ≥ 14 (`apt install clang-14`) for C++20 support. CMake ≥ 3.20 (`apt install cmake` or from cmake.org).
- **Windows:** Visual Studio 2022 (Build Tools or full IDE) with the C++ workload. See the Windows section under Build commands.

On first use on a new machine, run `cmake --preset release && cmake --build build`, then verify the detected SIMD tier with `./build/tools/system_inspector` (if built with `BUILD_TOOLS=ON`). SIMD flag selection (`-march=native` on GCC/Clang, CPU-probed `/arch:` on MSVC) is automatic at compile time.

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

### Windows (MSVC 2022)

> **Windows tool paths vary** by installation method (direct installer, `winget`, `chocolatey`, Microsoft Store, etc.). The paths below are common defaults — adjust for your installation.

Activate the MSVC toolchain once per PowerShell session before building:

```powershell
# Default path for VS 2022 Build Tools:
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
# For full VS (Community/Professional/Enterprise), use instead:
# $vcvars = "C:\Program Files\Microsoft Visual Studio\2022\{edition}\VC\Auxiliary\Build\vcvars64.bat"
# Auto-detect any edition:
# $vsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
# $vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
$envVars = cmd /c "`"$vcvars`" > nul && set"
foreach ($line in $envVars) {
    if ($line -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

# Then build as normal:
cmake --preset release
cmake --build build
```

VS 2022 Build Tools or full VS is sufficient. Install from https://aka.ms/vs/17/release/vs_buildtools.exe, or `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`. GTest is fetched automatically via `FetchContent`; no vcpkg needed.

> macOS 13 (Ventura) is the minimum supported version in v4. macOS 12 and earlier are not
> supported; use v3.8.0 or fork. See MIGRATION.md.

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

## Tool prerequisites

The linting and pre-commit tools must be installed before use:
- **clang-format**: part of LLVM (`brew install llvm`, `apt install clang-format`, `choco install llvm`)
- **cmake-format**: `pip install cmake-format`
- **pre-commit**: `pip install pre-commit`
- **cppcheck**: OS-package-managed (`brew install cppcheck`, `apt install cppcheck`, `choco install cppcheck`)

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
  --suppress=toomanyconfigs --suppress=functionStatic \
  --std=c++20 -I include src/

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
| 4b | `training/` | `BaumWelchTrainer`, `MapBaumWelchTrainer`, `ViterbiTrainer`; `BasicSegmentalKMeansTrainer<Obs>` with aliases `SegmentalKMeansTrainer` (scalar) and `SegmentalKMeansTrainerMV` (MV) |
| — | `io/` | JSON (`hmm_json.h`, recommended), legacy XML, `FileIOManager` |
| — | `performance/` | `TranscendentalKernels` (FB recurrence), `detail/simd_math_helpers.h` (shared SIMD math helpers), `fb_recurrence_policy.h`, `simd_double_ops.h` (runtime-dispatch distribution batch kernels) |

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

- **Tier 2 (explicit intrinsics, runtime-dispatched)**: 11 of 16 scalar distributions route `getBatchLogProbabilities` through the `DoubleVecOps` dispatch table (`performance/simd_double_ops.h`). The table is built once at startup via CPUID and caches function pointers into 5 per-ISA TUs (`simd_double_ops_{scalar,sse2,avx2,avx512,neon}.cpp`), each compiled with a targeted flag rather than `-march=native`. The 5 remaining scalar distributions (Discrete, Poisson, Binomial, NegativeBinomial, Uniform) are tier-1 only. `ForwardBackwardCalculator` and `BaumWelchTrainer` also have explicit recurrence kernels via `TranscendentalKernels`, using the shared helpers in `detail/simd_math_helpers.h`.
- **Tier 1 (compiler auto-vectorization)**: Five scalar distributions remain tier-1 by design — the same assessment as libstats, which marks them `SIMD deferred` for identical reasons:
    - **Poisson, Binomial, NegativeBinomial**: `lgamma(k)` must be evaluated per element; no portable vectorized lgamma exists without a math-library dependency (SVML, libmvec, or Accelerate vvlgamma). Would become tier-2 if a vectorized lgamma is added to `simd_double_ops_*.cpp`.
    - **Discrete**: per-element integer floor + range check and table lookup by symbol index. Vectorizable in principle via AVX2 gather, but complex index arithmetic and no performance data justifying the effort.
    - **Uniform**: the entire batch evaluates to a single constant (log(1/(b−a))) inside bounds or −∞ outside. Already ~2 instructions per element; SIMD buys nothing.
  MV distributions (`DiagonalGaussian`, `FullCovGaussian`, `IndependentComponents`) call `getLogProbability(row_view(obs, t))` per timestep rather than a batch interface and are not in `LIBHMM_SIMD_SOURCES`.

`detail/simd_math_helpers.h` is the single source of truth for vectorized log/exp/cos/log1p helpers shared by the per-ISA distribution kernels and `TranscendentalKernels`. Tiny `log1p` inputs use a polynomial path for accuracy; general inputs reuse the shared vector log helper.

`getBatchLogProbabilities(std::span<const double> obs, std::span<double> out)` is the SIMD interface: calculators call it once per state per `compute()` and consume a flat row-major buffer of log-emission values.

Threading is **not used** in the production path. `ThreadPool` exists in `platform/thread_pool.h` but is consumed only by two diagnostic tools in `tools/`.

### Distribution fit quality

The weighted `fit(data, weights)` method is the Baum-Welch M-step. Fit quality varies by distribution:

- **Tier A — exact weighted MLE/EM**: Gaussian, Exponential, Poisson, Discrete, LogNormal, Pareto,
  Rayleigh, VonMises, Binomial, ChiSquared (Newton MLE, v4.2.1), Gamma, Weibull, NegativeBinomial,
  Beta, StudentT (Newton/ECME, corrected in v4.2.1 — were implemented before v4.2.0 but mis-labelled)
- **Tier C — MOM (defensible in EM context)**: Uniform (fixed-range; MOM is exact for uniform support)

Priority M-step improvements are documented in `docs/GOLD_STANDARD_CHECKLIST.md`.

All `fit(data, weights)` implementations guard against near-zero weight by preserving current parameters (not calling `reset()`):
```cpp
if (sumW < precision::ZERO || std::isnan(sumW)) return;
```
`reset()` is called only for genuinely degenerate *data*.

### Model selection

`count_free_parameters(hmm)`, `compute_aic()`, `compute_bic()`, `compute_aicc()`, and `evaluate_model()` are declared in `include/libhmm/training/model_selection.h`.

### I/O

JSON is the recommended format—exact IEEE 754 round-trip, no external dependencies. Scalar: `save_json`/`load_json`. MV: `save_json_mv`/`load_json_mv` (v4 schema with `obs_type: "multivariate"`). Legacy XML (`XMLFileReader`/`XMLFileWriter`) is scalar-only and deprecated; retained for reading existing `.xml` files. Reference HMM files live in `samples/`.

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
