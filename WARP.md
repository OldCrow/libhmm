# WARP.md

This file provides guidance to Warp (warp.dev) when working in this repository.

# libhmm — Modern C++20 Hidden Markov Model Library

## Current Status

**Version**: v3.0.1 release prep on `main` (`v3.0.0` is latest published tag/release).
**Tests**: 36/36 passing (`ctest`). No `known_broken` labels remain.
**Active phase**: Complete. All phases through Post-Phase 5 (CI/tooling, benchmarks) are done.

See the Warp plan artifact **"libhmm Modernization: Architecture & Refactoring Plan"** for the full roadmap.

---

## Architecture Overview (v3.0)

Phases 0–4.5 are complete. The library has been fully modernised:

```
include/libhmm/
├── platform/       # Layer 0: compile-time SIMD macros, CPU feature constants
├── math/           # Layer 1: constants, log-space ops, numerical stability
├── linalg/         # Layer 2: Matrix, Vector, ObservationSet, StateSequence
├── common/         # Shared types and serialization helpers
├── distributions/  # Layer 3: EmissionDistribution + DistributionBase + 15 distributions
│   ├── emission_distribution.h   # Abstract base (const-correct, getBatchLogProbabilities)
│   ├── distribution_base.h       # Shared atomic cache, math helpers
│   └── [15 concrete distributions]
├── hmm.h           # Core HMM class (setDistribution / getDistribution API)
├── calculators/    # Layer 4 — inference
│   ├── forward_backward_calculator.h  # Canonical log-space FB
│   └── viterbi_calculator.h           # Canonical log-space Viterbi
├── training/       # Layer 4 — parameter estimation
│   ├── baum_welch_trainer.h           # Log-space EM; weighted fit()
│   ├── viterbi_trainer.h              # Hard-assignment with TrainingConfig presets
│   └── segmental_kmeans_trainer.h     # Discrete-state initialisation
└── io/             # XML I/O
src/                # Implementation (mirrors include/)
tests/              # GTest suite — levels 0–7 (see tests/CMakeLists.txt)
examples/           # 12 usage demonstrations (all canonical API)
tools/              # Standalone diagnostic/benchmarking executables
benchmarks/         # Comparative benchmarks (requires external HMM libs — deferred)
```

### What was removed in the refactor

- All SIMD calculator variants (`ScaledSIMD*`, `LogSIMD*`, `AdvancedLog*`)
- `AutoCalculator` / `CalculatorSelector` / `CalculatorTraits`
- `ScaledBaumWelchTrainer`, `RobustViterbiTrainer`
- `ProbabilityDistribution` base class (replaced by `EmissionDistribution`)
- `HmmTrainer` base (replaced by `Trainer`)

---

## Key Design Decisions

1. **`EmissionDistribution` base** — const-correct, `std::span` params, `getBatchLogProbabilities()` batch hook for SIMD, `fit(data, weights)` for Baum-Welch M-step.

2. **Two canonical calculators** — `ForwardBackwardCalculator` (log-space, precomputed log-trans) and `ViterbiCalculator`. Both call `getBatchLogProbabilities()` per state per time step.

3. **Compile-time SIMD dispatch** — source-distributed; each machine builds for its own CPU. GCC/Clang: `-march=native`. MSVC: `check_cxx_source_runs`-verified `/arch:AVX512`/`AVX2`/`AVX`. All 15 distribution TUs in `LIBHMM_SIMD_SOURCES`. Tier 2 explicit intrinsics: Gaussian + Exponential via `detail::` free functions (extractable to separate TU for future runtime dispatch).

4. **Thread-safe cache** — `std::atomic<bool> cacheValid_` in `DistributionBase`. Avoids mutex; safe for concurrent const reads from the calculator thread pool.

5. **`TrainingConfig` presets** — `training_presets::fast()`, `balanced()`, `precise()` in `viterbi_trainer.h`.

6. **`setDistribution(state, unique_ptr<EmissionDistribution>)` / `getDistribution(state) → EmissionDistribution&`** — the canonical `Hmm` API.

---

## Development Machines

| Machine | OS | CPU | SIMD | Role |
|---|---|---|---|---|
| Asus TUF A16 (2025) | Windows 11 Pro | AMD Ryzen 7 7745 (Zen 4) | SSE2 + AVX + AVX2 + AVX-512 | Primary Windows/MSVC dev |
| MacBook Pro 9,1 | macOS | Intel Ivy Bridge | SSE2 + AVX (no AVX2) | Secondary — macOS/Clang |
| MacBook Pro 14,1 | macOS | Apple M1 Pro | NEON | Secondary — ARM/NEON |
| Mac Mini M1 | macOS | Apple M1 | NEON | Secondary — ARM/NEON |

---

## CI

`.github/workflows/ci.yml` runs on every push to `main`.

| Job | Runner | Compiler | Notes |
|---|---|---|---|
| Linux / GCC | ubuntu-latest | GCC | Release, `ctest` |
| Linux / Clang | ubuntu-latest | Clang | Release, `ctest` |
| macOS | macos-latest | AppleClang | Release, `ctest` |
| Windows | windows-latest | MSVC | Release, `ctest` |
| Lint | ubuntu-latest | — | clang-format dry-run + cppcheck (warning-only) |

GTest is fetched via `FetchContent` if not found locally — no vcpkg required on CI.

---

## Windows Session Setup (Asus TUF A16)

### Configure and Build

The Visual Studio generator handles MSVC internally — do NOT call `vcvars64.bat` before cmake.

```powershell
$repo = "C:\Users\gdwol\Development\libhmm"

# Configure (GTest fetched automatically via FetchContent if not found)
cmake -S $repo -B "$repo\build" -G "Visual Studio 17 2022" -A x64

# Build
cmake --build "$repo\build" --config Release --parallel 4
```

### Run Tests

GTest is compiled from source via FetchContent — no DLL copying required.

```powershell
# Standard run (mirrors CI)
ctest --test-dir C:\Users\gdwol\Development\libhmm\build -C Release `
      --parallel 4 --output-on-failure

# Custom targets
cmake --build C:\Users\gdwol\Development\libhmm\build --target check
```

### Run Tools

```powershell
$tools = "C:\Users\gdwol\Development\libhmm\build\tools\Release"
& "$tools\simd_inspection.exe"               # SIMD ISA report + smoke tests
& "$tools\batch_performance.exe"             # FB + Viterbi throughput
& "$tools\hmm_validator.exe" model.xml 100  # Load + validate + infer
```

### Git Commits

GPG signing may timeout in non-interactive shells:
```powershell
git -c commit.gpgsign=false commit -m "message"
```

CRLF: `.gitattributes` enforces LF. CRLF warnings on `git add` are normal.

---

## macOS Session Notes

- Homebrew: `/opt/homebrew` (Apple Silicon) or `/usr/local` (Intel)
- SIMD: `-march=native` automatically selects NEON on arm64, AVX/AVX2 on Intel
- Configure: `cmake -S libhmm -B libhmm/build`
- Tests: `ctest --test-dir libhmm/build --parallel 4`

---

## Test Suite Structure

Tests in `tests/CMakeLists.txt` use `add_hmm_test()` helper organized into 8 levels:

| Level | Content |
|---|---|
| 1 | Math & Numerics |
| 2 | Linear Algebra |
| 3 | Distributions (all 15 + traits/header/type_safety) |
| 4 | Core HMM |
| 5 | Calculators (canonical + continuous + edge cases) |
| 6 | Trainers (canonical + training + edge cases + BW convergence) |
| 7 | IO + Integration (stream IO + end-to-end casino) |

Custom targets: `check` (correctness, parallel), `check_timing` (serial).
Note: named `check` not `run_tests` to avoid cmake's built-in `RUN_TESTS` on Windows.

---

## Known Issues

| Issue | Status | Notes |
|---|---|---|
| `test_xml_file_io` | Resolved | Fixed in v3.0.0 — platform-guarded test path now correctly provokes the error on both Windows and Unix. |
| Benchmarks | Complete | Comparative suite run on macOS (April 2026). Results in `benchmarks/docs/BENCHMARKING_RESULTS.md`. libhmm throughput ~8–14k obs/ms vs old ~1k obs/ms baseline; now faster than StochHMM and JAHMM. HMMLib ~3×, GHMM ~5× faster than libhmm. |
| StochHMM PI typo | Resolved | `source/src/stochMath.h` had `3.145926...`; corrected to `3.141592653589793`. Post-fix continuous log-likelihoods match libhmm to machine precision. Pre-fix continuous results in `BENCHMARKING_RESULTS.md` are marked invalid. |

---

## Useful Commands

```powershell
# Check for stale old-API references
Select-String -Recurse `
    -Path "C:\Users\gdwol\Development\libhmm\src","C:\Users\gdwol\Development\libhmm\include" `
    -Pattern "setProbabilityDistribution|ProbabilityDistribution|AutoCalculator|RobustViterbi"

# Build and run a single test
cmake --build C:\Users\gdwol\Development\libhmm\build --config Release --target test_canonical_calculators
C:\Users\gdwol\Development\libhmm\build\tests\Release\test_canonical_calculators.exe

# Check SIMD configuration
cmake -B C:\Users\gdwol\Development\libhmm\build C:\Users\gdwol\Development\libhmm 2>&1 | Select-String "SIMD"
```
