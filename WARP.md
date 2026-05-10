# WARP.md

This file provides guidance to Warp (warp.dev) when working in this repository.

# libhmm — Modern C++20 Hidden Markov Model Library

## Current Status

**Version**: v3.5.3 — pending merge on `refactor/code-quality-phase7`; v3.5.2 is the latest published tag on `main`.
**Tests**: 37/37 passing on all four CI platforms (Linux/GCC, Linux/Clang, macOS/AppleClang, Windows/MSVC).
**Active phase**: Code quality roadmap complete. All lizard warnings triaged; Tier 6 (SIMD boilerplate in `transcendental_kernels.cpp`) is the only deferred structural item.

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
tests/              # GTest suite — semantic groups (see tests/CMakeLists.txt)
examples/           # 13 usage demonstrations (all canonical API)
tools/              # Standalone diagnostic/benchmarking executables
benchmarks/         # Comparative benchmarks
│   ├── src/        #   libhmm vs HMMLib / LAMP / JAHMM (Windows+Unix)
│   │               #   libhmm vs GHMM / HTK (macOS/Linux only)
│   └── docs/       #   BENCHMARKING_RESULTS.md, Library_Compatibility_Guide.md
```

### What was removed in the refactor

- All SIMD calculator variants (`ScaledSIMD*`, `LogSIMD*`, `AdvancedLog*`)
- `AutoCalculator` / `CalculatorSelector` / `CalculatorTraits`
- `ScaledBaumWelchTrainer`, `RobustViterbiTrainer`
- `ProbabilityDistribution` base class (replaced by `EmissionDistribution`)
- `HmmTrainer` base (replaced by `Trainer`)

### Build outputs

Sources compile once into `hmm_objects` (OBJECT library), then linked into:
- `hmm` — shared library (`hmm.dll` / `libhmm.dylib` / `libhmm.so`)
- `hmm_static` — static archive (`hmm_static.lib` / `libhmm.a`)

Both are always produced regardless of `BUILD_SHARED_LIBS`. Tests link against
`hmm_static` to avoid Windows DLL path issues at test runtime.

---

## Key Design Decisions

1. **`EmissionDistribution` base** — const-correct, `std::span` params, `getBatchLogProbabilities()` batch hook for SIMD, `fit(data, weights)` for Baum-Welch M-step.

2. **Two canonical calculators** — `ForwardBackwardCalculator` (log-space, precomputed log-trans) and `ViterbiCalculator`. Both call `getBatchLogProbabilities()` per state per time step.

3. **Compile-time SIMD dispatch** — source-distributed; each machine builds for its own CPU. GCC/Clang: `-march=native`. MSVC: `check_cxx_source_runs`-verified `/arch:AVX512`/`AVX2`/`AVX`. All 15 distribution TUs plus transcendental kernels, FB calculator, and BW trainer in `LIBHMM_SIMD_SOURCES`. Tier 2 explicit intrinsics: Gaussian, Exponential, LogNormal, Pareto via `detail::` free functions; recurrence kernels (FB max-reduce, BW xi) via `TranscendentalKernels` in `src/performance/`. Shared vector exp/log helpers in `include/libhmm/performance/simd_kernels_internal.h`.

4. **Thread-safe cache** — `std::atomic<bool> cacheValid_` in `DistributionBase`. Avoids mutex; safe for concurrent const reads if the library is invoked from multiple threads (calculators and trainers themselves run single-threaded — see `performance/PERFORMANCE_ARCHITECTURE.md`).

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
| Catalina guard | macos-latest | AppleClang | Configure-only pass/fail guard checks with `CMAKE_OSX_DEPLOYMENT_TARGET=10.15` |
| Pre-commit | ubuntu-latest | — | `pre-commit run --all-files` (gating) |
| Cppcheck | ubuntu-latest | — | Static analysis on `src/` (gating) |

GTest is fetched via `FetchContent` if not found locally — no vcpkg required on CI.

---

## Session Start Baseline Workflow (Required)

Run this sequence at the start of every `libhmm` session:

1. Verify host architecture and CPU family before configuring/building.
2. Choose the platform-specific build path for this machine (macOS non-Catalina, macOS Catalina, or Windows MSVC).
3. If the machine/architecture changed since the previous session, reconfigure from a clean build directory before comparing performance or SIMD behavior.

Architecture checks:

```bash
# macOS/Linux shells
uname -m
uname -s
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
```

```powershell
# PowerShell
[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
[System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
$env:PROCESSOR_IDENTIFIER
```

Routing:
- **Windows/MSVC:** follow `## Windows Session Setup (Asus TUF A16)` and use Visual Studio 2022 x64 Release commands.
- **macOS (non-Catalina):** follow `## macOS Session Notes` with standard `cmake -S ... -B ...` flow.
- **macOS Catalina (10.15):** follow `### Catalina startup workflow (fresh clone/sync)` exactly (`./scripts/configure_catalina.sh build`, no Homebrew LLVM/libc++ hints unless troubleshooting).

`libhmm` uses compile-time SIMD dispatch (`-march=native` on GCC/Clang; CPU-selected `/arch` on MSVC), so architecture mismatches directly change generated binaries and observed behavior.

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

Tests link against `hmm_static`; no DLL on PATH required.

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

### Git Commits and Tags

`commit.gpgSign` and `tag.gpgSign` are both `true`. If the GPG agent times out,
restart it first:
```powershell
gpgconf --kill all
git commit -m "message"
```
For tags, use `--no-sign` as a fallback if the agent keeps timing out:
```powershell
git tag -a vX.Y.Z --no-sign -m "message"
```
When moving an existing tag: add `-f` to both the local `tag` command and the
`push --force`.

CRLF: `.gitattributes` enforces LF. CRLF warnings on `git add` are normal.

---

## macOS Session Notes

- Homebrew: `/opt/homebrew` (Apple Silicon) or `/usr/local` (Intel)
- SIMD: `-march=native` automatically selects NEON on arm64, AVX/AVX2 on Intel
- Configure (non-Catalina): `cmake -S libhmm -B libhmm/build`
- Tests: `ctest --test-dir libhmm/build --parallel 4`

### Catalina startup workflow (fresh clone/sync)

- Always run `./scripts/configure_catalina.sh build` for the first configure.
- The script sanitizes toolchain-related environment variables, pins AppleClang via `xcrun`, and sets `CMAKE_OSX_DEPLOYMENT_TARGET=10.15`.
- **Build type:** the script defaults to `Release` (`-O3`). This is required for correctness: at `-O0`, AppleClang inserts `VZEROUPPER` in the prologue of large-frame AVX functions before saving the `__m256d` argument, silently zeroing `x[2]` and `x[3]`. For debuggable builds use `RelWithDebInfo` (`-O2 -g`) — SIMD helpers inline at `-O2` so the issue cannot occur: `./scripts/configure_catalina.sh build -DCMAKE_BUILD_TYPE=RelWithDebInfo`. Pure `Debug` (`-O0`) is unsafe for any code path that passes `__m256d` through a real call boundary.
- Do not point Catalina builds at Homebrew LLVM/libc++ (`/usr/local/opt/llvm`, `Cellar/llvm*`, libc++ include paths). The root `CMakeLists.txt` guard fails configure when those hints are detected.
- Use `-DLIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX=ON` only for explicit troubleshooting; runtime stability is not guaranteed.

---

## Test Suite Structure

Tests in `tests/CMakeLists.txt` use `add_hmm_test()` helper organized into semantic groups:

| Group | Content |
|---|---|
| Platform Capabilities | No tests yet (placeholder) |
| Math & Numerics | constants, numerical stability, common types |
| Performance Primitives | transcendental kernels (SIMD parity vs `std::exp`) |
| Distributions | all 15 + traits/header/type_safety |
| Core HMM | HMM construction and state management |
| Calculators | canonical + continuous + edge cases + FB mode parity |
| Trainers | canonical + training + edge cases + BW convergence + BW parity |
| IO & Integration | stream IO + end-to-end casino |

Custom targets: `check` (correctness, parallel), `check_timing` (serial).
Note: named `check` not `run_tests` to avoid cmake's built-in `RUN_TESTS` on Windows.

---

## Known Issues

| Issue | Status | Notes |
|---|---|---|
| `test_xml_file_io` | Resolved | Fixed in v3.0.0 — platform-guarded test path now correctly provokes the error on both Windows and Unix. |
| Benchmarks | Complete | Comparative suite run on macOS and Windows (April 2026). Results in `benchmarks/docs/BENCHMARKING_RESULTS.md`. LAMP and JAHMM benchmarks now build and run on Windows. GHMM and HTK require macOS/Linux (POSIX/Autotools dependencies); CMake skips their targets on Windows. libhmm throughput ~8–14k obs/ms; faster than StochHMM and JAHMM. HMMLib ~3×, GHMM ~5× faster than libhmm. |
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
