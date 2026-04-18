# WARP.md

This file provides guidance to Warp (warp.dev) when working in this repository.

# libhmm — Modern C++ Hidden Markov Model Library

## Project Overview

libhmm is a C++17 Hidden Markov Model engine with 15 emission distributions, three Forward-Backward variants (standard, scaled SIMD, log SIMD), three Viterbi variants, and Baum-Welch and Viterbi training algorithms. Zero external runtime dependencies.

**Current status**: v2.9.1 on `main`. Active development is on `refactor/modern-architecture`.

### Development History

libhmm v2.9.1 reached a complete feature set but halted because the distribution layer had foundational design problems (non-const `getProbability`, no batch interface for SIMD calculators, manual per-class cache flags, no weighted `fit()` for Baum-Welch). **libstats** was built as a companion project to learn and demonstrate the solutions. The `refactor/modern-architecture` branch applies those lessons back here.

See the Warp plan artifact "libhmm Modernization: Architecture & Refactoring Plan" for the full refactoring roadmap.

## Architecture Overview

### Current (v2.9.1) Layer Structure

```
include/libhmm/
├── common/         # Mixed bag: matrix/vector types, god header (common.h 21KB), numerics
├── performance/    # SIMD detection, threading, log-space ops, benchmarking — all mixed together
├── distributions/  # 15 distributions + ProbabilityDistribution base (C++17, #ifndef guards)
├── calculators/    # Forward-Backward (3 variants), Viterbi (3 variants), AutoCalculator
├── training/       # Baum-Welch, scaled Baum-Welch, Viterbi trainer, segmented K-Means
└── io/             # XML I/O
```

### Target (refactor/modern-architecture) Layer Structure

```
include/libhmm/
├── platform/       # Layer 0: SIMD detection, CPU features, arch-specific dispatch thresholds
├── math/           # Layer 1: constants, special functions, log-space ops, numerical utils
├── linalg/         # Layer 2: matrix/vector types (consolidated from common/)
├── distributions/  # Layer 3: EmissionDistribution interface + DistributionBase + 15 distributions
├── calculators/    # Layer 4: Forward-Backward, Viterbi variants, AutoCalculator
├── training/       # Layer 4: Baum-Welch, Viterbi training, K-Means
└── io/             # XML I/O
```

### Key Design Decisions (refactor targets)

- `EmissionDistribution` replaces `ProbabilityDistribution`: const-correct, with `getBatchLogProbabilities()` for calculator SIMD paths and weighted `fit(data, weights)` for Baum-Welch
- `#pragma once` everywhere (was `#ifndef` guards)
- C++20 standard (was C++17)
- CMake SIMD detection gates on **compiler capability** (`check_cxx_compiler_flag`), not build machine CPU — per-TU flags via `set_source_files_properties`, dispatch TU compiled with no SIMD flags

## Development Ecosystem

| Machine | OS | CPU | SIMD | Role |
|---|---|---|---|---|
| Asus TUF A16 (2025) | Windows 11 Pro | AMD Ryzen 7 7445 (Zen 4) | SSE2 + AVX + AVX2 + AVX-512 | Primary Windows/MSVC dev machine |

## Windows Session Setup (Asus TUF A16)

### Configure and Build

The Visual Studio generator handles the MSVC toolchain internally — **vcvars64.bat does not need to be called before cmake**. Do not call vcvars in a shell that will then run cmake; the combined environment exceeds cmd.exe's line-length limit.

```powershell
$repo = "C:\Users\gdwol\Development\libhmm"
$vcpkgPrefix = "C:/vcpkg/installed/x64-windows"

# Configure (run from any PowerShell — no MSVC activation needed for VS generator)
cmake -S $repo -B "$repo\build" `
    -G "Visual Studio 17 2022" -A x64 `
    -DBUILD_SHARED_LIBS=OFF `
    "-DCMAKE_PREFIX_PATH=$vcpkgPrefix"

# Build
cmake --build "$repo\build" --config Release
```

### Run Tests

GTest is installed as a DLL. Copy the DLLs alongside test executables before running ctest:

```powershell
Copy-Item "C:\vcpkg\installed\x64-windows\bin\gtest.dll" "C:\Users\gdwol\Development\libhmm\build\tests\Release\" -Force
Copy-Item "C:\vcpkg\installed\x64-windows\bin\gtest_main.dll" "C:\Users\gdwol\Development\libhmm\build\tests\Release\" -Force

ctest --test-dir "C:\Users\gdwol\Development\libhmm\build" -C Release --output-on-failure --timeout 60
```

### vcvars Activation (for command-line tools only)

Only needed if you are running MSVC command-line tools directly (e.g., `cl`, `link`, `dumpbin`):

```powershell
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$envVars = cmd /c "`"$vcvars`" > nul && set"
foreach ($line in $envVars) {
    if ($line -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
```

### GTest Location

GTest is installed via vcpkg at `C:\vcpkg\installed\x64-windows`.
Pass `-DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows` at configure time.

The vcpkg toolchain file (`C:/vcpkg/scripts/buildsystems/vcpkg.cmake`) is present but this vcpkg instance is in manifest mode. Using `CMAKE_PREFIX_PATH` directly is more reliable.

### One-Time Setup Notes

- Visual Studio 2022 Build Tools (not full VS) — sufficient for cmake + MSVC
- **Smart App Control must be Off** (Windows Security → App & Browser Control → SAC settings). SAC blocks locally compiled executables and cannot be re-enabled without a Windows reset.
- GTest installed via vcpkg (`gtest:x64-windows`) — headers and DLLs at `C:\vcpkg\installed\x64-windows`
- CMake 4.x installed

## Test Baseline (Windows/MSVC, 2026-04-18)

Recorded on `main` (v2.9.1) before any refactoring, using Release static build:

**40/45 tests passing**

Known failures carried into the refactor branch:

| Test | Status | Cause |
|---|---|---|
| `test_simd_viterbi_calculators` | Not Run | Compile error: missing `<chrono>` include — MSVC C3861 |
| `test_viterbi_selection` | Not Run | Startup failure — investigate |
| `test_viterbi_traits` | Not Run | Link error: `AdvancedLogSIMDViterbiCalculator` constructor undefined |
| `test_modernized_apis` | Not Run | Same link error as above |
| `test_xml_file_io` | Failed | Runtime failure — investigate |

These pre-existing failures are **not regressions from the refactor**. The exit criterion for each refactoring phase is that the passing count does not decrease from 40/45.

## Project Structure (Current)

```
libhmm/
├── include/libhmm/     # Public headers
├── src/                # Implementation (mirrors include structure)
├── tests/              # GTest-based test suite (mirrors include structure)
├── examples/           # Usage demonstrations (10 examples, 9 build on Windows)
├── benchmarks/         # Benchmarking suite
├── performance/        # Root-level performance analysis tools
├── docs/               # Documentation (GOLD_STANDARD_CHECKLIST.md, STYLE_GUIDE.md, etc.)
├── cmake/              # CMake helper files
├── build_windows.bat   # Windows build helper (configure only — see Session Setup above)
└── WARP.md             # This file
```

## Key Source Files

- `include/libhmm/distributions/probability_distribution.h` — current base class (to be replaced by `EmissionDistribution`)
- `include/libhmm/distributions/distribution_traits.h` — C++17 type traits (keep as-is)
- `include/libhmm/common/common.h` — 21KB god header (to be decomposed in Phase 1)
- `src/performance/simd_support.cpp` — runtime SIMD detection
- `docs/GOLD_STANDARD_CHECKLIST.md` — per-distribution implementation checklist
