# Cross-Platform Build Guide

libhmm builds and tests on Windows (MSVC), macOS (AppleClang), Linux (GCC), and Linux (Clang).
All four platforms are verified by CI on every push.

## Requirements

- **C++20** compiler: GCC 12+, Clang 14+, Apple Clang 14+ (Xcode 14, macOS 13+), MSVC 2022 (17.x) or later
- **CMake 3.25+**
- **Zero external dependencies** at runtime — GTest is fetched automatically via FetchContent

## Platforms

### Windows (MSVC)

CMake selects the newest installed Visual Studio; pin one with e.g.
`-G "Visual Studio 17 2022"` if several are installed.

```powershell
cmake -B build -A x64
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --parallel 4
```

Notes:
- Do NOT call `vcvars64.bat` before cmake; the VS generator handles it
- SIMD: `check_cxx_source_runs` selects `/arch:AVX512`, `/arch:AVX2`, or `/arch:AVX` by running a test binary to verify CPU support — prevents ILLEGAL INSTRUCTION crashes on cloud VMs that accept the flag but can't execute it

### macOS (AppleClang)

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build
```

Notes:
- Homebrew prefix: `/opt/homebrew` (Apple Silicon) or `/usr/local` (Intel)
- CMake detects the architecture automatically via `uname -m`
- SIMD: `-march=native` — selects NEON on AArch64, AVX/AVX2 on Intel Macs

### Linux (GCC)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build
```

Notes:
- Math library (`-lm`) linked explicitly on Linux
- SIMD: `-march=native` — same as macOS

## Configuration Summary

When cmake runs, the SIMD configuration is reported:

```
-- SIMD support — AVX-512: 1  AVX2: 1  AVX: 1        (Windows, Ryzen 7)
-- SIMD optimization: /arch:AVX512 (CPU verified)

-- SIMD compiler support — NEON: ON (AArch64 baseline) (macOS Apple Silicon)
-- SIMD optimization: -march=native (AArch64/NEON)

-- SIMD support — AVX-512: 0  AVX2: 1  AVX: 1         (macOS Intel)
-- SIMD optimization: -march=native
```

## Build Options

```bash
cmake -DLIBHMM_BUILD_EXAMPLES=OFF ..    # Skip examples
cmake -DLIBHMM_BUILD_TESTS=OFF ..       # Skip tests
cmake -DLIBHMM_BUILD_TOOLS=OFF ..       # Skip tools
```

Both `hmm` (shared) and `hmm_static` (static) always build from one OBJECT
target — there is no `BUILD_SHARED_LIBS` toggle to suppress the shared
library.

## Library Output

Both shared and static libraries are produced in every build. Sources compile once into an
OBJECT library and link into both targets — no double compilation overhead.

| Platform | Files produced | Notes |
|---|---|---|
| macOS | `libhmm.dylib`, `libhmm.a` | Shared: `@rpath`; static: archive |
| Linux | `libhmm.so`, `libhmm.a` | Shared: `$ORIGIN` RPATH; static: archive |
| Windows | `hmm.dll`, `hmm.lib`, `hmm_static.lib` | `hmm.lib` = DLL import lib; `hmm_static.lib` = static archive |

## CI Matrix

See `.github/workflows/ci.yml`. Four build jobs (Linux/GCC, Linux/Clang, macOS/AppleClang,
Windows/MSVC) plus a lint job (clang-format + cppcheck). All 47 tests pass on every platform.

## CMake Standard

Full rules: [CMake House Style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md)
in the fleet standards repo; this section is self-sufficient for this repo. Deviations from that
standard, current as of Phase 3A (target-first + option rename):
- Target-first scoping and `LIBHMM_`-prefixed options are landed: includes
  and warning flags are applied via `target_include_directories`/
  `target_compile_options` on `hmm_objects` (and per-target on tests/tools/
  examples), gated on `PROJECT_IS_TOP_LEVEL`; component-toggle options
  default `${PROJECT_IS_TOP_LEVEL}`. `LIBHMM_WERROR` (default `OFF`) is the
  `-Werror`/`/WX` vehicle, enabled by CI.
- `BUILD_SHARED_LIBS` is removed (both `hmm`/`hmm_static` always build from
  one OBJECT target — there was never a real toggle to preserve).
- Install contract already conforms: GNUInstallDirs, `libhmm-targets` export
  (namespace `libhmm::`), kebab `libhmm-config.cmake`, `SameMajorVersion`.
- Presets (`CMakePresets.json`, schema 6, min CMake 3.25): `release` →
  `build/`, `debug` → `build-debug/`, `rel-with-debug` →
  `build-relwithdebinfo/`. No project-specific extras.
- **A configure-time fact that a PUBLIC header branches on goes in the
  generated `libhmm/config.h`, never in `target_compile_definitions`.** House
  rule from [CMake House Style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md).
  Template `cmake/libhmm_config.h.in` → `${CMAKE_BINARY_DIR}/include/libhmm/
  config.h`, installed beside the hand-written headers. A `PRIVATE`
  definition reaches the library's own TUs and nothing else, so test TUs and
  installed consumers compile a *different* body for the same `inline`
  function — an ODR violation, and one that hides real defects because no
  test ever compiles the shipped branch. A header also covers pkg-config and
  plain-include-path consumers, which a target property cannot reach.
  `LIBHMM_HAS_CXX17_BESSEL` is currently the only such fact; add new ones to
  the same header. `consumer_example/main.cpp` asserts the installed tier
  two-sidedly, so a regression fails CI rather than going quiet.

## Tooling prerequisites

Moved here from AGENTS.md on 2026-09-07: one-time install commands, needed
when setting a machine up rather than in every session.

The linting and pre-commit tools must be installed before use:
- **clang-format**: part of LLVM (`brew install llvm`, `apt install clang-format`, `choco install llvm`)
- **cmake-format**: `pip install cmake-format`
- **pre-commit**: `pip install pre-commit`
- **cppcheck**: OS-package-managed (`brew install cppcheck`, `apt install cppcheck`, `choco install cppcheck`)
- **mpmath** (`pip install mpmath`): only for regenerating the checked-in trig tables/references with `scripts/gen_trig_cleanroom_table.py` and `scripts/gen_trig_ulp_vectors.py`; not needed to build or test
