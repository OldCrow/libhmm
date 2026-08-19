# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project Overview

C++20 Hidden Markov Model library. Zero external dependencies (C++20 standard library only). GTest is fetched via `FetchContent` only for the test suite. Produces both a shared (`hmm`) and static (`hmm_static`) library from a single OBJECT target.

`main` is the stable v4 branch (current release: v4.3.0). Multivariate HMM support is provided via `BasicHmm<Obs>` and `BasicEmissionDistribution<Obs>` templates. `using Hmm = BasicHmm<double>` and `using EmissionDistribution = BasicEmissionDistribution<double>` preserve v3 source compatibility; users consuming only the v3 API can build from `main` unchanged.

## Session Start

At the start of every session, perform these steps in order:

1. Verify machine architecture before making SIMD assumptions — SIMD flag selection (`-march=native` on GCC/Clang, CPU-probed `/arch:` on MSVC) is automatic at compile time, but active tier affects which code paths run.
2. Select the matching build path (see Platform-Specific Notes).
3. On first use on a new machine, run `cmake --preset release && cmake --build build`, then verify the detected SIMD tier with `./build/tools/system_inspector` (if built with `LIBHMM_BUILD_TOOLS=ON`).

Quick architecture checks:

```bash
# macOS/Linux shells
uname -m
uname -s
```

```powershell
# PowerShell (Windows)
[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
[System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
```

## Agent Workflow

- When reviewing repository state or "what's changed" (e.g., syncing after time away, catching up on a branch), start with `git diff --stat` and `git log` rather than reading full file contents. Read complete files only for items you've determined are directly relevant to the task at hand.
- For any subagent expected to run more than ~30 minutes, structure its brief to report interim progress at natural milestones (e.g., after each major deliverable) rather than running silently to a single final report.

## Build Commands

CMake presets map to fixed binary directories:

```bash
# Release (default; output in build/)
cmake --preset release && cmake --build build

# Debug (output in build-debug/)
cmake --preset debug && cmake --build build-debug

# RelWithDebInfo — preferred for profiling (output in build-relwithdebinfo/)
cmake --preset rel-with-debug && cmake --build build-relwithdebinfo
```

`RelWithDebInfo` uses the same optimization flags as `Release`, adding only
debug symbols for profiler resolution — measured performance numbers are
equivalent to Release; use RelWithDebInfo only when the profiler needs
symbol information.

```bash
# Manual configure (no preset)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Build options: `LIBHMM_BUILD_EXAMPLES`, `LIBHMM_BUILD_TESTS`, `LIBHMM_BUILD_TOOLS` (all `ON` by default, i.e. `${PROJECT_IS_TOP_LEVEL}`), `LIBHMM_BUILD_BENCHMARKS` (`OFF`), `LIBHMM_ENABLE_CLANG_TIDY` (`OFF`), `LIBHMM_WERROR` (`OFF`). The old unprefixed names (`BUILD_EXAMPLES`, `BUILD_TESTS`, `BUILD_TOOLS`, `BUILD_BENCHMARKS`, `ENABLE_CLANG_TIDY`) were retired in v4.3.0 and are **no longer honoured** — the v4.2.x mapping shim is gone. Passing one while libhmm is the top-level project warns that it is being ignored; as a subproject libhmm does not react to them at all, since an unprefixed `BUILD_TESTS` belongs to the superproject and that collision is what the rename existed to end. `ENABLE_STATIC_ANALYSIS`/`ENABLE_CPPCHECK` were deleted outright (never consumed).

### CMake standard

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
  generated `libhmm/config.h`, never in `target_compile_definitions`.**
  Template `cmake/libhmm_config.h.in` → `${CMAKE_BINARY_DIR}/include/libhmm/
  config.h`, installed beside the hand-written headers. A `PRIVATE`
  definition reaches the library's own TUs and nothing else, so test TUs and
  installed consumers compile a *different* body for the same `inline`
  function — an ODR violation, and one that hides real defects because no
  test ever compiles the shipped branch. That is exactly what happened with
  `LIBHMM_HAS_CXX17_BESSEL` (issue #75, and it is why #72 survived). A header
  also covers pkg-config and plain-include-path consumers, which a target
  property cannot reach. `LIBHMM_HAS_CXX17_BESSEL` is currently the only such
  fact; add new ones to the same header. `consumer_example/main.cpp` asserts
  the installed tier two-sidedly, so a regression fails CI rather than going
  quiet.

## Test Commands

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

## Platform-Specific Notes

**Compiler prerequisites:**
- **macOS:** Xcode Command Line Tools (`xcode-select --install`) provides AppleClang. Full Xcode is not required for the library build. macOS 13 (Ventura) is the minimum supported version in v4. macOS 12 and earlier are not supported; use v3.8.0 or fork. See MIGRATION.md.
- **Linux:** GCC ≥ 12 (`apt install g++-12`) or Clang ≥ 14 (`apt install clang-14`) for C++20 support. CMake ≥ 3.25 (`apt install cmake` or from cmake.org).
- **Windows:** Visual Studio 2022 (Build Tools or full IDE) with the C++ workload. VS 2022 Build Tools or full VS is sufficient. Install from https://aka.ms/vs/17/release/vs_buildtools.exe, or `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`. GTest is fetched automatically via `FetchContent`; no vcpkg needed.

### Windows toolchain setup

> **Windows tool paths vary** by installation method (direct installer, `winget`, `chocolatey`, Microsoft Store, etc.). The paths below are common defaults — adjust for your installation. VS Build Tools and full VS editions use different default directories.

Activate the MSVC toolchain once per PowerShell session before building:

```powershell
# Default path for VS 2022 Build Tools. For full VS (Community/Professional/Enterprise),
# replace "BuildTools" with your edition under "C:\Program Files\Microsoft Visual Studio\2022\".
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
# Auto-detect any edition instead:
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

**One-time setup:**
- Visual Studio 2022 Build Tools (not full IDE) is sufficient. Install from https://aka.ms/vs/17/release/vs_buildtools.exe, `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`.
  - Build Tools default path: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\`
  - Full VS default path: `C:\Program Files\Microsoft Visual Studio\2022\{edition}\`
- **Smart App Control must be Off** (Windows Security → App & Browser Control → SAC settings). SAC blocks locally compiled executables and cannot be re-enabled without a Windows reset.
- CMake ≥ 3.25: https://cmake.org/download/, `winget install Kitware.CMake`, or `choco install cmake`.

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

Top-level headers sit between layers 3 and 4: `basic_hmm.h` (the
`BasicHmm<Obs>` model itself), `hmm.h` (`Hmm`/`HmmMV` aliases, clone/sample
helpers), and `topology.h` (structural transition masks —
`initialize_topology`/`enforce_topology` over `BasicHmm`, #46).
`libhmm.h` is the single umbrella include.

### v4 template parameterization

`BasicHmm<Obs>` and `BasicEmissionDistribution<Obs>` are the new v4 base types:

- `using Hmm = BasicHmm<double>` — scalar HMM (v3 API preserved)
- `using HmmMV = BasicHmm<ObservationVectorView>` — multivariate HMM (v4 addition); `ObservationVectorView = std::span<const double>`; emission slots start null and must be set explicitly
- `using EmissionDistribution = BasicEmissionDistribution<double>`

`Hmm` is non-copyable but movable. Default construction creates a 4-state model with `GaussianDistribution` emissions on the scalar path.

### SIMD strategy

SIMD compile flags (`LIBHMM_BEST_SIMD_FLAGS` = `-march=native` on GCC/Clang, CPU-probed `/arch:AVX512|AVX2|AVX` on MSVC) are applied **per-TU** to `LIBHMM_SIMD_SOURCES`—not globally—so non-SIMD code compiles at the platform baseline ISA. Since #58 (v4.4.0), that list holds only the distribution batch-override TUs: the FB calculators, BW/MAP trainers, and `transcendental_kernels.cpp` compile at baseline and reach their SIMD kernels through the dispatch table, so `LIBHMM_PORTABLE` costs them nothing.

There are two tiers of SIMD implementation:

- **Tier 2 (explicit intrinsics, runtime-dispatched)**: 11 of 16 scalar distributions route `getBatchLogProbabilities` through the `DoubleVecOps` dispatch table (`performance/simd_double_ops.h`). The table is built once at startup via CPUID and caches function pointers into 5 per-ISA TUs (`simd_double_ops_{scalar,sse2,avx2,avx512,neon}.cpp`), each compiled with a targeted flag rather than `-march=native`. The 5 remaining scalar distributions (Discrete, Poisson, Binomial, NegativeBinomial, Uniform) are tier-1 only. `ForwardBackwardCalculator` and `BaumWelchTrainer` reach their recurrence/accumulation kernels through the same table: `TranscendentalKernels` is a thin facade whose six kernels live per-ISA in the `simd_double_ops_*.cpp` TUs (#58), using the shared helpers in `detail/simd_math_helpers.h`.
- **Tier 1 (compiler auto-vectorization)**: Five scalar distributions remain tier-1 by design. Four of the five are blocked on **gather**, not on any missing transcendental:
    - **Poisson, Binomial**: the observation is an integer count, so the log-factorial terms are a **table lookup** (`math/log_factorial.h`, exact to k = 18, tabulated to k = 1023), not a transcendental call. Binomial never calls `lgamma` at all — `logBinomialCoefficient` is three lookups. The blocker is therefore the **gather** to index by k, the same one as Discrete below, and libstats settled empirically (its #33) that x86 hardware gather is too expensive to pay for; table kernels are a NEON technique, not an x86 one. Until 2026-08-16 this entry claimed all three were blocked on a portable vectorized `lgamma`; that was wrong for these two — see the Poisson/Binomial batch-path comments.
    - **NegativeBinomial**: genuinely needs a vectorized `lgamma`, and is the only one of the three that does. `log Γ(k + r)` has a continuous `r` so it cannot be tabulated, while `log k!` already is and `log Γ(r)` is a per-parameter constant — one `lgamma` per element. This is the single concrete case for a vectorized-lgamma dependency (e.g. corvus); size any such proposal against one distribution, not three.
    - **Discrete**: per-element integer floor + range check and table lookup by symbol index. Vectorizable in principle via AVX2 gather, but complex index arithmetic and no performance data justifying the effort.
    - **Uniform**: the entire batch evaluates to a single constant (log(1/(b−a))) inside bounds or −∞ outside. Already ~2 instructions per element; SIMD buys nothing.
  MV distributions (`DiagonalGaussian`, `FullCovGaussian`, `IndependentComponents`) call `getLogProbability(row_view(obs, t))` per timestep rather than a batch interface and are not in `LIBHMM_SIMD_SOURCES`.

`detail/simd_math_helpers.h` is the single source of truth for vectorized log/exp/cos/sin/log1p helpers shared by the per-ISA distribution kernels and `TranscendentalKernels`. log/exp are SLEEF-derived (< 1 ULP). cos/sin are the clean-room quadrant-reduction kernel (#74, constants from `scripts/gen_trig_cleanroom_table.py`): faithfully rounded (max 1 ULP, mean ~0.03) for |x| ≤ 2²³, per-lane scalar libm fixup beyond, gated per tier against checked-in mpmath references in `tests/performance/test_trig_ulp_gates.cpp`. Tiny `log1p` inputs use a polynomial path for accuracy; general inputs reuse the shared vector log helper (note the `log1p_batch` table entry does NOT have that small-|x| path — issue #77).

`getBatchLogProbabilities(std::span<const double> obs, std::span<double> out)` is the SIMD interface: calculators call it once per state per `compute()` and consume a flat row-major buffer of log-emission values.

**FP contraction, and what libhmm does not promise** (audited 2026-08-16, issue #70). The build sets no `-ffp-contract` flag, so every TU takes its compiler's default — GCC `fast`, AppleClang `on`, MSVC/clang-cl `precise` (off). That is safe here — but as of v4.4.0 the reason changed, so restate it precisely. The trig kernels (#74) DO carry compensated/EFT-style sequences: the (r, rlo) reduction recovers each subtraction's rounding error, and the cos core keeps 1 − u/2 as an exact head+tail pair. Those sequences are nonetheless contraction-immune, because every product feeding them is exact by construction (n·p_k under the 30-bit split; u·0.5 is power-of-two scaling) — fusing an exact product changes nothing, so no proof depends on an intermediate rounding. Beyond the trig kernels the earlier audit still applies: no Kahan/Neumaier, no TwoSum/Fast2Sum, no `fma(a,b,-a*b)` residual trick. `logSumExp` is the plain `max + log1p(exp(Δ))` form and the FB/BW reductions are plain accumulations. The `ln2_hi`/`ln2_lo` Cody-Waite splits in `log_pd`/`exp_pd` look like the hazard but are not it — they compensate a *constant's* representation error, not an *operation's* rounding, so no proof depends on an intermediate rounding as written. Every deliberate fusion in those kernels is already an explicit `_mm*_fmadd_pd`/`fnmadd` intrinsic: fusion requested, never inferred.

**libhmm claims no bit-reproducibility of likelihoods or trained parameters across compilers, platforms, or machines, and cannot** — the tier dispatch itself prevents it. `TranscendentalKernels::sum_exp_sum2_minus_max` and friends accumulate into 8-, 4-, or 2-wide partial sums depending on which tier CPUID selects, so the summation tree and its rounding change with the CPU the binary runs on at fixed compiler and fixed flags. `-ffp-contract=off` would therefore buy nothing here (unlike corvus, which needs it for its double-double primitives) while costing FMA in accumulations where fusion is accuracy-positive.

**If you add a compensated sequence, it stops being safe.** Anything whose correctness rests on an intermediate rounding happening exactly as written must be contraction-proofed at the point of introduction — either spell every fusion explicitly, or scope `-ffp-contract=off` to that source file. Do not assume the audit above still holds after such a change.

Threading is **not used** in the production path — a deliberate, settled decision since the Phase 4 refactor replaced the Plan-A `WorkStealingPool`-based hierarchy with per-distribution batch SIMD (Plan B). `ThreadPool` was subsequently moved out of the library entirely, from `libhmm/platform/thread_pool.h` into `tools/thread_pool.h`, since no production code (calculators, trainers, distributions, HMM core) ever instantiated it; today it is consumed only by two diagnostic tools in `tools/`. See PLAN.md for the tracked GitHub issue proposing to revisit this for parallel E-step accumulation.

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

## Coding Conventions

See `docs/STYLE_GUIDE.md` for the full guide. Key points:

- `#pragma once` always (no `#ifndef` guards); enforced by pre-commit hook
- Classes: `PascalCase`; methods and local variables: `camelCase`; private members: `camelCase_` (trailing underscore); constants: `SCREAMING_SNAKE_CASE`
- All distributions must implement the separate `validateParameters()` pattern: called in constructors and setters, throws `std::invalid_argument`
- Each distribution `.cpp` uses `using namespace constants;` inside `namespace libhmm` — no magic numbers
- Expensive per-call values (normalization constants, log parameters) are cached via `mutable` members invalidated by setters; the cache flag is `std::atomic<bool>` for read-thread safety
- K&R brace style, 4-space indentation, 100-character line limit (enforced by `.clang-format`)
- Use `[[nodiscard]]`, `noexcept`, `std::span`, `std::optional` where appropriate (C++20)

### Tool prerequisites

The linting and pre-commit tools must be installed before use:
- **clang-format**: part of LLVM (`brew install llvm`, `apt install clang-format`, `choco install llvm`)
- **cmake-format**: `pip install cmake-format`
- **pre-commit**: `pip install pre-commit`
- **cppcheck**: OS-package-managed (`brew install cppcheck`, `apt install cppcheck`, `choco install cppcheck`)

### Linting and formatting

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

## Common Development Tasks

### Adding a new distribution

Use an existing distribution (e.g. `src/distributions/rayleigh_distribution.cpp` for single-parameter, `src/distributions/gamma_distribution.cpp` for two-parameter) as a template. Required checklist (`docs/GOLD_STANDARD_CHECKLIST.md`):

1. Concrete non-virtual `getBatchLogProbabilities` override (tier 1 minimum)
2. Weighted `fit(data, weights)` with near-zero weight guard
3. `reset()`, `clone()`, `sample()`, `to_json()` / `from_json()` (registered in `src/io/hmm_json.cpp`), `getNumParameters()`
4. `std::atomic<bool> cacheValid_` thread-safe cache
5. Add source to `LIBHMM_SIMD_SOURCES` and `LIBHMM_SOURCES` in `CMakeLists.txt`
6. Add a test file under `tests/distributions/` and register it in `tests/CMakeLists.txt`

## CI / Validation

Fleet-wide workflow rules (runner budget, bounded parallelism, ISA hazards on
hosted runners, action pinning):
[CI House Style](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md).

Four parallel build-matrix jobs: Linux/GCC, Linux/Clang, macOS/AppleClang, Windows/MSVC 2022. Additional quality jobs (ubuntu): pre-commit, cppcheck, and clang-tidy. Tests run with `-LE "known_broken|benchmark"`.

`LIBHMM_ENABLE_CLANG_TIDY` (CMake option, `OFF` by default) wires clang-tidy into the normal build via the `CXX_CLANG_TIDY` target property; enable locally with `cmake --preset release -DLIBHMM_ENABLE_CLANG_TIDY=ON` when needed. The dedicated CI `clang-tidy` job instead runs `run-clang-tidy` against `compile_commands.json` as a single fast analysis pass and is **advisory (non-blocking)**: `continue-on-error: true`. Decision record (issue #62): the full default check set produces ~3765 warnings dominated by checks that conflict with libhmm's documented architecture — six checks are disabled in `.clang-tidy` (see that file for the full rationale) covering the pragma-once convention, intentional SIMD intrinsics/pointer arithmetic in perf-critical hot paths, the v4 template+virtual pattern, and a false-positive move-ctor idiom. **The residual is 86 UNIQUE sites, not the ~1344 the raw log suggests** — `run-clang-tidy` re-reports every header diagnostic once per including TU, so a single flagged line in a widely-included header can appear 20+ times. Count unique `file:line:col + check` tuples before drawing conclusions from this job. #63 (the `modernize-use-nodiscard` cluster in the linalg headers) is closed; measured 2026-08-16, what remains is `modernize-use-auto` (16), `modernize-concat-nested-namespaces` (9), `modernize-raw-string-literal` (8), `modernize-use-default-member-init` (7), `modernize-avoid-c-arrays` (7), and a long tail — all mechanical. 86 mechanical sites is a tractable target, so promoting this job to blocking is now a scheduling question rather than a scale one (see PLAN.md).

## Open Items
See PLAN.md for current status, in-progress work, and open questions.
Distribution fit-quality roadmap specifically lives in
docs/GOLD_STANDARD_CHECKLIST.md — PLAN.md points to it rather than
duplicating it.
