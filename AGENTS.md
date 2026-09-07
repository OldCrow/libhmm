# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project Overview

C++20 Hidden Markov Model library. Zero external dependencies (C++20 standard library only). GTest is fetched via `FetchContent` only for the test suite. Produces both a shared (`hmm`) and static (`hmm_static`) library from a single OBJECT target.

`main` is the stable v4 branch (current release: v4.4.1). Multivariate HMM support is provided via `BasicHmm<Obs>` and `BasicEmissionDistribution<Obs>` templates. `using Hmm = BasicHmm<double>` and `using EmissionDistribution = BasicEmissionDistribution<double>` preserve v3 source compatibility; users consuming only the v3 API can build from `main` unchanged.

## Session Start

Fleet-wide session-start steps (architecture check, build-path selection):
[Session Start](https://github.com/OldCrow/standards/blob/main/SESSION-START.md).

On first use on a new machine, run `cmake --preset release && cmake --build build`, then verify the detected SIMD tier with `./build/tools/simd_inspection` (if built with `LIBHMM_BUILD_TOOLS=ON`).

## Agent Workflow

- Before pushing C++ changes from a Windows/MSVC machine, sweep every
  changed TU with the local mingw `g++ -std=c++20 -fsyntax-only -I include
  -I build/include -I build/_deps/googletest-src/googletest/include -I src`.
  MSVC and AppleClang provide transitive standard-library includes that
  libstdc++ does not (v4.4.1 example: `<cstring>` for `std::memcpy` —
  compiled clean on MSVC and macOS, failed all four Linux CI legs), and
  this one-minute sweep catches the whole class before a CI round-trip.

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

Build options: `LIBHMM_BUILD_EXAMPLES`, `LIBHMM_BUILD_TESTS`, `LIBHMM_BUILD_TOOLS` (all `ON` by default, i.e. `${PROJECT_IS_TOP_LEVEL}`), `LIBHMM_BUILD_BENCHMARKS` (`OFF`), `LIBHMM_ENABLE_CLANG_TIDY` (`OFF`), `LIBHMM_WERROR` (`OFF`), `LIBHMM_PORTABLE` (`OFF`; swaps `-march=native` for a portable baseline ISA on `LIBHMM_SIMD_SOURCES` — for distributable wheels, leave `OFF` for local builds). The old unprefixed names (`BUILD_EXAMPLES`, `BUILD_TESTS`, `BUILD_TOOLS`, `BUILD_BENCHMARKS`, `ENABLE_CLANG_TIDY`) were retired in v4.3.0 and are **no longer honoured** — the v4.2.x mapping shim is gone. Passing one while libhmm is the top-level project warns that it is being ignored; as a subproject libhmm does not react to them at all, since an unprefixed `BUILD_TESTS` belongs to the superproject and that collision is what the rename existed to end. `ENABLE_STATIC_ANALYSIS`/`ENABLE_CPPCHECK` were deleted outright (never consumed).

### CMake standard

Deviations from [CMake House Style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md)
(target-first scoping, `LIBHMM_`-prefixed options, no `BUILD_SHARED_LIBS`, install
contract, presets): `docs/CROSS_PLATFORM.md`.

**A configure-time fact that a PUBLIC header branches on goes in the
generated `libhmm/config.h`, never in `target_compile_definitions`.** A
`PRIVATE` definition reaches the library's own TUs and nothing else, so test
TUs and installed consumers compile a *different* body for the same
`inline` function — an ODR violation, and one that hides real defects
because no test ever compiles the shipped branch. Template:
`cmake/libhmm_config.h.in` → `${CMAKE_BINARY_DIR}/include/libhmm/config.h`.
`LIBHMM_HAS_CXX17_BESSEL` is currently the only such fact; add new ones to
the same header. `consumer_example/main.cpp` asserts the installed tier
two-sidedly, so a regression fails CI rather than going quiet.

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
- **Windows:** toolchain floor, install routes, and the per-version install
  paths are in [WINDOWS-TOOLCHAIN.md §1](https://github.com/OldCrow/standards/blob/main/WINDOWS-TOOLCHAIN.md#1-one-time-setup). GTest is fetched
  automatically via `FetchContent`; no vcpkg needed.

### Windows toolchain setup

Fleet-wide MSVC activation, one-time setup, and Smart App Control notes:
[Windows Toolchain](https://github.com/OldCrow/standards/blob/main/WINDOWS-TOOLCHAIN.md).

After activating the toolchain, build with libhmm's own presets:

```powershell
cmake --preset release
cmake --build build
```

## Architecture

Headers layer strictly downward: `platform/` (0, SIMD CPU detection) →
`math/` (1, constants/Bessel/digamma) → `linalg/` (2, `BasicMatrix`/`BasicVector`)
→ `distributions/` (3, `BasicEmissionDistribution<Obs>` + 16 concrete
distributions) → `calculators/`/`training/` (4, FB/Viterbi/Baum-Welch).
`io/`, `performance/`, `detail/` sit alongside, not in the numbered stack.
Top-level: `basic_hmm.h`, `hmm.h`, `topology.h` (#46); `libhmm.h` is the
umbrella include.

`BasicHmm<Obs>`/`BasicEmissionDistribution<Obs>` are the v4 template base
types. `Hmm`/`EmissionDistribution` alias the `double` (scalar, v3-API-compatible)
instantiation; `HmmMV = BasicHmm<ObservationVectorView>` is the multivariate one.
`Hmm` is non-copyable but movable.

SIMD has two tiers: 11 of 16 distributions dispatch through `DoubleVecOps`
(runtime CPUID) at tier 2; 5 (Discrete, Poisson, Binomial, NegativeBinomial,
Uniform) stay tier-1 (compiler auto-vectorized) by design — mostly blocked on
gather cost, not on a missing transcendental.

Full layer table, v4 alias details, SIMD dispatch mechanics and per-distribution
tier rationale, fit-quality tiers, model selection, and I/O formats:
`docs/ARCHITECTURE.md`.

`getBatchLogProbabilities(std::span<const double> obs, std::span<double> out)` is the SIMD interface: calculators call it once per state per `compute()` and consume a flat row-major buffer of log-emission values. Since v4.4.1 (#86) the precondition `out.size() >= obs.size()` is enforced: every concrete override and the CRTP fallback call `checkBatchSpans()` and throw `std::invalid_argument` on a short out span. The `DoubleVecOps` raw-pointer layer below it remains unchecked by design.

**FP contraction.** The build sets no `-ffp-contract` flag, so every TU takes its compiler's default. Audited safe under #70: every deliberate fusion in the SIMD kernels is already an explicit `_mm*_fmadd_pd`/`fnmadd` intrinsic, and no proof depends on an intermediate rounding. Full audit, kernel by kernel: `docs/ARCHITECTURE.md`.

**libhmm claims no bit-reproducibility of likelihoods or trained parameters across compilers, platforms, or machines, and cannot** — the tier dispatch itself prevents it. `TranscendentalKernels::sum_exp_sum2_minus_max` and friends accumulate into 8-, 4-, or 2-wide partial sums depending on which tier CPUID selects, so the summation tree and its rounding change with the CPU the binary runs on at fixed compiler and fixed flags. `-ffp-contract=off` would therefore buy nothing here (unlike corvus, which needs it for its double-double primitives) while costing FMA in accumulations where fusion is accuracy-positive.

**If you add a compensated sequence, it stops being safe.** Anything whose correctness rests on an intermediate rounding happening exactly as written must be contraction-proofed at the point of introduction — either spell every fusion explicitly, or scope `-ffp-contract=off` to that source file. Do not assume the audit above still holds after such a change.

Threading is **not used** in the production path — a deliberate, settled decision since the Phase 4 refactor. The supported model is **caller-level parallelism**: concurrent training of distinct model instances is a documented, TSan-tested contract (`basic_hmm.h` thread-safety Doxygen, `tests/test_concurrent_training.cpp`). Const evaluation on a shared instance is also safe (mutex-serialised double-checked cache fill in `distribution_base.h`); mutation is not. How it got here, and the #48 decision: `docs/ARCHITECTURE.md`.

Fit-quality tiers per distribution, model-selection entry points
(`compute_aic`/`compute_bic`/`compute_aicc`), and I/O format details
(JSON recommended, legacy XML deprecated): `docs/ARCHITECTURE.md`.

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
- **mpmath** (`pip install mpmath`): only for regenerating the checked-in trig tables/references with `scripts/gen_trig_cleanroom_table.py` and `scripts/gen_trig_ulp_vectors.py`; not needed to build or test

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

Full checklist (template file, `fit()` guard, CMake registration, tests):
use the `add-hmm-distribution` skill.

## CI / Validation

Fleet-wide workflow rules (runner budget, bounded parallelism, ISA hazards on
hosted runners, action pinning):
[CI House Style](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md).

CI triggers on pushes to `main`, PRs targeting `main`, `workflow_dispatch`, and a monthly cron — a push to a `dev/*` branch does NOT run CI; open the PR to get the matrix (the v4.4.x release flow relies on this).

Four parallel build-matrix jobs: Linux/GCC, Linux/Clang, macOS/AppleClang, Windows/MSVC (`windows-latest`, whichever VS the runner image ships). Additional jobs (ubuntu): ThreadSanitizer, AddressSanitizer, pre-commit, cppcheck, and clang-tidy — nine legs in total. Tests run with `-LE "known_broken|benchmark"`.

`LIBHMM_ENABLE_CLANG_TIDY` (CMake option, `OFF` by default) wires clang-tidy into the normal build via the `CXX_CLANG_TIDY` target property; enable locally with `cmake --preset release -DLIBHMM_ENABLE_CLANG_TIDY=ON` when needed. The dedicated CI `clang-tidy` job instead runs `run-clang-tidy` against `compile_commands.json` as a single fast analysis pass and is **advisory (non-blocking)**: `continue-on-error: true`. Six checks are disabled in `.clang-tidy` (see that file for the full rationale), covering the pragma-once convention, intentional SIMD intrinsics/pointer arithmetic in perf-critical hot paths, the v4 template+virtual pattern, and a false-positive move-ctor idiom. `run-clang-tidy` re-reports every header diagnostic once per including TU, so a single flagged line in a widely-included header can appear 20+ times — count unique `file:line:col + check` tuples before drawing conclusions from this job. Current counts and promoting this job to blocking are tracked in PLAN.md.

## Reading map — load on demand, not preemptively
- Orienting on layering, v4 templates, SIMD dispatch tiers, fit quality,
  model selection, or I/O formats → `docs/ARCHITECTURE.md`.
- Cross-platform build setup, per-OS troubleshooting, or the CMake-standard
  deviations in full → `docs/CROSS_PLATFORM.md`.
- Working with `LIBHMM_*` experimental/feature flags → `docs/EXPERIMENTAL_FLAGS.md`.
- Planning or prioritizing performance work → `docs/Future_Performance_Work.md`.
- Adding or auditing a distribution's `fit()` quality → `docs/GOLD_STANDARD_CHECKLIST.md`.
- Naming, formatting, or other code-style questions beyond the summary above → `docs/STYLE_GUIDE.md`.
- What each repo document is for, and how they cross-reference →
  [DOC-CONVENTIONS.md](https://github.com/OldCrow/standards/blob/main/DOC-CONVENTIONS.md).
- Session state, in-progress work, open questions → `PLAN.md`.

## Open Items
See PLAN.md for current status, in-progress work, and open questions.
Distribution fit-quality roadmap specifically lives in
docs/GOLD_STANDARD_CHECKLIST.md — PLAN.md points to it rather than
duplicating it.
