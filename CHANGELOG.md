# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.5.1] - 2026-05-09

Test infrastructure patch. 37 test executables / 37 passing.

### Added

- **GTest migration** of 15 legacy distribution-specific test files
  (`tests/distributions/test_*_distribution.cpp`): the `assert()`/`main()`
  pattern replaced with `TEST()`/`EXPECT_*` macros throughout. The old
  pattern silently produced false-green results in Release builds because
  `-DNDEBUG` compiles `assert()` to a no-op; tests now fire
  unconditionally. Closes #10.
  - Includes the v3.5.0 post-release hotfix (six IO assertions that had
    contained wrong format strings since v2.7.0, first detected in a
    non-Release build — commit 56014e2).
- **`WeightedStats` unit tests** added to `tests/common/test_common.cpp`:
  direct tests for `detail::compute_weighted_stats` and
  `detail::compute_weighted_mean`, which underpin the weighted `fit()` path
  in 9+ distributions but had no unit test of their own. Covers known-value
  cases (uniform and non-uniform weights), single-element (variance = 0),
  empty spans, zero-weight sum, and NaN weight (all `nullopt` guard paths).
- **`json::Reader` unit tests** added to `tests/io/test_hmm_json.cpp`:
  direct tests for the handwritten JSON parser, exercising error paths
  (`consume` mismatch, EOF on consume/peek/read_double, unterminated string,
  non-numeric double, array size-cap throw) not reachable through the
  HMM-level `from_json()` tests. Also tests `write_double` and `write_array`
  round-trip at the parser level.

## [3.5.0] - 2026-05-06

cppcheck quality pass and V2 numerical framework removal. 37/37 tests pass.

### Removed

- **V2 numerical safety framework** (`include/libhmm/math/numerical_stability.h`,
  `src/common/numerical_stability.cpp`, `tests/common/test_numerical_stability.cpp`):
  `NumericalSafety`, `ConvergenceDetector`, `AdaptivePrecision`, `ErrorRecovery`,
  and `NumericalDiagnostics` were built for the pre-V3 scaled-calculator architecture
  and had no callers since the log-space-only pivot in V3. cppcheck confirmed every
  method was unreachable; dead callee chain back to the deleted scaled trainers.
  Files and associated test removed. 37/38 → 37 tests.

## [3.4.0] - 2026-05-05

Code quality refactoring (Phases 1–3) and JSON serialization. 38/38 tests pass.

### Added

- **JSON serializer/deserializer** (`include/libhmm/io/hmm_json.h`,
  `src/io/json_utils.h/cpp`, `src/io/hmm_json.cpp`):
  - `to_json(Hmm)`, `from_json(string_view)`, `save_json(Hmm, path)`,
    `load_json(path)` free functions — no external dependencies.
  - `json::Reader` schema-aware tokenizer + `write_double`/`write_array`/
    `write_matrix`/`write_distribution` helpers at `max_digits10` precision
    for exact IEEE 754 round-trip.
  - All 15 distributions gain `to_json() const override` and
    `static from_json(json::Reader&)` factory methods.
  - Anonymous-namespace `unordered_map<string, FactoryFn>` dispatch (CC 2,
    constant regardless of distribution count).
  - Input sanitization: `kMaxHmmStates=4096` (≈128 MB matrix cap),
    `kMaxJsonInputBytes=10 MB`, `kMaxDiscreteSymbols=65536`, bounded
    `read_double_array(N)` / `read_double_matrix(N,N)` to prevent heap
    growth before post-read dimension checks.
- **`libhmm.h`** now includes `hmm_json.h` as the recommended I/O entry point.
- **`samples/`** directory: `two_state_gaussian.{json,xml}` and
  `casino.{json,xml}` — validated reference HMM files in both formats.
- **`test_hmm_json`** (new test): type-field format check for all 15
  distributions, 15-state all-distributions round-trip (bit-exact via
  max_digits10), file save/load, 8 sanitization boundary tests.
- **Weighted statistics helper** (`src/common/weighted_stats.cpp`):
  `detail::compute_weighted_stats` / `detail::compute_weighted_mean`
  consolidate repeated weighted preamble logic across 8 distributions.
- **`test_hmm_json`**, extended `test_hmm_stream_io`, and updated
  `test_xml_file_io` (all 4 GTEST_SKIP instances removed — round-trips now
  pass with corrected parsers).

### Changed

- **`operator>>(istream, Hmm)` decomposed**: 258 lines / CC 18 → 76 lines /
  CC 7. Extracted 14 named `parse_*` functions to the anonymous namespace;
  replaced `std::function<>` map with a plain `StreamParserFn` function-pointer
  map. Added `parse_binomial` and `parse_rayleigh` (previously absent from
  dispatch). Zero lizard warnings in `hmm.cpp`.
- **All 15 `operator<<`** now delegate to `toString()` for consistent output;
  all 15 standalone `operator>>`** rewritten to token-scan the `toString()`
  format, including skipping derived-value lines (Mean, Variance, etc.).
- **`StudentT::operator>>`**: CC 22 key=value parser (65 lines) replaced with
  3-line token scan (CC 2).
- **`Discrete::operator>>`**: legacy bare-integer format replaced with current
  `toString()` / `Number of symbols = N` format.
- **`XMLFileReader::isValidXMLFile`** renamed `canParseAsHmm`; dead boost TODO
  comment blocks removed from both XML classes; classes marked deprecated
  (prefer `hmm_json.h`).
- **`hmm_validator` tool**: auto-detects format from file extension
  (`.json` → `load_json`; anything else → `XMLFileReader`).
- **`basic_hmm_example`** section 4: raw `ofstream`/`ifstream` round-trip
  replaced with `save_json`/`load_json` demonstration.
- **`segmental_kmeans_trainer.cpp`**: implementation-level documentation
  added (algorithm overview, partitioning strategy, 1e-10 floor rationale,
  convergence indicator, flatten rationale).
- **`common/common.h` split** (Phase 2, PR #5): linalg headers removed;
  distribution headers replaced `common.h` dependency with targeted includes;
  explicit linalg includes added only where genuinely required. Fan-in
  reduced from 24 to the headers that actually need the linalg types.
- **`json_utils.cpp`**: `std::from_chars` (floating-point, not available on
  AppleClang/libc++) replaced with `std::strtod`; `write_double` imbues
  `std::locale::classic()` for locale safety.
- **Code quality (Phase 1, PR #4)**:
  - `detail::compute_weighted_stats` eliminates duplicated weighted fit
    preambles across 8 distributions.
  - `precomputeLogTransitions` consolidated into the shared calculator base.
  - `BaumWelchTrainer::train` decomposed from CC 32 to CC 13.
  - `ViterbiTrainer::runIteration` decomposed.
  - Tool includes normalised to `distributions.h` umbrella header.
  - `FileIOManager` write-path extracted to shared `do_write` helper.
  - `NumericalDiagnostics` value-classification helper extracted.

### Fixed

- **9 distributions** had `operator<<` writing a format different from
  `toString()` (e.g. `GaussianDistribution` wrote `"Normal Distribution:
  Mean = V"` while `toString()` writes `"Gaussian Distribution: μ (mean) = V"`).
- **6 `parse_*` functions** in `hmm.cpp` had wrong token counts
  (e.g. `parse_poisson` expected `λ =` but `toString()` writes
  `λ (rate parameter) =`).
- **`Binomial`/`NegativeBinomial` `operator>>`** parsed a parenthesised
  `Dist(r,p)` format that no `operator<<` ever produced.
- **`Poisson::operator>>`** parsed 4 tokens before value; `toString()` writes 6.
- **`std::from_chars` (float)** not available in AppleClang/libc++ — caused CI
  failure; replaced with portable `std::strtod`.

### Removed

- Dead `include/libhmm/common/serialization.h` (zero callers post-Phase 2).
- Dead `include/libhmm/distributions/distribution_io_utils.h` (zero callers
  since Phase 1).
- All `GTEST_SKIP()` instances (was 4 in IO tests; hidden bugs now surface as
  failures and have been fixed).

## [3.3.0] - 2026-05-03

SIMD performance phase: explicit vector kernels for transcendental
operations and two additional Tier-2 distributions. 37/37 tests pass.

### Added

- **SIMD transcendental kernels** (`src/performance/transcendental_kernels.cpp`):
  five inner-loop kernels used by `ForwardBackwardCalculator` (FB max-reduce
  recurrence) and `BaumWelchTrainer` (dense-xi accumulation) now have
  AVX-512 / AVX / SSE2 / NEON backends. The vector `exp` helper uses a
  13-term Horner polynomial with Cephes `ln2` range reduction and branch-free
  underflow masking at `MIN_LOG_PROBABILITY`. AVX path stays AVX-1 compatible
  for Ivy Bridge / Catalina. Benchmarks on Zen 4 / AVX-512 (T=1000):
  FB max-reduce 5.7× faster at N=32; BW xi accumulation 1.03–1.15×.
- **LogNormal and Pareto promoted to Tier 2** (`src/distributions/`): explicit
  SIMD `getBatchLogProbabilities` via a vector `log` helper (IEEE-754 exponent
  extraction, 7-term Horner, split-LN2 reconstruction, ≤5 ULP).
- **`simd_kernels_internal.h`**: single source of truth for vector exp/log
  primitives shared by all Tier-2 distribution TUs and the transcendental
  kernels TU.
- **FB recurrence crossover retuned** (`fb_recurrence_policy.h`): threshold
  moved from N≥5 to N≥4 on x86 after profiling post-SIMD (MaxReduce is 1.7×
  faster at N=4).
- **New tests** (37 total, up from 33):
  - `test_simd_platform`: compile-time ISA hierarchy invariants (`#error`) and
    runtime contracts on `simd_platform.h` utility functions.
  - `test_transcendental_kernels`: SIMD vs `std::exp` parity for all five
    kernels across 11 sizes; 1e-12 rel / 1e-15 abs tolerance.
  - `test_fb_mode_parity`: Pairwise vs MaxReduce FB log-likelihood agreement.
  - `test_bw_parity`: BW determinism (bit-exact) and EM monotonicity.
- **New tools**: `bw_hotspot` (BW E-step phase breakdown), `hotspot_breakdown`
  (FB phase-level timings), `fb_crossover_sweep` (Pairwise vs MaxReduce
  timing across N), `fb_contour_sweep` (2-D N×T timing heatmap data).

### Changed

- `fb_recurrence_policy.h` moved from `include/libhmm/calculators/` to
  `include/libhmm/performance/` (cross-cutting primitive, not calculator-specific).
- Test group labels in `tests/CMakeLists.txt` changed from numeric Level N
  notation to semantic names; Performance Primitives group reordered before
  Distributions to reflect dependency order.
- `performance/PERFORMANCE_ARCHITECTURE.md` updated: Tier-2 coverage,
  delivered recurrence-kernel SIMD, corrected `LIBHMM_SIMD_SOURCES` list.
- `*.ps1` line-ending rule in `.gitattributes` changed from `eol=crlf` to
  `eol=lf` (PowerShell handles LF on all platforms; avoids CI pre-commit
  mixed-line-ending failures).

## [3.2.1] - 2026-05-02

CI hygiene fix; no functional changes.

### Fixed

- **`pre-commit` `cmake-format` hook crash on Ubuntu CI**
  (`.pre-commit-config.yaml`): `cmakelang` lazy-imports `yaml` from
  `load_yaml()` but does not declare `pyyaml` as a runtime dependency.
  With Python 3.14 in the pre-commit hook venv (the current
  `ubuntu-latest` default), this surfaced as
  `ModuleNotFoundError: No module named 'yaml'` and caused the gating
  pre-commit job to fail on every push. Added
  `additional_dependencies: ['pyyaml']` to the `cmake-format` hook so
  pre-commit injects pyyaml into the hook's venv.
- **Trailing whitespace in v3.2.0 edits** (`examples/README.md`,
  `examples/CMakeLists.txt`, `tests/CMakeLists.txt`,
  `include/libhmm/libhmm.h`, `include/libhmm/platform/simd_platform.h`,
  `include/libhmm/common/common.h`): six files committed in v3.2.0 had
  trailing whitespace on otherwise-blank indented lines. The
  `trailing-whitespace` pre-commit hook auto-fixed them on Ubuntu CI but
  exited non-zero; the fixes are now committed directly so the hook
  passes cleanly.

## [3.2.0] - 2026-05-02

Completes the cleanup that the v3.0.0-alpha refactor ("Modern C++20
Architecture Refactor") started but did not finish. That release pivoted
libhmm from a planned "calculators consume `Optimized*` containers backed by
a `WorkStealingPool`" architecture to the canonical "per-distribution batch
SIMD via `getBatchLogProbabilities`" architecture. The pivot was sound, but
the abandoned Plan-A infrastructure was retained in the tree as nominal
"future hooks" that no production code ever wired up. This release deletes
that orphaned material and fixes the small set of associated edits that
followed from the deletions.

No behavioural changes to the live API: the canonical calculators, trainers,
distributions, HMM core, examples, tools, benchmarks, and tests are all
unaffected. 33/33 tests pass.

### Added

- **Catalina configure helper** (`scripts/configure_catalina.sh`): one-command
  macOS 10.15 setup that sanitizes Homebrew-sensitive environment variables,
  pins AppleClang via `xcrun`, sets sysroot, and configures
  `CMAKE_OSX_DEPLOYMENT_TARGET=10.15`.
- **Segmental k-means example** (`examples/segmental_kmeans_example.cpp`):
  closes the only example-coverage gap. Demonstrates standalone use, the
  Baum-Welch warm-start pattern, and the discrete-only constraint.

### Changed

- **Catalina runtime guard hardening** (`CMakeLists.txt`): Catalina fail-fast
  detection now inspects compiler origin (`CMAKE_CXX_COMPILER`, `CC`, `CXX`),
  additional environment vectors (`SDKROOT`, `MACOSX_DEPLOYMENT_TARGET`,
  `CPATH`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`, `DYLD_LIBRARY_PATH`,
  `PKG_CONFIG_PATH`), and implicit include/link directories to catch Homebrew
  LLVM/libc++ contamination earlier.
- **Catalina test dependency policy** (`tests/CMakeLists.txt`): on Catalina,
  GTest is forced through FetchContent instead of Homebrew discovery to avoid
  mixed-runtime linkage.
- **ThreadPool relocation** (`tools/thread_pool.{h,cpp}`): moved out of the
  library and into `tools/`. Only `analyze_overhead` and `debug_parallel`
  consumed it; no production code path uses thread pools today. Trimmed
  dead-on-arrival `ThreadAffinityGuard` and unused `CpuInfo` cache-size
  helpers while moving.
- **Phase-1 forwarding-stub migration**: deleted six 4-line stubs in
  `include/libhmm/common/` and `include/libhmm/performance/` (`basic_*`,
  `optimized_*`, `linear_algebra`, `numerical_stability`, `simd_platform`,
  `simd_support`); migrated the six remaining call sites to the canonical
  `linalg/`, `math/`, and `platform/` paths.
- **Performance docs rewrite** (`performance/PERFORMANCE_ARCHITECTURE.md`):
  replaced the abandoned Plan-A four-level-hierarchy description with the
  canonical Plan-B per-distribution batch-SIMD strategy (matches what
  `getBatchLogProbabilities` actually does in production).

### Fixed

- **Windows examples crash** (`examples/CMakeLists.txt`): switched all
  examples to link against `hmm_static` instead of the `hmm` DLL. Mixed
  `std::vector`-based types across the .exe / .dll boundary on MSVC produced
  `STATUS_STACK_BUFFER_OVERRUN` at runtime in Release (and an access
  violation in Debug) before any output was emitted. Tests already linked
  statically for the same reason; examples now follow suit.

### Removed

- **Dead `LogSpaceOps`** (`include/libhmm/math/log_space_ops.h`,
  `src/performance/log_space_ops.cpp`): zero callers outside the file pair
  itself; the misleading SIMD path computed a mask but never consumed it,
  and a header-defined `static LogSpaceInitializer globalLogSpaceInit`
  created per-TU static initialisers. Removed entirely.
- **Dead `WorkStealingPool`** (`include/libhmm/platform/work_stealing_pool.h`,
  `src/performance/work_stealing_pool.cpp`): only used inside its own
  file pair; the .cpp was not even in `LIBHMM_SOURCES` so the class could
  not link. Speculative Plan-A parallelism layer with no consumer.
- **Dead `Benchmark` framework** (`include/libhmm/platform/benchmark.h`):
  declared `Timer`, `Benchmark`, `BenchmarkResult`, `BenchmarkStats`,
  `HmmBenchmarkUtils`, and `RegressionTester` but nothing implemented or
  used them. Benchmarks under `benchmarks/src/` define their own local
  timing structs.
- **Dead `Optimized*` class family** (`include/libhmm/linalg/optimized_*.h`,
  `src/common/optimized_*_simd.cpp`, three dedicated test files): the
  Plan-A high-performance data layer that the Phase-4 refactor superseded
  with per-distribution batch SIMD via `getBatchLogProbabilities`. ~3,200
  lines removed; production `Matrix`/`Vector` typedefs continue to resolve
  to `BasicMatrix<double>`/`BasicVector<double>`.
- **Dead training scaffolding**: `Centroid`, `Cluster`, the `HmmTrainer =
  Trainer` alias, the `SegmentedKMeansTrainer = SegmentalKMeansTrainer`
  alias, and the isolated tests that exercised `Centroid`/`Cluster` only
  through their own getters/setters.
- **Orphaned parallelism constants/utilities**
  (`include/libhmm/platform/parallel_{constants,execution}.h`): zero
  consumers after `Optimized*` removal; ten size/grain-size thresholds
  and a `safe_*` algorithm-wrapper family that nothing called.
- **Dead `SIMDOps`** (`include/libhmm/platform/simd_support.h`,
  `src/performance/simd_support.cpp`): ~600 lines of SSE2/AVX/NEON
  dot/add/multiply specialisations with no callers. Distributions and
  the `simd_inspection` tool depend only on `simd_platform.h` directly.
- **Uncompiled `optimized_simd_stubs.cpp`**: 198 lines of `_simd`
  template-method specialisations for the now-removed `Optimized*` types.
  Was never in `LIBHMM_SOURCES` so it never built.
- **Stale design docs**: `include/libhmm/common/matrix_architecture.md`
  (described the abandoned Plan-A `Optimized*` hierarchy) and
  `tests/performance/README.md` (referenced five development tools, three
  of which were removed long ago, plus calculator variants and constants
  removed in this cleanup).

Net effect: roughly 6,500 lines of dead code and stale infrastructure
removed; 33/33 tests still pass; production behaviour unchanged.

## [3.1.2] - 2026-04-26

### Changed

- **CI workflow hardening** (`.github/workflows/ci.yml`): added least-privilege
  `permissions: contents: read`, branch-scoped run cancellation via
  `concurrency`, explicit job timeouts, and manual `workflow_dispatch`.
- **Quality gate hardening** (`.github/workflows/ci.yml`): replaced ad-hoc
  format checks with a dedicated gating `precommit` job
  (`pre-commit run --all-files`) and kept `cppcheck` as a separate gating job.
- **Build/test flow cleanup** (`.github/workflows/ci.yml`): removed duplicate
  build steps and aligned CI test execution with project labels by excluding
  `known_broken|benchmark` from the default test run.
- **Pre-commit local hook execution metadata**
  (`scripts/check-pragma-once.sh`): marked executable to keep script-hook
  execution consistent on Linux CI.

### Fixed

- **Benchmark source formatting consistency** (`benchmarks/src/*.cpp`): applied
  clang-format normalization during CI-hardening prep; no benchmark logic
  changes.

## [3.1.1] - 2026-04-26

### Fixed

- **Discrete XML roundtrip parsing** (`src/hmm.cpp`): updated `operator>>` to
  parse the current labeled `DiscreteDistribution::toString()` format
  (`Number of symbols = N`, `P(i) = value`) instead of assuming a legacy fixed
  11-value payload. Kept legacy numeric parsing as fallback for older model
  files.

## [3.1.0] - 2026-04-25

### Added

- **LAMP_HMM Windows port** (`LAMP_HMM/`): native MSVC build via a minimal
  `CMakeLists.txt`. Three patches applied to the 1999–2003 legacy source:
  pre-standard header modernisation (`.h` headers → standard equivalents + `using
  namespace std`); stream-state check (`assert(stream != NULL)` → `if (!stream)`);
  `utils.h` portability fix — POSIX `drand48`/`srand48` replaced with the built-in
  Numerical Recipes RNG, and `ostream`/`ifstream`/`cout`/`cerr`/`endl`
  using-declarations added so downstream headers resolve these names without explicit
  `std::` qualification under MSVC. No algorithm changes; numerical output is
  identical to the Unix build.
- **LAMP benchmark Windows support** (`benchmarks/src/libhmm_vs_lamp_benchmark.cpp`):
  `<unistd.h>` / POSIX `getcwd` / `mkdir -p` / `rm -rf` replaced with `<filesystem>`,
  `_mkdir`, and `std::filesystem::remove_all`; relative paths resolved via
  `std::filesystem::absolute().generic_string()` to eliminate `./` components that
  cause cmd.exe syntax errors on the redirect target; executable invoked as
  `hmmFind.exe` via an unquoted absolute path to avoid the cmd.exe `/C`
  leading-quote mangling bug (quoting a command that starts with `"` causes cmd.exe
  to strip the outer `"` pair and produce a malformed command line).
- **Subprocess warmup pattern** (`LAMPBenchmark::warmup()`): untimed throw-away
  invocation added before all timed LAMP runs to absorb OS cold-start and AV scan
  latency. On Windows, the first `system()` call for a new executable takes ~1–2 s
  regardless of algorithm complexity (observed: 1,440 ms first run vs ~45 ms
  steady-state). Documented in `benchmarks/docs/BENCHMARKING_RESULTS.md` (methodology
  section) and `benchmarks/docs/Library_Compatibility_Guide.md` (troubleshooting
  section) as a required pattern for any subprocess-based comparator added in future.

### Changed

- **Build system**: sources compile once into `hmm_objects`
  both `hmm` (shared) and `hmm_static` (static) — no double compilation. Both targets are
  always produced regardless of `BUILD_SHARED_LIBS`.
  - Windows: `hmm.dll` + `hmm.lib` (DLL import library) + `hmm_static.lib` (static archive).
    `WINDOWS_EXPORT_ALL_SYMBOLS` enables automatic symbol export on MSVC.
  - macOS: `libhmm.dylib` + `libhmm.a`.
  - Linux: `libhmm.so` + `libhmm.a`.
  - `BUILD_SHARED_LIBS=OFF` suppresses the shared library; the static library is always built.
- **JAHMM benchmark** (`benchmarks/src/libhmm_vs_jahmm_benchmark.cpp`): Windows
  compatibility — `std::filesystem` path helpers replace POSIX equivalents; Windows Java
  binary resolution; classpath separator is `;` on Windows; `_popen` aliased for MSVC.
- `benchmarks/CMakeLists.txt`: `-lm` not linked on MSVC (math functions are in the CRT).
- clang-format applied to `libhmm_vs_jahmm_benchmark.cpp`, `libhmm_vs_lamp_benchmark.cpp`,
  and `log_normal_distribution.cpp` (style-only, no logic changes).

### Fixed

- **Windows test suite** (`tests/CMakeLists.txt`): tests now link against `hmm_static`
  instead of `hmm`. The OBJECT-library build system always produces `hmm.dll` on Windows
  regardless of `BUILD_SHARED_LIBS`, so tests that imported symbols from `hmm.dll` failed
  at startup with `0xc0000135 STATUS_DLL_NOT_FOUND` when the DLL was not on `PATH`.
  Linking against the static archive avoids the runtime path requirement and is consistent
  with the CI flag `-DBUILD_SHARED_LIBS=OFF`.

## [3.0.1]

### Fixed

- **PoissonDistribution**: improved log-factorial accuracy for larger `k` by using `std::lgamma(k + 1)` in the non-cached path.
- **LogNormalDistribution**: corrected exponent scaling in probability calculations to use squared deviation (`(x - μ)^2`) with the cached `-1/(2σ^2)` factor.

## [3.0.0] - 2026-04-23

### Fixed

- **test_xml_file_io**: platform-guarded invalid paths; test now passes on all platforms.
  Removed from `known_broken` — no `known_broken` labels remain in the suite.
- **cmake RUN_TESTS collision**: Visual Studio generator creates a built-in `RUN_TESTS`
  target; renamed custom target to `check` / `check_timing` to avoid the conflict.
- **calculator.h**: removed deprecated `hmm_`/`hmm_legacy_ptr_` alias members that
  triggered `-Wdeprecated-declarations` on GCC and AppleClang.
- **Precision**: `log_space_ops.cpp` — `log(1+exp(x))` → `log1p(exp(x))` (two sites);
  `log_normal_distribution.h` — `exp(σ²)-1` → `expm1(σ²)` in variance formula.
- **cppcheck findings** (all source): explicit constructors on 5 distributions
  (`BinomialDistribution`, `NegativeBinomialDistribution`, `LogNormalDistribution`,
  `ParetoDistribution`, `WeibullDistribution`); `TrainingConfig` passed by const
  reference in `ViterbiTrainer`; dead `matrix_vector_multiply_fallback` removed;
  `thread_pool.cpp` cache-size returns typed as `std::size_t`; `starts_with()` in
  Linux `/proc/cpuinfo` parser; const loop vars in `work_stealing_pool.cpp`;
  variable scope fixes in 13 distribution `operator>>` methods.
- **cluster.h C4267**: `static_cast<int>` on four `size_t→int` conversions.

### Added

- **`.clang-format`**: K&R brace style, 4-space indent, 100-col limit. Applied to all
  source files; `.git-blame-ignore-revs` lists the bulk-reformat commit.
- **CI lint job**: `clang-format --dry-run` + `cppcheck` on `src/` (ubuntu-latest;
  warning-only until fully hardened).
- **Linux/Clang matrix entry**: CI now validates on 4 platforms (Linux/GCC,
  Linux/Clang, macOS/AppleClang, Windows/MSVC).
- **`BUILD_TOOLS` CMake option**: all four optional subdirectories
  (tests/examples/tools/benchmarks) are now consistently option-gated.
- **Pre-commit hooks**: `clang-format` and `cmake-format` added alongside existing
  file-hygiene and `#pragma once` checks.

### Changed

- All source files bulk-reformatted to `.clang-format` style (whitespace-only).
- `tests/CMakeLists.txt`: `add_hmm_test()` now applies `-Wall -Wextra -Wpedantic`
  (GCC/Clang) or `/W4 /permissive-` (MSVC) to every test target.
- CI trigger: removed stale `refactor/modern-architecture` branch.
- `build_windows.bat` removed — hardcoded vcpkg path, vcpkg no longer required.

## [3.0.0-alpha.1] - 2026-04-22

### Changed

- Modernized remaining benchmark sources to canonical v3 APIs and removed references to removed calculator/distribution APIs.
- Consolidated benchmark documentation under `benchmarks/docs/BENCHMARKING_RESULTS.md` and `benchmarks/docs/Library_Compatibility_Guide.md`; removed stale overlapping benchmark docs.
- Redirected benchmark runtime artifacts to benchmark build-log directories instead of writing outputs into the repository root.
- Added `build*/` gitignore coverage and cleaned obsolete benchmark artifacts tracked from earlier runs.
- Tightened benchmark-results wording to distinguish historical vs current sections and align terminology with C++20 and canonical calculators.

### Fixed

- `ViterbiCalculator` constructor now explicitly discards the `decode()` return value to satisfy `[[nodiscard]]` and keep the core `hmm` target warning-free.

## [3.0.0-alpha] - 2026-04-22

### Modern C++20 Architecture Refactor

Complete rewrite of the distribution, calculator, and training layers.
All external dependencies removed. C++ standard raised to C++20.

### Breaking Changes

- **`Hmm` API**: `setProbabilityDistribution(state, ptr)` → `setDistribution(state, unique_ptr)`;
  `getProbabilityDistribution(state)` → `getDistribution(state)` (returns `EmissionDistribution&`, not a pointer)
- **`ProbabilityDistribution`** removed; replaced by `EmissionDistribution` abstract base
- **All SIMD calculator variants removed**: `ScaledSIMDForwardBackwardCalculator`,
  `LogSIMDForwardBackwardCalculator`, `ScaledSIMDViterbiCalculator`, `LogSIMDViterbiCalculator`,
  `AdvancedLog*` variants
- **`AutoCalculator` / `CalculatorSelector` / `CalculatorTraits`** removed
- **`ScaledBaumWelchTrainer`** removed; `BaumWelchTrainer` now log-space and numerically stable
- **`RobustViterbiTrainer`** removed; absorbed into `ViterbiTrainer` with `TrainingConfig` presets
- **`HmmTrainer`** base class renamed to `Trainer`
- **`Observation` type alias** removed; use `double` directly
- **`two_state_hmm.h`** moved from public include tree to `examples/support/`

### Added

#### EmissionDistribution Interface (Layer 3)
- New abstract base `EmissionDistribution` replacing `ProbabilityDistribution`
- `getBatchLogProbabilities(span<const double>, span<double>)` — batch evaluation for SIMD
- `fit(span<const double>, span<const double> weights)` — weighted MLE for Baum-Welch M-step
- `DistributionBase` provides thread-safe `std::atomic<bool>` cache and shared math helpers
- All 15 distributions implement concrete non-virtual `getBatchLogProbabilities()` loops (tier 1)
- `GaussianDistribution` and `ExponentialDistribution` have explicit SIMD intrinsics (tier 2):
  AVX-512 (8-wide), AVX2 (4-wide), SSE2 (2-wide), NEON (2-wide) with scalar tail

#### Canonical Calculators (Layer 4)
- `ForwardBackwardCalculator` — canonical log-space; calls `getBatchLogProbabilities()` per state;
  pre-computes log transition matrix; no underflow on any sequence length
- `ViterbiCalculator` — same architecture; returns MAP state sequence

#### Canonical Trainers (Layer 4)
- `BaumWelchTrainer` — canonical log-space EM; uses weighted `fit()` for M-step; accumulates γ
  across multiple sequences; works with any `EmissionDistribution`
- `ViterbiTrainer` — hard-assignment training with `TrainingConfig` and named presets
  (`training_presets::fast()`, `balanced()`, `precise()`); reports `hasConverged()` /
  `reachedMaxIterations()` after training
- `SegmentalKMeansTrainer` (canonical name; `SegmentedKMeansTrainer` retained as alias)

#### SIMD Infrastructure
- `LIBHMM_BEST_SIMD_FLAGS` selected at configure time: MSVC uses `check_cxx_source_runs`
  (verifies CPU can execute instructions, not just that the compiler accepts the flag);
  GCC/Clang use `-march=native`
- AArch64 architecture detection in CMakeLists gates x86 checks correctly
- `simd_inspection` tool reports active ISA and runs 6 functional smoke tests

#### Test Suite (36/36 canonical tests)
- Tests organised into 8 architectural levels with `add_hmm_test()` helper
- 4 new tests: `test_calculator_continuous`, `test_calculator_edge_cases`,
  `test_baum_welch_convergence` (EM monotonicity), `test_end_to_end` (casino problem)
- Custom targets: `check` (parallel, correctness), `check_timing` (serial)

#### Tools Directory
- `simd_inspection` — SIMD ISA report + 6 functional smoke tests
- `batch_performance` — FB + Viterbi throughput at varied (N, T) with Gaussian HMM
- `hmm_validator` — loads XML HMM, validates, runs ForwardBackward + Viterbi with diagnostics

#### Examples (12 total, all canonical API)
- New: `baum_welch_example` (EM convergence table, BW vs Viterbi comparison)
- New: `viterbi_trainer_example` (TrainingConfig preset comparison)
- New: `student_t_hmm_example` (financial risk regime detection, BW training)
- Updated: all 9 existing examples to canonical `setDistribution`/`getDistribution` API

### Changed

- C++ standard raised from C++17 to **C++20**
- All `#ifndef`/`#define`/`#endif` header guards replaced with `#pragma once`
- SIMD detection in CMakeLists is architecture-aware (AArch64 / x86 separate paths)
- `tests/CMakeLists.txt`: `add_hmm_test()` helper, 8-level organisation, `run_tests` target
- `examples/CMakeLists.txt`: `add_hmm_example()` helper; examples re-enabled
- All distribution `fit()` methods accept `std::span<const double>` (was `std::vector<Observation>&`)

### Fixed

- ForwardBackwardCalculator allocation bug: `obsVec` was allocated N times inside the
  N-state emission loop; moved outside the loop (matched ViterbiCalculator behaviour)
- MSVC SIMD detection: `check_cxx_compiler_flag` accepted `/arch:AVX512` unconditionally
  even on VMs without AVX-512; replaced with `check_cxx_source_runs` to verify at runtime
- Apple Silicon false-positive SSE4.2 detection: Clang on arm64 accepted x86 flags silently;
  x86 checks now gated on `CMAKE_SYSTEM_PROCESSOR`

### Removed

- All SIMD calculator variants (12 test files, 3 performance tools also removed)
- `docs/CALCULATOR_MIGRATION.md` (migration complete)
- `robust_viterbi_trainer_example.cpp`, `robust_financial_hmm_example.cpp`

---

## [2.9.1] - 2025-07-02

### Cross-Platform Architecture Support & Build Quality Release

This release delivers Apple Silicon support across the HMM ecosystem and eliminates build warnings through architecture-aware configuration.

### Added

####  Apple Silicon Ecosystem Support
- **ARM64 HMM Library Ecosystem**: Successfully ported and validated several major HMM libraries for Apple Silicon to support benchmarking
  - **HMMLib**: Intel SSE to ARM NEON intrinsic mapping with full SIMD performance preservation
  - **GHMM**: Python environment and build system compatibility for ARM64
  - **StochHMM**: Cross-platform compilation with architecture detection
  - **HTK**: Speech recognition toolkit ARM64 compatibility
  - **LAMP HMM**: Pre-C++98 updated to compile and run with modern Apple Clang on Apple Silicon
  - **JAHMM**: Early 2000s Java HMM library confirmed to work with minimal tweaks
  - **libhmm**: Native Apple Silicon optimization with automatic Homebrew path detection

#### Architecture-Aware Build System
- **Automatic Architecture Detection**: CMake now detects and configures for Apple Silicon vs Intel Mac
  - Apple Silicon (arm64): Uses `/opt/homebrew` Homebrew path
  - Intel Mac (x86_64): Uses `/usr/local` Homebrew path
  - Linux: Uses standard package manager paths (`/usr`, `/usr/local`)
- **Cross-Platform Documentation**: Enhanced [CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md) with comprehensive architecture support details

#### ARM NEON SIMD Optimization
- **Intel SSE to ARM NEON Mapping**: Complete intrinsic translation for maintaining SIMD performance
  - Double precision: `_mm_add_pd` → `vaddq_f64`, `_mm_mul_pd` → `vmulq_f64`
  - Single precision: `_mm_add_ps` → `vaddq_f32`, `_mm_mul_ps` → `vmulq_f32`
  - Horizontal operations: Custom ARM NEON implementations for complex reductions
- **Memory Alignment**: Cross-platform aligned memory allocation (`_mm_malloc` → `posix_memalign`)
- **Performance Preservation**: Maintains equivalent SIMD vectorization capabilities on ARM64

### Enhanced

#### Build System Robustness
- **Zero Build Warnings**: Eliminated all compiler and linker warnings across platforms
- **Modern GTest Integration**: Updated to use `GTest::gtest_main` and `GTest::gtest` targets
- **Dependency Management**: Improved library detection and linking strategies
- **CMake Modernization**: Platform-specific configurations with proper feature detection

#### Benchmarking Infrastructure
- **Multi-Architecture Validation**: All benchmark libraries now compile and run on both Intel and ARM64
- **Performance Verification**: Cross-platform performance characteristics documented
- **Ecosystem Compatibility**: Complete validation of numerical agreement across architectures

### Fixed

#### Compiler Warnings Elimination
- **Unused Variable Warnings**: Fixed in test and performance files
  - `parallel_constants_tuning.cpp:168`: Removed unused `probability` variable
  - `test_weibull_distribution.cpp`: Made accumulator variables `volatile` to prevent optimization
- **Linker Warnings**: Eliminated "ignoring duplicate libraries" messages
  - Root cause: Architecture-specific Homebrew paths were hardcoded for Intel Mac
  - Solution: Dynamic architecture detection and path configuration

#### Cross-Platform Compatibility
- **HMMLib ARM64 Port**: Complete Intel SSE to ARM NEON intrinsic translation
- **Template Dependencies**: Fixed C++17 template compatibility issues across all platforms
- **Build Dependencies**: Resolved library detection issues on different architectures
- **Memory Allocation**: Cross-platform aligned memory allocation strategies

### Technical Implementation

#### Architecture Detection System
```cmake
# Automatic Apple Silicon vs Intel Mac detection
execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE APPLE_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(APPLE_ARCH STREQUAL "arm64")
    set(HOMEBREW_PREFIX "/opt/homebrew")
    message(STATUS "Detected Apple Silicon (${APPLE_ARCH}) - using Homebrew at ${HOMEBREW_PREFIX}")
else()
    set(HOMEBREW_PREFIX "/usr/local")
    message(STATUS "Detected Intel Mac (${APPLE_ARCH}) - using Homebrew at ${HOMEBREW_PREFIX}")
endif()
```

#### ARM NEON SIMD Mapping
```cpp
// Architecture-specific SIMD operations
#if defined(__aarch64__) || defined(__arm64__)
    #include <arm_neon.h>
    typedef float32x4_t __m128;
    typedef float64x2_t __m128d;

    // Intel SSE to ARM NEON translations
    #define _mm_add_pd(a, b) vaddq_f64(a, b)
    #define _mm_mul_pd(a, b) vmulq_f64(a, b)
    #define _mm_set_pd1(val) vdupq_n_f64(val)
#elif defined(__x86_64__)
    #include <pmmintrin.h>
    // Use native Intel SSE intrinsics
#endif
```

#### Cross-Platform Library Integration
```cpp
// Memory allocation compatibility
#ifdef HMM_SIMD_X86
    ptr = static_cast<float_type *>(_mm_malloc(size, 16));
#else
    if (posix_memalign(reinterpret_cast<void**>(&ptr), 16, size) != 0) {
        ptr = static_cast<float_type *>(malloc(size));
    }
#endif
```

### Validation Results

#### Cross-Platform Build Quality
```
Architecture Support:
├─ Intel Mac (x86_64): ✅ Full compatibility with /usr/local Homebrew
├─ Apple Silicon (arm64): ✅ Full compatibility with /opt/homebrew Homebrew
├─ Linux (x86_64/arm64): ✅ Standard package manager support
└─ Generic Unix: ✅ Fallback compatibility

Build Quality:
├─ Compiler Warnings: 0/0 (100% clean)
├─ Linker Warnings: 0/0 (100% clean)
├─ Test Suite: All 47 test executables compile and run successfully
└─ SIMD Performance: Maintained across Intel and ARM architectures
```

#### HMM Ecosystem Validation
```
Library ARM64 Compatibility:
├─ libhmm: ✅ Native ARM64 with automatic configuration
├─ HMMLib: ✅ Complete Intel SSE to ARM NEON port
├─ GHMM: ✅ Python environment and build compatibility
├─ StochHMM: ✅ Cross-platform compilation
└─ HTK: ✅ Speech toolkit ARM64 support

Performance Characteristics:
├─ ARM NEON: Equivalent vectorization to Intel SSE
├─ Memory Alignment: 16-byte alignment preserved
├─ SIMD Width: 4 floats / 2 doubles per vector (same as SSE)
└─ Benchmarks: 100% numerical agreement across architectures
```

### Performance Impact

#### SIMD Performance Preservation
- **ARM64 Performance**: Maintains equivalent SIMD acceleration to Intel x86_64
- **Vector Operations**: 4-element float and 2-element double vectorization preserved
- **Memory Bandwidth**: Optimized aligned memory access patterns maintained
- **Cross-Platform Consistency**: Identical performance characteristics across architectures

#### Ecosystem Performance
- **HMMLib on ARM64**: ~14x average speedup over libhmm maintained
- **SIMD Optimization**: ARM NEON provides comparable performance to Intel SSE
- **Memory Efficiency**: Cross-platform aligned allocation strategies
- **Build Performance**: Reduced compilation overhead through improved dependency detection

### Breaking Changes

**None** - All changes are internal build system and compatibility improvements maintaining full API compatibility.

### Migration Notes

No action required for existing code. The improvements provide automatic benefits:

#### For Apple Silicon Users
- Build system automatically detects ARM64 and uses `/opt/homebrew`
- All HMM ecosystem libraries now compile and run natively
- SIMD performance maintained through ARM NEON optimizations
- Cross-compilation between Intel and ARM64 supported

#### For Intel Mac Users
- Continues to use `/usr/local` Homebrew path automatically
- All existing build configurations work unchanged
- Enhanced cross-platform compatibility for team environments

#### For Linux Users
- Improved package manager integration
- Better support for ARM64 Linux distributions
- Enhanced build system robustness

### Documentation Updates

- **[CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md)**: Comprehensive cross-platform build guide
- **[Library_Compatibility_Guide.md](benchmarks/docs/Library_Compatibility_Guide.md)**: Complete ARM64 porting documentation
- **Architecture Detection**: CMake configuration examples and troubleshooting
- **Performance Analysis**: Cross-platform SIMD optimization details

### Dependencies

- **C++17 Compatible Compiler**: GCC 7+, Clang 6+, MSVC 2017+
- **CMake**: 3.15 or later with improved platform detection
- **Homebrew** (macOS): Automatically detected at architecture-specific path
- **Platform Support**: macOS (Intel/Apple Silicon), Linux (x86_64/ARM64), Unix-like systems
- **SIMD Support**: Intel SSE/AVX, ARM NEON with automatic detection

### Future Roadmap

This release establishes the foundation for:
- **Useable HMM Ecosystem**: Complete cross-platform compatibility across many major HMM libraries
- **SIMD Optimization**: Advanced vectorization strategies leveraging platform-specific capabilities
- **Performance Scaling**: Architecture-aware optimization for different processor types
- **Cloud Deployment**: Enhanced compatibility for containerized and cloud-native deployments

---

## [2.9.0] - 2025-06-30

### Significant Enhancements to HMM Calculator Infrastructure

This release brings modern APIs, SIMD optimizations, multi-threaded execution, benchmarking and testing improvements, along with improved documentation and architecture. Changes include:

- **Modern API Extensions**: Enhancements with backward compatibility.
- **SIMD Optimizations**: Improved vectorization across various distributions.
- **Multi-threaded Execution**: Parallel processing capabilities added.
- **Benchmarking Improvements**: Updated benchmark tests for more precise performance measurement.
- **Documentation**: Revised documentation aligning with new functionalities.

This release addresses all critical issues identified in previous versions and lays a strong foundation for future development. Backwars compatibility is maintained while offering substantial performance gains.

## [2.8.0] - 2025-06-29

### Comprehensive Distribution Testing Framework Release

This release includes a complete overhaul of the test suite for all 16 statistical distributions, enhancing mathematical correctness and addressing critical bugs. Highlights include:

- **Standardized Test Patterns**: 174 total cases organized by distribution families (location-scale, gamma family, etc.)
- **Mathematical Validation**: Coverage for parameter validation, statistical properties, probability calculations, and edge case handling
- **Bug Fixes**: Critical patch for Pareto distribution boundary conditions ensuring accurate PDF calculations
- **Library Integration**: Ensures all distributions accessible through main header with the added Rayleigh distribution

All 174 distribution tests pass with a 100% success rate, confirming rigorous validation for the entire distribution library.

### Complete Gold Standard Distribution Optimizations Release

This release delivers comprehensive performance optimizations across all probability distributions while maintaining mathematical correctness and adding new functionality. Major improvements include the addition of the Rayleigh distribution, significant performance gains through standard library integration, and enhanced numerical accuracy.

### Added

#### New Distribution
- **Rayleigh Distribution**: Complete implementation with Gold Standard compliance
  - Specialized case of Weibull distribution (k=2) for modeling vector magnitudes
  - Applications in communications, wind modeling, and signal processing
  - Full test suite with performance benchmarks and edge case coverage
  - Optimized PDF/CDF calculations with caching mechanisms

#### Enhanced Performance Features
- **Vectorized Batch Processing**: Added batch computation methods for Beta distribution
  - `getProbabilityBatch()` and `getLogProbabilityBatch()` for efficient bulk operations
  - Optimized for processing multiple values with enhanced cache reuse
  - Foundation for future SIMD vectorization across all distributions

#### Mathematical Correctness Improvements
- **Proper Beta CDF Implementation**: Fixed critical mathematical error
  - Replaced incorrect incomplete gamma function with proper incomplete beta function
  - Implemented continued fraction expansion for numerical accuracy
  - Ensures mathematically correct cumulative probability calculations

### Enhanced

#### Distribution Performance Optimizations
- **Beta Distribution**: 24% PDF improvement, 62% log PDF improvement
  - Enhanced caching system with `invBeta_`, `alphaMinus1_`, `betaMinus1_`
  - Binary exponentiation for fast integer power calculations
  - Optimized boundary case handling and numerical stability
- **Log-Normal Distribution**: 61% PDF improvement, 53% fitting improvement
  - Welford's algorithm integration for numerically stable fitting
  - Enhanced caching mechanisms and optimized log-space calculations
- **Gaussian Distribution**: Enhanced CDF using `std::erf` for improved accuracy
  - 31% performance improvement in error function calculations
  - Better numerical stability and hardware optimization

#### Standard Library Integration
- **Mathematical Function Optimization**: Replaced custom implementations with standard library
  - `std::lgamma` replaces custom `loggamma` (31% faster)
  - `std::erf` replaces custom `errorf` implementation
  - Leverages hardware-optimized mathematical functions
  - Maintains compatibility while improving performance and maintainability

#### Numerical Enhancements
- **Binary Exponentiation**: Fast integer power calculations across distributions
  - Optimized small case handling (powers 0-4) with direct computation
  - Logarithmic complexity for larger integer exponents
  - Significant speedup for distributions with integer shape parameters
- **Enhanced Caching Systems**: Comprehensive pre-computation strategies
  - Distribution-specific cached values for frequently used calculations
  - Automatic cache invalidation on parameter changes
  - Reduced redundant mathematical operations

### Fixed

#### Mathematical Correctness
- **Beta Distribution CDF**: Corrected from incorrect gamma-based to proper beta function implementation
- **Boundary Value Handling**: Improved edge case processing across all distributions
- **Numerical Stability**: Enhanced precision in extreme value scenarios

#### Code Quality and Maintainability
- **Dead Code Removal**: Eliminated unused custom mathematical implementations
- **Function Declarations**: Updated headers to reflect standard library usage
- **Compilation Optimization**: Resolved build warnings and enhanced compiler optimization

### Performance Results

#### Current Performance Benchmarks
```
Beta Distribution:     0.097μs PDF, 0.047μs log PDF, 0.029μs fitting
Log-Normal:           0.079μs PDF, 0.045μs log PDF, 0.037μs fitting
Gaussian:             0.045μs PDF, 0.027μs log PDF, 0.017μs fitting
Rayleigh:             Sub-microsecond performance across all operations
```

#### Optimization Impact Summary
- **Beta PDF**: 24% improvement (118ns → 90ns per call)
- **Beta Log PDF**: 62% improvement (93ns → 35ns per call)
- **Standard Library Functions**: 31% improvement in mathematical operations
- **Overall**: Maintained sub-microsecond performance while adding functionality

### Technical Implementation

#### Binary Exponentiation Algorithm
```cpp
auto fastPower = [](double base, int exp) -> double {
    if (exp <= 4) return directComputation(base, exp);  // Optimized small cases
    // Binary exponentiation for larger powers
    double result = 1.0;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
};
```

#### Enhanced Caching Strategy
```cpp
void updateCache() const noexcept {
    logBeta_ = std::lgamma(alpha_) + std::lgamma(beta_) - std::lgamma(alpha_ + beta_);
    invBeta_ = std::exp(-logBeta_);  // Direct computation cache
    alphaMinus1_ = alpha_ - 1.0;      // Frequent calculation cache
    betaMinus1_ = beta_ - 1.0;        // Frequent calculation cache
    cacheValid_ = true;
}
```

### Quality Assurance

#### Test Coverage
```
Distribution Tests: 15/15 distributions with complete Gold Standard coverage
Performance Tests: Comprehensive benchmarking across all optimized functions
Numerical Tests: Edge cases, boundary values, and extreme parameter validation
Batch Processing: Vectorized operations testing for future SIMD integration
```

#### Compatibility
- **API Compatibility**: All changes maintain full backward compatibility
- **Performance**: Enhanced speed while maintaining mathematical correctness
- **Reliability**: Improved numerical stability and error handling

### Breaking Changes

**None** - All optimizations are internal improvements maintaining full API compatibility.

### Migration Notes

Existing code continues to work unchanged with enhanced performance and accuracy:
- Beta distribution CDF calculations now return mathematically correct results
- All distributions benefit from faster mathematical function evaluation
- New batch processing methods available for performance-critical applications

### Future Roadmap

This release establishes the foundation for:
- **SIMD Vectorization**: Batch processing infrastructure ready for vector instructions
- **Calculator Optimizations**: Next phase focusing on algorithm-level improvements
- **Advanced Features**: Enhanced parallel processing and specialized distributions

---

## [2.7.1] - 2025-06-28

### Discrete Distributions Gold Standard & HTK Benchmark Enhancement Release

This release elevates all 4 discrete distributions (Discrete, Binomial, Negative Binomial, Poisson) to the gold standard of exception safety, input validation, and naming consistency. Additionally, the benchmarking suite now distinctly separates discrete from continuous benchmarks and includes extended performance scaling analysis.

### Added

#### Discrete Distribution Enhancements
- **Gold Standard Upgrades**: All discrete distributions updated to gold standard
  - Comprehensive exception handling in stream input operators
  - Input validation for edge cases, e.g., k=0 for Poisson, negative probabilities
  - Consistent variable naming following the project guidelines

#### Continuous Benchmarking Improvements
- **HTK Benchmark Separation**: Clean discrete and continuous benchmark separation
- **Performance Scaling Analysis**: Extended sequences ranging 100 to 1,000,000 observations
- **Gaussian Distribution Fixes**: Proper handling of mean/variance pairs

### Enhanced

#### Benchmarking and Accuracy
- **Numerical Validation**: Accurate log-likelihood computation for continuous models
- **Performance Benchmarking**: Updated benchmarks showcase distinct advantages of libhmm and HTK under different scenarios

#### Code Reliability and Consistency
- **Error Messages**: Improved clarity and consistency in error reporting
- **Documentation**: Updated all affected components to align with enhanced functionality

### Fixed

#### Stream I/O and Naming Conventions
- **Variable Naming**: Consistent patterns across all discrete distributions
- **Stream Input Resilience**: No crashes on malformed input for discrete distributions
- **Numerical Accuracy**: Precise computation ensures 100% correctness in tests

### Breaking Changes

**None** - All modifications maintain full backward compatibility and primarily elevate internal robustness.

### Migration Notes

No actions required from prior versions. Code continues to operate correctly with improved internal robustness. Enhanced error safety in all four discrete distributions offers better reliability for production systems.

### Quality Improvements

This release attains the next level of quality, highlighting our ongoing commitment to continuous improvement and excellence in HMM library development:
- **Systematic Upgrades**: Completing goals outlined in the Q4 roadmap of 2024
- **Continuous Refactoring**: Focus on internal, non-breaking refactoring for long-term maintainability
- **Code Quality Enforcements**: Automatized checks via clang-tidy ensure adherence to standards

---

## [2.7.0] - 2024-06-27

### Stream I/O Robustness & Code Quality Release

This release focuses on implementing robust exception handling in distribution stream input operators and establishing consistent coding standards across the library.

### Added

#### Exception Handling Infrastructure
- **Robust Stream Input Operators**: All distribution `operator>>` implementations now include comprehensive exception handling
  - Try-catch blocks around `std::stod()` calls to handle malformed input gracefully
  - Stream failbit setting on parsing errors instead of throwing exceptions
  - Consistent error recovery patterns across all 10 distributions
- **Input Validation**: Enhanced validation for boundary values and special cases
  - Proper handling of negative values, NaN, and infinity in stream parsing
  - Consistent use of library constants instead of magic numbers

#### Code Quality Standards
- **Gold Standard Checklist**: Comprehensive quality standards document for distribution implementations
  - 43-point checklist covering robustness, consistency, and maintainability
  - Guidelines for exception handling, variable naming, and documentation
  - Progressive upgrade path for bringing all distributions to gold standard
- **Variable Naming Consistency**: Standardized variable naming in stream input operators
  - Discarded tokens consistently named "token" across all distributions
  - Value tokens use meaningful names ("alpha_str", "lambda_str", etc.)
  - Enhanced code readability and maintainability

### Enhanced

#### Distribution Robustness
- **Beta Distribution**: Fixed boundary value handling to return exact 0.0 for mathematical correctness
- **Stream Parsing**: All distributions now handle malformed input gracefully without crashes
- **Error Messages**: Improved error handling with consistent stream state management
- **Constants Usage**: Proper use of `math::ZERO_DOUBLE` vs `precision::ZERO` for exact comparisons

#### Code Consistency
- **Uniform Patterns**: All stream input operators follow identical exception handling patterns
- **Naming Standards**: Consistent variable naming improves code readability across distributions
- **Documentation**: Clear standards established for future distribution implementations

### Fixed

#### Stream I/O Reliability
- **Exception Safety**: Eliminated potential crashes from malformed stream input across all distributions
- **Boundary Cases**: Fixed Beta distribution boundary value handling for Google Test compatibility
- **Input Validation**: Robust handling of edge cases in all distribution stream operators

#### Code Quality
- **Magic Numbers**: Replaced hardcoded values with proper library constants
- **Variable Naming**: Standardized naming conventions across all distribution implementations
- **Error Handling**: Consistent error recovery patterns prevent undefined behavior

### Technical Implementation

#### Exception Handling Pattern
```cpp
// Before (prone to crashes):
std::string s;
is >> s;
double value = std::stod(s);  // Could throw exception

// After (robust):
std::string value_str;
try {
    is >> token >> token >> value_str;
    double value = std::stod(value_str);
    if (is.good()) {
        distribution.setParameter(value);
    }
} catch (const std::exception& e) {
    is.setstate(std::ios::failbit);
}
```

#### Distributions Enhanced (10 Total)
- **Beta Distribution** - Added robust exception handling and fixed boundary values
- **Log-Normal Distribution** - Enhanced stream parsing with proper error handling
- **Pareto Distribution** - Added comprehensive exception handling
- **Poisson Distribution** - Implemented robust stream input parsing
- **Weibull Distribution** - Added exception safety to stream operators
- **Uniform Distribution** - Enhanced with robust error handling
- **Chi-squared Distribution** - Implemented consistent exception handling
- **Gaussian Distribution** - Added robust stream input parsing
- **Exponential Distribution** - Enhanced with exception safety
- **Gamma Distribution** - Implemented comprehensive error handling

### Test Results

#### Quality Assurance
```
Test Suite Results: 40/40 tests passing (100%)
Distribution Tests: 10/10 distributions with robust stream I/O
Code Quality: All distributions follow consistent patterns
Exception Handling: 100% coverage across stream input operators
```

#### Validation Coverage
- **Stream I/O Testing**: All distributions tested with malformed input
- **Edge Cases**: Boundary values, NaN, and infinity handling verified
- **Error Recovery**: Consistent failbit setting confirmed across all operators
- **Variable Naming**: Consistent patterns verified across all implementations

### Breaking Changes

**None** - All changes are internal implementation improvements that maintain full API compatibility.

### Migration Notes

Existing code continues to work unchanged. The improvements provide:
- More robust stream input parsing with graceful error handling
- Better error reporting through proper stream state management
- Enhanced reliability for applications using stream I/O operations

### Quality Standards

This release establishes the foundation for systematic quality improvements:
- **Gold Standard Checklist**: Clear criteria for distribution quality
- **Progressive Enhancement**: Roadmap for upgrading remaining distributions
- **Consistent Patterns**: Established templates for robust implementations
- **Maintainability**: Simplified code patterns for easier maintenance

---

## [2.6.0] - 2024-06-26

### Major Release - Boost Elimination & Benchmarking Framework

This version removes all Boost library dependencies and introduces a comprehensive benchmarking suite for validating libhmm against other HMM implementations.

### Key Accomplishments

#### Complete Boost Dependency Removal (Phase 8.3)
- **Self-Contained Library**: Now requires only C++17 standard library
- **Custom Matrix/Vector Classes**: Replaced `boost::numeric::ublas` with efficient custom implementations
  - Contiguous memory layout optimized for cache performance
  - Template-based design supporting extensible numeric types
  - SIMD-friendly memory alignment for vectorization
- **Custom XML Serialization**: Replaced `boost::serialization` with lightweight implementation
  - Compact code footprint with clear, readable output
  - Full support for all 17 distribution types and model structures
- **Build System Modernization**: Simplified CMake configuration
  - Reduced compilation time and binary size
  - Enhanced cross-platform compatibility

#### Comprehensive Benchmarking Framework
- **Multi-Library Integration**: Successfully integrated 5 HMM libraries (libhmm, HMMLib, GHMM, StochHMM, HTK)
- **Numerical Validation**: Achieved 100% numerical agreement across libraries at machine precision
- **Performance Characterization**: Established baseline performance metrics across sequence lengths from 1,000 to 1,000,000 observations
- **Compatibility Documentation**: Complete integration guides with fixes for each library

### Added

#### Core Infrastructure
- **Custom Matrix/Vector Classes** (`BasicMatrix<T>`, `BasicVector<T>`)
  - Template-based design with type aliases for clean API
  - Standard mathematical operators and efficient memory management
  - Zero external dependencies with move semantics support

- **XML Serialization System**
  - Direct XML generation with proper formatting
  - Support for all distribution types and model components
  - Human-readable output format

#### Benchmarking Suite
- **22 Benchmark Programs**: Comprehensive testing across multiple libraries and scenarios
- **7 Documentation Files**: Detailed analysis, compatibility guides, and methodology
- **Library Integration Solutions**:
  - HMMLib: Fixed C++17 template compatibility issues
  - GHMM: Resolved indexing assumptions and Python environment setup
  - StochHMM: Dynamic model file generation and format conversion
  - HTK: File I/O wrappers for speech recognition toolkit integration

### Enhanced

#### Performance & Quality
- **Memory Layout**: Optimized for better cache locality and SIMD operations
- **Compilation Speed**: Significant improvement without Boost template instantiation
- **Code Maintainability**: Clean separation of concerns and modern C++17 practices

#### Numerical Validation Results
```
Library Performance vs libhmm:
├─ GHMM:      23x faster (100% numerical agreement)
├─ HMMLib:    17-20x faster (100% numerical agreement)
├─ HTK:       Variable performance (intentionally rounded results)
└─ StochHMM:  2x faster (100% numerical agreement)

Test Coverage: 32 test cases across 4 classic HMM problems
Numerical Accuracy: Machine precision agreement (≤1e-14)
```

### Fixed

#### Library Compatibility
- **Template Dependencies**: Fixed modern C++ template inheritance issues in HMMLib
- **API Integration**: Corrected indexing assumptions and format handling across libraries
- **Build Conflicts**: Clean separation of internal vs external dependencies

#### Repository Organization
- **Git Configuration**: Proper `.gitignore` setup for benchmarks and build artifacts
- **Directory Structure**: Organized source code vs third-party library separation
- **CMake Integration**: Removed generated `Testing/` directory from version control

### Performance Analysis

#### Key Insights
- **Numerical Correctness**: libhmm maintains perfect accuracy across all test scenarios
- **Dependency Independence**: Unique among tested libraries for complete self-containment
- **Modern Architecture**: Contemporary C++17 codebase with extensible design
- **Performance Position**: Establishes baseline for future optimization work

### Technical Implementation

```cpp
// Migration from Boost to custom implementation
// Before:
#include <boost/numeric/ublas/matrix.hpp>
using Matrix = boost::numeric::ublas::matrix<double>;

// After:
#include "libhmm/common/common.h"
using Matrix = libhmm::BasicMatrix<double>;
```

### Breaking Changes

**None** - Full API compatibility maintained while removing dependencies.

### Migration Notes

Existing code works unchanged:
```cpp
Matrix transition_matrix(2, 2);
transition_matrix(0, 1) = 0.3;
auto hmm = std::make_unique<Hmm>(num_states);
```

Benefits are automatic:
- Faster compilation without Boost dependencies
- Smaller binaries and easier deployment
- Enhanced performance through optimized memory layout

### Dependencies

**Before (v2.5.0)**: C++17, CMake 3.15+, Boost Libraries
**After (v2.6.0)**: C++17, CMake 3.15+ only

### Future Development

This release establishes:
- Foundation for advanced SIMD optimization
- Benchmarking framework for measuring improvements
- Clean architecture for extending distributions and algorithms
- Validation infrastructure for continuous development

---

## [2.5.0] - 2024-06-25

### 🎯 Calculator Modernization & Benchmark Validation Release

This release focuses on AutoCalculator system validation, benchmark suite modernization, and significant performance improvements through validated SIMD optimizations.

### Added

#### AutoCalculator System Validation
- **Complete API Modernization**: All calculator code migrated to current `libhmm::forwardbackward::AutoCalculator` and `libhmm::viterbi::AutoCalculator` APIs
- **Intelligent Algorithm Selection**: Enhanced calculator selection with detailed performance rationale
  - Automatic Scaled-SIMD selection for appropriate problem sizes
  - Smart fallback strategies based on problem characteristics
  - Numerical stability prioritization for long sequences (≥1000 observations)
- **Performance Transparency**: Calculator selection rationale now visible in debug output

#### Benchmark Suite Modernization
- **Algorithm Performance Benchmark**: Updated to use AutoCalculator APIs with enhanced debug output
- **Classic Problems Benchmark**: Comprehensive 16-test validation suite
  - 4 Classic HMM Problems: Dishonest Casino, Weather Model, CpG Island Detection, Speech Recognition
  - 4 Sequence Lengths: 100, 500, 1000, 2000 observations per problem
  - Both Forward-Backward and Viterbi algorithm validation
- **API Compatibility**: All benchmark code uses current libhmm API patterns

#### Performance Validation Infrastructure
- **Numerical Accuracy Verification**: 100% accuracy maintained across all 16 benchmark comparisons
- **Performance Measurement**: Reliable timing infrastructure for ongoing optimization work
- **Calculator Selection Validation**: Verified AutoCalculator system works correctly across all problem sizes

### Enhanced

#### Performance Improvements
- **Major Performance Gains**: ~17x improvement from previous performance gaps
  - Forward-Backward: Reduced from ~540x to 31.3x gap vs HMMLib (average)
  - Viterbi: Improved to 20.9x gap vs HMMLib (average)
  - Range: 18x-47x depending on problem size and algorithm type
- **SIMD Effectiveness**: Clear evidence that SIMD optimizations provide substantial performance benefits
- **Algorithm Maturity**: AutoCalculator system selecting appropriate algorithms effectively

#### API Modernization
- **Namespace Consolidation**: Migrated from deprecated `libhmm::calculators` to current `libhmm::forwardbackward` and `libhmm::viterbi` namespaces
- **Simplified Calculator Usage**: Removed manual SIMD vs scalar selection logic - now handled automatically
- **Enhanced Debug Information**: Detailed calculator selection explanations with performance predictions

#### Code Quality
- **Future-Ready Infrastructure**: Benchmark system ready for ongoing optimization work
- **Maintainable Codebase**: Simplified calculator instantiation while maintaining full functionality
- **Development Workflow**: Reliable performance measurement tools for continuous improvement

### Fixed

#### Benchmark Compatibility
- **Include Path Updates**: Updated from old `calculator_traits.h` to new `forward_backward_traits.h` and `viterbi_traits.h`
- **API Deprecation**: Removed usage of deprecated calculator classes and selection patterns
- **Build System**: All benchmarks compile successfully with current API

#### Performance Measurement
- **Accurate Timing**: Validated benchmark timing infrastructure provides consistent results
- **Numerical Stability**: Perfect agreement between libhmm and HMMLib across all test cases (≤ 2e-10 precision)
- **Calculator Selection**: Verified AutoCalculator chooses optimal algorithms based on problem characteristics

### Performance Analysis

#### Detailed Performance Breakdown
- **Dishonest Casino**: 15-25x gap (Forward-Backward), 10-19x gap (Viterbi)
- **Weather Model**: 20-35x gap (Forward-Backward), 19-20x gap (Viterbi)
- **CpG Island Detection**: 18-37x gap (Forward-Backward), 18-24x gap (Viterbi)
- **Speech Recognition**: 39-47x gap (Forward-Backward), 18-34x gap (Viterbi)

#### Key Performance Insights
- **Trend Analysis**: Performance gap decreases with larger, more complex problems
- **SIMD Impact**: ScaledSIMD calculator correctly selected for largest problems showing optimization benefits
- **Remaining Opportunity**: 20-31x gap suggests room for further architectural improvements
- **Optimization Evidence**: Clear demonstration that SIMD work is providing substantial real-world gains

### Validation Results

#### Numerical Accuracy
```
Successful comparisons: 16/16
Numerical matches: 16/16 (100.0%)
Viterbi likelihood differences: ≤ 2e-10 (machine precision)
```

#### Calculator Selection Examples
- Small problems (100 obs): "Predicted performance: 1.65x baseline"
- Medium problems (500 obs): "Predicted performance: 3.125x baseline"
- Large problems (1000+ obs): "Provides numerical stability for long sequences"

### Technical Specifications

#### Updated Files
- **algorithm_performance_benchmark.cpp**: Modernized to AutoCalculator API
- **classic_problems_benchmark.cpp**: Comprehensive test suite with current API
- **Phase 8 Documentation**: Updated to reflect current performance improvements and validation status

#### Infrastructure Improvements
- **API Consistency**: All calculator usage follows current best practices
- **Debug Visibility**: Calculator selection rationale available for development and optimization
- **Measurement Reliability**: Validated benchmark infrastructure for ongoing performance work

### Breaking Changes

None - this release maintains full backward compatibility while significantly improving performance measurement and validation capabilities.

### Migration Notes

Existing code continues to work unchanged. The improvements provide:
- Better performance through validated SIMD optimizations
- More intelligent algorithm selection with transparency
- Reliable benchmark infrastructure for measuring future improvements

### Future Work Foundation

This release establishes a solid foundation for:
1. **Algorithm Optimization**: Using reliable benchmark system to measure improvements
2. **Performance Profiling**: Identifying specific bottlenecks with confidence in measurement accuracy
3. **SIMD Enhancement**: Expanding vectorization with validated numerical stability checks

---

## [2.4.0] - 2024-06-24

### 🧹 Include Consolidation & Numerical Stability Release

This release focuses on code maintainability, numerical robustness, and developer experience improvements through header consolidation and comprehensive stability infrastructure.

### Added

#### Numerical Stability Infrastructure
- **NumericalSafety Class**: Comprehensive finite value validation and safe mathematical operations
  - `safeLog()` and `safeExp()` with underflow/overflow protection
  - Probability range validation and automatic normalization
  - Container validation for matrices and vectors
- **ConvergenceDetector**: Adaptive convergence detection with oscillation and stagnation detection
- **AdaptivePrecision**: Dynamic tolerance adjustment based on problem characteristics
- **ErrorRecovery**: Multiple recovery strategies (STRICT, GRACEFUL, ROBUST, ADAPTIVE)
- **NumericalDiagnostics**: Real-time health monitoring with actionable recommendations

#### Trainer Traits System
- **Compile-time Type Safety**: Distribution compatibility checking at build time
- **Template Metaprogramming**: Modern C++17 type traits with `constexpr` and SFINAE
- **Trainer Selection**: Automatic algorithm selection based on distribution capabilities
- **Zero Runtime Overhead**: All type checking resolved during compilation

#### Development Infrastructure
- **Internal Documentation**: Organized development docs in `.dev-docs/` (gitignored)
- **Future Work Parking Lot**: Comprehensive roadmap for v2.x and v3.x development
- **Phase Documentation**: Complete modernization history and rationale

### Enhanced

#### Include Structure Modernization
- **Umbrella Header Consolidation**: Replaced 70+ individual distribution includes with single `distributions.h`
- **Consistent Architecture**: Unified include pattern across 11 core files
- **Reduced Maintenance Overhead**: Single point of distribution header management
- **Build Efficiency**: Improved incremental compilation performance

#### Code Quality
- **Maintainability**: Significantly easier to add new distributions
- **Readability**: Cleaner, more professional header structure
- **Standards Compliance**: Aligned with C++ umbrella header best practices
- **Documentation**: Clear dependency relationships and API organization

#### Testing Infrastructure
- **Extended Test Suite**: 31 test suites (up from 28)
- **Numerical Stability Tests**: 24 new tests for edge case handling
- **Trainer Traits Tests**: 12 new tests for compile-time type safety
- **Comprehensive Coverage**: 100% pass rate with zero regressions

### Fixed

#### Robustness Improvements
- **Edge Case Handling**: Comprehensive protection against NaN, infinity, and underflow
- **Training Stability**: Enhanced convergence detection prevents infinite loops
- **Error Recovery**: Graceful handling of degenerate data and numerical issues
- **Memory Safety**: Continued adherence to RAII principles

### Performance Improvements

- **Build Times**: Potential improvement through optimized include structure
- **Runtime Stability**: Proactive numerical issue detection and correction
- **Zero Overhead**: Type safety checking with no runtime cost
- **Adaptive Precision**: Dynamic adjustment based on problem characteristics

### Technical Specifications

#### Files Modified
- **Header Files**: 4 core library headers consolidated
- **Test Files**: 7 test files with simplified includes
- **Lines Reduced**: 70+ redundant include lines eliminated
- **Maintainability**: Single point of distribution header management

#### New Infrastructure
- **Numerical Constants**: Carefully tuned for different scenarios
- **Error Recovery Strategies**: 4 different approaches based on requirements
- **Diagnostic Capabilities**: Real-time numerical health assessment
- **Future-Ready**: Foundation for advanced trainer selection (Phase 6)

### Breaking Changes

None - this release maintains full backward compatibility while significantly improving maintainability.

### Migration Notes

Existing code works unchanged. The improvements are transparent:

```cpp
// Existing includes still work
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/poisson_distribution.h"

// But now you can simply use:
#include "libhmm/distributions/distributions.h"  // All distributions available
```

### Dependencies

- **C++17 Compatible Compiler**: GCC 7+, Clang 6+, MSVC 2017+
- **CMake**: 3.15 or later
- **Boost Libraries**: For matrix operations
- **Platform**: macOS, Linux, Unix-like systems

---

## [2.3.0] - 2024-06-23

### 🚀 Major Feature Release - Advanced Statistical Distributions & Performance Optimization

This release adds powerful new statistical distributions and comprehensive performance optimizations, making libhmm suitable for advanced statistical modeling and high-performance applications.

### Added

#### New Statistical Distributions
- **Student's t-distribution**: Complete implementation for robust statistical modeling
  - Location (μ), scale (σ), and degrees of freedom (ν) parameters
  - Heavy-tailed distribution perfect for financial modeling and outlier-robust analysis
  - MLE parameter fitting and comprehensive validation
- **Chi-squared distribution**: Essential for goodness-of-fit testing and statistical analysis
  - Degrees of freedom parameter
  - Used in hypothesis testing and categorical data analysis
  - Efficient implementation with numerical stability

#### Performance & Optimization Framework
- **SIMD Support**: Platform-specific vectorized operations
  - AVX support for Intel/AMD processors
  - SSE2 fallback for older x86 systems
  - ARM NEON support for Apple Silicon and ARM processors
  - Automatic CPU feature detection and optimization selection
- **Thread Pool**: Modern C++17 concurrent processing
  - Work-stealing algorithm for optimal load balancing
  - Thread affinity support for NUMA systems
  - Configurable thread count with automatic detection
- **Optimized Forward-Backward Calculator**:
  - SIMD-accelerated matrix-vector operations
  - Cache-optimized memory layouts
  - Blocked algorithms for large matrices
  - Up to 3x performance improvement on compatible hardware
- **Calculator Traits System**:
  - Automatic algorithm selection based on problem size
  - Runtime optimization based on CPU capabilities
  - Performance profiling and reporting

#### Advanced Examples
- **Robust Financial HMM**: Demonstrates Student's t-distribution for modeling heavy-tailed financial returns
- **Statistical Process Control HMM**: Quality control monitoring using comprehensive statistical methods

#### Infrastructure Improvements
- **Distribution Traits**: Compile-time distribution analysis for type safety
- **Convenience Headers**: `distributions.h` umbrella header for easy inclusion
- **Memory Management**: Aligned allocators for SIMD operations
- **CPU Detection**: Runtime CPU feature detection and optimization

### Enhanced

#### Build System
- **CMake Policy Compliance**: Fixed FindBoost deprecation warnings
- **Cross-Platform Optimization**: Platform-specific SIMD compilation flags
- **Zero-Warning Builds**: Eliminated all compiler warnings

#### Testing Framework
- **Comprehensive Unit Tests**: Full coverage for new distributions
- **Performance Testing**: Benchmarking and optimization validation
- **Edge Case Validation**: Robust handling of boundary conditions
- **Integration Testing**: Cross-distribution compatibility verification

#### Parser & I/O
- **Multi-line Format Support**: Enhanced HMM stream parser for complex distribution outputs
- **Token-based Parsing**: Robust parsing for all distribution types
- **Serialization Consistency**: Reliable round-trip serialization for all distributions

### Fixed

#### Critical Issues
- **Stream Parser**: Fixed "stod: no conversion" errors in HMM I/O
- **Gaussian Distribution Parser**: Corrected multi-line format token consumption
- **Memory Safety**: Enhanced validation for edge cases and invalid inputs

#### Code Quality
- **Unused Variables**: Eliminated all unused variable warnings
- **Deprecated Functions**: Replaced deprecated API calls with modern equivalents
- **Exception Handling**: Improved error messages and exception safety

### Performance Improvements

- **Matrix Operations**: Up to 3x speedup with SIMD vectorization
- **Memory Access**: Cache-optimized layouts reduce memory latency
- **Parallel Processing**: Multi-core training algorithms for large datasets
- **Algorithm Selection**: Automatic optimization based on problem characteristics

### Technical Specifications

#### Supported Distributions (17 total)
**Discrete**: Discrete, Poisson, Binomial, Negative Binomial
**Continuous**: Gaussian, Gamma, Exponential, Log-Normal, Pareto, Beta, Weibull, Uniform, **Student's t**, **Chi-squared**

#### SIMD Support
- **Intel/AMD**: AVX, SSE2 instruction sets
- **ARM**: NEON instruction set (Apple Silicon, ARM processors)
- **Automatic Detection**: Runtime CPU feature detection
- **Fallback**: Scalar implementations for unsupported hardware

#### Threading
- **Work-Stealing Thread Pool**: Optimal load distribution
- **NUMA Awareness**: Thread affinity for multi-socket systems
- **Scalable Design**: Efficient scaling from 1 to 64+ cores

### Breaking Changes

None - this release maintains full backward compatibility while adding new features.

### Migration Notes

All existing code continues to work unchanged. New features are opt-in:

```cpp
// New distributions
auto studentT = std::make_unique<StudentTDistribution>(3.0, 0.0, 1.0);
auto chiSquared = std::make_unique<ChiSquaredDistribution>(5.0);

// Performance optimization (automatic)
OptimizedForwardBackwardCalculator calc(hmm.get(), observations);
```

### Dependencies

- **C++17 Compatible Compiler**: GCC 7+, Clang 5+, MSVC 2017+
- **CMake**: 3.15 or later
- **Boost Libraries**: For matrix operations
- **Platform**: macOS, Linux, Unix-like systems

---

## [2.0.0] - 2024-06-21

### 🎉 Major Release - C++17 Modernization

This release represents a complete modernization of the libhmm library with critical bug fixes and enhanced memory safety.

### Added
- **C++17 Standard Compliance**: Full modernization to C++17 standards
- **Smart Pointer Memory Management**: Replaced all raw pointers with `std::unique_ptr` and `std::shared_ptr`
- **Modern CMake Build System**: Enhanced CMake configuration with proper target management
- **Comprehensive Test Suite**: Expanded unit tests with better coverage
- **Enhanced Type Safety**: Explicit type casting and bounds checking throughout
- **Memory Safety**: RAII principles implemented consistently
- **Modern Loop Constructs**: Range-based for loops and auto type deduction

### Fixed
- **CRITICAL**: Fixed segmentation fault in `ViterbiTrainer::train()`
  - **Root Cause**: Double ownership of `ProbabilityDistribution*` objects in `viterbi_trainer.cpp:124`
  - **Solution**: Modified distributions in place rather than reassigning ownership
  - **Impact**: ViterbiTrainer now runs successfully without crashes
- **Memory Leaks**: Eliminated all raw pointer memory leaks through smart pointer adoption
- **Deprecated Functions**: Updated `prepare_hmm()` to `prepareTwoStateHmm()` calls
- **Build Warnings**: Resolved all compilation warnings in C++17 mode

### Changed
- **API Modernization**:
  - `int main(void)` → `int main()`
  - Replaced global `using namespace` with selective imports
  - Modern function parameter styles
- **Memory Management**:
  - `new`/`delete` → `std::make_unique<>()`
  - Raw pointers → Smart pointers throughout
- **Loop Syntax**:
  - C-style loops → Modern C++17 range-based loops
  - Manual indexing → Iterator-based approaches where appropriate
- **Error Handling**: Enhanced exception safety and error reporting

### Removed
- **Raw Pointer Usage**: Eliminated unsafe manual memory management
- **C-Style Constructs**: Removed outdated C-style function signatures
- **Memory Unsafe Patterns**: Cleaned up potential double-free scenarios

### Performance
- **Memory Efficiency**: Smart pointers provide better memory locality and automatic cleanup
- **Compilation Speed**: Modern C++17 features enable better compiler optimizations
- **Runtime Safety**: Bounds checking and type safety prevent runtime errors

### Technical Details

#### ViterbiTrainer Bug Fix
```cpp
// BEFORE (Buggy - caused segfault):
ProbabilityDistribution* pdist = hmm_->getProbabilityDistribution(i);
pdist->fit(clusterObservations);
hmm_->setProbabilityDistribution(i, pdist);  // Double ownership!

// AFTER (Fixed):
ProbabilityDistribution* pdist = hmm_->getProbabilityDistribution(i);
pdist->fit(clusterObservations);
// No reassignment needed - HMM already owns the distribution
```

#### Smart Pointer Migration
```cpp
// BEFORE:
Hmm* hmm = new Hmm(2);
// ... use hmm
delete hmm;

// AFTER:
auto hmm = std::make_unique<Hmm>(2);
// Automatic cleanup when out of scope
```

### Migration Guide

For users upgrading from v1.x:

1. **Update Compiler**: Ensure C++17 compatible compiler (GCC 7+, Clang 6+, MSVC 2017+)
2. **Memory Management**: Replace any direct `new`/`delete` with smart pointers
3. **Function Calls**: Update `prepare_hmm()` to `prepareTwoStateHmm()`
4. **Build System**: Use CMake for modern builds (legacy Makefile still supported)

### Compatibility
- **Backwards Compatible**: API remains largely unchanged
- **ABI Breaking**: Memory management changes require recompilation
- **C++17 Required**: No longer compatible with pre-C++17 compilers

---

## [1.0.0] - Previous Version

### Features
- Basic HMM implementation
- Viterbi training and decoding
- Multiple probability distributions
- Forward-Backward algorithms
- File I/O support

### Known Issues (Fixed in 2.0.0)
- Segmentation faults in ViterbiTrainer
- Memory leaks from raw pointer usage
- Non-standard C++ constructs
- Build warnings in modern compilers
