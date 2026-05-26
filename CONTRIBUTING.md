# Contributing to libhmm

Thank you for your interest in contributing to libhmm. We welcome bug reports,
distribution implementations, performance improvements, benchmark examples, and
documentation improvements.

## Quick Start

### Prerequisites

- **C++20 compiler**: GCC 11+, Clang 14+, MSVC 2019 16.11+, or AppleClang 12+
- **CMake 3.20+**
- **Git**
- **GTest** (fetched automatically via CMake FetchContent if not found locally)

### Build and Test — macOS / Linux

```bash
git clone https://github.com/OldCrow/libhmm.git
cd libhmm

# Standard build
cmake -B build
cmake --build build --config Release
ctest --test-dir build --output-on-failure

# macOS Catalina (10.15) — use the Catalina-safe script
./scripts/configure_catalina.sh build
cmake --build build --config Release
ctest --test-dir build
```

### Build and Test — Windows (Visual Studio 2022)

```powershell
git clone https://github.com/OldCrow/libhmm.git
cd libhmm

# Configure with the Visual Studio generator
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build (Release)
cmake --build build --config Release --parallel 4

# Run all tests
ctest --test-dir build -C Release --output-on-failure --parallel 4

# Run a specific test
ctest --test-dir build -C Release -R test_gamma_distribution
```

Notes for Windows contributors:
- Do **not** call `vcvars64.bat` before CMake — the VS generator handles the
  toolchain automatically.
- SIMD capability is detected at configure time via `check_cxx_source_runs`
  (AVX-512 → AVX2 → AVX → SSE2 baseline). No manual flag selection is needed.
- The build produces both `hmm.dll` + `hmm.lib` (shared) and `hmm_static.lib`
  (static). Tests and examples link against `hmm_static.lib` to avoid
  runtime path issues.
- Source files containing non-ASCII characters (Greek letters, math symbols)
  require `/utf-8`, which is applied globally. No source file encoding changes
  are needed.

All existing tests must pass before submitting a pull request.

## How to Contribute

1. **Check existing issues** before starting — avoid duplicating work
2. **Open an issue** to discuss new distributions or significant changes
3. **Fork and create a feature branch**
4. **Implement your changes** following the coding standards below
5. **Add tests** (see Testing Requirements)
6. **Update documentation** as needed (headers, CHANGELOG.md)
7. **Submit a pull request** with a clear description

## Coding Standards

See `docs/STYLE_GUIDE.md` for the full style reference. Key points:

- **C++20 throughout**: `std::span`, `[[nodiscard]]`, `noexcept`, `#pragma once`
- **Naming**: `PascalCase` classes, `camelCase` methods, `snake_case_` private
  members with trailing underscore, `UPPER_CASE` compile-time constants
- **No magic numbers**: use `libhmm::constants` namespace
- **`using namespace constants;`** at the top of each `.cpp` file inside
  `namespace libhmm {`
- **No external dependencies**: C++20 stdlib only; no Boost, no BLAS

### Adding a New Distribution

See `docs/GOLD_STANDARD_CHECKLIST.md` for the complete requirements checklist.
In summary:

1. Inherit from `DistributionBase` in `include/libhmm/distributions/`
2. Implement all pure virtual methods including `getBatchLogProbabilities()`
3. Implement `fit(span<const double>)` — unweighted MLE
4. Implement `fit(span<const double>, span<const double>)` — weighted MLE
   for the Baum-Welch M-step. **Important**: return early (do not call
   `reset()`) when `sum(weights) ≈ 0`. Calling `reset()` on near-zero weight
   destroys valid parameters and causes EM state collapse.
5. Add to `LIBHMM_SIMD_SOURCES` and `LIBHMM_SOURCES` in `CMakeLists.txt`
6. Register in `include/libhmm/distributions/distributions.h` and
   `src/io/hmm_json.cpp` factory
7. Update `DISTRIBUTION_COUNT` and `CONTINUOUS_DISTRIBUTION_COUNT` in
   `distributions.h`
8. Add a test file in `tests/distributions/` and register in
   `tests/CMakeLists.txt`
9. Use `using namespace constants;` — no raw numeric literals

### Fit Quality

New distributions should use exact weighted MLE where available (see Tier A
in `docs/GOLD_STANDARD_CHECKLIST.md`). If MLE requires Newton–Raphson
(Gamma, Beta, Weibull, NegBin), use `include/libhmm/math/psi_functions.h`
for digamma and trigamma. For circular distributions, use
`include/libhmm/math/bessel.h`.

## Testing Requirements

- **All 41 existing tests must pass** — `ctest --test-dir build`
- **New distribution**: add `test_<name>_distribution.cpp` covering:
  - PDF normalisation (for continuous: numerical integration ≈ 1)
  - Log-PDF consistency with PDF
  - `fit()` parameter recovery on synthetic data
  - `fit(data, weights)` with zero weights keeps current parameters
  - JSON round-trip
  - `reset()` behaviour
- **New example**: must build cleanly with no warnings

```bash
# Run a specific distribution test
./build/tests/test_gamma_distribution

# Run with verbose output
ctest --test-dir build --verbose -R test_gamma
```

## Reporting Bugs

Include:

1. Clear description of the problem
2. Minimal reproducing example
3. Expected vs. actual behaviour
4. Environment: OS and version, compiler and version, CMake version, CPU

## Feature Requests

1. Search existing issues to avoid duplicates
2. Describe the use case and motivation
3. For new distributions: identify the target research domain and whether a
   published MLE M-step exists or Newton–Raphson is required
4. For new benchmark examples: identify the reference dataset and comparison
   package

## Build Configuration

```bash
cmake -DBUILD_EXAMPLES=OFF ..    # skip examples
cmake -DBUILD_TESTS=OFF ..       # skip tests
cmake -DBUILD_TOOLS=OFF ..       # skip tools
cmake -DBUILD_BENCHMARKS=ON ..   # build comparison benchmarks (needs external libs)

# macOS Catalina only: allow Homebrew LLVM (unsafe, may crash)
cmake -DLIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX=ON ..
```

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
`ci`, `chore`.

```bash
# Examples
git commit -m "feat(distributions): add DirichletDistribution with MLE fit"
git commit -m "fix(gamma): correct Newton step for near-zero k initialisation"
git commit -m "docs(examples): add CpG island bioinformatics benchmark"
git commit -m "perf(gaussian): promote to tier-2 SIMD on AArch64"
```

## Release Process

Releases are tagged manually by the maintainer after all CI checks pass.

1. All 41+ tests pass on all four CI platforms (Linux/GCC, Linux/Clang,
   macOS/AppleClang, Windows/MSVC)
2. `CHANGELOG.md` updated
3. Version bumped in `CMakeLists.txt` and `include/libhmm/libhmm.h`
4. Tag applied: `git tag vX.Y.Z && git push origin vX.Y.Z`

Versioning follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: breaking API changes
- **MINOR**: new distributions, algorithms, or benchmark examples
- **PATCH**: bug fixes and performance improvements

## Current Priority Areas

1. **CpG island bioinformatics example** — 8-state discrete HMM on human
   genome sequence (Durbin et al. 1998), comparison against StochHMM
2. **Multivariate Gaussian distribution** — required for PAMAP2/HARTH
   activity recognition benchmarks
3. **JOSS paper** — see `paper/paper.md` on the `joss-paper` branch
4. **ECM refinements** — post-convergence Newton polish for Student-t ν in
   near-Gaussian states

---

By contributing to libhmm, you agree that your contributions will be licensed
under the MIT License that governs this project.
