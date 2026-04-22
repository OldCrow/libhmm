# HMM Library Compatibility Guide

## Purpose
This guide captures the **current** setup and compatibility expectations for comparative benchmarks in `benchmarks/src/libhmm_vs_*`.

Primary results and interpretation are maintained in `BENCHMARKING_RESULTS.md`. This file focuses on build/runtime integration.

## External dependency layout

The benchmark CMake configuration expects external libraries under a configurable root:

- `LIBHMM_BENCHMARK_DEPS_ROOT` (default: `$HOME/Development`)
- `HMMLIB_DIR` (default: `${LIBHMM_BENCHMARK_DEPS_ROOT}/HMMLib`)
- `GHMM_DIR` (default: `${LIBHMM_BENCHMARK_DEPS_ROOT}/GHMM`)
- `STOCHHMM_DIR` (default: `${LIBHMM_BENCHMARK_DEPS_ROOT}/StochHMM`)
- `HTK_DIR` (default: `${LIBHMM_BENCHMARK_DEPS_ROOT}/HTK`)
- `JAHMM_DIR` (default: `${LIBHMM_BENCHMARK_DEPS_ROOT}/Jahmm`)
- `LAMP_DIR` (default: `${LIBHMM_BENCHMARK_DEPS_ROOT}/LAMP`)

Set these at configure time if your layout differs.

## Configure and build

```bash
cmake -S . -B build-benchmarks-release -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-benchmarks-release --parallel
```

Build only specific targets when iterating:

```bash
cmake --build build-benchmarks-release --target libhmm_vs_hmmlib_benchmark libhmm_vs_ghmm_benchmark --parallel
```

## Library-specific integration notes

### HMMLib
- Boost must be available for HMMLib-dependent targets.
- If Boost is missing, HMMLib targets are skipped by CMake.

### GHMM
- Requires GHMM headers and `libghmm`.
- On macOS, benchmark CMake normalizes GHMM `install_name` to reduce `dyld` path breakage when GHMM is relocated.

### StochHMM
- Requires StochHMM headers and `libstochhmm`.
- Continuous Gaussian comparisons require the corrected PI constant in StochHMM (`source/src/stochMath.h`).

### HTK
- Benchmarks assume HTK tools are available in runtime PATH.
- HTK output/log-likelihood behavior can be implementation-specific and may prioritize throughput over exact likelihood reporting.

### JAHMM
- Benchmark target supports configurable root via `JAHMM_DIR` and `LIBHMM_BENCH_JAHMM_DIR`.
- Runtime Java and Javac binaries are resolved from known locations with fallback to PATH.

### LAMP
- Benchmark target supports configurable root via `LAMP_DIR` and `LIBHMM_BENCH_LAMP_DIR`.
- Requires `hmmFind` under the configured LAMP root.

## Runtime artifacts

Benchmark logs and generated artifacts should be stored under the active build tree:

- Typical location: `build-benchmarks-release/benchmark-logs/`
- Avoid writing benchmark artifacts to repository root.

## Troubleshooting

- If a benchmark target is missing, check CMake configure output for dependency readiness messages.
- If a comparator target builds but fails at runtime, verify external tool/library paths first (JAHMM, LAMP, HTK).
- If numerical comparisons drift for StochHMM continuous runs, verify PI correction and rebuild StochHMM.

## Scope note

This guide intentionally omits historical migration details and legacy API notes. Historical benchmark interpretation remains in `BENCHMARKING_RESULTS.md`.
