# libhmm Benchmarking Results and Analysis

## Overview

This document summarizes benchmark results comparing libhmm against major HMM libraries for numerical accuracy, throughput, and scaling behavior. It includes historical benchmark snapshots and a current consolidated update (April 2026). For current comparisons, use the `Benchmark Update (April 2026): Consolidated libhmm_vs_* Results` and `Post-modernization validation signal` sections.

## Methodology

### Libraries Tested

1. **libhmm** - Modern C++20 implementation with zero external dependencies
2. **HMMLib** - High-performance C++ library with Boost dependencies
3. **StochHMM** - Bioinformatics-focused C++ library
4. **GHMM** - General Hidden Markov Model Library (C)
5. **HTK** - Hidden Markov Model Toolkit (command-line based)

### Comparator scope and caveats

These benchmarks are useful, but they are not perfectly apples-to-apples. Each
external library optimizes for a different problem shape, API style, or runtime model.
Interpret throughput and likelihood tables with these constraints in mind:

| Library | Important constraint for interpretation |
|---------|-----------------------------------------|
| **HMMLib** | Specialized discrete-HMM engine built around custom matrix/vector classes and SSE-oriented templates. In this benchmark workflow it is included header-only, and it solves a narrower problem than libhmm's general emission-distribution framework. |
| **StochHMM** | Designed for bioinformatics workflows and text-defined models. Continuous-distribution comparisons require source patches; in particular, Gaussian results from builds without the PI correction are not valid for numerical-accuracy conclusions. |
| **GHMM** | Mature C library with a lower-level API and strong performance. Comparisons reflect a more specialized C implementation, not equal developer ergonomics or API scope. **macOS/Linux only** — a native Windows build would require porting Autotools to CMake, Windows-compatible GSL and libxml2, and POSIX API replacement across the full C source; the benchmark links `libghmm` directly so the whole library must build under MSVC. |
| **HTK** | Toolkit-style workflow invoked through executables and file I/O rather than a native in-process API. It uses deliberately rounded log-likelihood values (often multiples of 1000), which is acceptable for relative scoring in speech workflows but not for exact-likelihood comparisons. **macOS/Linux only** — HTK carries an X11 dependency and a POSIX Makefile build system; unlike LAMP (a small self-contained tree), a Windows MSVC port would be a substantial effort with no upstream support path. |
| **JAHMM** | Pure Java library invoked through a subprocess bridge in this suite. Results reflect both the Java runtime model and the benchmark bridge, not just core HMM arithmetic. |
| **LAMP** | Executable/config-file workflow rather than a linkable library. Timings include a much heavier process-and-file-I/O model than in-process C/C++ libraries. |

### Test Problems

Two classic HMM benchmark problems were used across all libraries:

#### 1. Dishonest Casino Problem
- **States**: 2 (Fair Die, Loaded Die)
- **Observations**: 6 symbols (dice faces 0-5)
- **Transitions**: Fair→Fair (0.95), Fair→Loaded (0.05), Loaded→Fair (0.10), Loaded→Loaded (0.90)
- **Emissions**: Fair die (uniform 1/6), Loaded die (symbol 5 favored at 0.50)

#### 2. Weather Model Problem
- **States**: 2 (Sunny, Rainy)
- **Observations**: 2 symbols (Hot, Cold)
- **Transitions**: Sunny→Sunny (0.7), Sunny→Rainy (0.3), Rainy→Sunny (0.4), Rainy→Rainy (0.6)
- **Emissions**: Sunny→Hot (0.8), Sunny→Cold (0.2), Rainy→Hot (0.3), Rainy→Cold (0.7)

### Sequence Lengths Tested

All libraries were tested with identical observation sequences of varying lengths:
- 100, 500, 1,000, 2,000, 5,000, 10,000, 50,000, 100,000, 500,000, 1,000,000 observations

### Numerical Accuracy Validation

To ensure fair comparison and detect numerical issues:
- **Shared sequences**: All libraries used identical pre-generated observation sequences
- **Fixed random seed**: Reproducible results across runs (seed = 42)
- **Log-likelihood comparison**: Precision validation to machine epsilon
- **Deep numerical analysis**: Step-by-step forward variable comparison for critical validation

### Performance Metrics

- **Forward-Backward algorithm timing**: Primary performance metric
- **Viterbi algorithm timing**: Secondary performance metric
- **Throughput**: Observations processed per millisecond
- **Scaling behavior**: Performance across different sequence lengths

### Subprocess comparator timing — warmup requirement

Comparators invoked via `system()` or `popen()` (HTK, JAHMM, LAMP) are subject to
OS-level cold-start latency on the first execution in a session. On Windows this is
primarily the security scanner loading and verifying the executable image; the effect
also exists on Linux/macOS but is typically smaller.

**Observed magnitude**: first LAMP invocation on Windows took ~1,440 ms vs ~45 ms for
all subsequent calls — a >30x inflation unrelated to HMM computation.

**Mitigation**: every subprocess-based benchmark in this suite calls a `warmup()`
method before the timed loops. The warmup runs the executable once with a minimal
problem (2 states, 2 symbols, 10 observations) and discards the result. This primes
the OS page cache and satisfies any one-time security scan so timed runs reflect
steady-state subprocess overhead.

**General rule**: any benchmark that measures a comparator through a `system()` call
must include a warmup invocation before timed measurements. Omitting it conflates OS
scheduling and security overhead with algorithm performance. This applies to any new
subprocess-based comparator added in future.

## Key Findings

### Numerical Accuracy Results

libhmm shows machine-precision agreement with key reference libraries:

| Library | Numerical Accuracy vs libhmm | Status |
|---------|------------------------------|---------|
| **HMMLib** | Identical to machine epsilon | ✅ Machine precision |
| **StochHMM** | Identical to machine epsilon (after PI correction; see note below) | ✅ Machine precision |
| **GHMM** | Identical to machine epsilon | ✅ Machine precision |
| **HTK** | Rounded log-probabilities | ⚠️ Rounded output |

**Example numerical comparison** (Casino Problem, 1000 observations):
- libhmm: -1.815e+03
- HMMLib: -1.815e+03
- StochHMM: -1.815e+03
- GHMM: -1.815e+03
- HTK: -2.000e+03 ← **Deliberately rounded for computational efficiency**

**HTK Numerical Approach**: HTK appears to use rounded log-probabilities (multiples of 1000) as a design choice, likely for computational efficiency and file-based processing. This is consistent with HTK's focus on speech recognition where exact likelihood values are less critical than relative comparisons.

### StochHMM Continuous Gaussian Accuracy Correction (April 2026)

During comparative benchmarking, we identified a deterministic log-likelihood offset in StochHMM continuous Gaussian runs. The root cause was an incorrect π constant in `source/src/stochMath.h` in the external StochHMM dependency:

- Incorrect: `#define PI 3.145926535897932`
- Corrected: `#define PI 3.141592653589793238463`

StochHMM's Gaussian PDF normalization uses this constant, so the error introduced a fixed per-observation shift in log-space. The observed discrepancy scaled linearly with sequence length:

- 100: `6.893e-02`
- 500: `3.446e-01`
- 1000: `6.893e-01`
- 2000: `1.379e+00`
- 5000: `3.446e+00`

After correcting the constant and rebuilding/reinstalling `libstochhmm.a`, the same benchmark matched libhmm to machine precision:

- 100: `2.842e-14`
- 500: `0.000e+00`
- 1000: `0.000e+00`
- 2000: `9.095e-13`
- 5000: `0.000e+00`

Interpretation guidance:

- Continuous Gaussian StochHMM accuracy comparisons from builds with the incorrect PI constant are not valid for numerical-accuracy conclusions.
- Discrete-model benchmark conclusions are unaffected by this specific issue.
- Reproducible StochHMM continuous benchmarking requires rebuilding the external dependency after applying the PI correction.

### Performance Rankings
Historical snapshot from earlier benchmark runs; use the April 2026 consolidated table for current comparisons.

#### Overall Throughput (observations/millisecond)

| Rank | Library | Avg Throughput | Speedup vs libhmm |
|------|---------|----------------|-------------------|
| 1 | **GHMM** | 25,164.9 obs/ms | **24.25x faster** |
| 2 | **HMMLib** | 18,889.2 obs/ms | **17.83x faster** |
| 3 | **StochHMM** | 2,075.7 obs/ms | **2.10x faster** |
| 4 | **libhmm** | 1,037.6 obs/ms | *baseline* |
| 5 | **HTK** | Variable* | *See scaling analysis* |

*HTK shows unique scaling characteristics - see detailed analysis below.

#### Performance by Sequence Length

**Short Sequences (≤1,000 observations):**
- GHMM: 25-30x faster than libhmm
- HMMLib: 15-20x faster than libhmm
- StochHMM: 2x faster than libhmm
- HTK: 5-50x slower than libhmm (high initialization overhead)

**Medium Sequences (1,000-10,000 observations):**
- GHMM: 20-25x faster than libhmm
- HMMLib: 15-20x faster than libhmm
- StochHMM: 2x faster than libhmm
- HTK: Approaching libhmm performance

**Long Sequences (≥50,000 observations):**
- GHMM: 20-25x faster than libhmm
- HMMLib: 15-20x faster than libhmm
- StochHMM: 2x faster than libhmm
- HTK: 10-200x faster than libhmm (strong scaling)

### HTK Scaling Analysis

HTK exhibits unique performance characteristics reflecting its design for speech recognition workflows:

**Performance Pattern:**
- **Constant overhead**: ~4-5ms regardless of sequence length
- **Strong scaling**: Minimal per-observation cost for long sequences
- **Crossover point**: Becomes competitive around 5,000 observations
- **Very long sequences**: Up to 200x faster than libhmm for 1M+ observations

**Design Trade-offs:**
- Uses rounded log-likelihood values (multiples of 1000) for efficiency
- File-based operations optimized for batch processing
- **Suitable for relative likelihood comparisons and classification tasks**
- **Less suitable for applications requiring precise likelihood values**

## Detailed Analysis

### Algorithm Selection and Optimization

**libhmm canonical calculator path (current architecture):**
- Uses canonical log-space ForwardBackward and Viterbi calculator implementations
- Uses distribution-level batch hooks (`getBatchLogProbabilities()`) for SIMD acceleration where available
- Emphasizes numerical stability and implementation consistency across sequence lengths

**Library-Specific Optimizations:**
- **GHMM**: Highly optimized C implementation with strong scaling
- **HMMLib**: Boost-based optimizations, particularly strong matrix operations
- **StochHMM**: Moderate optimizations, designed for bioinformatics workflows
- **HTK**: File-based operations with efficiency-focused approximations

### Memory and Scalability

All libraries successfully processed sequences up to 1,000,000 observations without memory issues, indicating robust implementation across the ecosystem.

## Implementation Quality Assessment

### Code Quality and Maintainability

| Library | Language | Dependencies | Modern Features | Maintainability |
|---------|----------|--------------|-----------------|-----------------|
| **libhmm** | C++20 | None | ✅ Modern C++ | ✅ High |
| **HMMLib** | C++ | Boost | ⚠️ Legacy patterns | ⚠️ Moderate |
| **StochHMM** | C++ | None | ⚠️ Mixed patterns | ⚠️ Moderate |
| **GHMM** | C | None | ❌ C-style API | ❌ Difficult |
| **HTK** | C | None | ❌ Command-line only | ❌ Very difficult |

### Numerical Stability

**libhmm Implementation Validation:**
- ✅ Identical results to academic reference implementations
- ✅ Proper scaling for numerical stability
- ✅ Consistent behavior across all sequence lengths
- ✅ No numerical underflow or overflow issues detected

## Conclusions and Recommendations

### Primary Findings

1. **libhmm is numerically correct**: Machine-precision agreement with established academic libraries indicates mathematically sound implementation.

2. **Performance is competitive**: While not the fastest, libhmm provides reasonable performance for a modern, dependency-free library.

3. **Ecosystem validation**: The benchmarking confirms libhmm's place as a reliable, modern alternative in the HMM library landscape.

### Use Case Recommendations

#### Choose **GHMM** when:
- Maximum raw performance is critical
- C integration is acceptable
- Numerical precision is important
- You can handle complex C API

#### Choose **HMMLib** when:
- High performance is needed
- C++ integration is required
- Boost dependencies are acceptable
- Legacy code compatibility is needed

#### Choose **libhmm** when:
- Modern C++20 features are desired
- Zero external dependencies are required
- Code maintainability is important
- Moderate performance is sufficient
- Cross-platform compatibility is needed
- Precise log-likelihood values are required

#### Choose **StochHMM** when:
- Bioinformatics applications are the focus
- Moderate performance is acceptable
- Specialized biological sequence features are needed

#### Choose **HTK** when:
- Speech recognition or classification tasks are the focus
- Very long sequence processing is required
- Relative likelihood comparisons are sufficient
- Batch processing workflows are preferred

### Performance Context

libhmm's performance should be evaluated in context:

- **1,045 observations/ms** means processing 1 million observations in ~1 second
- For most practical applications, this performance is more than adequate
- The ~20x speed difference with top performers matters primarily for:
  - High-frequency real-time applications
  - Massive batch processing workflows
  - Training on extremely large datasets

### Future Development

The benchmarking reveals optimization opportunities for libhmm:

1. **SIMD optimizations**: Further vectorization could improve performance
2. **Memory layout**: Cache-friendly data structures could reduce overhead
3. **Algorithm variants**: Additional specialized calculators could be beneficial
4. **Parallel processing**: Multi-threading support for very large sequences

## Technical Notes

### Benchmarking Environment
- **Platform**: macOS (Apple Silicon/Intel compatible)
- **Compiler**: Modern C++ compiler with -O3 optimization
- **Libraries**: All built from source with optimizations enabled

### Reproducibility
All benchmark code and configurations are available in the `benchmarks/` directory. The benchmarks use fixed random seeds and shared observation sequences to ensure reproducible results across runs and platforms.

**Note on External Libraries**: The original source code for HMMLib, StochHMM, GHMM, and HTK is not included in this repository. To reproduce these benchmarks, these libraries must be obtained from their respective developers/maintainers and built according to their official documentation:

- **HMMLib**: Available from original authors/research institutions
- **StochHMM**: https://github.com/KorfLab/StochHMM
- **GHMM**: http://ghmm.org
- **HTK**: http://htk.eng.cam.ac.uk (requires registration)

The benchmark implementations in this repository provide the integration code necessary to test these libraries once properly installed.

### Validation Methodology
The numerical accuracy validation included:
- Direct log-likelihood comparison to machine precision
- Step-by-step forward algorithm verification
- Cross-validation between multiple reference implementations
- Deep numerical analysis of scaling factors and intermediate values

This validation indicates that libhmm's implementation is mathematically equivalent to established academic reference implementations.

### Library-Specific Notes

**HTK Implementation**: HTK's use of rounded log-probabilities reflects its optimization for speech recognition applications where computational efficiency and relative likelihood comparisons are more important than absolute precision. This is a valid design choice for its intended domain but should be considered when selecting HTK for applications requiring exact likelihood values.

---

## Continuous Distribution Benchmarking Results (HTK vs libhmm)
Historical HTK-focused snapshot from earlier runs; compare with the April 2026 consolidated section for current multi-library throughput context.

### Test Configuration
**Problems Tested:**
- **Gaussian Speech**: 2-state vowel/consonant model (means: 2.0, 8.0; variances: 0.5, 1.0)
- **Gaussian Temperature**: 2-state normal/overheating model (means: 22.0, 45.0; variances: 2.0, 8.0)

**Sequence Lengths:** 100, 500, 1,000, 5,000, 10,000, 50,000, 100,000, 500,000, 1,000,000 observations

### Continuous HMM Performance Results

#### Performance Crossover Analysis
| Sequence Length | libhmm Performance | HTK Performance | Winner |
|-----------------|--------------------|-----------------|---------|
| 100 obs         | 20-40x faster      | High overhead   | **libhmm** |
| 500 obs         | 9-10x faster       | Moderate overhead| **libhmm** |
| 1,000 obs       | 4-5x faster        | Approaching parity| **libhmm** |
| 5,000 obs       | ~1x (equal)        | ~1x (equal)     | **Equal** |
| 10,000 obs      | 0.5x               | 2x faster       | **HTK** |
| 50,000 obs      | 0.1x               | 10x faster      | **HTK** |
| 100,000 obs     | 0.05x              | 20x faster      | **HTK** |
| 500,000 obs     | 0.01x              | 100x faster     | **HTK** |
| 1,000,000 obs   | 0.006x             | 177x faster     | **HTK** |

#### Key Performance Insights

**HTK Scaling Characteristics:**
- **Constant overhead**: ~4-5ms regardless of sequence length
- **Strong scaling**: Near-constant time complexity for very long sequences
- **Peak throughput**: Up to 201,734 observations/ms for 1M observation sequences
- **Architecture**: Likely uses streaming/batched algorithms optimized for speech recognition

**libhmm Scaling Characteristics:**
- **Linear scaling**: Traditional O(n) HMM algorithms with SIMD optimization
- **SIMD benefits**: Consistent 1.65x performance improvement over baseline
- **Small-scale performance**: Strong results for research-scale problems
- **Precision focus**: More accurate log-likelihood values (vs HTK's rounded estimates)

#### Numerical Accuracy Comparison
**libhmm**: Provides precise log-likelihood values (e.g., -1.774e+02)
**HTK**: Uses rounded log-likelihood estimates (e.g., -2.000e+02) for computational efficiency

**Implication**: HTK sacrifices some numerical precision for substantial performance gains on large sequences

### Continuous vs Discrete Performance

#### Continuous Distribution Support
| Library | 1D Gaussian | Multi-D Gaussian | Other Continuous |
|---------|-------------|------------------|-----------------|
| **libhmm** | ✅ Full | ❌ Not yet | ✅ Many distributions |
| **HTK** | ✅ Full | ✅ Full | ✅ Speech-focused |

**Note**: libhmm currently supports 1D continuous distributions only, while HTK provides full multi-dimensional continuous distribution support optimized for speech recognition.

### Architecture Trade-offs

#### libhmm Strengths
- **Research-friendly**: Precise numerical computations
- **Modern C++**: Easy integration and maintenance
- **Small-scale performance**: Strong for sequences < 5,000 observations
- **SIMD optimization**: Effective vectorization for appropriate problem sizes

#### HTK Strengths
- **Production-ready**: Industrial-strength scaling for speech recognition
- **Large-scale performance**: Very strong for sequences > 10,000 observations
- **Mature ecosystem**: Decades of optimization for speech processing workflows
- **Multi-dimensional support**: Full continuous distribution capabilities

### Updated Recommendation Matrix

| Use Case | Sequence Length | Precision Needs | Distribution Type | Recommended Library |
|----------|----------------|-----------------|-------------------|--------------------|
| Research | < 5,000 obs | High precision | Any | **libhmm** |
| Prototyping | < 1,000 obs | Moderate | Discrete/1D Continuous | **libhmm** |
| Production Speech | > 10,000 obs | Relative comparisons | Multi-D Continuous | **HTK** |
| Batch Processing | > 50,000 obs | Efficiency focus | Any | **HTK** |
| Multi-dimensional | Any length | Continuous features | Multi-D Gaussian | **HTK** |
| Modern C++ Integration | Any length | Developer productivity | Discrete/1D Continuous | **libhmm** |

This continuous distribution analysis outlines the performance envelope for both libraries and provides guidance for selecting a library based on application requirements.

---

## Benchmark Update (April 2026): Consolidated `libhmm_vs_*` Results

This section adds an updated snapshot without removing any prior content. Earlier benchmark tables above remain as historical context.

### Scope of this update

- Re-ran and/or validated all available `libhmm_vs_*` benchmark targets in the current benchmark suite.
- Included post-fix JAHMM and LAMP runs after external path/runtime integration fixes.
- Consolidated throughput summaries into a single table covering every `libhmm_vs_*` target.

### Overall Throughput Across All `libhmm_vs_*` Benchmarks (Release Runs)

| Benchmark target | Comparator | libhmm avg throughput (obs/ms) | Comparator avg throughput (obs/ms) | Ratio (Comparator/libhmm) | Source log |
|------------------|------------|--------------------------------|------------------------------------|---------------------------|------------|
| `libhmm_vs_ghmm_benchmark` | GHMM | 9514.3 | 47232.7 | 4.96x | `build-benchmarks-release/benchmark-logs/libhmm_vs_ghmm_benchmark.log` |
| `libhmm_vs_ghmm_continuous_benchmark` | GHMM | 10017.4 | 26059.9 | 2.60x | `build-benchmarks-release/benchmark-logs/libhmm_vs_ghmm_continuous_benchmark.log` |
| `libhmm_vs_hmmlib_benchmark` | HMMLib | 8451.6 | 27196.3 | 3.22x | `build-benchmarks-release/benchmark-logs/libhmm_vs_hmmlib_benchmark_prior5.log` |
| `libhmm_vs_htk_benchmark` | HTK | 9333.1 | 41012.9 | 4.39x | `build-benchmarks-release/benchmark-logs/libhmm_vs_htk_benchmark_prior5.log` |
| `libhmm_vs_htk_continuous_benchmark` | HTK | 14217.4 | 45119.0 | 3.17x | `build-benchmarks-release/benchmark-logs/libhmm_vs_htk_continuous_benchmark_batch2.log` |
| `libhmm_vs_stochhmm_benchmark` | StochHMM | 9433.6 | 5825.6 | 0.62x | `build-benchmarks-release/benchmark-logs/libhmm_vs_stochhmm_benchmark.log` |
| `libhmm_vs_stochhmm_continuous_benchmark`* | StochHMM | 3605.3 | 5839.5 | 1.62x | `build-benchmarks-release/benchmark-logs/libhmm_vs_stochhmm_continuous_benchmark_after_pi_fix.log` |
| `libhmm_vs_jahmm_benchmark`** | JAHMM | 7161.5 | 3803.6 | 0.53x | `build-benchmarks-release/benchmark-logs/libhmm_vs_jahmm_benchmark_after_pathfix.log` |
| `libhmm_vs_lamp_benchmark` | LAMP | 6016.7 | 48.2 | 0.01x | Windows x86_64 run, April 2026 (post-warmup) |

\* Uses post-PI-correction StochHMM continuous results (`after_pi_fix`).
\** JAHMM benchmark log does not emit an average throughput summary line; values above are computed from per-run forward timings in the same log.

### Updated Code Quality and Maintainability Snapshot (All Evaluated Libraries)

| Library | Primary implementation | Dependency footprint | Interface style | Maintainability (relative) | Integration effort in this benchmark suite |
|---------|------------------------|----------------------|-----------------|----------------------------|--------------------------------------------|
| **libhmm** | Modern C++ (v3 architecture) | None | Native C++ API | High | Low |
| **HMMLib** | C++ (legacy-style template/matrix design) | Boost | Native C++ API | Medium | Medium |
| **StochHMM** | C++ (bioinformatics-oriented) | Low | C++ API + model/trellis abstractions | Medium | Medium |
| **GHMM** | C | Low | C API | Low | High |
| **HTK** | C toolkit | Low | CLI + file-based workflow | Low | High |
| **JAHMM** | Java | Java runtime/toolchain | Java API (bridge invoked from C++ benchmark) | Medium-Low | High |
| **LAMP** | C executable toolkit | Low | CLI + config-file workflow | Low-Medium | High |

### Notes on interpretation

- The consolidated table shows materially higher libhmm throughput than the original historical table in this document, reflecting updated code paths, release builds, and benchmark harness modernization.
- Throughput comparisons remain benchmark-specific and should be interpreted per target (`discrete` vs `continuous`, API path, and runtime model differences).
- Comparator rankings should be read together with the library-specific constraints above. A faster result may reflect narrower scope, coarser numerical output, or a workflow optimized for a different domain rather than a strictly better general-purpose HMM implementation.

### Post-modernization validation signal (April 2026)

To capture correctness signal separately from throughput, three updated diagnostic benchmarks were re-run:

- `build-benchmarks-release/benchmark-logs/diagnostic_accuracy_test_modernized.log`
- `build-benchmarks-release/benchmark-logs/deep_numerical_analysis_modernized.log`
- `build-benchmarks-release/benchmark-logs/gaussian_distribution_comparison_modernized.log`

Key outcomes:

- **Canonical numerical parity with HMMLib** (`deep_numerical_analysis_modernized.log`):
  Across sequence lengths 10, 50, 100, 200, 500, 1000, and 2000, libhmm and HMMLib log-likelihoods match to near machine precision. Maximum absolute difference observed: `5.093170e-11` (length 2000), with no length-dependent drift pattern.

- **Step-level forward-pass agreement** (`deep_numerical_analysis_modernized.log`):
  Normalized per-step forward-variable differences are in floating-point noise range (`~1e-16`, max shown `4.163e-16`), and final log-probability difference is `0.000000e+00`.

- **Distribution-layer Gaussian agreement across libraries** (`gaussian_distribution_comparison_modernized.log`):
  libhmm, GHMM, and StochHMM report `MATCH` across all tested Gaussian cases (standard, shifted mean, negative mean, high variance), indicating aligned PDF/log-PDF behavior at the distribution layer.

- **Constructor semantics validated for reproducibility** (`diagnostic_accuracy_test_modernized.log`):
  `GaussianDistribution(mean, second_parameter)` uses **standard deviation** semantics (not variance). This check avoids silent benchmark misconfiguration when mapping model parameters.

- **Canonical calculator self-consistency checks pass** (`diagnostic_accuracy_test_modernized.log`):
  ForwardBackward pointer/reference constructors and `getLogProbability()` vs `log(probability())` are numerically identical on the test model; a manual forward calculation also matches libhmm (`probability diff 6.939e-18`, `log diff 0.000e+00`).
