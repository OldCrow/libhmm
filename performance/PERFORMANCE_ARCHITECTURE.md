# Performance Architecture
Where SIMD lives, where threading lives, and what's reserved for future work.
## SIMD strategy: per-distribution batch evaluation
SIMD in libhmm targets the natural batch unit of HMM inference: "compute log P(O\_t | state\_i) for all T observations of a given state." Each `EmissionDistribution` exposes:
```cpp path=null start=null
void getBatchLogProbabilities(std::span<const double> observations,
                              std::span<double> out) const;
```
The canonical calculators (`ForwardBackwardCalculator`, `ViterbiCalculator`) call this once per state per `compute()`, producing T contiguous log-emission values that the recurrences then consume from a flat row-major buffer.
Two tiers of implementation:
- **Tier 2 — explicit intrinsics.** `GaussianDistribution` and `ExponentialDistribution` ship hand-written `detail::` free functions with an AVX-512 → AVX/AVX2 → SSE2 → NEON → scalar dispatch chain. See `src/distributions/gaussian_distribution.cpp` `detail::gaussian_logpdf_batch` for the canonical shape. The free-function pattern is deliberately extractable to a separate TU for future runtime dispatch without API changes.
- **Tier 1 — auto-vectorization-friendly loops.** The other 13 distributions implement `getBatchLogProbabilities` as concrete non-virtual loops over plain arrays, compiled with `LIBHMM_BEST_SIMD_FLAGS` (the highest CPU-verified ISA on the build machine). Whether the compiler actually emits vector instructions depends on the loop body — transcendentals like `std::exp` are not auto-vectorized by MSVC even with `/arch:AVX2`, so tier 1 is best read as "well-shaped scalar code" rather than "guaranteed SIMD."
All 15 distribution TUs are listed in `LIBHMM_SIMD_SOURCES` in the top-level `CMakeLists.txt` and receive the SIMD compile flags.
## Where SIMD does and doesn't live today
- ✅ **Distribution batch emission evaluation** — `getBatchLogProbabilities`. Effective for emission-bound workloads (continuous distributions, large T). Tier 2 in particular delivers measurable speedups; tier 1 depends on compiler heuristics.
- ⚠️ **Recurrence kernels** — FB max-reduce, BW xi accumulation, Viterbi inner loop. These are state×state inner loops dominated by `exp` / `log1p` calls. Currently scalar. The active perf-branch work introduces an internal `TranscendentalKernels` abstraction in `include/libhmm/performance/transcendental_kernels.h` with scalar today and AVX2/NEON backends planned, so future explicit vector-math implementations can plug in without rewriting the call sites.
- The runtime `Matrix`/`Vector` typedefs in `common/common.h` resolve to `BasicMatrix<double>`/`BasicVector<double>`. The library no longer ships separate "optimized" container variants (see Historical context).
## Threading: not currently used
Production calculators and trainers run single-threaded on every workload. Specifically:
- No code in `src/` instantiates a thread pool or invokes `std::execution::par_unseq` on the production path.
- `ThreadPool` (in `platform/thread_pool.h`) is alive but only consumed by two diagnostic tools (`tools/analyze_overhead.cpp`, `tools/debug_parallel.cpp`); nothing in the library itself uses it.
- The `std::atomic<bool>` cache in `DistributionBase` exists for read-safety in case a downstream consumer multiplexes a single HMM instance across threads, not because libhmm threads internally.
### Where threading would have leverage if reintroduced
- **Per-sequence work in Baum-Welch** — multiple training sequences are embarrassingly parallel; each computes its own α/β/γ/ξ contribution before a synchronized M-step accumulate. Determinism requires a stable reduction order.
- **Per-state work inside a recurrence timestep** — each state's column in the FB max-reduce is independent. N is typically small (2-64), so a persistent pool with cache-line-aligned per-state buffers is the only way thread-launch overhead pays off; parallel reductions break determinism unless explicitly ordered.
Any reintroduction of threading should come with an explicit determinism story and per-kernel scoping. A general-purpose work-stealing pool was tried in an earlier design pass and abandoned without a consumer; it should not be reintroduced without a target kernel in mind.
## Build-time SIMD selection
The build system picks the highest CPU-verified ISA per machine and applies it as a per-TU compile flag to `LIBHMM_SIMD_SOURCES`:
- **GCC/Clang on all platforms**: `-march=native`. Selects NEON on AArch64, the highest available x86 ISA on Intel/AMD.
- **MSVC on x86_64**: probes `/arch:AVX512`, `/arch:AVX2`, `/arch:AVX` via `check_cxx_source_runs` and selects the highest one the build machine can actually execute (not just the highest the compiler accepts). Falls back to SSE2 baseline in cross-compilation.
- **AArch64**: NEON is the mandatory ISA baseline; no flag needed.
See the `# SIMD DETECTION` block in `CMakeLists.txt` for details. The non-distribution sources (`src/common/`, `src/calculators/`, `src/training/`, `src/io/`, `src/performance/`) compile at the platform baseline ISA so that explicit intrinsics in the distribution TUs are the only place SIMD codegen is committed to.
## Historical context
An earlier draft of this document described a four-level hierarchy in which calculators consumed `OptimizedMatrix`/`OptimizedVector` containers and a `WorkStealingPool` provided per-state parallelism. That plan was superseded by the v3.0.0-alpha (Phase 4) refactor (see `CHANGELOG.md`), which removed the per-calculator SIMD variants (`ScaledSIMD*`, `LogSIMD*`, `AdvancedLog*`) in favor of the per-distribution batch interface documented above. The Optimized\* containers, `WorkStealingPool`, the per-library `Benchmark` framework, and the parallel-execution constants/utilities they depended on were retained for several releases as "future hooks" but never wired into the canonical calculator/trainer pipeline; they were removed in a subsequent dead-code cleanup. The SIMD investment in `getBatchLogProbabilities` is the canonical and current strategy.
