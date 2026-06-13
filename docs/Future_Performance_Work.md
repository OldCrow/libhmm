# libhmm — Future Performance Work

Current SIMD and threading strategy: `performance/PERFORMANCE_ARCHITECTURE.md`.
M-step quality improvements: `docs/GOLD_STANDARD_CHECKLIST.md`.

---

## What is already done

1. **Batch emission evaluation** — Tier 1 (auto-vectorisable concrete loops) and
   Tier 2 (explicit AVX-512/AVX2/SSE2/NEON intrinsics) implemented for all 16
   scalar distributions via `getBatchLogProbabilities()`.  Gaussian and
   Exponential have hand-written Tier 2 kernels; the other 14 are Tier 1.

2. **Forward-backward recurrence** — `TranscendentalKernels` max-reduce path
   eliminates redundant `exp()` calls in the inner loop.  Adaptive policy selects
   max-reduce over pairwise log-sum-exp for N ≥ 4.  Both the scalar and MV
   explicit-instantiation TUs are compiled with `LIBHMM_BEST_SIMD_FLAGS`.

3. **Log-transition precomputation** — `BasicCalculator::precompute_log_transitions`
   builds both logTrans and logTransT once at construction; reused across
   `compute()` calls on the same HMM.

4. **Zero-copy observation accumulation in trainers** — MV trainers accumulate
   non-owning `ObservationVectorView` row spans; scalar trainers pre-allocate with
   `reserve()` to avoid repeated reallocation.  FBC emission buffer is reused by
   BW trainer via `getLogEmitByTime()` to avoid a second emission pass.

---

## Remaining high-value opportunities

### 1. Per-sequence parallelism in Baum-Welch (largest impact)

Each training sequence in `BasicBaumWelchTrainer::train()` runs an independent
FBC before contributing to the shared M-step accumulators.  This is embarrassingly
parallel: the E-step per sequence can run on a thread pool; the M-step remains
single-threaded.

Expected gain: near-linear in the number of training sequences, bounded by
cache pressure on the shared accumulator reduction.

Design constraint: reduction order must be deterministic for reproducible results.
Use per-thread accumulators with a final reduce rather than a shared atomic.
See `PERFORMANCE_ARCHITECTURE.md §Threading` for the full design note.

### 2. Runtime SIMD dispatch

Build-time selection (`-march=native` / CPU-probed `/arch:`) produces a binary
optimised for the build machine only.  The `detail::` free-function pattern in
Tier 2 (`GaussianDistribution`, `ExponentialDistribution`) was deliberately
designed to be extractable — moving dispatch from link-time to runtime (function
pointer or `ifunc`) requires no API changes.

Prerequisite: decide whether portable / package-manager distribution is a library
goal before investing here.

### 3. MV distribution batch evaluation

The three multivariate distributions call `getLogProbability(row_view(obs, t))`
per observation — N×T virtual calls per `compute()`.  For large T, a
`getBatchLogProbabilities(span<const ObservationVectorView>, span<double>)` override
with an auto-vectorisable inner loop would be significantly faster.
`DiagonalGaussianDistribution` is the best first candidate: the inner product
`Σ_d (x_d - μ_d)² / σ²_d` per observation is SIMD-friendly with a fixed D.

### 4. Tier C M-step quality (fewer EM iterations = effective speedup)

See `docs/GOLD_STANDARD_CHECKLIST.md §Outstanding M-step improvements`:

1. `StudentTDistribution` — ECM scale-mixture (highest priority; impacts DAX/S&P benchmarks)
2. `GammaDistribution` — Newton MLE for k (impacts elk step-length fitting quality)
3. `WeibullDistribution` — Newton MLE for k
4. `NegativeBinomialDistribution` — profile likelihood for r

Better M-steps reduce the iteration count to convergence, which is the dominant
cost in large training runs.

---

## Benchmark infrastructure

Comparative benchmarks against external libraries: `benchmarks/`.
Internal throughput tools: `tools/batch_performance`, `tools/simd_inspection`.
All performance work should start from measurement, not assumption.
