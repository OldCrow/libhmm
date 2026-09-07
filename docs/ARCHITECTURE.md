# Architecture

Reference detail for libhmm's header layering, v4 template design, SIMD
dispatch strategy, distribution fit-quality tiers, model selection, and I/O
formats. AGENTS.md carries the guardrails (batch-API exception contract, FP
contraction, threading); this doc carries the orientation and reasoning
behind them.

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
| — | `performance/` | `TranscendentalKernels` (FB recurrence), `fb_recurrence_policy.h`, `simd_double_ops.h` (runtime-dispatch distribution batch kernels) |
| — | `detail/` | Internal: `simd_math_helpers.h` (shared SIMD math helpers) and `trig_cleanroom_data.inc` (its trig constants) — not installed; `log_utils.h` (shared log-space utilities for the calculators/trainers) — installed, since public headers include it |

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
    - **Poisson, Binomial**: the observation is an integer count, so the log-factorial terms are a **table lookup** (`math/log_factorial.h`, exact to k = 18, tabulated to k = 1023), not a transcendental call. Binomial never calls `lgamma` at all — `logBinomialCoefficient` is three lookups. The blocker is therefore the **gather** to index by k, the same one as Discrete below, and libstats settled empirically (its #33) that x86 hardware gather is too expensive to pay for; table kernels are a NEON technique, not an x86 one.
    - **NegativeBinomial**: genuinely needs a vectorized `lgamma`, and is the only one of the three that does. `log Γ(k + r)` has a continuous `r` so it cannot be tabulated, while `log k!` already is and `log Γ(r)` is a per-parameter constant — one `lgamma` per element. This is the single concrete case for a vectorized-lgamma dependency (e.g. corvus); size any such proposal against one distribution, not three.
    - **Discrete**: per-element integer floor + range check and table lookup by symbol index. Vectorizable in principle via AVX2 gather, but complex index arithmetic and no performance data justifying the effort.
    - **Uniform**: the entire batch evaluates to a single constant (log(1/(b−a))) inside bounds or −∞ outside. Already ~2 instructions per element; SIMD buys nothing.
  MV distributions (`DiagonalGaussian`, `FullCovGaussian`, `IndependentComponents`) call `getLogProbability(row_view(obs, t))` per timestep rather than a batch interface and are not in `LIBHMM_SIMD_SOURCES`.

`detail/simd_math_helpers.h` is the single source of truth for vectorized log/exp/cos/sin/log1p helpers shared by the per-ISA distribution kernels and `TranscendentalKernels`. log/exp are SLEEF-derived (< 1 ULP). cos/sin are the clean-room quadrant-reduction kernel (#74, constants from `scripts/gen_trig_cleanroom_table.py`): faithfully rounded (max 1 ULP, mean ~0.03) for |x| ≤ 2²³, per-lane scalar libm fixup beyond, gated per tier against checked-in mpmath references in `tests/performance/test_trig_ulp_gates.cpp`. Tiny `log1p` inputs use a polynomial path for accuracy; general inputs reuse the shared vector log helper. The `log1p_batch` table entry is add-then-log with no small-|x| path — that is its documented contract (`simd_double_ops.h`; #77 closed 2026-08-19 with no in-library consumer found), and `log1p_inplace` is the entry that carries the polynomial path.

For the batch-API exception contract, the FP-contraction analysis, and the
threading/parallelism contract, see the `## Architecture` section in
AGENTS.md — those are guardrails, kept there verbatim.

### Distribution fit quality

The weighted `fit(data, weights)` method is the Baum-Welch M-step. Fit quality varies by distribution:

- **Tier A — exact weighted MLE/EM**: Gaussian, Exponential, Poisson, Discrete, LogNormal, Pareto,
  Rayleigh, VonMises, Binomial, ChiSquared (Newton MLE), Gamma, Weibull, NegativeBinomial,
  Beta, StudentT (Newton/ECME)
- **Tier C — MOM (defensible in EM context)**: Uniform (fixed-range; MOM is exact for uniform support)

Priority M-step improvements are documented in `docs/GOLD_STANDARD_CHECKLIST.md`.

All `fit(data, weights)` implementations guard against near-zero weight by preserving current parameters (not calling `reset()`):
```cpp
if (sumW < precision::ZERO || std::isnan(sumW)) return;
```
`reset()` is called only for genuinely degenerate *data*. The three MV fits use the same guard (a subnormal `sumW` used to pass `<= 0.0` and overflow `1/sumW`). Known drift, tracked as an issue: LogNormal and Student-t normalise by the *total* weight while skipping out-of-support points, which deflates the fitted moments when such points carry weight.

### Model selection

`count_free_parameters(hmm)`, `compute_aic()`, `compute_bic()`, `compute_aicc()`, and `evaluate_model()` are declared in `include/libhmm/training/model_selection.h`.

### I/O

JSON is the recommended format—exact IEEE 754 round-trip, no external dependencies. Scalar: `save_json`/`load_json`. MV: `save_json_mv`/`load_json_mv` (v4 schema with `obs_type: "multivariate"`). Legacy XML (`XMLFileReader`/`XMLFileWriter`) is scalar-only and deprecated; retained for reading existing `.xml` files. Reference HMM files live in `samples/`.
