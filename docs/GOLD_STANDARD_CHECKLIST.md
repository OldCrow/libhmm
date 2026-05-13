# LibHMM Gold Standard Distribution Implementation Checklist

This document tracks the implementation status of all probability distributions in libhmm
according to the "Gold Standard" requirements, plus a fit quality survey added in May 2026.

---

## Gold Standard Requirements (v3.0)

### v3.0 Interface Requirements (EmissionDistribution — added Phase 2/4.5)

All distributions must implement the `EmissionDistribution` abstract interface. New in v3.0:

- ✅ **`getBatchLogProbabilities(std::span<const double> obs, std::span<double> out) const override`**
  - **Tier 1 (all 16)**: Concrete non-virtual loop using cached parameters directly.
    Virtual dispatch in the `DistributionBase` default defeats compiler auto-vectorization.
    The override removes that overhead and allows `-march=native` / `/arch:AVX2` to vectorize.
  - **Tier 2 (Gaussian, Exponential)**: Explicit SIMD intrinsics via `detail::` free function
    taking only plain data pointers. AVX-512 (8-wide), AVX2 (4-wide), SSE2 (2-wide), NEON (2-wide),
    scalar tail. NaN inputs yield `-Inf` to match scalar behaviour.
  - **Tier 2 upgrade paths for other distributions** are documented inline in each `.cpp` file.
    Distributions with `log(x)` (LogNormal, Rayleigh, Pareto, Weibull, Gamma) require vectorised log
    (Intel SVML / GNU libmvec / Apple Accelerate vvlog). Distributions with `lgamma` (Beta,
    ChiSquared, StudentT, Poisson, Binomial, NegBinomial) require vectorised lgamma. Defer until
    a portable math-library dependency is chosen.

- ✅ **`fit(std::span<const double> data, std::span<const double> weights) override`**
  - Weighted MLE (or MOM approximation — see Fit Quality section) for the Baum-Welch M-step.
  - Weights are unnormalised γ values; each distribution normalises by `sum(weights)` internally.
  - When `sum(weights) ≈ 0` (state has near-zero responsibility), current parameters are
    **preserved** — not reset. Resetting would destroy valid parameters and cause EM state
    collapse (the state gets default params, attracts no observations, and cannot recover).
  - `reset()` is still called for genuinely degenerate *data* (e.g. non-positive values for
    Exponential, zero-variance for distributions that require positive spread). This is correct
    because the data itself is pathological, not the weight distribution.

- ✅ **`std::span<const double>` parameter types** — replaces `std::vector<Observation>&`.
  No copies in the hot path.

- ✅ **Thread-safe cache** — `std::atomic<bool> cacheValid_` in `DistributionBase`
  with `acquire`/`release` memory ordering.

- ✅ **C++20 standard** — `[[nodiscard]]`, `noexcept`, `std::span`, `#pragma once`.

### Implementation Requirements
- ✅ **Core Methods:**
  - `getProbability()` — PDF/PMF (const)
  - `getLogProbability()` — log PDF/PMF with numerical stability (const, noexcept)
  - `getCumulativeProbability()` — CDF (where mathematically meaningful)
  - `fit(std::span<const double>)` — unweighted MLE
  - `fit(std::span<const double>, std::span<const double>)` — weighted MLE / M-step
  - `getBatchLogProbabilities(span, span)` — concrete non-virtual batch loop (tier 1 minimum)
  - `reset()` — reset to default parameters
  - `toString()` — human-readable string representation

- ✅ **Rule of Five** — copy/move constructors and assignment operators, virtual destructor.

- ✅ **Caching System** — expensive calculations cached; automatic invalidation on parameter change.

- ✅ **Input Validation** — robust parameter validation; NaN/infinity handling throughout.

- ✅ **Constants Usage** — `using namespace constants;` at the top of each `.cpp`; no magic numbers.

- ✅ **I/O Operators** — `operator==`, `operator<<`, `operator>>`, `to_json()`, `from_json()`.

---

## Fit Quality Survey (May 2026)

The weighted `fit(data, weights)` method is the Baum-Welch M-step for each distribution.
The quality of this estimate determines how well EM converges to the true maximum likelihood.

### Tier A — Exact weighted MLE (closed-form)

These distributions have closed-form weighted MLE. The M-step is correct and optimal.

| Distribution | M-step estimate | Notes |
|---|---|---|
| GaussianDistribution | Weighted Welford mean + variance | Exact MLE for (μ, σ²) |
| ExponentialDistribution | λ = 1 / weighted_mean | Exact MLE |
| PoissonDistribution | λ = weighted_mean | Exact MLE |
| DiscreteDistribution | P(k) = Σ w_i·I(x_i=k) / Σ w_i | Exact MLE |
| LogNormalDistribution | Weighted mean/variance of log(x) | Exact MLE in log-space |
| ParetoDistribution | k = sumW/sumWLog, x_m = weighted_min | Exact MLE |
| RayleighDistribution | σ = √(Σ w_i·x_i² / 2·Σ w_i) | Exact MLE |
| VonMisesDistribution | μ via atan2, κ via Mardia-Jupp + Newton | Near-optimal; Mardia-Jupp error < 0.003 |
| BinomialDistribution | p = weighted_mean / n | Exact MLE for p (n fixed as max observation) |

### Tier B — MOM ≈ MLE: gap is small in practice

Single-parameter distributions where MOM and MLE are very close for typical parameter values.

| Distribution | M-step estimate | Gap to MLE |
|---|---|---|
| ChiSquaredDistribution | k = weighted_mean | MOM (E[X]=k). True MLE requires ψ(k/2) = log(mean) − mean_of_log. Gap is small for k > 2. |

### Tier C — MOM: gap can be material

Multi-parameter distributions where MOM diverges from MLE for certain parameter ranges.
These converge to a valid local optimum but may arrive at a suboptimal one.

| Distribution | M-step estimate | Known limitation |
|---|---|---|
| GammaDistribution | k = mean²/var, θ = var/mean | MOM. MLE for k requires solving log(k) − ψ(k) = log(mean) − mean_of_log. MOM is close for k > 1; degrades for k < 0.5. |
| NegativeBinomialDistribution | p = mean/var, r = mean²/(var−mean) | MOM. MLE for r has no closed form; needs profile likelihood or Newton. |
| UniformDistribution | a = mean − √(3·var), b = mean + √(3·var) | MOM. MLE (weighted min/max) is unstable in EM context; MOM is defensible. |
| BetaDistribution | α = mean·f, β = (1−mean)·f | MOM. Degrades when α or β < 0.5. MLE requires digamma Newton. |
| WeibullDistribution | k and λ from weighted mean/variance | MOM. Gap is significant for k outside [0.5, 3]. MLE for k needs Newton. |
| StudentTDistribution | μ = weighted_mean, σ = corrected_std, ν from kurtosis | Kurtosis-based ν is noisy when a state covers few observations. ECM algorithm would give exact joint (μ, σ, ν) updates. |

### Outstanding M-step improvements (priority order)

1. **StudentTDistribution — ECM algorithm**: Replace kurtosis MOM with the Gaussian scale-mixture
   EM (Dempster 1977; Murphy 2012 §11.4.5). Per-observation scale weights u_i = (ν+1)/(ν + z_i²/σ²)
   give closed-form μ and σ updates; ν requires a Newton step. Directly improves financial HMM
   quality (demonstrated on DAX 2000–2022 vs fHMM benchmark).

2. **GammaDistribution — Newton MLE for k**: One Newton step from the MOM starting point:
   k_new = k − (log k − ψ(k) − s) / (1/k − ψ'(k)), s = log(mean) − mean_of_log.
   Improves ecological movement models (step-length fitting) for sparse states with k < 1.

3. **WeibullDistribution — Newton MLE for k**: Analogous single Newton step. Relevant for
   reliability and predictive-maintenance HMMs.

4. **NegativeBinomialDistribution — profile likelihood for r**: One-dimensional numerical
   optimisation starting from MOM estimate. Relevant for count-data HMMs with high dispersion.

5. **BetaDistribution — digamma Newton for (α, β)**: Standard two-parameter iterative MLE.
   Low priority since Beta is rarely the primary emission in EM-fitted HMMs.

---

## Current Status Matrix (16 distributions)

All 16 distributions meet the Gold Standard v3.0 interface. 41/41 tests pass on all platforms.

| Feature | G | E | Ga | U | Chi | W | Ra | Bi | NB | T | Be | LN | Pa | Po | Di | VM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `getProbability()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `getLogProbability()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `getCumulativeProbability()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fit()` unweighted | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fit()` weighted | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `getBatchLogProbabilities()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `reset()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| JSON I/O | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rule of Five | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Thread-safe cache | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Fit quality tier | A | A | C | C | B | C | A | A | C | C | C | A | A | A | A | A |

Key: G=Gaussian, E=Exponential, Ga=Gamma, U=Uniform, Chi=ChiSquared, W=Weibull, Ra=Rayleigh,
Bi=Binomial, NB=NegativeBinomial, T=StudentT, Be=Beta, LN=LogNormal, Pa=Pareto, Po=Poisson,
Di=Discrete, VM=VonMises.

Fit tiers: **A** = exact weighted MLE; **B** = MOM ≈ MLE; **C** = MOM, gap can be material.

---

## Legend
- ✅ Complete — fully implemented and tested
- ❌ Missing — needs implementation
- ⚠️ Known limitation — see Fit Quality section

---

## Distribution inventory (v3.1)

### Discrete (4)
1. ✅ DiscreteDistribution — Tier A fit
2. ✅ BinomialDistribution — Tier A fit
3. ✅ NegativeBinomialDistribution — ⚠️ Tier C fit (MOM for r)
4. ✅ PoissonDistribution — Tier A fit

### Continuous (12)
5. ✅ GaussianDistribution — Tier A fit
6. ✅ ExponentialDistribution — Tier A fit
7. ✅ GammaDistribution — ⚠️ Tier C fit (MOM for k)
8. ✅ LogNormalDistribution — Tier A fit
9. ✅ BetaDistribution — ⚠️ Tier C fit (MOM for α, β)
10. ✅ UniformDistribution — ⚠️ Tier C fit (MOM is defensible for uniform)
11. ✅ WeibullDistribution — ⚠️ Tier C fit (MOM for k)
12. ✅ ParetoDistribution — Tier A fit
13. ✅ RayleighDistribution — Tier A fit
14. ✅ StudentTDistribution — ⚠️ Tier C fit (ECM is the priority improvement)
15. ✅ ChiSquaredDistribution — Tier B fit
16. ✅ VonMisesDistribution — Tier A fit (Mardia-Jupp ≈ MLE)

---

## Notes and conventions

### Fit guard convention (added May 2026)
All weighted `fit(data, weights)` implementations use early-return on near-zero weight:
```cpp
// Guard: near-zero weight — keep current parameters (not reset).
if (sumW < precision::ZERO || std::isnan(sumW)) return;
```
`reset()` is only called for genuinely degenerate *data* (non-positive values, zero spread, etc.).

### Constants convention
Each distribution `.cpp` file uses `using namespace constants;` inside `namespace libhmm {`.
All numeric literals are replaced with named constants from `libhmm::constants`.

### Performance tiers
- Tier 1 (all 16): concrete non-virtual `getBatchLogProbabilities` loop
- Tier 2 (Gaussian, Exponential): explicit SIMD intrinsics
- Tier 2 upgrades for other distributions deferred until a portable vectorised math library
  (log, lgamma) is available

---

*Last updated: 2026-05-13 (libhmm v3.6.0 current; VonMisesDistribution and EM fixes targeting v3.7.0)*
