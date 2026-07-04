# LibHMM Gold Standard Distribution Implementation Checklist

This document tracks the implementation status of all probability distributions in libhmm
according to the "Gold Standard" requirements, plus a fit quality survey added in May 2026.

---

## Gold Standard Requirements (v4.0)

### v4.0 Interface Requirements (BasicEmissionDistribution<Obs>)

All distributions must implement the `EmissionDistribution` abstract interface. New in v3.0:

- ✅ **`getBatchLogProbabilities(std::span<const double> obs, std::span<double> out) const override`**
  - **Tier 1 (5 distributions)**: Concrete non-virtual loop using cached parameters. The
    override removes virtual-dispatch overhead so `-march=native` can auto-vectorize the loop.
    Applies to: Discrete, Poisson, Binomial, NegativeBinomial, Uniform.
    SIMD is deferred for these: `lgamma` per element (Poisson, Binomial, NegBinomial) has no
    portable vectorized form; Discrete requires integer gather; Uniform is already ~2 instructions
    per element with no arithmetic to vectorize.
  - **Tier 2 (11 distributions)**: Routes through the `DoubleVecOps` runtime dispatch table
    (`include/libhmm/performance/simd_double_ops.h`), which selects the best ISA at startup via
    CPUID — AVX-512 (8-wide), AVX2 (4-wide), SSE2 (2-wide), NEON (2-wide), scalar. Each ISA
    tier is a separate TU compiled with a targeted flag rather than `-march=native`, so prebuilt
    binaries degrade gracefully on older CPUs rather than SIGILL.
    Primitives available: `log`, `exp`, `cos`, `log1p` (SLEEF-based, < 1 ULP for log/exp).
    Applies to: Gaussian, Exponential, LogNormal, Gamma, ChiSquared, Rayleigh, Pareto, Weibull,
    Beta, StudentT, VonMises.

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

### Tier B

No distributions remain in Tier B as of v4.2.1.

### Tier C — MOM: gap can be material

One distribution remains Tier C.

| Distribution | M-step estimate | Notes |
||---|---|---|
|| UniformDistribution | a = mean - sqrt(3*var), b = mean + sqrt(3*var) | MOM. MLE (weighted min/max) is numerically unstable under EM weights; MOM is defensible. |

### Previously Tier B/C — promoted to Tier A (all resolved by v4.2.1)

| Distribution | Algorithm | Since |
||---|---|---|
|| ChiSquaredDistribution | Newton on psi(k/2) = mean_log_x - log(2) | v4.2.1 |
|| GammaDistribution | Cheng & Feast (1979) starter + Newton on log(k) - psi(k) = s | pre-v4.2 |
|| WeibullDistribution | MOM seed + Newton on profile score for k (100 iters) | pre-v4.2 |
|| NegativeBinomialDistribution | MOM seed + digamma/trigamma Newton for r (200 iters) | pre-v4.2 |
|| BetaDistribution | MOM seed + digamma Newton for (alpha, beta) (200 iters) | pre-v4.2 |
|| StudentTDistribution | Full ECM: closed-form mu, sigma; Newton for nu (Liu & Rubin 1994) | pre-v4.2 |

### Outstanding M-step work

None. All 15 non-Uniform scalar distributions are Tier A.

---

## Current Status Matrix (16 distributions)

All 16 scalar distributions meet the Gold Standard v4.1 interface. 47/47 tests pass on all platforms.
15 of 16 are Tier A (exact weighted MLE). UniformDistribution is the sole Tier C (MOM defensible in EM context).
The 3 MV distributions (DiagonalGaussian, FullCovGaussian, IndependentComponents) implement
`BasicEmissionDistribution<ObservationVectorView>` and are covered by `test_multivariate_distributions`.

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
| Fit quality tier | A | A | A | C | A | A | A | A | A | A | A | A | A | A | A | A |

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

## Distribution inventory (v4.0)

### Discrete (4)
1. ✅ DiscreteDistribution — Tier A fit
2. ✅ BinomialDistribution — Tier A fit
3. ✅ NegativeBinomialDistribution — Tier A fit (Newton MLE for r)
4. ✅ PoissonDistribution — Tier A fit

### Continuous (12)

5. ✅ GaussianDistribution — Tier A fit
6. ✅ ExponentialDistribution — Tier A fit
7. ✅ GammaDistribution — Tier A fit (Newton MLE for k)
8. ✅ LogNormalDistribution — Tier A fit
9. ✅ BetaDistribution — Tier A fit (digamma Newton for α, β)
10. ✅ UniformDistribution — Tier C fit (MOM defensible; MLE min/max unstable in EM)
11. ✅ WeibullDistribution — Tier A fit (Newton MLE for k)
12. ✅ ParetoDistribution — Tier A fit
13. ✅ RayleighDistribution — Tier A fit
14. ✅ StudentTDistribution — Tier A fit (ECM, Liu & Rubin 1994)
15. ✅ ChiSquaredDistribution — Tier A fit (Newton MLE, v4.2.1)
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
- **Tier 1 (5)**: Discrete, Poisson, Binomial, NegativeBinomial, Uniform — concrete non-virtual
  loop; compiler auto-vectorizes under `-march=native`. SIMD permanently deferred: `lgamma` has
  no portable vectorized form; Discrete needs integer gather; Uniform has trivial per-element work.
- **Tier 2 (11)**: Gaussian, Exponential, LogNormal, Gamma, ChiSquared, Rayleigh, Pareto,
  Weibull, Beta, StudentT, VonMises — runtime-dispatched via `DoubleVecOps` CPUID table.
  Primitives: `log`/`exp`/`cos`/`log1p` from `detail/simd_math_helpers.h`
  (SLEEF-based, < 1 ULP for log/exp; polynomial small-input path for `log1p`).

---

*Last updated: 2026-07-04 (libhmm v4.2.1; 11/16 scalar distributions tier-2 SIMD-dispatched; 15/16 Tier A fit quality)*
