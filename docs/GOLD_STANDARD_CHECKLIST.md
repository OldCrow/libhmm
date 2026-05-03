# LibHMM Gold Standard Distribution Implementation Checklist

This document tracks the implementation status of all probability distributions in libhmm according to our "Gold Standard" requirements, based on the Gaussian and Exponential distributions as reference implementations.

## Gold Standard Requirements (v3.0)

### v3.0 Interface Requirements (EmissionDistribution — added Phase 2/4.5)

All distributions must implement the `EmissionDistribution` abstract interface. New in v3.0:

- ✅ **`getBatchLogProbabilities(std::span<const double> obs, std::span<double> out) const override`**
  - **Tier 1 (all 15)**: Concrete non-virtual loop using cached parameters directly.
    Virtual dispatch in the `DistributionBase` default defeats compiler auto-vectorization.
    The override removes that overhead and allows `-march=native` / `/arch:AVX2` to vectorize.
  - **Tier 2 (Gaussian, Exponential)**: Explicit SIMD intrinsics via `detail::` free function
    taking only plain data pointers. AVX-512 (8-wide), AVX2 (4-wide), SSE2 (2-wide), NEON (2-wide),
    scalar tail. NaN inputs yield `-Inf` to match scalar behaviour.
  - **Tier 2 upgrade paths for other distributions** are documented inline in each `.cpp` file.
    Distributions with `log(x)` (LogNormal, Rayleigh, Pareto, Weibull, Gamma) require vectorised log
    (Intel SVML / GNU libmvec / Apple Accelerate vvlog). Distributions with `lgamma` (Beta, ChiSquared,
    StudentT, Poisson, Binomial, NegBinomial) require vectorised lgamma. Defer until a portable
    math-library dependency is chosen.

- ✅ **`fit(std::span<const double> data, std::span<const double> weights) override`**
  - Weighted MLE for Baum-Welch M-step. Weights are unnormalised γ values; each distribution
    normalises by `sum(weights)` internally. Falls back to `reset()` if `sum(weights) ≈ 0`.

- ✅ **`std::span<const double>` parameter types** — replaces `std::vector<Observation>&`.
  No copies in the hot path.

- ✅ **Thread-safe cache** — `std::atomic<bool> cacheValid_` in `DistributionBase`
  with `acquire`/`release` memory ordering. Required because the calculator thread pool
  can trigger concurrent const reads on the same distribution.

- ✅ **C++20 standard** — `[[nodiscard]]`, `noexcept`, `std::span`, `#pragma once`.

### Implementation Requirements
- ✅ **Core Methods:**
  - `getProbability()` — PDF/PMF (const)
  - `getLogProbability()` — log PDF/PMF with numerical stability (const, noexcept)
  - `getCumulativeProbability()` — CDF (where mathematically meaningful)
  - `fit(std::span<const double>)` — unweighted MLE (Welford's algorithm)
  - `fit(std::span<const double>, std::span<const double>)` — weighted MLE (Baum-Welch)
  - `getBatchLogProbabilities(span, span)` — concrete non-virtual batch loop (tier 1 minimum)
  - `reset()` — reset to default parameters
  - `toString()` — human-readable string representation

- ✅ **Rule of Five:**
  - Copy Constructor
  - Move Constructor
  - Copy Assignment Operator
  - Move Assignment Operator
  - Destructor (virtual, defaulted)

- ✅ **Caching System:**
  - Comprehensive caching of expensive calculations
  - Cache validation flags
  - Automatic cache invalidation on parameter changes

- ✅ **Input Validation:**
  - Robust parameter validation with appropriate exceptions
  - NaN/infinity handling
  - Data validation in fitting methods

- ✅ **Constants Usage:**
  - All numeric literals replaced with constants from `libhmm::constants`
  - No hardcoded magic numbers

- ✅ **I/O  Operators:**
  - `operator==` - Equality comparison with tolerance
  - `operator<<` - Stream output
  - `operator>>` - Stream input (recommended)

### Test Requirements
- ✅ **Core Tests:**
  - Basic Functionality
  - Probability Calculations
  - Parameter Fitting
  - Parameter Validation
  - Copy/Move Semantics
  - Invalid Input Handling
  - Reset Functionality

- ✅ **Advanced Tests:**
  - Log Probability calculations
  - String Representation
  - Fitting Validation
  - Performance characteristics (recommended)
  - Mathematical Correctness (recommended)
  - Numerical Stability (recommended)

- ✅ **Gold Standard Tests:**
  - CDF calculations (where applicable)
  - Equality/I-O operators
  - Caching mechanism verification

---

## Current Status Matrix

### Feature Implementation Status

|| Feature | Gaussian | Exponential | Gamma | Uniform | Chi-Squared | Weibull | Rayleigh | Binomial | Negative-Binomial | Student-t | Beta | Log-Normal | Pareto | Poisson | Discrete |
||----------|------------|---------------|----------|-----------|----------------|----------|-----------|------------|----------------------|-------------|------|---------------|----------|----------|-----------|
|| **Core Methods** |
|| `getProbability()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| `getLogProbability()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| `getCumulativeDensity()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| `fit()` (Welford) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| `reset()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| `toString()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| **Rule of Five** |
|| Copy Constructor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Move Constructor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Copy Assignment | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Move Assignment | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Destructor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| **Caching** |
|| Comprehensive Cache | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Cache Validation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Auto-invalidation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| **I/O  Operators** |
|| `operator==` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| `operator<<` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| `operator>>` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| **Constants** |
|| Uses `libhmm::constants` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Test Coverage Status

|| Test Category | Gaussian | Exponential | Gamma | Uniform | Chi-Squared | Weibull | Rayleigh | Binomial | Negative-Binomial | Student-t | Beta | Log-Normal | Pareto | Poisson | Discrete |
||---------------|----------|-------------|-------|---------|-------------|---------|----------|----------|-------------------|-----------|------|------------|--------|---------|----------|
|| **Core Tests** |
|| Basic Functionality | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Probability Calculations | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Parameter Fitting | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Parameter Validation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Copy/Move Semantics | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Invalid Input Handling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Reset Functionality | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| **Advanced Tests** |
|| Log Probability | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| String Representation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Performance Tests | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Mathematical Correctness | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Fitting Validation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Numerical Stability | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| **Gold Standard Tests** |
|| CDF Tests | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Equality/I-O Tests | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|| Caching Tests | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Legend
- ✅ **Complete**: Fully implemented and tested
- ❌ **Missing**: Needs to be implemented/added
- ❓ **Unknown**: Needs assessment
- 🔄 **In Progress**: Currently being worked on

---

## Current Status (v3.0)

All 15 distributions fully meet the Gold Standard checklist. The v3.0 interface additions
(`getBatchLogProbabilities`, weighted `fit()`, `std::span` params, atomic cache) are also
complete for all 15. No outstanding action items.

---

## Planned Update Order
1. ✅ **Gaussian** - Reference implementation (constants applied, comprehensive tests verified)
2. ✅ **Exponential** - Reference implementation (constants applied, comprehensive tests verified)
3. ✅ **Gamma** - Updated (constants applied, comprehensive tests verified)
4. ✅ **Uniform** - Updated (constants applied, comprehensive tests verified, performance tests added)
5. ✅ **Chi-squared** - Updated to Gold standard (constants applied, comprehensive tests verified)
6. ✅ **Weibull** - Updated to Gold standard (all features implemented, comprehensive tests verified)
7. ✅ **Rayleigh** - Updated to Gold standard (all features implemented, comprehensive tests verified)
8. ✅ **Pareto** - Updated to Gold standard (all features implemented, comprehensive tests verified)
8. ✅ **Binomial** - Updated to Gold standard (all features implemented, comprehensive tests verified)
9. ✅ **Negative Binomial** - Updated to Gold standard (all features implemented, comprehensive tests verified)
10. ✅ **Student-t** - Updated to Gold standard (all features implemented, comprehensive tests verified)
11. ✅ **Beta** - Updated to Gold standard (all features implemented, comprehensive tests verified)
12. ✅ **Log-Normal** - Updated to Gold standard (all features implemented, comprehensive tests verified)
13. ✅ **Pareto** - Updated to Gold standard (all features implemented, comprehensive tests verified)
14. ✅ **Poisson** - Updated to Gold standard (all features implemented, comprehensive tests verified)
15. ✅ **Discrete** - Updated to Gold standard (all features implemented, comprehensive tests verified)

---

## Notes  Conventions

### C++20 Features to Use
- `[[nodiscard]]` for getter methods
- `noexcept` specifications where appropriate
- Default member initializers
- Structured bindings where helpful
- `constexpr` for compile-time constants
- `std::span<const double>` for read-only data parameters (no copies)
- `#pragma once` (replaces `#ifndef` guards)

### Performance Considerations
- Cache expensive calculations (log values, normalization constants)
- `std::atomic<bool> cacheValid_` (thread-safe, not `mutable bool`)
- Use Welford’s algorithm for numerical stability in fitting
- Avoid repeated computations in hot paths
- Implement `getBatchLogProbabilities()` as a concrete non-virtual loop (tier 1)
  to enable compiler auto-vectorization under `-march=native` / `/arch:AVX2`

### Testing Conventions
- Each test function should be self-contained
- Use descriptive test names
- Include edge cases and boundary conditions
- Test both success and failure paths
- Verify numerical accuracy with known values

### Variable Naming Conventions
- Standardize common meaningful internal variable names, such as using "token" for discardable tokens in the  operator implementation

---

*Last Updated: 2026-04-22 (v3.0.0-alpha — all 15 distributions complete)*
