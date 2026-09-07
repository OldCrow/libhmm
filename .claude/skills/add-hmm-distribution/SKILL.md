---
name: add-hmm-distribution
description: Use when adding a new emission distribution to libhmm (a new BasicEmissionDistribution subclass under src/distributions/) — walks the full checklist from templating an existing distribution through registering it in CMake, JSON I/O, and the test suite.
---

# Adding a new distribution to libhmm

Use an existing distribution (e.g. `src/distributions/rayleigh_distribution.cpp` for single-parameter, `src/distributions/gamma_distribution.cpp` for two-parameter) as a template. Required checklist (`docs/GOLD_STANDARD_CHECKLIST.md`):

1. Concrete non-virtual `getBatchLogProbabilities` override (tier 1 minimum)
2. Weighted `fit(data, weights)` with near-zero weight guard
3. `reset()`, `clone()`, `sample()`, `to_json()` / `from_json()` (registered in `src/io/hmm_json.cpp`), `getNumParameters()`
4. `std::atomic<bool> cacheValid_` thread-safe cache
5. Add source to `LIBHMM_SOURCES` in `CMakeLists.txt`. Add it to `LIBHMM_SIMD_SOURCES` in `cmake/SimdDispatch.cmake` only if it is tier-1 (compiler auto-vectorization); a distribution that dispatches through `DoubleVecOps` does not belong there
6. Add a test file under `tests/distributions/` and register it in `tests/CMakeLists.txt`
