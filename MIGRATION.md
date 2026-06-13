# libhmm v3 → v4 Migration Guide

v4.0.0 introduces multivariate emission distributions and a templated type
system.  **Most v3.x user code recompiles unchanged** — the scalar API is
preserved through type aliases.  Read the relevant section below for your
situation.

---

## Platform requirements raised

| Compiler | v3 minimum | v4 minimum |
|---|---|---|
| Apple Clang (macOS) | Any (with Catalina workarounds) | **14** (Xcode 14, macOS 13+) |
| GCC (Linux) | 11 | **12** |
| Clang (Linux) | 14 | 14 (unchanged) |
| MSVC (Windows) | 2019 16.11 | **2022 17.x** (`/std:c++20`) |

macOS 12 (Monterey) and earlier are no longer supported.
The `LIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX` CMake option is removed.

If your project must target older compilers, you are welcome to fork libhmm
and reintroduce the compatibility machinery.  The last release that supported
older toolchains is **v3.8.0**, which remains available on the `main` branch
history and as a tagged release.  Upstream will not accept pull requests that
restore support for compilers below the v4 minimums listed above.

---

## Normal scalar users — no changes needed

If you use `Hmm`, `EmissionDistribution`, `ForwardBackwardCalculator`,
`ViterbiCalculator`, `BaumWelchTrainer`, `MapBaumWelchTrainer`,
`ViterbiTrainer`, and the 16 existing scalar distributions, your code
recompiles under v4 unchanged:

```cpp
// v3 code — compiles unchanged under v4
#include "libhmm/libhmm.h"
using namespace libhmm;

Hmm hmm(2);
hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
BaumWelchTrainer trainer(hmm, obs);
trainer.train();
ForwardBackwardCalculator fbc(hmm, obs[0]);
```

All v3 names are now type aliases for the underlying templates:

```
Hmm                    = BasicHmm<double>
EmissionDistribution   = BasicEmissionDistribution<double>
ForwardBackwardCalculator = BasicForwardBackwardCalculator<double>
ViterbiCalculator      = BasicViterbiCalculator<double>
BaumWelchTrainer       = BasicBaumWelchTrainer<double>
MapBaumWelchTrainer    = BasicMapBaumWelchTrainer<double>
ViterbiTrainer         = BasicViterbiTrainer<double>
Trainer                = BasicTrainer<double>
Calculator             = BasicCalculator<double>
```

Existing scalar JSON files (schema without `obs_type`) load unchanged.

---

## Custom distributions — one declaration change required

If you have a custom distribution derived from `DistributionBase`, you must
add the CRTP parameter:

```cpp
// v3 (no longer compiles)
class MyDistribution : public DistributionBase { ... };

// v4 — add your class name as the template argument
class MyDistribution : public DistributionBase<MyDistribution> { ... };
```

If your distribution overrides `clone()` manually, remove the override —
`DistributionBase<Derived>` now provides a CRTP implementation automatically
via `std::make_unique<Derived>(static_cast<const Derived&>(*this))`.
Your distribution must remain copy-constructible (all concrete distributions
already are).

The virtual interface is otherwise unchanged: `getProbability`,
`getLogProbability`, `getBatchLogProbabilities`, `fit`, `reset`, `sample`,
`to_json`, `isDiscrete`, `getNumParameters`.

One new non-pure virtual is added:

```cpp
// Returns 1 by default; override for multivariate distributions.
virtual std::size_t getDimension() const noexcept { return 1; }
```

Scalar distributions get this for free.

---

## Subclasses of `Hmm` — must update

`Hmm` is now an alias for `BasicHmm<double>`, not a concrete class.
Subclasses of `Hmm` require one of:

```cpp
// Option A: derive from the scalar alias (most common case)
class MyHmm : public BasicHmm<double> { ... };

// Option B: switch to composition
class MyHmm {
    Hmm hmm_;   // embed rather than extend
};
```

If you only override helper methods (e.g. a custom `validate()`), prefer
option A. If you were using inheritance for code sharing without genuine
IS-A semantics, prefer option B.

---

## JSON IO — backward compatible; new MV schema for v4 models

Existing scalar JSON files are read unchanged by `load_json()`.
New multivariate HMM files use a distinct schema detected by `obs_type`:

```json
{
  "libhmm_version": "4",
  "obs_type": "multivariate",
  "dimensions": 2,
  "states": 3,
  "pi": [ ... ],
  "trans": [ ... ],
  "distributions": [ ... ]
}
```

Use `save_json_mv` / `load_json_mv` for multivariate models.
Using `load_json` on a multivariate file throws `std::runtime_error` because
the schema lacks `libhmm_version` / `obs_type`.

XML IO remains scalar-only and is deprecated. Read existing `.xml` files with
`XMLFileReader`; prefer JSON for all new code.

---

## `count_free_parameters` — now a template

The function is now a template on `BasicHmm<Obs>` and works for both
`Hmm` and `HmmMV`. Existing calls to `count_free_parameters(hmm)` where
`hmm` is a `const Hmm&` continue to compile unchanged.

---

## New features in v4

### Multivariate emission distributions

Three new distributions for `HmmMV = BasicHmm<ObservationVectorView>`:

| Class | Parameters | Use case |
|---|---|---|
| `IndependentComponentsDistribution` | D independent scalar emissions | Mixed-family multivariate |
| `DiagonalGaussianDistribution` | D means + D variances | Uncorrelated Gaussian features |
| `FullCovarianceGaussianDistribution` | D means + D×D covariance | Correlated Gaussian features |

### k-means initialisation for MV HMMs

```cpp
std::mt19937_64 rng(42);
kmeans_init(hmm_mv, training_data, rng);   // from libhmm/training/kmeans_init.h
```

Runs k-means++ seeding + Lloyd's algorithm, then calls `fit()` on each state's
emission distribution from its assigned cluster. Always call this before
`BaumWelchTrainer<ObservationVectorView>` on a fresh model.

### MV JSON IO

```cpp
save_json_mv(hmm_mv, "model_mv.json");
HmmMV loaded = load_json_mv("model_mv.json");
```

### Model selection for MV

`count_free_parameters(hmm_mv)` works for `HmmMV` using the same
formula: N*(N-1) + (N-1) + Σ dist_i.getNumParameters().
