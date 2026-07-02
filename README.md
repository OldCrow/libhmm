# libhmm — C++20 Hidden Markov Model Library

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-4.1.3-brightgreen.svg)](https://github.com/OldCrow/libhmm/releases)
[![Tests](https://img.shields.io/badge/Tests-46%2F46_Passing-success.svg)](tests/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX--512%2FAVX2%2FSSE2%2FNEON-blue.svg)](src/distributions/)
[![CI](https://github.com/OldCrow/libhmm/actions/workflows/ci.yml/badge.svg)](https://github.com/OldCrow/libhmm/actions)

Fit Hidden Markov Models with Baum-Welch EM, decode sequences with Viterbi or posterior decoding,
and run exact probabilistic inference at native C++ speed — no external runtime required.

**Zero external dependencies** — C++20 standard library only.

## Use cases

HMMs model systems where discrete latent states drive observable outputs. libhmm fits naturally when you need to:

- **Movement ecology** — classify behavioral states (foraging, travelling, resting) from GPS step lengths and turning angles using Gamma, Rayleigh, or Weibull emissions
- **Financial regime detection** — identify bull/bear/volatile market states from return series with Gaussian or Student-t emissions
- **Sensor analysis** — decode activity states from accelerometer or IMU data in embedded or real-time C++ systems
- **Bioinformatics** — fit discrete-emission HMMs to biological sequences (CpG islands, gene prediction, splice sites)
- **Any C++ application** — embed HMM inference without pulling in an R or Python runtime as a dependency

If you are doing exploratory analysis, R packages (`moveHMM`, `depmixS4`) or Python libraries
(`hmmlearn`, `pomegranate`) may be more ergonomic. libhmm is the right choice when the model
needs to run *inside* a C++ application or pipeline.

## Features

### Training Algorithms

- **Baum-Welch** — canonical log-space EM; works with any `EmissionDistribution` via weighted `fit()` and exposes `getLastLogProbability()` for per-iteration convergence tracking
- **MAP Baum-Welch** — adds symmetric Dirichlet priors on A, π, and discrete emissions; prevents
  degenerate zero-probability transitions on sparse data. `c = 0` recovers standard MLE exactly.
  Use `computeLogPrior()` for the correct convergence criterion (likelihood alone is not monotone
  when `c > 0`).
- **Viterbi Training** — hard-assignment with `TrainingConfig` presets (`fast`, `balanced`, `precise`)
- **Segmental K-Means** — for discrete HMMs; useful as initialiser before EM

### Inference

- **ForwardBackward** — canonical log-space calculator; returns `probability()`, `getLogProbability()`,
  and `decodePosterior()` (per-step argmax-γ decoding, minimising per-step state error rate)
- **Viterbi** — canonical log-space decoder; returns the MAP joint state sequence

Use `decodePosterior()` when per-step annotation accuracy matters (e.g. gene prediction).
Use Viterbi when whole-sequence structural coherence is required (e.g. speech alignment).

Both calculators call `getBatchLogProbabilities()` per state per time step, enabling
SIMD acceleration directly at the distribution layer.

### Model Selection

- `count_free_parameters(hmm)` — free parameters = N*(N-1) transitions + (N-1) initial + sum of emission params
- `compute_aic(logL, k)`, `compute_bic(logL, k, n)`, `compute_aicc(logL, k, n)` — lower is better
- `evaluate_model(hmm, logL, n)` — returns `HmmModelCriteria{aic, bic, aicc}` in one call

```cpp
ForwardBackwardCalculator fbc(hmm, obs);
HmmModelCriteria mc = evaluate_model(hmm, fbc.getLogProbability(), obs.size());
std::cout << "AIC: " << mc.aic << "  BIC: " << mc.bic << "\n";
```

### Probability Distributions (16 scalar + 3 multivariate)

**Discrete:** `DiscreteDistribution`, `BinomialDistribution`, `NegativeBinomialDistribution`, `PoissonDistribution`

**Continuous:** `GaussianDistribution`, `ExponentialDistribution`, `GammaDistribution`, `LogNormalDistribution`,
`BetaDistribution`, `UniformDistribution`, `WeibullDistribution`, `ParetoDistribution`,
`RayleighDistribution`, `StudentTDistribution`, `ChiSquaredDistribution`, `VonMisesDistribution`

**Multivariate** (`Obs = ObservationVectorView = std::span<const double>`):
`DiagonalGaussianDistribution`, `FullCovarianceGaussianDistribution`, `IndependentComponentsDistribution`

All distributions implement `getBatchLogProbabilities()` for SIMD-accelerated batch evaluation.
11 of 16 scalar distributions route through a **runtime-dispatched SIMD kernel table** (tier 2)
selected at startup via CPUID — AVX-512 (8-wide), AVX2 (4-wide), SSE2 (2-wide), NEON (2-wide), scalar.
The 5 remaining distributions (Discrete, Poisson, Binomial, NegativeBinomial, Uniform) use
concrete non-virtual loops that the compiler auto-vectorizes under `-march=native` (tier 1);
explicit SIMD is deferred because `lgamma` per element has no portable vectorized form.

### I/O

- **JSON serialization** (recommended): exact IEEE 754 round-trip, no external dependencies, locale-safe.
  ```cpp
  libhmm::save_json(hmm, "model.json");       // scalar
  auto hmm2 = libhmm::load_json("model.json");

  libhmm::save_json_mv(hmm_mv, "model_mv.json"); // multivariate
  auto hmm_mv2 = libhmm::load_json_mv("model_mv.json");
  ```
- **Legacy XML** (`XMLFileReader` / `XMLFileWriter`): retained for reading existing `.xml` files;
  deprecated in favour of JSON for new code.
- See [`samples/`](samples/) for ready-to-use reference HMM files in both formats.

### Performance

- **Runtime SIMD dispatch for distribution kernels**: ISA tier selected at startup via CPUID;
  the same binary gracefully degrades from AVX-512 to SSE2 or scalar on older CPUs.
  Each ISA tier is compiled into its own TU with a targeted flag (`-mavx2`, `-mavx512f`, etc.)
  rather than `-march=native`, so prebuilt binaries are safe to distribute.
- **`-march=native` for auto-vectorization**: tier-1 distributions and the FB recurrence
  kernel still use compiler auto-vectorization under the native ISA for that build machine.
- **MSVC**: `/arch:AVX512`, `/arch:AVX2`, or `/arch:AVX` (CPU-verified at configure time)
- **Log-space throughout**: no numerical underflow on long sequences
- **Pre-computed log transition matrices**: amortised once per `compute()` call

## Quick Start

### Build

```bash
git clone https://github.com/OldCrow/libhmm.git
cd libhmm
cmake -B build
cmake --build build --config Release
ctest --test-dir build
```

On Windows with Visual Studio:
```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --parallel 4
```

### Basic Usage

```cpp
#include "libhmm/libhmm.h"
using namespace libhmm;

// Create a 2-state HMM with Gaussian emissions
Hmm hmm(2);

Matrix trans(2, 2);
trans(0, 0) = 0.9; trans(0, 1) = 0.1;
trans(1, 0) = 0.2; trans(1, 1) = 0.8;
hmm.setTrans(trans);

Vector pi(2); pi(0) = 0.6; pi(1) = 0.4;
hmm.setPi(pi);

hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
hmm.setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.5));

// Train with Baum-Welch (works with any EmissionDistribution)
ObservationLists obs = { /* your sequences */ };
BaumWelchTrainer trainer(&hmm, obs);
trainer.train();
double train_log_p = trainer.getLastLogProbability(); // total E-step log-probability

// Evaluate
ForwardBackwardCalculator fbc(hmm, obs[0]);
double log_p = fbc.getLogProbability();

// Decode - two strategies:
ViterbiCalculator vc(hmm, obs[0]);
StateSequence map_path      = vc.decode();           // MAP joint path
StateSequence marginal_path = fbc.decodePosterior(); // per-step argmax-gamma
```

### MAP Baum-Welch

```cpp
// c=1 (Laplace smoothing) prevents zero transitions on sparse data.
MapBaumWelchTrainer map_trainer(&hmm, obs, /*pseudo_count=*/1.0);

double prev = -std::numeric_limits<double>::infinity();
for (int i = 0; i < 50; ++i) {
    map_trainer.train();
    // Correct MAP convergence criterion: logL + log P(lambda|c)
    double logL   = ForwardBackwardCalculator(hmm, obs[0]).getLogProbability();
    double mapObj = logL + map_trainer.computeLogPrior();
    if (mapObj - prev < 1e-6) break;
    prev = mapObj;
}
```

### ViterbiTrainer Presets

```cpp
// Fast convergence for interactive use
ViterbiTrainer fast_trainer(&hmm, obs, training_presets::fast());
fast_trainer.train();

// Precise convergence for final models
ViterbiTrainer precise_trainer(&hmm, obs, training_presets::precise());
precise_trainer.train();
std::cout << "Converged: " << precise_trainer.hasConverged() << "\n";
```

## Project Structure

```
libhmm/
├── include/libhmm/    # Public headers (layered architecture)
│   ├── platform/      # Layer 0: SIMD CPU detection (compile-time + runtime CPUID)
│   ├── math/          # Layer 1: constants, log-space, numerics
│   ├── linalg/        # Layer 2: Matrix, Vector, ObservationMatrix types
│   ├── distributions/ # Layer 3: 16 scalar + 3 multivariate distributions
│   ├── performance/   # Runtime dispatch table (simd_double_ops.h) + FB recurrence policy
│   ├── basic_hmm.h    # BasicHmm<Obs> template; Hmm and HmmMV aliases
│   ├── calculators/   # Layer 4: ForwardBackward, Viterbi (scalar + MV)
│   ├── training/      # Layer 4: BaumWelch, MapBaumWelch, Viterbi, kmeans_init
│   └── io/            # JSON (hmm_json.h, scalar + MV) + legacy XML I/O
├── src/
│   ├── distributions/ # Distribution implementations
│   ├── performance/   # simd_double_ops_{scalar,sse2,avx2,avx512,neon}.cpp + simd_dispatch.cpp
│   └── platform/      # cpu_detection.cpp (runtime CPUID)
├── tests/             # 46-test GTest suite
├── examples/          # 20 usage demonstrations
├── tools/             # simd_inspection, batch_performance, hmm_validator (.json/.xml)
├── samples/           # Reference HMM files (two_state_gaussian, casino) in JSON and XML
├── benchmarks/        # Comparative benchmarks (requires external libraries)
├── docs/              # Documentation and checklists
└── CMakeLists.txt
```

## Examples

See [examples/](examples/) for demonstrations:

| Example | Distribution(s) | Trainer |
|---|---|---|
| `basic_hmm_example` | Discrete, Gaussian, Poisson | Viterbi + JSON I/O |
| `baum_welch_example` | Gaussian | BaumWelch (with EM convergence table) |
| `viterbi_trainer_example` | Gaussian | Viterbi (preset comparison) |
| `student_t_hmm_example` | StudentT | BaumWelch |
| `poisson_hmm_example` | Poisson | Viterbi |
| `financial_hmm_example` | Beta, LogNormal | Viterbi |
| `reliability_hmm_example` | Weibull, Exponential | Viterbi |
| `quality_control_hmm_example` | Binomial, Uniform | Viterbi |
| `economics_hmm_example` | NegBinomial, Pareto | Viterbi |
| `queuing_theory_hmm_example` | Poisson, Exponential, Gamma | Viterbi |
| `statistical_process_control_hmm_example` | ChiSquared | Viterbi |
| `swarm_coordination_example` | Discrete (243 symbols) | — |
| `posterior_decoding_example` | Discrete | ForwardBackward (`decodePosterior` vs Viterbi) |
| `map_baum_welch_example` | Discrete | MAP Baum-Welch (c=0 vs c=1, MAP convergence table) |
| `elk_movement_example` | Gamma, VonMises | BaumWelch (step length + turning angle) |
| `dax_regime_example` | StudentT | BaumWelch (DAX log-returns 2000–2022) |
| `mv_gaussian_example` | DiagonalGaussian (MV) | BaumWelch (2D synthetic) |
| `elk_mv_example` | IndependentComponents (MV) | BaumWelch (GPS tracks vs moveHMM R) |
| `mv_regime_example` | DiagonalGaussian + FullCovGaussian (MV) | BaumWelch (SPY+QQQ vs hmmlearn) |

## Requirements

- **C++20** compiler: GCC 12+, Apple Clang 14+ (macOS 13+), Clang 14+, MSVC 2022 17.x
- **CMake 3.20+**

No external dependencies. GTest is fetched automatically via CMake `FetchContent` for the test suite.

## Documentation

- [MIGRATION.md](MIGRATION.md) — v3→v4 upgrade guide
- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution guidelines and toolchain policy
- [docs/CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md) — build options, library output, CI matrix
- [docs/GOLD_STANDARD_CHECKLIST.md](docs/GOLD_STANDARD_CHECKLIST.md) — distribution implementation requirements
- [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) — coding conventions

## License

MIT License — see [LICENSE](LICENSE) for details.
