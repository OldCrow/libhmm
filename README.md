# libhmm — Modern C++20 Hidden Markov Model Library

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-3.1.0-brightgreen.svg)](https://github.com/OldCrow/libhmm/releases)
[![Tests](https://img.shields.io/badge/Tests-36%2F36_Passing-success.svg)](tests/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX--512%2FAVX2%2FSSE2%2FNEON-blue.svg)](src/distributions/)
[![CI](https://github.com/OldCrow/libhmm/actions/workflows/ci.yml/badge.svg)](https://github.com/OldCrow/libhmm/actions)

A modern, high-performance C++20 implementation of Hidden Markov Models with 15 emission distributions,
canonical log-space algorithms, and compile-time SIMD acceleration.

**Zero external dependencies** — C++20 standard library only.

## Features

### Training Algorithms

- **Baum-Welch** — canonical log-space EM; works with any `EmissionDistribution` via weighted `fit()`
- **Viterbi Training** — hard-assignment with `TrainingConfig` presets (`fast`, `balanced`, `precise`)
- **Segmental K-Means** — for discrete HMMs; useful as initialiser before EM

### Inference

- **ForwardBackward** — canonical log-space calculator; returns `probability()` and `getLogProbability()`
- **Viterbi** — canonical log-space decoder; returns the MAP state sequence

Both calculators call `getBatchLogProbabilities()` per state per time step, enabling
SIMD acceleration directly at the distribution layer.

### Probability Distributions (15)

**Discrete:** `DiscreteDistribution`, `BinomialDistribution`, `NegativeBinomialDistribution`, `PoissonDistribution`

**Continuous:** `GaussianDistribution`, `ExponentialDistribution`, `GammaDistribution`, `LogNormalDistribution`,
`BetaDistribution`, `UniformDistribution`, `WeibullDistribution`, `ParetoDistribution`,
`RayleighDistribution`, `StudentTDistribution`, `ChiSquaredDistribution`

All distributions implement `getBatchLogProbabilities()` for SIMD-accelerated batch evaluation.
`GaussianDistribution` and `ExponentialDistribution` have explicit AVX-512/AVX2/SSE2/NEON intrinsics (tier 2);
the remaining 13 use concrete non-virtual loops that the compiler auto-vectorizes under `-march=native`.

### Performance

- **Compile-time SIMD dispatch**: each machine builds for its own CPU
  - GCC/Clang: `-march=native` (AVX-512 on capable x86, NEON on AArch64)
  - MSVC: `/arch:AVX512`, `/arch:AVX2`, or `/arch:AVX` (CPU-verified at configure time)
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

// Evaluate
ForwardBackwardCalculator fbc(hmm, obs[0]);
double log_p = fbc.getLogProbability();

// Decode
ViterbiCalculator vc(hmm, obs[0]);
StateSequence path = vc.decode();
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
│   ├── platform/      # Layer 0: SIMD detection
│   ├── math/          # Layer 1: constants, log-space, numerics
│   ├── linalg/        # Layer 2: Matrix, Vector types
│   ├── distributions/ # Layer 3: 15 distributions + base
│   ├── hmm.h          # Core HMM class
│   ├── calculators/   # Layer 4: ForwardBackward, Viterbi
│   └── training/      # Layer 4: BaumWelch, Viterbi, SegmentalKMeans
├── src/               # Implementation (mirrors include/)
├── tests/             # 36-test GTest suite (7 architectural levels)
├── examples/          # 12 usage demonstrations
├── tools/             # simd_inspection, batch_performance, hmm_validator
├── benchmarks/        # Comparative benchmarks (requires external libraries)
├── docs/              # Documentation and checklists
└── CMakeLists.txt
```

## Examples

See [examples/](examples/) for demonstrations:

| Example | Distribution(s) | Trainer |
|---|---|---|
| `basic_hmm_example` | Discrete, Gaussian | Viterbi |
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

## Requirements

- **C++20** compiler: GCC 11+, Clang 14+, MSVC 2019 16.11+
- **CMake 3.20+**

No external dependencies. GTest is fetched automatically via CMake `FetchContent` for the test suite.

## Documentation

- [WARP.md](WARP.md) — session guide for Warp AI agent
- [docs/CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md) — build options, library output, CI matrix
- [docs/GOLD_STANDARD_CHECKLIST.md](docs/GOLD_STANDARD_CHECKLIST.md) — distribution implementation requirements
- [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) — coding conventions

## License

MIT License — see [LICENSE](LICENSE) for details.
