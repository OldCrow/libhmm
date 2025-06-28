# libhmm - Modern C++17 Hidden Markov Model Library

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.15%2B-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.7.1-brightgreen.svg)](https://github.com/OldCrow/libhmm/releases)
[![Tests](https://img.shields.io/badge/Tests-31/31_Passing-success.svg)](tests/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX%2FSSE2%2FNEON-blue.svg)](src/performance/)
[![Threading](https://img.shields.io/badge/Threading-C%2B%2B17-orange.svg)](src/performance/thread_pool.cpp)

A modern, high-performance C++17 implementation of Hidden Markov Models with advanced statistical distributions, SIMD optimization, and parallel processing capabilities.

**🚀 Latest Release v2.7.1**: Discrete distributions gold standard upgrade and HTK benchmarking enhancement release. Brings all 4 discrete distributions (Discrete, Binomial, Negative Binomial, Poisson) to gold standard with comprehensive exception handling, robust boundary value management, and consistent variable naming. Enhances benchmarking suite with clean discrete/continuous HTK separation and performance scaling analysis. Maintains 100% test pass rate while significantly improving discrete distribution reliability and numerical robustness.

## Major Achievements

✅ **Complete Boost dependency elimination** - Replaced all Boost.uBLAS and Boost.Serialization dependencies with custom C++17 implementations  
✅ **Custom Matrix/Vector implementations** - SIMD-friendly contiguous memory layout with full API compatibility  
✅ **Comprehensive 5-library benchmarking suite** - 100% numerical agreement across HMMLib, GHMM, StochHMM, HTK, and libhmm  
✅ **Zero external dependencies** - C++17 standard library only, simplified build and deployment  
✅ **Performance baseline established** - Validated numerical accuracy with comprehensive performance characterization across HMM libraries  

## Features

### 🎯 **Training Algorithms**
- **Viterbi Training** - Segmented k-means with clustering
- **Segmented K-Means** - Alternative k-means implementation
- **Baum-Welch** - Standard expectation-maximization
- **Scaled Baum-Welch** - Numerically stable implementation

### 📊 **Probability Distributions** 
**Discrete Distributions:**
- Discrete Distribution (categorical)
- Binomial Distribution (success/failure trials)
- Negative Binomial Distribution (success probability modeling)
- Poisson Distribution (count data and rare events)

**Continuous Distributions:**
- Gaussian (Normal) Distribution (symmetric, bell-curve)
- Beta Distribution (probabilities and proportions on [0,1])
- Gamma Distribution (positive continuous variables)
- Exponential Distribution (waiting times and reliability)
- Log-Normal Distribution (multiplicative processes)
- Pareto Distribution (power-law phenomena)
- Uniform Distribution (continuous uniform random variables)
- Weibull Distribution (reliability analysis and survival modeling)
- **Student's t-Distribution** (robust modeling with heavy tails)
- **Chi-squared Distribution** (goodness-of-fit and categorical analysis)

**Quality Standards:** All distributions are being progressively updated to meet the [Gold Standard Checklist](docs/GOLD_STANDARD_CHECKLIST.md) for consistency, robustness, and maintainability.

### 🧮 **Calculators**
**Forward-Backward Algorithms:**
- **Standard Forward-Backward** - Classic algorithm for probability computation
- **Scaled SIMD Forward-Backward** - Numerically stable with SIMD optimization and automatic CPU fallback
- **Log SIMD Forward-Backward** - Log-space computation with SIMD optimization and automatic CPU fallback

**Viterbi Algorithms:**
- **Standard Viterbi** - Most likely state sequence decoding
- **Scaled SIMD Viterbi** - Numerically stable Viterbi with SIMD optimization and automatic CPU fallback
- **Log SIMD Viterbi** - Log-space Viterbi with SIMD optimization and automatic CPU fallback

**Automatic Calculator Selection:**
- **AutoCalculator** - Intelligent algorithm selection based on problem characteristics
- **Performance Prediction** - CPU feature detection and optimal calculator selection
- **Traits-Based Selection** - Automatic fallback from SIMD to scalar implementations

### ⚡ **Performance Optimizations**
- **SIMD Support**: AVX, SSE2, and ARM NEON vectorization
- **Thread Pool**: Modern C++17 work-stealing thread pool
- **Automatic Optimization**: CPU feature detection and algorithm selection
- **Memory Efficiency**: Aligned allocators and memory pools
- **Cache Optimization**: Blocked algorithms for large matrices

### 💾 **I/O Support**
- XML file reading/writing
- Extensible file I/O manager
- Model serialization

### 🧪 **Testing Infrastructure**
- **Distribution Tests**: 14 comprehensive distribution tests in `tests/distributions/`
- **Calculator Tests**: 10 SIMD calculator and performance tests in `tests/calculators/`
- **Integration Tests**: Core HMM functionality testing
- **100% Test Coverage**: All distributions with complete functionality testing
- **CMake/CTest Integration**: Automated testing framework
- **Continuous Validation**: Parameter fitting, edge cases, and error handling
- **Code Quality Enforcement**: clang-tidy integration for automated style guide compliance

### 📈 **Benchmarking Suite**
- **Multi-Library Validation**: Integration with HMMLib, GHMM, StochHMM, HTK
- **Numerical Accuracy**: 100% agreement at machine precision across libraries
- **Performance Baseline**: Comprehensive performance characterization
- **Compatibility Documentation**: Complete integration guides and fixes

## Quick Start

### Building with CMake

```bash
# Clone and build
git clone <repository-url>
cd libhmm
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
ctest

# Install
sudo make install
```

**Cross-Platform Support**: For detailed cross-platform building instructions including macOS, Linux, and configuration options, see [Cross-Platform Build Guide](docs/CROSS_PLATFORM.md).

### Building with Make (Legacy)

```bash
make
```

### Basic Usage

```cpp
#include <libhmm/libhmm.h>

// Create HMM with 2 states and discrete observations
auto hmm = std::make_unique<Hmm>(2, 6); // 2 states, 6 observation symbols

// Set up training data
ObservationLists trainingData = { /* your observation sequences */ };

// Train using Baum-Welch
BaumWelchTrainer trainer(hmm.get(), trainingData);
trainer.train();

// Decode most likely state sequence
ViterbiCalculator viterbi(hmm.get(), observations);
StateSequence states = viterbi.decode();
```

### Examples

See the [examples/](examples/) directory for comprehensive usage examples:

- **[basic_hmm_example.cpp](examples/basic_hmm_example.cpp)** - Basic HMM usage with modern C++17 features
- **[poisson_hmm_example.cpp](examples/poisson_hmm_example.cpp)** - Website traffic modeling using Poisson distribution
- **[financial_hmm_example.cpp](examples/financial_hmm_example.cpp)** - Market volatility modeling with Beta and Log-Normal distributions
- **[reliability_hmm_example.cpp](examples/reliability_hmm_example.cpp)** - Component lifetime analysis with Weibull and Exponential distributions
- **[quality_control_hmm_example.cpp](examples/quality_control_hmm_example.cpp)** - Manufacturing process monitoring with Binomial and Uniform distributions
- **[economics_hmm_example.cpp](examples/economics_hmm_example.cpp)** - Economic modeling with Negative Binomial and Pareto distributions
- **[queuing_theory_hmm_example.cpp](examples/queuing_theory_hmm_example.cpp)** - Service systems and queuing analysis with Poisson, Exponential, and Gamma distributions
- **[robust_financial_hmm_example.cpp](examples/robust_financial_hmm_example.cpp)** - Heavy-tailed financial modeling with Student's t-distribution
- **[statistical_process_control_hmm_example.cpp](examples/statistical_process_control_hmm_example.cpp)** - Advanced quality control with Chi-squared distribution
- **[swarm_coordination_example.cpp](examples/swarm_coordination_example.cpp)** - Discrete state drone swarm coordination and formation control

## Project Structure

```
libhmm/
├── include/libhmm/           # Public headers
│   ├── calculators/          # Algorithm implementations
│   ├── distributions/        # Probability distributions
│   ├── training/            # Training algorithms
│   ├── io/                  # File I/O support
│   └── common/              # Utilities and types
├── src/                     # Implementation files
├── tests/                   # Test suite
├── examples/                # Usage examples
├── docs/                    # Documentation
└── CMakeLists.txt          # Modern build system
```

## Requirements

- **C++17** compatible compiler (GCC 7+, Clang 6+, MSVC 2017+)
- **CMake 3.15+** (for CMake builds)
- **Make** (for legacy Makefile builds)

**Zero External Dependencies** - libhmm now requires only the C++17 standard library!

## Documentation

- [API Documentation](docs/api/)
- [Tutorials](docs/tutorials/)
- [Examples](examples/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Follow the [Gold Standard Checklist](docs/GOLD_STANDARD_CHECKLIST.md) for distribution implementations
5. Ensure code complies with the project style guide (enforced via clang-tidy)
6. Submit a pull request

## Acknowledgments

- Original implementation modernized to C++17 standards
- Inspired by JAHMM and other HMM libraries
- Mathematical foundations from Rabiner & Juang tutorials
