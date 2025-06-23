# libhmm - Modern C++17 Hidden Markov Model Library

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.15%2B-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.2.1-brightgreen.svg)](https://github.com/OldCrow/libhmm/releases)
[![Tests](https://img.shields.io/badge/Tests-19/19_Passing-success.svg)](tests/)

A modern, type-safe C++17 implementation of Hidden Markov Models with comprehensive training algorithms and probability distributions.

**🎉 Latest Release v2.2.1**: Complete statistical modeling framework with 12 probability distributions, comprehensive unit test coverage (19/19 tests passing), enhanced numerical stability, and 7 comprehensive real-world examples covering all major application domains.

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

### 🧮 **Calculators**
- Forward-Backward Algorithm
- Scaled Forward-Backward (numerical stability)
- Log Forward-Backward (log-space computation)
- Viterbi Algorithm (most likely path)

### 💾 **I/O Support**
- XML file reading/writing
- Extensible file I/O manager
- Model serialization

### 🧪 **Testing Infrastructure**
- **Comprehensive Unit Tests**: 19 standalone distribution tests
- **Integration Tests**: Core HMM functionality testing
- **100% Test Coverage**: All distributions with complete functionality testing
- **CMake/CTest Integration**: Automated testing framework
- **Continuous Validation**: Parameter fitting, edge cases, and error handling

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
- **Boost** libraries
- **Make** (for legacy Makefile builds)

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
4. Submit a pull request

## Acknowledgments

- Original implementation modernized to C++17 standards
- Inspired by JAHMM and other HMM libraries
- Mathematical foundations from Rabiner & Juang tutorials
