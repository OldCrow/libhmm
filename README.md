# libhmm - Modern C++17 Hidden Markov Model Library

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.15%2B-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern, type-safe C++17 implementation of Hidden Markov Models with comprehensive training algorithms and probability distributions.

**🆕 Recently Updated**: This library has been modernized with C++17 standards, smart pointer memory management, and critical bug fixes including resolution of segmentation faults in ViterbiTrainer.

## Features

### 🎯 **Training Algorithms**
- **Viterbi Training** - Segmented k-means with clustering
- **Segmented K-Means** - Alternative k-means implementation
- **Baum-Welch** - Standard expectation-maximization
- **Scaled Baum-Welch** - Numerically stable implementation

### 📊 **Probability Distributions** 
- Discrete Distribution
- Gaussian (Normal) Distribution
- Gamma Distribution
- Exponential Distribution
- Log-Normal Distribution
- Pareto Distribution

### 🧮 **Calculators**
- Forward-Backward Algorithm
- Scaled Forward-Backward (numerical stability)
- Log Forward-Backward (log-space computation)
- Viterbi Algorithm (most likely path)

### 💾 **I/O Support**
- XML file reading/writing
- Extensible file I/O manager
- Model serialization

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
