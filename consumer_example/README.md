# Consumer Example: find_package

Demonstrates consuming libhmm from an external project after installation.

## Prerequisites

Build and install libhmm:

```bash
cd /path/to/libhmm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix /tmp/libhmm-install
```

## Build this example

```bash
cd consumer_example
cmake -S . -B build -DCMAKE_PREFIX_PATH=/tmp/libhmm-install
cmake --build build
./build/consumer_demo
```

## What it tests

- `find_package(libhmm REQUIRED)` locates the installed package
- `libhmm::hmm` target provides headers and the shared library
- `#include "libhmm/libhmm.h"` resolves correctly from the install tree
- A 2-state Gaussian HMM constructs and its emission PDF evaluates correctly
