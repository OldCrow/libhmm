# Consumer Example: FetchContent

Demonstrates consuming libhmm via CMake's `FetchContent` module — no separate install step needed.

## Build this example

```bash
cd consumer_example_fetchcontent
cmake -S . -B build
cmake --build build --parallel
./build/consumer_demo
```

The first configure will clone the libhmm repository. Subsequent builds use the cached source.

## Using a local checkout

For development, point FetchContent at a local directory instead of GitHub:

```cmake
FetchContent_Declare(libhmm SOURCE_DIR /path/to/libhmm)
```

## What it tests

- `FetchContent_MakeAvailable(libhmm)` builds libhmm as a subdirectory
- `libhmm::hmm` target is available without `find_package`
- `#include "libhmm/libhmm.h"` resolves correctly from the build tree
- A 2-state Gaussian HMM constructs and its emission PDF evaluates correctly
