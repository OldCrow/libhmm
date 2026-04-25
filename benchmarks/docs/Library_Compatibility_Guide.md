# HMM Library Compatibility Guide

## Purpose

This guide provides the steps needed to build each external HMM library used in the
comparative benchmarks and link it against benchmark code. For each library it covers:

1. Source patches — bugs discovered during integration that must be applied before building
2. Build instructions — per platform where the steps differ
3. Linking — static and dynamic linking against the compiled library, or subprocess
   integration where the library provides executables rather than a linkable target

Performance figures and benchmark interpretation are in `BENCHMARKING_RESULTS.md`.
CMake dependency paths for the benchmark build system are in the top-level `README.md`.

## External dependency layout

The benchmark CMake configuration locates external libraries via these variables
(defaults assume `$HOME/Development` as the root):

| CMake variable              | Default path                                  |
|-----------------------------|-----------------------------------------------|
| `LIBHMM_BENCHMARK_DEPS_ROOT`| `$HOME/Development`                           |
| `HMMLIB_DIR`                | `${LIBHMM_BENCHMARK_DEPS_ROOT}/HMMLib`        |
| `GHMM_DIR`                  | `${LIBHMM_BENCHMARK_DEPS_ROOT}/GHMM`          |
| `STOCHHMM_DIR`              | `${LIBHMM_BENCHMARK_DEPS_ROOT}/StochHMM`      |
| `HTK_DIR`                   | `${LIBHMM_BENCHMARK_DEPS_ROOT}/HTK`           |
| `JAHMM_DIR`                 | `${LIBHMM_BENCHMARK_DEPS_ROOT}/Jahmm`         |
| `LAMP_DIR`                  | `${LIBHMM_BENCHMARK_DEPS_ROOT}/LAMP`          |

Override at configure time if your layout differs:

```bash
cmake -S . -B build -DBUILD_BENCHMARKS=ON -DHMMLIB_DIR=/opt/hmmlib
```

---

## HMMLib

HMMLib 1.0.1 (2010) targets C++98/C++03 with Intel SSE intrinsics.
Two categories of patches are required: C++ standard compliance and ARM64 SIMD porting.

### Source patches

#### Patch 1 — Template base class method resolution (required on C++11 and later)

Modern C++ requires explicit `this->` qualification when calling methods from a
dependent template base class. Three files need this fix.

**`HMMlib/hmm_table.hpp`, constructor body (line 246):**
```cpp
// Before
allocator::allocate(no_rows, no_columns, *this);
reset(val);

// After
allocator::allocate(no_rows, no_columns, *this);
this->reset(val);
```

**`HMMlib/hmm_vector.hpp`, `operator=` body (line 141):**
```cpp
// Before
reset(val);
return *this;

// After
this->reset(val);
return *this;
```

**`HMMlib/hmm_matrix.hpp`, `operator=` body (line 99):**
```cpp
// Before
reset(val);
return *this;

// After
this->reset(val);
return *this;
```

#### Patch 2 — ARM64 (Apple Silicon / AArch64) SIMD port

HMMLib's compute kernels use Intel SSE intrinsics exclusively. On ARM64 these must
be replaced with ARM NEON equivalents. This patch is only required when building on
Apple Silicon or other AArch64 machines; x86_64 builds are unaffected.

**Architecture detection block — add to the top of each header that includes SIMD
(`hmm_table.hpp`, `hmm_vector.hpp`, `hmm_matrix.hpp`, `hmm.hpp`, `allocator_traits.hpp`,
`sse_operator_traits.hpp`):**

```cpp
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    #include <pmmintrin.h>
    #define HMM_SIMD_X86
#elif defined(__aarch64__) || defined(__arm64__)
    #include <arm_neon.h>
    #define HMM_SIMD_ARM
    typedef float32x4_t __m128;
    typedef float64x2_t __m128d;
#else
    #define HMM_SIMD_NONE
#endif
```

**`HMMlib/sse_operator_traits.hpp` — Intel to NEON intrinsic substitutions:**

| Intel SSE              | ARM NEON replacement                                                        |
|------------------------|-----------------------------------------------------------------------------|
| `_mm_set_pd1(v)`       | `vdupq_n_f64(v)`                                                            |
| `_mm_set_ps1(v)`       | `vdupq_n_f32(v)`                                                            |
| `_mm_add_pd(a,b)`      | `vaddq_f64(a,b)`                                                            |
| `_mm_add_ps(a,b)`      | `vaddq_f32(a,b)`                                                            |
| `_mm_mul_pd(a,b)`      | `vmulq_f64(a,b)`                                                            |
| `_mm_mul_ps(a,b)`      | `vmulq_f32(a,b)`                                                            |
| `_mm_hadd_pd(a,a)`     | `vgetq_lane_f64(a,0) + vgetq_lane_f64(a,1)`                                |
| `_mm_hadd_ps(a,a)`     | `vget_lane_f32(vpadd_f32(vget_low_f32(a), vget_high_f32(a)), 0)`           |

**`HMMlib/allocator_traits.hpp` — replace Intel-specific aligned allocation:**
```cpp
// Before
T.table = static_cast<float_type*>(_mm_malloc(size, 16));

// After
#ifdef HMM_SIMD_X86
    T.table = static_cast<float_type*>(_mm_malloc(size, 16));
#else
    if (posix_memalign(reinterpret_cast<void**>(&T.table), 16, size) != 0)
        T.table = static_cast<float_type*>(malloc(size));
#endif
```

**`CMakeLists.txt` — architecture-aware SIMD flags:**
```cmake
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(CMAKE_CXX_FLAGS "-Wall -Wconversion -O3 -march=armv8-a")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(CMAKE_CXX_FLAGS "-Wall -Wconversion -O3 -msse4")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wconversion -O3")
endif()
```

### Building

#### macOS and Linux

HMMLib requires Boost.

```bash
# macOS (Homebrew)
brew install boost

# Linux (Debian/Ubuntu)
sudo apt-get install libboost-all-dev

# Build (apply patches first)
cmake . -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### Windows

Not validated. The x86_64 build path uses `_mm_malloc` (available on MSVC via
`<xmmintrin.h>`) but the CMake configuration applies GCC/Clang-only SIMD flags
(`-msse4`). Windows support requires adapting the CMake SIMD flag selection for MSVC
(`/arch:SSE4.2` or `/arch:AVX`). No attempt has been made to validate this.

### Linking

HMMLib builds as a static library. Boost must also be linked.

**CMake:**
```cmake
find_library(HMMLIB_LIB HMMLib PATHS ${HMMLIB_DIR})
target_link_libraries(your_target PRIVATE ${HMMLIB_LIB} Boost::program_options)
target_include_directories(your_target PRIVATE ${HMMLIB_DIR})
```

**Compiler flags (GCC/Clang):**
```bash
g++ benchmark.cpp \
    -I${HMMLIB_DIR} \
    -L${HMMLIB_DIR} -lHMMLib \
    -lboost_program_options \
    -O3 -msse4
```

HMMLib does not build a shared library.

---

## GHMM

GHMM is a C library with Python bindings built via GNU Autotools.
It requires GSL and libxml2 as external dependencies.

### Source patches

#### ARM64 only — `tests/mcmc.c`

The GHMM library itself compiles cleanly on ARM64. One test binary (`mcmc`) fails
to compile due to missing headers and an incorrect pointer type. This patch is only
needed if you intend to run the full GHMM test suite.

**Add missing includes (after existing includes at the top of `tests/mcmc.c`):**
```c
#include <ghmm/fbgibbs.h>    /* init_priors */
#include <ghmm/cfbgibbs.h>   /* ghmm_dmodel_cfbgibbs */
```

**Fix incorrect pointer type and add null check:**
```c
// Before
int * Q = ghmm_dmodel_cfbgibbs(mo, my_output, pA, pB, pPi, 2, iter, 0);
printf("viterbi prob mcmc%f \n",
    ghmm_dmodel_viterbi_logp(mo, my_output->seq[0], my_output->seq_len[0], Q[0]));

// After
int ** Q = ghmm_dmodel_cfbgibbs(mo, my_output, pA, pB, pPi, 2, iter, 0);
if (Q != NULL) {
    printf("viterbi prob mcmc%f \n",
        ghmm_dmodel_viterbi_logp(mo, my_output->seq[0], my_output->seq_len[0], Q[0]));
}
```

### Building

#### macOS

GHMM requires Python development headers. Use a virtual environment to isolate the
Python dependency.

```bash
brew install autoconf automake libtool gsl libxml2

python3 -m venv /path/to/GHMM/venv
source /path/to/GHMM/venv/bin/activate

# Apply patches, then configure:
# Apple Silicon
./configure --with-python=/opt/homebrew/bin/python3
# Intel Mac
./configure --with-python=/usr/local/bin/python3

make -j$(nproc)
make install
```

#### Linux

```bash
sudo apt-get install autoconf automake libtool libgsl-dev libxml2-dev python3-dev

./configure
make -j$(nproc)
sudo make install
```

#### Windows

Not supported. GHMM depends on POSIX APIs and GNU Autotools. Build under WSL if a
Windows host is required.

### Linking

GHMM installs both a static archive and a shared library.

**Static (GCC/Clang):**
```bash
gcc benchmark.c \
    -I/path/to/ghmm/include \
    -L/path/to/ghmm/lib -lghmm \
    -lgsl -lxml2 -lm
```

**Dynamic:**
```bash
# Linux
export LD_LIBRARY_PATH=/path/to/ghmm/lib:$LD_LIBRARY_PATH
# macOS
export DYLD_LIBRARY_PATH=/path/to/ghmm/lib:$DYLD_LIBRARY_PATH

gcc benchmark.c -I/path/to/ghmm/include -L/path/to/ghmm/lib -lghmm -lgsl -lxml2 -lm
```

**CMake:**
```cmake
find_library(GHMM_LIB ghmm PATHS ${GHMM_DIR}/lib)
target_link_libraries(your_target PRIVATE ${GHMM_LIB})
target_include_directories(your_target PRIVATE ${GHMM_DIR}/include)
```

On macOS, the benchmark CMake normalizes the GHMM shared library install name
automatically to avoid `dyld` path errors when GHMM is relocated.

### API notes

GHMM uses 0-based indexing throughout for states, observations, and matrix access,
despite some documentation suggesting 1-based. Verify against source rather than
documentation.

```c
ghmm_dmodel* model = ghmm_dmodel_calloc(num_states, num_symbols);

for (int i = 0; i < num_states; i++) {
    for (int j = 0; j < num_states; j++)
        model->s[i].out_a[j] = prob;         /* 0-based */
    for (int k = 0; k < num_symbols; k++)
        model->s[i].b[k] = emission_prob;    /* 0-based */
}
```

---

## StochHMM

StochHMM (KorfLab/StochHMM) is a C++11 HMM library with a text-based model
definition format. Three bugs have been identified and filed as upstream PRs.
All four patches below should be applied before building.

### Source patches

#### Patch 1 — PI constant precision (upstream PR #25)

`src/stochMath.h`, line 49:
```cpp
// Before — transposed digit: 3.145... instead of 3.141...
#define PI 3.145926535897932

// After
#define PI 3.141592653589793238463
```

This error introduces a systematic offset in PDF normalizations for any distribution
that uses `PI`, including Gaussian and several other continuous PDFs.

#### Patch 2 — Replace `std::random_shuffle` (upstream PR #24)

`std::random_shuffle` was removed in C++17.

**`src/sequence.cpp`** — add `#include <random>` directly after `#include "sequence.h"`,
then replace the `sequence::shuffle()` body:

```cpp
void sequence::shuffle(){
    thread_local std::mt19937 rng(std::random_device{}());
    if (realSeq){
        std::shuffle(real->begin(), real->end(), rng);
    }
    else if (seq!=NULL){
        std::shuffle(seq->begin(), seq->end(), rng);
    }
    else{
        std::shuffle(undigitized.begin(), undigitized.end(), rng);
    }
    return;
}
```

**`src/sequence.h`**, line 179 — update the comment:
```cpp
// Before
//! Shuffles the sequence using std::random_shuffle

// After
//! Shuffles the sequence using std::shuffle
```

#### Patch 3 — Poisson PMF denominator (upstream PR #23)

`src/PDF.cpp`, `poisson_pdf()`:
```cpp
// Before — Stirling-based factorial introduces measurable drift at moderate k
return (pow(lambda,(double)k) * exp(-1*lambda)) / factorial(k);

// After — exact result via the gamma function identity Γ(k+1) = k!
return (pow(lambda,(double)k) * exp(-1*lambda)) / tgamma((double)k + 1.0);
```

PRs #23, #24, and #25 are pending upstream merge. Apply them as local patches until
they appear in the upstream repository.

#### Patch 4 — `M_PI` undefined on MSVC

`src/stochMath.cpp`, line 237, inside `factorial()`:
```cpp
// Before — MSVC does not define M_PI without _USE_MATH_DEFINES
double f = sqrt((2*x+(1/3))*M_PI)*pow(x,x)*exp(-1*x);

// After — PI is already defined in stochMath.h
double f = sqrt((2*x+(1/3))*PI)*pow(x,x)*exp(-1*x);
```

GCC and Clang define `M_PI` from `<math.h>`. Applying this patch on all platforms is
harmless.

### Building

#### macOS and Linux (Autotools)

StochHMM has no external dependencies.

```bash
./configure
make -j$(nproc)
```

Result: `src/libstochhmm.a` (static library), `src/StochHMM` (executable).

#### All platforms including Windows (CMake)

StochHMM's upstream build system is Autotools and does not support Windows.
Create the following `CMakeLists.txt` at the repository root, then build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Result: `build/libstochhmm.a` or `build/Release/stochhmm.lib` (static library),
`build/StochHMM` or `build/Release/StochHMM.exe` (executable).

**`CMakeLists.txt`:**

```cmake
cmake_minimum_required(VERSION 3.15)
project(StochHMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(
        /W3
        /wd4244   # int/size_t narrowing
        /wd4267   # size_t -> int
        /wd4305   # double -> float truncation
        /wd4458   # declaration hides class member
        /wd4456   # declaration hides previous local
        /wd4309   # truncation of constant value
        /wd4554   # operator precedence
        /D_CRT_SECURE_NO_WARNINGS
    )
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# Source list mirrors src/Makefile.am libstochhmm_a_SOURCES.
# Files present in src/ but absent from the upstream Makefile
# (alignment.cpp, statistics.cpp, distributions.cpp, CDF.cpp, Counter.cpp,
# MotifScorer.cpp, sequenceStream.cpp, trainingSeq.cpp, output.cpp)
# are intentionally excluded — they are not wired into any build target upstream.
set(STOCHHMM_LIB_SOURCES
    src/pwm.cpp          src/PDF.cpp           src/trellis.cpp
    src/viterbi.cpp      src/stoch_viterbi.cpp src/stoch_forward.cpp
    src/nth_best.cpp     src/stochTable.cpp    src/backward.cpp
    src/forward.cpp      src/baum_welch.cpp    src/forward_viterbi.cpp
    src/posterior.cpp    src/traceback_path.cpp src/externDefinitions.cpp
    src/index.cpp        src/stochMath.cpp     src/text.cpp
    src/userFunctions.cpp src/hmm.cpp          src/state.cpp
    src/lexicalTable.cpp src/track.cpp         src/emm.cpp
    src/externalFuncs.cpp src/modelTemplate.cpp src/transitions.cpp
    src/weight.cpp       src/options.cpp       src/seqJobs.cpp
    src/seqTracks.cpp    src/sequence.cpp      src/sequences.cpp
    src/bitwise_ops.cpp  src/dynamic_bitset.cpp
)

add_library(stochhmm STATIC ${STOCHHMM_LIB_SOURCES})
target_include_directories(stochhmm PUBLIC src)

option(BUILD_STOCHHMM_EXE "Build the StochHMM command-line executable" ON)
if(BUILD_STOCHHMM_EXE)
    # Target is named StochHMM_cli to avoid an MSBuild circular dependency
    # (CMake project and executable target cannot share a name under the VS generator).
    # OUTPUT_NAME preserves the StochHMM binary name on disk.
    add_executable(StochHMM_cli src/StochHMM.cpp)
    set_target_properties(StochHMM_cli PROPERTIES OUTPUT_NAME StochHMM)
    target_link_libraries(StochHMM_cli PRIVATE stochhmm)
endif()
```

### Linking

StochHMM does not build a shared library.

**CMake (benchmark integration):**
```cmake
# If building StochHMM as a subdirectory:
add_subdirectory(${STOCHHMM_DIR} stochhmm_build EXCLUDE_FROM_ALL)
target_link_libraries(your_target PRIVATE stochhmm)
# target_include_directories is propagated via PUBLIC on the stochhmm target

# If linking against a pre-built archive:
find_library(STOCHHMM_LIB stochhmm
    PATHS ${STOCHHMM_DIR}/build/Release ${STOCHHMM_DIR}/src)
target_link_libraries(your_target PRIVATE ${STOCHHMM_LIB})
target_include_directories(your_target PRIVATE ${STOCHHMM_DIR}/src)
```

**Compiler flags (GCC/Clang):**
```bash
g++ benchmark.cpp \
    -I${STOCHHMM_DIR}/src \
    -L${STOCHHMM_DIR}/build -lstochhmm \
    -std=c++17
```

### API notes

StochHMM uses a text-based model definition format and a FASTA-like sequence format.
There is no programmatic model construction API; models are loaded from files.

```
# Model file format (*.hmm)
MODEL:
    NAME:    WeatherModel
    STATES:  2
    ALPHABET: 2

STATES:
    NAME: Sunny
    TRANSITIONS: Standard: Sunny:0.8 Rainy:0.2
    EMISSION: 0:0.6 1:0.4

    NAME: Rainy
    TRANSITIONS: Standard: Sunny:0.4 Rainy:0.6
    EMISSION: 0:0.2 1:0.8
```

```cpp
StochHMM::model hmm;
hmm.import("model_file.hmm");

StochHMM::sequences seqs;
seqs.import("sequence_file.fa");

double log_likelihood = hmm.forward(*seq);
StochHMM::traceback_path path = hmm.viterbi(*seq);
```

Benchmark code generates model and sequence files dynamically, then invokes the
StochHMM API to load them. See `benchmarks/src/libhmm_vs_stochhmm_benchmark.cpp`
for the current pattern.

---

## HTK

HTK (Hidden Markov Model Toolkit) is a speech recognition toolkit written in C.
It provides stand-alone executables only; there is no linkable library.
Integration in benchmarks is via subprocess calls and file I/O.

### Source patches

Both patches are required on all platforms with a C99-or-later compiler.

#### Patch 1 — Missing `ARCH` definition (`HTKLib/esignal.c`)

Add after line 29 (after `#include "esignal.h"`):

```c
#ifndef ARCH
#  if defined(__APPLE__)
#    if defined(__aarch64__)
#      define ARCH "ARM64"
#    elif defined(__x86_64__)
#      define ARCH "X86_64"
#    else
#      define ARCH "UNKNOWN"
#    endif
#  elif defined(__linux__)
#    if defined(__aarch64__)
#      define ARCH "ARM64"
#    elif defined(__x86_64__)
#      define ARCH "X86_64"
#    else
#      define ARCH "UNKNOWN"
#    endif
#  else
#    define ARCH "UNKNOWN"
#  endif
#endif
```

#### Patch 2 — Deprecated `finite()` function (`HTKLib/HTrain.c`)

`finite()` was removed in C99. Replace both occurrences (lines 1517 and 1536):

```c
// Before
if (!finite(cTemp[m]))
if (!finite(vTemp[k]))

// After
if (!isfinite(cTemp[m]))
if (!isfinite(vTemp[k]))
```

### Building

HTK uses Makefiles and requires X11 (even when only HMM inference tools are used).

#### macOS

```bash
brew install --cask xquartz   # X11; installs to /opt/X11

cd /path/to/HTK/source
make clean
# Apply patches
./configure --prefix=/path/to/HTK/install
make all CPPFLAGS="-I/opt/X11/include" LDFLAGS="-L/opt/X11/lib"
make install
export PATH="/path/to/HTK/install/bin:$PATH"
HVite -V   # verify
```

#### Linux

```bash
sudo apt-get install libx11-dev

cd /path/to/HTK/source
make clean
# Apply patches
./configure --prefix=/path/to/HTK/install
make all
make install
export PATH="/path/to/HTK/install/bin:$PATH"
```

#### Windows

Not practical. HTK's build system is POSIX Makefile-based and the source uses POSIX
APIs throughout. Build under WSL if a Windows host is required.

### Integration

HTK does not expose a C/C++ API. Benchmark integration uses the `HVite` tool via
subprocess: create temporary model and sequence files in HTK format, invoke `HVite`,
then parse its output.

```cpp
std::string cmd = "HVite -C " + config + " -H " + model_file + " ...";
int rc = system(cmd.c_str());
double ll = parse_htk_log_likelihood(output_file);
```

HTK rounds log-likelihood output to multiples of 1000 by design (speech recognition
convention). Account for this in numerical comparisons.

---

## JAHMM

JAHMM is a pure Java HMM library (jahmm-0.6.2). It has no native code and
requires only a JDK. No platform-specific steps are necessary.

### Building (all platforms)

```bash
# Requires JDK 8 or later
java -version && javac -version

cd /path/to/Jahmm/source/jahmm-0.6.2
mkdir -p build
javac -d build $(find src -name "*.java")
jar cf jahmm.jar -C build .
```

On macOS with Homebrew OpenJDK:
```bash
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
```

On Windows (PowerShell):
```powershell
Get-ChildItem -Recurse -Filter *.java src | ForEach-Object { $_.FullName } |
    Set-Content sources.txt
javac -d build "@sources.txt"
jar cf jahmm.jar -C build .
```

### Integration

JAHMM does not expose a C/C++ API. Benchmark code invokes the compiled JAR
as a subprocess and parses output:

```bash
java -cp jahmm.jar be.ac.ulg.montefiore.run.jahmm.apps.cli.Cli [command] [args]
```

```cpp
std::string cmd = "java -cp " + jar_path + " " + class_name +
                  " " + model_file + " " + seq_file;
FILE* pipe = popen(cmd.c_str(), "r");
double result = parse_jahmm_output(pipe);
pclose(pipe);
```

Runtime and build paths are configurable via `JAHMM_DIR` and
`LIBHMM_BENCH_JAHMM_DIR` at CMake configure time. Java and Javac are resolved from
known Homebrew locations with fallback to PATH.

---

## LAMP_HMM

LAMP_HMM (University of Maryland, 1999–2003) is a C++ HMM implementation using
pre-C++98 coding conventions. It provides the `hmmFind` executable only; there is no
linkable library. Integration in benchmarks is via subprocess calls and file I/O.

### Source patches

Apply to all twelve `.C` source files:
`utils.C`, `readConfigFile.C`, `obsSeq.C`, `plainStateTrans.C`, `gammaProb.C`,
`explicitDurationTrans.C`, `initStateProb.C`, `hmm.C`, `discreteObsProb.C`,
`gaussianObsProb.C`, `vectorObsProb.C`, `hmmFind.C`.

#### Patch 1 — Replace pre-standard headers

```cpp
// Before (all .C files)
#include <iostream.h>
#include <fstream.h>
#include <iomanip.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// After
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace std;
```

A shell one-liner for macOS/Linux:

```bash
for f in *.C; do
    sed -i.bak \
        -e 's|#include <iostream\.h>|#include <iostream>|g' \
        -e 's|#include <fstream\.h>|#include <fstream>|g' \
        -e 's|#include <iomanip\.h>|#include <iomanip>|g' \
        "$f"
done
```

Then add `using namespace std;` at the end of the include block in each file.

#### Patch 2 — Stream state checking

Modern C++ streams cannot be compared to `NULL`.

```cpp
// Before
ifstream hmmFile(filename);
if (hmmFile == NULL) { cerr << "File not found"; exit(-1); }

// After
ifstream hmmFile(filename);
if (hmmFile.fail()) { cerr << "File not found"; exit(-1); }
```

### Building

#### macOS and Linux

```bash
# Apply patches first
make hmmFind
```

The original Makefile is compatible with modern GCC and Clang without changes.
Result: `hmmFind` executable (approx. 165 KB). Eight non-critical warnings
(deprecated `sprintf`, non-virtual destructors) are expected.

#### Windows

Not validated. The Makefile targets GCC; an MSVC build would require a CMakeLists.txt.

### Integration

LAMP_HMM uses configuration-file-driven execution. All file paths in the
configuration must be absolute.

**Configuration file format** (each key and value on separate lines):
```
sequenceName=
/absolute/path/to/sequence.seq
skipLearning=
0
hmmInputName=
0
outputName=
/absolute/path/to/output_prefix
nbDimensions=
1
nbSymbols=
6
nbStates=
2
seed=
42
explicitDuration=
0
gaussianDistribution=
0
EMMethod=
2
```

**Sequence file format** (P5 magic header required — P6 will fail):
```
P5
nbSequences= 1
T= 100
0 1 0 1 1 0 ...
```

**Benchmark integration pattern:**
```cpp
std::string cmd = "/path/to/hmmFind " + config_file + " > " + log_file + " 2>&1";
system(cmd.c_str());
// Parse output_prefix.dis for distance metrics
```

LAMP_HMM is configured via `LAMP_DIR` and `LIBHMM_BENCH_LAMP_DIR` at CMake
configure time.

---

## Platform support summary

| Library  | macOS | Linux | Windows        | Linkable target     |
|----------|-------|-------|----------------|---------------------|
| HMMLib   | Yes   | Yes   | Not validated  | Static (`.a`)       |
| GHMM     | Yes   | Yes   | WSL only       | Static + dynamic    |
| StochHMM | Yes   | Yes   | Yes (CMake)    | Static (`.a`/`.lib`)|
| HTK      | Yes   | Yes   | WSL only       | Executable only     |
| JAHMM    | Yes   | Yes   | Yes            | JAR (subprocess)    |
| LAMP_HMM | Yes   | Yes   | Not validated  | Executable only     |

## Troubleshooting

- If a benchmark target is absent from the build, check CMake configure output for
  dependency resolution messages. Each external target prints a status line at
  configure time.
- If a target builds but fails at runtime, verify external tool or library paths
  first (PATH for HTK/JAHMM/LAMP, library install prefix for GHMM/HMMLib).
- If StochHMM continuous distribution comparisons produce unexpected offsets,
  confirm Patch 1 (PI constant) has been applied and StochHMM has been rebuilt.
- If GHMM install-name errors occur at runtime on macOS, rebuild the benchmark.
  The benchmark CMake normalizes the GHMM shared library install name automatically.
