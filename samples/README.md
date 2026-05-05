# libhmm sample HMM files

Pre-built HMM files for testing, validation, and experimentation.
Each model is provided in both the recommended JSON format and the legacy XML format.

## Files

### two_state_gaussian — 2-state Gaussian HMM
Simple two-state continuous HMM. Good for validating basic load/inference
pipelines with continuous observations.

| Parameter | Value |
|-----------|-------|
| States | 2 |
| Pi | [0.75, 0.25] |
| Transition | [[0.875, 0.125], [0.25, 0.75]] |
| State 0 | Gaussian(μ=0, σ=1) |
| State 1 | Gaussian(μ=2.5, σ=0.5) |

### casino — 2-state discrete HMM (dishonest casino)
Classic discrete HMM from Durbin et al. (1998). Two dice: fair (uniform over
6 outcomes) and loaded (biased toward face 6). Good for Viterbi decoding
tests and discrete distribution validation.

| Parameter | Value |
|-----------|-------|
| States | 2 |
| Pi | [0.75, 0.25] |
| Transition | [[0.875, 0.125], [0.25, 0.75]] |
| State 0 (fair) | Discrete(n=6, uniform ≈ 1/6 each) |
| State 1 (loaded) | Discrete(n=6, [0.125×5, 0.375]) |

## Formats

- **`.json`** — Recommended. Use `libhmm::load_json()` / `libhmm::save_json()`.
  Exact IEEE 754 round-trip via `max_digits10` precision.
- **`.xml`** — Legacy CDATA-wrapped text format. Use `XMLFileReader` for
  reading existing files; prefer JSON for new code.

## Usage

```cpp
#include "libhmm/io/hmm_json.h"

// JSON (recommended)
auto hmm = libhmm::load_json("samples/two_state_gaussian.json");

// Legacy XML
#include "libhmm/io/xml_file_reader.h"
libhmm::XMLFileReader reader;
auto hmm = reader.read("samples/two_state_gaussian.xml");
```

Or from the command line with the validator tool:

```
hmm_validator samples/two_state_gaussian.json
hmm_validator samples/casino.xml
```
