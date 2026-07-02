# Experimental Compiler Flags

This document describes compile-time preprocessor macros that gate alternative
algorithms in libhmm. These are **not CMake options** and are **not part of the
public API or build system**. They exist solely for algorithm comparison during
profiling work.

## Forward-Backward Recurrence Experiments

Two macros control recurrence kernel selection in
`BasicForwardBackwardCalculator` (`include/libhmm/calculators/basic_forward_backward_calculator.h`):

### `LIBHMM_EXPERIMENT_FB_MAX_REDUCE`

Forces `FbRecurrenceMode::MaxReduce` for every sequence, regardless of
sequence length or state count. Use this to benchmark the max-reduce
log-sum-exp kernel in isolation without the overhead of runtime mode selection.

### `LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR`

Applies a simplified heuristic: select `FbRecurrenceMode::MaxReduce` when the
model has more than 2 states, `FbRecurrenceMode::Pairwise` otherwise. Use this
to evaluate a cheaper selection rule against the full policy in
`selectFbRecurrenceMode()`.

## Activation

Neither macro is exposed through CMake. Activate via a compile definition on a
local build tree:

```bash
# Force MaxReduce for all sequences
cmake -B build-focus-maxreduce \
  -DCMAKE_CXX_FLAGS="-DLIBHMM_EXPERIMENT_FB_MAX_REDUCE" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Test the N>2 adaptive heuristic
cmake -B build-focus-adaptive \
  -DCMAKE_CXX_FLAGS="-DLIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

The `tools/fb_contour_sweep` and `tools/hotspot_breakdown` programs use these
macros to compare kernel performance across different sequence lengths and state
counts. The local build trees `build-focus-*` and `build-bench-adaptive-*` are
gitignored.

## When to use

Do not use these in production or release builds. They exist to answer the
question "does the max-reduce kernel help or hurt for this workload?" before
committing to a policy change in `selectFbRecurrenceMode()`. After any such
change, delete the local build tree and rebuild clean.
