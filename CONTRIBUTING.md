# Contributing to libhmm

Thank you for your interest in contributing to libhmm.

## What we welcome

- Bug fixes and correctness improvements
- New emission distributions (scalar or multivariate) with tests and Doxygen
- Performance improvements to existing algorithms
- Additional real-world examples with reference comparisons
- Documentation and clarity improvements
- CI / cross-platform build fixes within the supported toolchain matrix

## Toolchain policy

libhmm v4 requires:

| Platform | Minimum |
|---|---|
| macOS | 13 (Ventura), Apple Clang 14 / Xcode 14 |
| Linux | GCC 12 or Clang 14 |
| Windows | MSVC 2022 17.x (`/std:c++20`) |

**Pull requests that restore support for compilers below these minimums will
not be accepted into upstream.** The v4 minimums are load-bearing: they
guarantee `<concepts>`, `<ranges>`, `<span>`, and libc++ C++20 completeness
without conditional compilation or fallback paths.

If you need support for an older toolchain, the right approach is to fork the
repository. The last release that supported pre-v4 compilers is **v3.8.0**,
which is available as a tagged release and remains on the `main` branch history.

## Code standards

- C++20 throughout; no C++17-or-earlier fallback branches
- `#pragma once` for all headers
- Doxygen `/** */` comments on all public interfaces
- `[[nodiscard]]` on functions whose return value must not be ignored
- `noexcept` where the implementation genuinely cannot throw
- Cyclomatic complexity ≤ 10 per function (measured with lizard)
- Zero compiler warnings under `-Wall -Wextra -Wpedantic` (GCC/Clang) and `/W4` (MSVC)
- All new code covered by GTest tests in the appropriate `tests/` subdirectory
- Follow existing naming conventions: PascalCase for classes, camelCase for
  instance methods, snake_case for static helpers and free functions

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Write tests first when fixing bugs; confirm the test fails without your fix.
3. Run the full test suite: `cmake --build build && ctest --test-dir build`.
4. Check CCN: `python3 -m lizard src/ include/ --CCN 10 -w` — fix any violations.
5. Open a PR against `main` with a clear description of the change and its motivation.
