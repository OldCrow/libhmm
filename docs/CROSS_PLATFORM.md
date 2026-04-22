# Cross-Platform Build Guide

libhmm builds and tests on Windows (MSVC), macOS (AppleClang), and Linux (GCC).
All three platforms are verified by CI on every push.

## Requirements

- **C++20** compiler: GCC 11+, Clang 14+, MSVC 2019 16.11+ (Visual Studio 2022 recommended)
- **CMake 3.20+**
- **Zero external dependencies** at runtime — GTest is fetched automatically via FetchContent

## Platforms

### Windows (MSVC)

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --parallel 4 -LE known_broken
```

Notes:
- Do NOT call `vcvars64.bat` before cmake; the VS generator handles it
- SIMD: `check_cxx_source_runs` selects `/arch:AVX512`, `/arch:AVX2`, or `/arch:AVX` by running a test binary to verify CPU support — prevents ILLEGAL INSTRUCTION crashes on cloud VMs that accept the flag but can't execute it
- See `WARP.md` for full Windows session setup

### macOS (AppleClang)

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -LE known_broken
```

Notes:
- Homebrew prefix: `/opt/homebrew` (Apple Silicon) or `/usr/local` (Intel)
- CMake detects the architecture automatically via `uname -m`
- SIMD: `-march=native` — selects NEON on AArch64, AVX/AVX2 on Intel Macs
- `test_xml_file_io` is marked `known_broken` on Windows only; it passes on macOS

### Linux (GCC)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build -LE known_broken
```

Notes:
- Math library (`-lm`) linked explicitly on Linux
- SIMD: `-march=native` — same as macOS

## Configuration Summary

When cmake runs, the SIMD configuration is reported:

```
-- SIMD support — AVX-512: 1  AVX2: 1  AVX: 1        (Windows, Ryzen 7)
-- SIMD optimization: /arch:AVX512 (CPU verified)

-- SIMD compiler support — NEON: ON (AArch64 baseline) (macOS Apple Silicon)
-- SIMD optimization: -march=native (AArch64/NEON)

-- SIMD support — AVX-512: 0  AVX2: 1  AVX: 1         (macOS Intel)
-- SIMD optimization: -march=native
```

## Build Options

```bash
cmake -DBUILD_EXAMPLES=OFF ..    # Skip examples
cmake -DBUILD_TESTS=OFF ..       # Skip tests
cmake -DBUILD_SHARED_LIBS=OFF .. # Static library (default is shared)
```

## Library Output

| Platform | Library format | Notes |
|---|---|---|
| macOS | `.dylib` | `@rpath` for flexible loading |
| Linux | `.so` | `$ORIGIN` RPATH |
| Windows | `.dll` + `.lib` | Import library for linking |

## CI Matrix

See `.github/workflows/ci.yml`. All jobs run `ctest -LE known_broken` to keep CI green.
The one `known_broken` test (`test_xml_file_io`) is a pre-existing Windows-specific failure.
