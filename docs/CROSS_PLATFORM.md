# Cross-Platform Build Guide

libhmm builds and tests on Windows (MSVC), macOS (AppleClang), Linux (GCC), and Linux (Clang).
All four platforms are verified by CI on every push.

## Requirements

- **C++20** compiler: GCC 11+, Clang 14+, MSVC 2019 16.11+ (Visual Studio 2022 recommended)
- **CMake 3.20+**
- **Zero external dependencies** at runtime — GTest is fetched automatically via FetchContent

## Platforms

### Windows (MSVC)

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --parallel 4
```

Notes:
- Do NOT call `vcvars64.bat` before cmake; the VS generator handles it
- SIMD: `check_cxx_source_runs` selects `/arch:AVX512`, `/arch:AVX2`, or `/arch:AVX` by running a test binary to verify CPU support — prevents ILLEGAL INSTRUCTION crashes on cloud VMs that accept the flag but can't execute it
- See `WARP.md` for full Windows session setup

### macOS (AppleClang)

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build
```

Notes:
- Homebrew prefix: `/opt/homebrew` (Apple Silicon) or `/usr/local` (Intel)
- CMake detects the architecture automatically via `uname -m`
- SIMD: `-march=native` — selects NEON on AArch64, AVX/AVX2 on Intel Macs

#### macOS Catalina (10.15) runtime guard

On Catalina, Homebrew LLVM/libc++ paths can produce runtime segfaults. Use the
helper script to configure with a sanitized environment and pinned AppleClang
toolchain:

```bash
./scripts/configure_catalina.sh build
cmake --build build --config Release
ctest --test-dir build
```

If CMake detects Homebrew LLVM/libc++ contamination on Catalina, configure
fails fast with an error describing the detected sources and remediation. A
temporary escape hatch is available:

```bash
-DLIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX=ON
```

### Linux (GCC)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build
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
cmake -DBUILD_TOOLS=OFF ..       # Skip tools
cmake -DBUILD_SHARED_LIBS=OFF .. # Suppress shared library (static is always built)
```

## Library Output

Both shared and static libraries are produced in every build. Sources compile once into an
OBJECT library and link into both targets — no double compilation overhead.

| Platform | Files produced | Notes |
|---|---|---|
| macOS | `libhmm.dylib`, `libhmm.a` | Shared: `@rpath`; static: archive |
| Linux | `libhmm.so`, `libhmm.a` | Shared: `$ORIGIN` RPATH; static: archive |
| Windows | `hmm.dll`, `hmm.lib`, `hmm_static.lib` | `hmm.lib` = DLL import lib; `hmm_static.lib` = static archive |

## CI Matrix

See `.github/workflows/ci.yml`. Four build jobs (Linux/GCC, Linux/Clang, macOS/AppleClang,
Windows/MSVC) plus a lint job (clang-format + cppcheck). All 36 tests pass on every platform.
