# Cross-Platform Build Support

The libhmm library now supports cross-platform compilation for macOS and Linux with automatic platform detection and appropriate library generation.

## Platform Support

### macOS
- **Library Format**: `.dylib` (dynamic library)
- **Versioning**: `libhmm.2.0.0.dylib` with symbolic links
- **RPATH**: Uses `@rpath` for flexible library loading
- **Dependencies**: Automatically finds Homebrew packages (`/usr/local`, `/opt/homebrew`)
- **Math Library**: Integrated into system library (no explicit `-lm` needed)

### Linux
- **Library Format**: `.so` (shared object)
- **Versioning**: `libhmm.so.2.0.0` with symbolic links
- **RPATH**: Uses `$ORIGIN` for relative library loading
- **Dependencies**: Searches standard package locations (`/usr`, `/usr/local`)
- **Math Library**: Explicitly links math library (`-lm`)

### Generic Unix
- **Library Format**: Platform default shared library
- **Fallback**: Uses standard Unix conventions

## Build Instructions

### Standard Build (All Platforms)
```bash
mkdir build && cd build
cmake ..
make
ctest  # Run tests
```

### Platform-Specific Features

#### macOS
- Automatically detects Homebrew Boost installations
- Generates proper `.dylib` files with macOS conventions
- Supports both Intel (`/usr/local`) and Apple Silicon (`/opt/homebrew`) Homebrew

#### Linux
- Searches standard package manager locations
- Generates `.so` files with proper versioning
- Links math library explicitly for better compatibility

## Configuration Summary

When you run `cmake ..`, you'll see a configuration summary:

```
=== libhmm Configuration Summary ===
Platform: macOS
Version: 2.0.0
C++ Standard: 17
Build Type: Release
Shared Libraries: ON
Library Type: .dylib (macOS dynamic library)
Examples: ON
Tests: ON
Compiler: AppleClang
```

## Cross-Compilation

The CMake configuration automatically:

1. **Detects the target platform** (`APPLE`, `LINUX`, or generic `UNIX`)
2. **Sets appropriate compiler flags** (`-fPIC`, platform-specific linker flags)
3. **Configures library naming** (`.dylib` vs `.so`)
4. **Handles dependencies** (Boost location, math library linking)
5. **Sets up RPATH** for proper library loading

## Dependencies

- **C++17 compatible compiler** (GCC 7+, Clang 5+, AppleClang 12+)
- **CMake 3.15+**
- **Boost** (any recent version)
- **Google Test** (optional, for tests)

## Build Options

```bash
# Build without examples
cmake -DBUILD_EXAMPLES=OFF ..

# Build without tests
cmake -DBUILD_TESTS=OFF ..

# Build static libraries instead of shared
cmake -DBUILD_SHARED_LIBS=OFF ..

# Specify build type
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Testing Cross-Platform

The comprehensive test suite runs on both platforms:
- 129 total tests across 7 test suites
- Platform-agnostic C++17 code
- Automatic platform-specific adjustments for edge cases

## Installation

```bash
# Install to system directories
make install

# Or specify custom prefix
cmake -DCMAKE_INSTALL_PREFIX=/custom/path ..
make install
```

This installs:
- Libraries to `lib/`
- Headers to `include/libhmm/`
- CMake config files to `lib/cmake/libhmm/`
