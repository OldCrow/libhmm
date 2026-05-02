#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build}"

if [[ $# -gt 0 ]]; then
    shift
fi

if ! command -v xcrun >/dev/null 2>&1; then
    echo "error: xcrun not found. Install Xcode Command Line Tools first." >&2
    exit 1
fi

CC_BIN="$(xcrun --sdk macosx --find clang)"
CXX_BIN="$(xcrun --sdk macosx --find clang++)"
SYSROOT="$(xcrun --sdk macosx --show-sdk-path)"

echo "Configuring libhmm for macOS Catalina compatibility"
echo "  Repo:    ${REPO_ROOT}"
echo "  Build:   ${BUILD_DIR}"
echo "  CC:      ${CC_BIN}"
echo "  CXX:     ${CXX_BIN}"
echo "  Sysroot: ${SYSROOT}"

env -u CC \
    -u CXX \
    -u CFLAGS \
    -u CXXFLAGS \
    -u CPPFLAGS \
    -u LDFLAGS \
    -u SDKROOT \
    -u MACOSX_DEPLOYMENT_TARGET \
    -u CPATH \
    -u CPLUS_INCLUDE_PATH \
    -u LIBRARY_PATH \
    -u DYLD_LIBRARY_PATH \
    -u PKG_CONFIG_PATH \
    cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
        -DCMAKE_C_COMPILER="${CC_BIN}" \
        -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
        -DCMAKE_OSX_SYSROOT="${SYSROOT}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
        "$@"

echo "Catalina-safe configure complete."
