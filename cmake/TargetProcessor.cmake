# TargetProcessor.cmake - derive the architecture the build targets
#
# Sets LIBHMM_TARGET_PROCESSOR and LIBHMM_TARGET_IS_CROSS. Include before any code that branches on
# architecture (Homebrew prefix selection, cmake/SimdDispatch.cmake); such code reads
# LIBHMM_TARGET_PROCESSOR, never CMAKE_SYSTEM_PROCESSOR or `uname -m`.
#
# CMAKE_SYSTEM_PROCESSOR names the target only when a toolchain file sets it. On macOS, where
# CMAKE_OSX_ARCHITECTURES selects the architecture without a toolchain file, it keeps naming the
# build host. Branching on it gave an x86_64 build on Apple Silicon the NEON sources and
# -march=armv8-a ("unknown target CPU 'armv8-a'").
#
# Apple, CMAKE_OSX_ARCHITECTURES unset: the host architecture (CMAKE_SYSTEM_PROCESSOR).
# Apple, one architecture: that architecture.
# Apple, two or more (a universal build): configure fails. The per-ISA sources and flags are chosen
# once per configure and applied per source file, so one build tree cannot give each slice its own
# set. Build each architecture in its own tree and join the results with lipo.
#
# LIBHMM_TARGET_IS_CROSS is ON when the build machine cannot be assumed to run, or to share a CPU
# with, the target: CMAKE_CROSSCOMPILING, or an Apple target architecture other than the host's
# (which Rosetta may run, so CMake does not call it cross-compiling).
set(LIBHMM_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
set(LIBHMM_TARGET_IS_CROSS "${CMAKE_CROSSCOMPILING}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _libhmm_osx_arch_count)
    if(_libhmm_osx_arch_count GREATER 1)
        message(
            FATAL_ERROR
            "libhmm: CMAKE_OSX_ARCHITECTURES=\"${CMAKE_OSX_ARCHITECTURES}\" names more than "
            "one architecture. Universal builds are not supported in a single build tree: "
            "SIMD sources and flags are selected once per configure. Configure one tree per "
            "architecture (e.g. -DCMAKE_OSX_ARCHITECTURES=arm64, then x86_64) and join the "
            "libraries with lipo."
        )
    endif()
    if(CMAKE_OSX_ARCHITECTURES MATCHES "^arm64e?$")
        set(LIBHMM_TARGET_PROCESSOR "arm64")
    elseif(CMAKE_OSX_ARCHITECTURES MATCHES "^x86_64h?$")
        set(LIBHMM_TARGET_PROCESSOR "x86_64")
    else()
        message(
            FATAL_ERROR
            "libhmm: unrecognised CMAKE_OSX_ARCHITECTURES=\"${CMAKE_OSX_ARCHITECTURES}\". "
            "Expected one of arm64, arm64e, x86_64, x86_64h."
        )
    endif()
    if(NOT LIBHMM_TARGET_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
        set(LIBHMM_TARGET_IS_CROSS ON)
    endif()
    unset(_libhmm_osx_arch_count)
endif()
message(STATUS "Target processor: ${LIBHMM_TARGET_PROCESSOR}")
