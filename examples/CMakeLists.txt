# Examples CMakeLists.txt

# Detect if using LLVM compiler and set appropriate linking flags
# NOTE: LLVM-specific linking temporarily commented out due to runtime issues
#       on macOS 10.15 (2012 MacBook Pro). Using system libc++ for stability.
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER MATCHES "llvm")
#     message(STATUS "Detected LLVM Clang compiler - configuring libc++ linking")
#     set(LIBCXX_LINK_FLAGS "-stdlib=libc++")
#     # Add explicit path to LLVM's libc++ if needed
#     get_filename_component(LLVM_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
#     get_filename_component(LLVM_ROOT_DIR ${LLVM_BIN_DIR} DIRECTORY)
#     set(LLVM_LIBCXX_PATH "${LLVM_ROOT_DIR}/lib/c++")
#     if(EXISTS "${LLVM_LIBCXX_PATH}")
#         set(LIBCXX_LINK_FLAGS "${LIBCXX_LINK_FLAGS} -L${LLVM_LIBCXX_PATH} -Wl,-rpath,${LLVM_LIBCXX_PATH}")
#         message(STATUS "Using LLVM libc++ from: ${LLVM_LIBCXX_PATH}")
#     endif()
# else()
#     set(LIBCXX_LINK_FLAGS "")
# endif()

# Use system libc++ for stability
set(LIBCXX_LINK_FLAGS "")

# Helper function to create example targets with consistent settings
function(add_hmm_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
    
    # Link with hmm library and platform-specific math library
    if(LINUX)
        target_link_libraries(${target_name} hmm m)
    else()
        target_link_libraries(${target_name} hmm)
    endif()
    
    # Apply LLVM-specific linking flags if needed
    if(LIBCXX_LINK_FLAGS)
        set_target_properties(${target_name} PROPERTIES
            LINK_FLAGS "${LIBCXX_LINK_FLAGS}"
        )
    endif()
    
    # Ensure C++20 standard
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD_REQUIRED ON)
endfunction()

# TODO(refactor): All examples need updating to the canonical API (setDistribution / getDistribution,
# removal of ScaledSIMDForwardBackwardCalculator / LogSIMDForwardBackwardCalculator,
# removal of RobustViterbiTrainer). Re-enable each here once its source is updated.

# install(TARGETS ...) placeholder — restore when examples are re-enabled
