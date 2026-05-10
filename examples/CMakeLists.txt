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

    # Link with hmm library and platform-specific math library.
    # Using hmm_static avoids Windows DLL boundary issues that can corrupt
    # std::vector iterator state across the .exe / .dll boundary in Release
    # builds (the same reason tests link statically; see tests/CMakeLists.txt).
    if(LINUX)
        target_link_libraries(${target_name} hmm_static m)
    else()
        target_link_libraries(${target_name} hmm_static)
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
    if(LIBHMM_LEGACY_WARNING_SUPPRESSIONS)
        target_compile_options(${target_name} PRIVATE ${LIBHMM_LEGACY_WARNING_SUPPRESSIONS})
    endif()
endfunction()

# Basic HMM example (casino problem, distributions, Viterbi training, file I/O)
add_hmm_example(basic_hmm_example basic_hmm_example.cpp)

# Baum-Welch EM training (primary training algorithm)
add_hmm_example(baum_welch_example baum_welch_example.cpp)

# Viterbi training with TrainingConfig presets
add_hmm_example(viterbi_trainer_example viterbi_trainer_example.cpp)

# Segmental k-means training (discrete-only; useful as a Baum-Welch warm-start)
add_hmm_example(segmental_kmeans_example segmental_kmeans_example.cpp)

# Continuous distributions
add_hmm_example(poisson_hmm_example              poisson_hmm_example.cpp)
add_hmm_example(financial_hmm_example            financial_hmm_example.cpp)
add_hmm_example(reliability_hmm_example          reliability_hmm_example.cpp)
add_hmm_example(quality_control_hmm_example      quality_control_hmm_example.cpp)
add_hmm_example(economics_hmm_example            economics_hmm_example.cpp)
add_hmm_example(queuing_theory_hmm_example       queuing_theory_hmm_example.cpp)
add_hmm_example(statistical_process_control_hmm_example statistical_process_control_hmm_example.cpp)
add_hmm_example(student_t_hmm_example            student_t_hmm_example.cpp)

# Discrete state example (drone swarm coordination)
add_hmm_example(swarm_coordination_example swarm_coordination_example.cpp)

# Posterior decoding vs Viterbi (casino HMM)
add_hmm_example(posterior_decoding_example posterior_decoding_example.cpp)

# MAP-EM Baum-Welch with Dirichlet priors
add_hmm_example(map_baum_welch_example map_baum_welch_example.cpp)

# Install all examples
install(TARGETS
    basic_hmm_example
    baum_welch_example
    viterbi_trainer_example
    segmental_kmeans_example
    poisson_hmm_example
    financial_hmm_example
    reliability_hmm_example
    quality_control_hmm_example
    economics_hmm_example
    queuing_theory_hmm_example
    statistical_process_control_hmm_example
    student_t_hmm_example
    swarm_coordination_example
    posterior_decoding_example
    map_baum_welch_example
    RUNTIME DESTINATION bin/examples
)
