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
    
    # Ensure C++17 standard
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD_REQUIRED ON)
endfunction()

# Basic HMM example (original hmmtest)
add_hmm_example(basic_hmm_example basic_hmm_example.cpp)

# Poisson HMM example (demonstrates count data modeling)
add_hmm_example(poisson_hmm_example poisson_hmm_example.cpp)

# Financial HMM example (Beta and Log-Normal distributions)
add_hmm_example(financial_hmm_example financial_hmm_example.cpp)

# Reliability HMM example (Weibull and Exponential distributions)
add_hmm_example(reliability_hmm_example reliability_hmm_example.cpp)

# Quality Control HMM example (Binomial and Uniform distributions)
add_hmm_example(quality_control_hmm_example quality_control_hmm_example.cpp)

# Economics HMM example (Negative Binomial and Pareto distributions)
add_hmm_example(economics_hmm_example economics_hmm_example.cpp)

# Queuing Theory HMM example (Poisson, Exponential, Gamma distributions)
add_hmm_example(queuing_theory_hmm_example queuing_theory_hmm_example.cpp)

# Robust Financial Risk HMM example (Student's t-distribution)
add_hmm_example(robust_financial_hmm_example robust_financial_hmm_example.cpp)

# Statistical Process Control HMM example (Chi-squared distribution)
add_hmm_example(statistical_process_control_hmm_example statistical_process_control_hmm_example.cpp)

# Robust Viterbi Trainer example
add_hmm_example(robust_viterbi_trainer_example robust_viterbi_trainer_example.cpp)

# Swarm Coordination HMM example (Discrete state drone formation control)
add_hmm_example(swarm_coordination_example swarm_coordination_example.cpp)
# Install examples
install(TARGETS 
    basic_hmm_example 
    poisson_hmm_example 
    financial_hmm_example 
    reliability_hmm_example 
    quality_control_hmm_example 
    economics_hmm_example 
    queuing_theory_hmm_example
    robust_financial_hmm_example
    statistical_process_control_hmm_example
    robust_viterbi_trainer_example
    swarm_coordination_example
    RUNTIME DESTINATION bin/examples
)
