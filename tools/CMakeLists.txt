# libhmm tools — performance inspection and validation utilities.
# These are standalone executables built against the hmm library.
# They are NOT registered with CTest; run them manually from build/tools/.

# Helper: create a tool executable with standard settings.
# Accepts one or more source files (the first is the canonical one;
# additional sources let tools pull in shared utilities like thread_pool.cpp).
function(add_hmm_tool TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    target_include_directories(${TARGET_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR})

    if(LINUX)
        target_link_libraries(${TARGET_NAME} hmm m)
    else()
        target_link_libraries(${TARGET_NAME} hmm)
    endif()

    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
endfunction()

# Tool-local thread pool (analyze_overhead and debug_parallel only).
# Lives here, not in the library, because no production code uses it.
set(LIBHMM_TOOL_THREADPOOL thread_pool.cpp)

add_hmm_tool(analyze_overhead  analyze_overhead.cpp ${LIBHMM_TOOL_THREADPOOL})
add_hmm_tool(debug_parallel    debug_parallel.cpp   ${LIBHMM_TOOL_THREADPOOL})
add_hmm_tool(simd_inspection   simd_inspection.cpp)
add_hmm_tool(batch_performance batch_performance.cpp)
add_hmm_tool(hmm_validator     hmm_validator.cpp)
add_hmm_tool(hotspot_breakdown hotspot_breakdown.cpp)
add_hmm_tool(fb_contour_sweep  fb_contour_sweep.cpp)
add_hmm_tool(bw_hotspot          bw_hotspot.cpp)
add_hmm_tool(fb_crossover_sweep  fb_crossover_sweep.cpp)
if(LIBHMM_EXPERIMENT_FB_MAX_REDUCE)
    target_compile_definitions(hotspot_breakdown PRIVATE LIBHMM_EXPERIMENT_FB_MAX_REDUCE=1)
    target_compile_definitions(fb_contour_sweep PRIVATE LIBHMM_EXPERIMENT_FB_MAX_REDUCE=1)
endif()
if(LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR)
    target_compile_definitions(hotspot_breakdown PRIVATE LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR=1)
    target_compile_definitions(fb_contour_sweep PRIVATE LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR=1)
endif()

# simd_inspection must be compiled with the same SIMD flags as the distribution
# TUs so that LIBHMM_HAS_AVX512 / AVX2 / NEON are correctly defined and the
# reported ISA and vector width match what getBatchLogProbabilities actually uses.
# fb_crossover_sweep also reports the active ISA; compile it with the same flags
# so its diagnostics match the hot-path TUs it benchmarks.
if(LIBHMM_BEST_SIMD_FLAGS)
    set_source_files_properties(simd_inspection.cpp fb_crossover_sweep.cpp
        PROPERTIES COMPILE_FLAGS "${LIBHMM_BEST_SIMD_FLAGS}")
endif()


message(STATUS "Tools: analyze_overhead debug_parallel simd_inspection batch_performance hmm_validator hotspot_breakdown fb_contour_sweep bw_hotspot")
