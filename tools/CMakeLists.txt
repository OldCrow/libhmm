# libhmm tools — performance inspection and validation utilities.
# These are standalone executables built against the hmm library.
# They are NOT registered with CTest; run them manually from build/tools/.

# Helper: create a tool executable with standard settings
function(add_hmm_tool TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

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

# =============================================================================
# Moved from tests/performance/ (clean — no removed-API dependencies)
# =============================================================================
add_hmm_tool(analyze_overhead  analyze_overhead.cpp)
add_hmm_tool(debug_parallel    debug_parallel.cpp)

# =============================================================================
# New tools (Phase 4.5.2)
# =============================================================================
add_hmm_tool(simd_inspection   simd_inspection.cpp)
add_hmm_tool(batch_performance batch_performance.cpp)
add_hmm_tool(hmm_validator     hmm_validator.cpp)

install(TARGETS
    analyze_overhead
    debug_parallel
    simd_inspection
    batch_performance
    hmm_validator
    RUNTIME DESTINATION bin/tools
    COMPONENT tools
)

message(STATUS "Tools: analyze_overhead debug_parallel simd_inspection batch_performance hmm_validator")
