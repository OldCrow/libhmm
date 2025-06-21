# Examples CMakeLists.txt

# Basic HMM example (original hmmtest)
add_executable(basic_hmm_example basic_hmm_example.cpp)
target_link_libraries(basic_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(basic_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Install examples
install(TARGETS basic_hmm_example
    RUNTIME DESTINATION bin/examples
)
