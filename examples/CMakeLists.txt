# Examples CMakeLists.txt

# Basic HMM example (original hmmtest)
add_executable(basic_hmm_example basic_hmm_example.cpp)
target_link_libraries(basic_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(basic_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Poisson HMM example (demonstrates count data modeling)
add_executable(poisson_hmm_example poisson_hmm_example.cpp)
target_link_libraries(poisson_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(poisson_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Financial HMM example (Beta and Log-Normal distributions)
add_executable(financial_hmm_example financial_hmm_example.cpp)
target_link_libraries(financial_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(financial_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Reliability HMM example (Weibull and Exponential distributions)
add_executable(reliability_hmm_example reliability_hmm_example.cpp)
target_link_libraries(reliability_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(reliability_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Quality Control HMM example (Binomial and Uniform distributions)
add_executable(quality_control_hmm_example quality_control_hmm_example.cpp)
target_link_libraries(quality_control_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(quality_control_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Economics HMM example (Negative Binomial and Pareto distributions)
add_executable(economics_hmm_example economics_hmm_example.cpp)
target_link_libraries(economics_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(economics_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Queuing Theory HMM example (Poisson, Exponential, Gamma distributions)
add_executable(queuing_theory_hmm_example queuing_theory_hmm_example.cpp)
target_link_libraries(queuing_theory_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(queuing_theory_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Robust Financial Risk HMM example (Student's t-distribution)
add_executable(robust_financial_hmm_example robust_financial_hmm_example.cpp)
target_link_libraries(robust_financial_hmm_example hmm ${Boost_LIBRARIES} m)
target_include_directories(robust_financial_hmm_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Statistical Process Control HMM example (Chi-squared distribution)
add_executable(statistical_process_control_hmm_example statistical_process_control_hmm_example.cpp)
target_link_libraries(statistical_process_control_hmm_example hmm ${Boost_LIBRARIES})
set_property(TARGET statistical_process_control_hmm_example PROPERTY CXX_STANDARD 17)
set_property(TARGET statistical_process_control_hmm_example PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(robust_viterbi_trainer_example robust_viterbi_trainer_example.cpp)
target_link_libraries(robust_viterbi_trainer_example hmm ${Boost_LIBRARIES})
set_property(TARGET robust_viterbi_trainer_example PROPERTY CXX_STANDARD 17)
set_property(TARGET robust_viterbi_trainer_example PROPERTY CXX_STANDARD_REQUIRED ON)

# Swarm Coordination HMM example (Discrete state drone formation control)
add_executable(swarm_coordination_example swarm_coordination_example.cpp)
target_link_libraries(swarm_coordination_example hmm ${Boost_LIBRARIES})
set_property(TARGET swarm_coordination_example PROPERTY CXX_STANDARD 17)
set_property(TARGET swarm_coordination_example PROPERTY CXX_STANDARD_REQUIRED ON)
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
