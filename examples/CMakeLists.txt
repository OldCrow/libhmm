# Examples CMakeLists.txt

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

    # Ensure C++20 standard
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD_REQUIRED ON)
endfunction()

# Basic HMM example (casino problem, distributions, Viterbi training, file I/O)
add_hmm_example(basic_hmm_example basic_hmm_example.cpp)

# Baum-Welch EM training (primary training algorithm)
add_hmm_example(baum_welch_example baum_welch_example.cpp)

# Viterbi training with TrainingConfig presets
add_hmm_example(viterbi_trainer_example viterbi_trainer_example.cpp)

# Segmental k-means training — scalar (any distribution) and MV warm-start pattern
add_hmm_example(segmental_kmeans_example    segmental_kmeans_example.cpp)
add_hmm_example(segmental_kmeans_mv_example segmental_kmeans_mv_example.cpp)

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

# Multivariate HMM demonstration (v4) — standalone, synthetic data
add_hmm_example(mv_gaussian_example mv_gaussian_example.cpp)

# Real-world ecology benchmark: elk movement vs moveHMM (R)
add_hmm_example(elk_movement_example elk_movement_example.cpp)

# Elk MV: IndependentComponents(Gamma+VonMises) vs DiagGaussian vs FullCovGaussian(log-step,angle) — BIC comparison
add_hmm_example(elk_mv_example elk_mv_example.cpp)

# MV regime: DiagonalGaussian vs FullCovGaussian on two-sector synthetic returns (no data download)
add_hmm_example(mv_regime_example mv_regime_example.cpp)

# Network anomaly POC: MV HMM on CTU-13 Scenario 1 (Neris botnet binetflow)
add_hmm_example(zeek_anomaly_poc zeek_anomaly_poc.cpp)

# Real-world finance benchmark: DAX regime detection vs fHMM (R)
add_hmm_example(dax_regime_example dax_regime_example.cpp)

# Cross-market: S&P 500 regime detection (same model as DAX)
add_hmm_example(sp500_regime_example sp500_regime_example.cpp)

# Seismicity: Poisson HMM on annual earthquake counts (Zucchini & MacDonald)
add_hmm_example(earthquake_example earthquake_example.cpp)

# Meteorology: wind direction regimes via VonMisesDistribution (NOAA ISD)
add_hmm_example(wind_direction_example wind_direction_example.cpp)
