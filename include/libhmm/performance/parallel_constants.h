#ifndef LIBHMM_PERFORMANCE_PARALLEL_CONSTANTS_H_
#define LIBHMM_PERFORMANCE_PARALLEL_CONSTANTS_H_

#include <cstddef>

namespace libhmm {
namespace performance {

/**
 * @brief Parallel processing optimization constants
 * 
 * This header defines constants specific to parallel processing optimizations
 * using thread pools. These constants complement the SIMD constants already
 * defined in common.h and are used by calculators, trainers, and other
 * performance-critical components that utilize parallel processing.
 * 
 * Note: SIMD-related constants are defined in common.h under constants::simd
 */
namespace parallel {

    /**
     * @brief Minimum number of states required to use parallel processing in calculators
     * 
     * Below this threshold, the overhead of thread pool task submission
     * and synchronization typically outweighs the benefits of parallelization
     * for Forward-Backward and Viterbi calculations.
     * 
     * Optimized to base-2 multiple (512) based on empirical performance testing
     * which showed 10-12% performance improvements for large problems.
     */
    inline constexpr std::size_t MIN_STATES_FOR_CALCULATOR_PARALLEL = 512;
    
    /**
     * @brief Minimum number of states for parallel emission probability computation
     * 
     * Emission probability calculations can be parallelized at a lower threshold
     * since they involve more computation per state (distribution calculations).
     * Used in both scaled-SIMD and log-SIMD calculators.
     * 
     * Optimized to base-2 multiple (256) based on empirical performance testing
     * for better cache alignment and SIMD efficiency.
     */
    inline constexpr std::size_t MIN_STATES_FOR_EMISSION_PARALLEL = 256;
    
    /**
     * @brief Default grain size for parallel calculator loops
     * 
     * This determines the minimum number of work items assigned to each thread
     * in parallel forward/backward calculations. Optimized to balance overhead
     * with load balancing effectiveness.
     * 
     * Optimized to base-2 multiple (64) based on empirical performance testing
     * for better cache line alignment and SIMD register efficiency.
     */
    inline constexpr std::size_t CALCULATOR_GRAIN_SIZE = 64;
    
    /**
     * @brief Grain size for simple parallel operations (scaling, element-wise ops)
     * 
     * For very simple operations like probability scaling or element-wise
     * arithmetic, a smaller grain size may be appropriate.
     * 
     * Optimized to base-2 multiple (32) based on empirical performance testing
     * for better cache alignment and SIMD efficiency.
     */
    inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 32;
    
    /**
     * @brief Minimum sequence length for parallel training algorithms
     * 
     * Training algorithms like Baum-Welch benefit from parallelization when
     * processing long observation sequences. Short sequences have too little
     * work to justify parallel overhead.
     */
    inline constexpr std::size_t MIN_SEQUENCE_LENGTH_FOR_PARALLEL = 100;
    
    /**
     * @brief Minimum number of observation sequences for parallel training
     * 
     * When training on multiple observation sequences, parallelization
     * across sequences becomes beneficial above this threshold.
     */
    inline constexpr std::size_t MIN_SEQUENCES_FOR_PARALLEL = 4;
    
    /**
     * @brief Minimum total work units for parallel training
     * 
     * Training benefits from parallelization when the total computational
     * work (states × sequence_length × iterations) exceeds this threshold.
     */
    inline constexpr std::size_t MIN_TOTAL_WORK_FOR_TRAINING_PARALLEL = 10000;
    
    /**
     * @brief Grain size for parallel training operations
     * 
     * Grain size optimized for training algorithms which typically involve
     * more computation per work item than calculators.
     */
    inline constexpr std::size_t TRAINING_GRAIN_SIZE = 25;
    
    /**
     * @brief Maximum grain size to ensure good load balancing
     * 
     * Prevents any single thread from getting too much work, which could
     * lead to load imbalancing on systems with many cores.
     */
    inline constexpr std::size_t MAX_GRAIN_SIZE = 1000;
    
    /**
     * @brief Minimum work per thread in parallel reductions
     * 
     * For parallel sum reductions and similar operations, ensure
     * each thread has at least this much work to justify overhead.
     */
    inline constexpr std::size_t MIN_WORK_PER_THREAD = 100;
    
    /**
     * @brief Batch size for parallel processing of observation sequences
     * 
     * When processing multiple observation sequences in training,
     * this determines the batch size for parallel processing.
     */
    inline constexpr std::size_t OBSERVATION_BATCH_SIZE = 16;

} // namespace parallel
} // namespace performance
} // namespace libhmm

#endif // LIBHMM_PERFORMANCE_PARALLEL_CONSTANTS_H_
