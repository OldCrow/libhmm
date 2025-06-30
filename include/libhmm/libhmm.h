#ifndef LIBHMM_H_
#define LIBHMM_H_

/**
 * @file libhmm.h
 * @brief Master header for the libhmm Hidden Markov Model library
 * 
 * This header provides a single include point for the complete libhmm library.
 * It includes all major components: core HMM classes, probability distributions,
 * calculation algorithms, training methods, and I/O functionality.
 * 
 * @section usage Basic Usage
 * @code
 * #include "libhmm/libhmm.h"
 * 
 * using namespace libhmm;
 * 
 * // Create HMM with Gaussian distributions
 * Hmm hmm(2);
 * hmm.setProbabilityDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
 * hmm.setProbabilityDistribution(1, std::make_unique<GaussianDistribution>(3.0, 2.0));
 * 
 * // Train with Viterbi algorithm
 * ViterbiTrainer trainer(&hmm, observations);
 * trainer.train();
 * @endcode
 * 
 * @section performance Performance Note
 * For large projects or when compilation time is critical, consider including
 * only the specific headers you need instead of this master header.
 * 
 * @version 2.8.0
 * @author libhmm development team
 */

//==============================================================================
// CORE FOUNDATION
//==============================================================================

/// Core types, constants, and mathematical utilities
#include "libhmm/common/common.h"

/// Main HMM class and state machine implementation
#include "libhmm/hmm.h"

//==============================================================================
// HIGH-PERFORMANCE LINEAR ALGEBRA (OPTIONAL)
//==============================================================================

/// Complete linear algebra with both basic and optimized classes
/// Comment out this line if you only need basic linear algebra functionality
/// and want to minimize compilation time and dependencies
#include "libhmm/common/linear_algebra.h"

//==============================================================================
// PROBABILITY DISTRIBUTIONS
//==============================================================================

/// All probability distributions (discrete and continuous) with type traits
#include "libhmm/distributions/distributions.h"

//==============================================================================
// CALCULATION ALGORITHMS
//==============================================================================

/// Forward-Backward, Viterbi, and SIMD-optimized calculators
#include "libhmm/calculators/calculators.h"

//==============================================================================
// TRAINING ALGORITHMS
//==============================================================================

/// All HMM training methods with automatic selection traits
#include "libhmm/training/trainers.h"

//==============================================================================
// INPUT/OUTPUT
//==============================================================================

/// File I/O and XML serialization
#include "libhmm/io/file_io_manager.h"
#include "libhmm/io/xml_file_reader.h"
#include "libhmm/io/xml_file_writer.h"

//==============================================================================
// PERFORMANCE OPTIMIZATION
//==============================================================================

/// SIMD support detection and vectorized operations (includes simd_platform.h)
#include "libhmm/performance/simd_support.h"

/// Optimized log-space arithmetic for HMM calculations (includes simd_support.h)
#include "libhmm/performance/log_space_ops.h"

/// Parallel processing constants and optimization thresholds
#include "libhmm/performance/parallel_constants.h"

/// Basic thread pool for parallel processing
#include "libhmm/performance/thread_pool.h"

/// Advanced work-stealing thread pool for high-performance computing
#include "libhmm/performance/work_stealing_pool.h"

/// Performance benchmarking and profiling utilities
#include "libhmm/performance/benchmark.h"

//==============================================================================
// CONVENIENCE UTILITIES
//==============================================================================

/// Pre-configured two-state HMM for common use cases
#include "libhmm/two_state_hmm.h"

/**
 * @namespace libhmm
 * @brief Main namespace for all libhmm functionality
 * 
 * The libhmm namespace contains all classes, functions, and utilities
 * provided by the Hidden Markov Model library. Key components include:
 * 
 * - **Core Classes**: Hmm, ObservationSet, StateSequence
 * - **Distributions**: 14+ probability distributions for discrete and continuous data
 * - **Calculators**: Forward-Backward and Viterbi algorithms with SIMD optimization
 * - **Trainers**: Baum-Welch, Viterbi, and clustering-based parameter estimation
 * - **I/O**: XML file reading/writing and general file management
 * - **Performance**: SIMD detection, thread pools, and numerical stability utilities
 */

#endif // LIBHMM_H_
