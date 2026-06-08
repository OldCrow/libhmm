#pragma once

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
 * hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
 * hmm.setDistribution(1, std::make_unique<GaussianDistribution>(3.0, 2.0));
 *
 * // Train with Viterbi algorithm
 * ViterbiTrainer trainer(hmm, observations);
 * trainer.train();
 * @endcode
 *
 * @section performance Performance Note
 * For large projects or when compilation time is critical, consider including
 * only the specific headers you need instead of this master header.
 *
 * @version 3.7.0
 * @author libhmm development team
 */

//==============================================================================
// CORE FOUNDATION
//==============================================================================

/// Core types, constants, and mathematical utilities.
/// linalg types (Matrix, Vector, ObservationSet, etc.) are provided
/// transitively through hmm.h → linalg/linalg_types.h.
#include "libhmm/common/common.h"

/// Main HMM class and state machine implementation
#include "libhmm/hmm.h"

//==============================================================================
// PROBABILITY DISTRIBUTIONS
//==============================================================================

/// All probability distributions: 16 scalar (discrete + continuous) and
/// 3 multivariate (Phase G) — see distributions/distributions.h for the list.
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

/// JSON serialization/deserialization — the recommended format for new code.
/// Provides to_json(), from_json(), save_json(), and load_json() free functions.
#include "libhmm/io/hmm_json.h"

/// File I/O utilities
/// reading existing .xml files).
#include "libhmm/io/file_io_manager.h"
#include "libhmm/io/xml_file_reader.h"
#include "libhmm/io/xml_file_writer.h"

//==============================================================================
// CONVENIENCE UTILITIES
//==============================================================================

/**
 * @namespace libhmm
 * @brief Main namespace for all libhmm functionality
 *
 * The libhmm namespace contains all classes, functions, and utilities
 * provided by the Hidden Markov Model library. Key components include:
 *
 * - **Core Classes**: Hmm (scalar), HmmMV (multivariate), ObservationSet, ObservationMatrix
 * - **Distributions**: 16 scalar + 3 multivariate emission distributions (v4)
 * - **Calculators**: Forward-Backward and Viterbi algorithms with SIMD optimization
 * - **Trainers**: Baum-Welch, Viterbi, and clustering-based parameter estimation
 * - **I/O**: XML file reading/writing and general file management
 * - **Performance**: SIMD detection, thread pools, and numerical stability utilities
 */
