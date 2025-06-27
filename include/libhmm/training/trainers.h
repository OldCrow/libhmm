#ifndef LIBHMM_TRAINERS_H_
#define LIBHMM_TRAINERS_H_

/**
 * @file trainers.h
 * @brief Convenience header that includes all libhmm HMM trainers
 * 
 * This header provides a single include point for all trainer implementations
 * available in libhmm. It follows the standard library convention of providing
 * umbrella headers for related functionality.
 * 
 * Usage:
 * @code
 * #include "libhmm/training/trainers.h"
 * 
 * // All trainers are now available:
 * BaumWelchTrainer trainer(hmm);
 * ViterbiTrainer viterbiTrainer(hmm);
 * ScaledBaumWelchTrainer scaledTrainer(hmm);
 * @endcode
 * 
 * @note For better compilation times, consider including only the specific
 *       trainer headers you need in performance-critical applications.
 */

// Base trainer interface
#include "libhmm/training/hmm_trainer.h"

// Baum-Welch based trainers
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/scaled_baum_welch_trainer.h"

// Viterbi-based trainers
#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/training/robust_viterbi_trainer.h"

// K-means based trainers
#include "libhmm/training/segmented_kmeans_trainer.h"

// Training utilities and support classes
#include "libhmm/training/centroid.h"
#include "libhmm/training/cluster.h"
#include "libhmm/common/basic_matrix3d.h"
#include "libhmm/common/optimized_matrix3d.h"

/**
 * @namespace libhmm
 * @brief All trainers are available in the libhmm namespace
 * 
 * After including this header, all trainer classes are available:
 * 
 * **Baum-Welch Trainers:**
 * - BaumWelchTrainer: Standard Baum-Welch algorithm for HMM parameter estimation
 * - ScaledBaumWelchTrainer: Numerically stable version using scaling
 * 
 * **Viterbi Trainers:**
 * - ViterbiTrainer: Viterbi training using most likely state sequences
 * - RobustViterbiTrainer: Enhanced Viterbi training with robustness features
 * 
 * **Clustering Trainers:**
 * - SegmentedKmeansTrainer: K-means clustering for sequence segmentation
 * 
 * **Training Utilities:**
 * - Centroid: Cluster centroid computation
 * - Cluster: Clustering support classes
 * - BasicMatrix3D: 3D matrix operations for training
 * - OptimizedMatrix3D: Performance-optimized 3D matrix operations
 */

// Trainer count for compile-time verification
namespace libhmm {
    namespace detail {
        /// Total number of concrete trainer types (excluding base class)
        inline constexpr std::size_t TRAINER_COUNT = 5;
        
        /// Number of Baum-Welch trainer variants
        inline constexpr std::size_t BAUM_WELCH_TRAINER_COUNT = 2;
        
        /// Number of Viterbi trainer variants
        inline constexpr std::size_t VITERBI_TRAINER_COUNT = 2;
        
        /// Number of clustering trainer types
        inline constexpr std::size_t CLUSTERING_TRAINER_COUNT = 1;
        
        static_assert(BAUM_WELCH_TRAINER_COUNT + VITERBI_TRAINER_COUNT + CLUSTERING_TRAINER_COUNT == TRAINER_COUNT,
                     "Trainer counts must be consistent");
    }
}

#endif // LIBHMM_TRAINERS_H_
