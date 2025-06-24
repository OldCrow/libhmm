#ifndef VITERBITRAINER_H_
#define VITERBITRAINER_H_

#include <unordered_map>
#include <cassert>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <functional>
#include "libhmm/common/common.h"
#include "libhmm/training/hmm_trainer.h"
#include "libhmm/training/cluster.h"
#include "libhmm/training/centroid.h"
#include "libhmm/distributions/distributions.h"
#include "libhmm/calculators/scaled_forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"

namespace libhmm
{

/// Implements the Segmented k-Means (aka Viterbi) training algorithm.
/// This implementation is based on the description given in Dugad & Desai
/// (1996), which uses logarithms to prevent mathematical operations on
/// very small numbers. Rabiner(1986) does mention this problem, but
/// does not provide a solution.
class ViterbiTrainer : public HmmTrainer
{
protected:
    /// Training termination flag
    bool terminated_{false};

    /// Array of clusters for k-means algorithm
    std::unique_ptr<Cluster[]> clusters_;

    /// Modern type-safe association key for observation-cluster mapping
    struct AssociationKey {
        std::size_t sequence_id;    ///< Index of observation sequence
        std::size_t observation_id; ///< Index of observation within sequence
        
        /// Equality operator for C++17 compatibility
        bool operator==(const AssociationKey& other) const noexcept {
            return sequence_id == other.sequence_id && observation_id == other.observation_id;
        }
        
        /// Inequality operator for C++17 compatibility
        bool operator!=(const AssociationKey& other) const noexcept {
            return !(*this == other);
        }
        
        /// Less-than operator for ordered containers (C++17 compatible)
        bool operator<(const AssociationKey& other) const noexcept {
            if (sequence_id != other.sequence_id) {
                return sequence_id < other.sequence_id;
            }
            return observation_id < other.observation_id;
        }
    };
    
    /// Hash function for AssociationKey to enable unordered_map usage
    struct AssociationKeyHash {
        std::size_t operator()(const AssociationKey& key) const noexcept {
            // Combine two hash values using a standard technique
            std::size_t h1 = std::hash<std::size_t>{}(key.sequence_id);
            std::size_t h2 = std::hash<std::size_t>{}(key.observation_id);
            return h1 ^ (h2 << 1); // Simple but effective hash combination
        }
    };

    /// Modern type-safe map for observation-cluster associations
    /// Maps (sequence_id, observation_id) -> cluster_index
    std::unordered_map<AssociationKey, std::size_t, AssociationKeyHash> associations_;

    /// Number of changes made to the clusters during training
    std::size_t numChanges_{0};

    /// Concatenates all the ObservationSets in the ObservationLists collection
    /// @return Flattened observation set
    ObservationSet flattenObservationLists();

    /// Sorts an ObservationSet using quicksort algorithm
    /// @param set ObservationSet to sort
    void quickSort(ObservationSet& set);
    
    /// Internal quicksort implementation
    /// @param set ObservationSet to sort
    /// @param left Left boundary for sorting
    /// @param right Right boundary for sorting
    void quickSort(ObservationSet& set, int left, int right);

    /// Returns the closest Cluster to an Observation, based on its Centroid
    /// @param o Observation to find closest cluster for
    /// @return Index of closest cluster
    std::size_t findClosestCluster(Observation o);

    /// Associates an Observation with a Cluster and records the association
    /// @param i Index of observation set
    /// @param j Index of observation within set
    /// @param c Index of cluster to associate with
    void associateObservation(std::size_t i, std::size_t j, std::size_t c);

    /// Returns cluster index that an observation is associated with
    /// @param i Index of observation set
    /// @param j Index of observation within set
    /// @return Cluster index
    std::size_t findAssociation(std::size_t i, std::size_t j);
    
    /// Computes a Pi vector and assigns it to the HMM
    void calculatePi();

    /// Computes a Transmission matrix and assigns it to the HMM
    void calculateTrans();

    /// Compares a StateSequence with the association list and reassigns if needed
    /// @param sequence State sequence to compare
    /// @param i Index of observation set
    void compareSequence(const StateSequence& sequence, std::size_t i);

public:
    /// Constructor with HMM and observation lists
    /// @param hmm Pointer to the HMM to train (must not be null)
    /// @param obsLists List of observation sequences for training
    /// @throws std::invalid_argument if hmm is null or obsLists is empty
    ViterbiTrainer(Hmm* hmm, const ObservationLists& obsLists)
        : HmmTrainer(hmm, obsLists),
          clusters_(std::make_unique<Cluster[]>(static_cast<std::size_t>(hmm->getNumStates()))) {
    }

    /// Virtual destructor (using RAII, no manual cleanup needed)
    virtual ~ViterbiTrainer() = default;

    /// Implements Segmented k-Means training
    /// Updates HMM parameters using the Viterbi training algorithm
    virtual void train() override;

}; // class ViterbiTrainer

}//namespace

#endif
