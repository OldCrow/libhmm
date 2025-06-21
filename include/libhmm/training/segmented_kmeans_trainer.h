#ifndef SEGMENTEDKMEANSTRAINER_H_
#define SEGMENTEDKMEANSTRAINER_H_

#include <map>
#include <vector>
#include <memory>
#include <stdexcept>
#include "libhmm/common/common.h"
#include "libhmm/training/hmm_trainer.h"
#include "libhmm/calculators/viterbi_calculator.h"

namespace libhmm
{

/// Modern cluster management class for segmented k-means training
/// Manages the mapping between observations and cluster assignments
class Clusters
{
private:
    /// Internal value class to store cluster numbers
    class Value {
    private:
        std::size_t clusterNb_;

    public:
        /// Default constructor
        Value() noexcept : clusterNb_{0} {}
        
        /// Constructor with cluster number
        /// @param clusterNb Initial cluster number
        explicit Value(std::size_t clusterNb) noexcept : clusterNb_{clusterNb} {}
        
        /// Set cluster number
        /// @param clusterNb New cluster number
        void setClusterNb(std::size_t clusterNb) noexcept {
            clusterNb_ = clusterNb;
        }
        
        /// Get cluster number
        /// @return Current cluster number
        std::size_t getClusterNb() const noexcept {
            return clusterNb_;
        }
    };
    
    std::map<std::size_t, Value> clustersHash_;
    std::vector<std::vector<std::size_t>> clusters_;

public:
    /// Default constructor
    Clusters() = default;

    /// Constructor with k-means initialization
    /// @param k Number of clusters
    /// @param observations Observation list for clustering
    /// @throws std::invalid_argument if k is zero or observations is empty
    Clusters(std::size_t k, const ObservationSet& observations);

    /// Check if observation is in specified cluster
    /// @param observation Observation index
    /// @param clusterNb Cluster number to check
    /// @return True if observation is in the cluster
    bool inCluster(std::size_t observation, std::size_t clusterNb) const {
        return clusterNumber(observation) == clusterNb;
    }
    
    /// Get cluster number for observation
    /// @param observation Observation index
    /// @return Cluster number
    /// @throws std::out_of_range if observation not found
    std::size_t clusterNumber(std::size_t observation) const;
    
    /// Get all observations in a cluster
    /// @param clusterNb Cluster number
    /// @return Vector of observation indices in the cluster
    /// @throws std::out_of_range if cluster number is invalid
    const std::vector<std::size_t>& cluster(std::size_t clusterNb) const;
    
    /// Remove observation from its current cluster
    /// @param observation Observation index to remove
    /// @param clusterNb Expected current cluster number
    /// @throws std::runtime_error if observation not found in expected cluster
    void remove(std::size_t observation, std::size_t clusterNb);
    
    /// Add observation to a cluster
    /// @param observation Observation index to add
    /// @param clusterNb Target cluster number
    /// @throws std::out_of_range if cluster number is invalid
    void put(std::size_t observation, std::size_t clusterNb);
    
    /// Get number of clusters
    /// @return Number of clusters
    std::size_t size() const noexcept { return clusters_.size(); }
}; //class Clusters

/// Modern implementation of the segmented K-Means (Viterbi) learning algorithm for HMMs
/// Uses k-means clustering combined with Viterbi decoding for HMM parameter estimation
class SegmentedKMeansTrainer : public HmmTrainer
{
private:
    bool terminated_{false};
    Clusters clusters_;
    
    /// Flatten observation lists into a single observation set
    /// @param observationLists Vector of observation sequences
    /// @return Single flattened observation set
    ObservationSet flattenObservationLists(const ObservationLists& observationLists);
    
    /// Perform one iteration of the segmented k-means algorithm
    void iterate();
    
    /// Learn initial state probabilities (pi vector)
    void learnPi();
    
    /// Learn transition probabilities
    void learnTrans();
    
    /// Learn emission probabilities (for discrete distributions)
    void learnEmis();
    
    /// Optimize cluster assignments using Viterbi decoding
    /// @return True if any assignments changed, false if converged
    bool optimizeCluster();
    
    /// Validate that HMM has discrete distributions
    /// @throws std::runtime_error if HMM doesn't have discrete distributions
    void validateDiscreteDistributions() const;

public:
    /// Constructor with HMM and observation lists
    /// @param hmm Pointer to the HMM to train (must not be null)
    /// @param obsLists List of observation sequences for training
    /// @throws std::invalid_argument if hmm is null or obsLists is empty
    SegmentedKMeansTrainer(Hmm* hmm, const ObservationLists& obsLists)
        : HmmTrainer(hmm, obsLists) {
        validateDiscreteDistributions();
    }

    /// Virtual destructor
    virtual ~SegmentedKMeansTrainer() = default;

    /// Execute segmented k-means training algorithm
    /// Updates HMM parameters using k-means clustering and Viterbi decoding
    virtual void train() override;

    /// Check if training has converged
    /// @return True if training has terminated
    bool isTerminated() const noexcept {
        return terminated_;
    }

};//class SegmentedKMeansTrainer


}//namespace

#endif
