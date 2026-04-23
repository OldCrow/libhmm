#pragma once

#include <map>
#include <vector>
#include <memory>
#include <stdexcept>
#include "libhmm/common/common.h"
#include "libhmm/training/trainer.h"

namespace libhmm {

/// Cluster bookkeeping used by segmental k-means training.
/// Manages the mapping between observations and cluster assignments.
class Clusters {
private:
    class Value {
    private:
        std::size_t clusterNb_;

    public:
        Value() noexcept : clusterNb_{0} {}
        explicit Value(std::size_t clusterNb) noexcept : clusterNb_{clusterNb} {}

        void setClusterNb(std::size_t clusterNb) noexcept { clusterNb_ = clusterNb; }

        std::size_t getClusterNb() const noexcept { return clusterNb_; }
    };

    std::map<std::size_t, Value> clustersHash_;
    std::vector<std::vector<std::size_t>> clusters_;

public:
    Clusters() = default;
    Clusters(std::size_t k, const ObservationSet &observations);

    bool inCluster(std::size_t observation, std::size_t clusterNb) const {
        return clusterNumber(observation) == clusterNb;
    }

    std::size_t clusterNumber(std::size_t observation) const;
    const std::vector<std::size_t> &cluster(std::size_t clusterNb) const;
    void remove(std::size_t observation, std::size_t clusterNb);
    void put(std::size_t observation, std::size_t clusterNb);
    std::size_t size() const noexcept { return clusters_.size(); }
}; // class Clusters

/// Segmental K-means trainer for HMMs.
/// This follows the academic literature's segmental/segmented k-means training
/// terminology for HMM parameter estimation. Relative to Baum-Welch, it is a
/// hard-assignment training method: state assignments are updated with Viterbi
/// decoding and parameters are re-estimated from those assignments directly.
/// It is often useful for initialization or approximate training before further
/// Baum-Welch refinement on suitable problems.
class SegmentalKMeansTrainer : public Trainer {
private:
    bool terminated_{false};
    Clusters clusters_;

    ObservationSet flattenObservationLists(const ObservationLists &observationLists);
    void iterate();
    void learnPi();
    void learnTrans();
    void learnEmis();
    bool optimizeCluster();
    void validateDiscreteDistributions() const;

public:
    SegmentalKMeansTrainer(Hmm &hmm, const ObservationLists &obsLists) : Trainer(hmm, obsLists) {
        validateDiscreteDistributions();
    }

    SegmentalKMeansTrainer(Hmm *hmm, const ObservationLists &obsLists) : Trainer(hmm, obsLists) {
        validateDiscreteDistributions();
    }

    ~SegmentalKMeansTrainer() override = default;

    void train() override;

    bool isTerminated() const noexcept { return terminated_; }
}; // class SegmentalKMeansTrainer

} // namespace libhmm
