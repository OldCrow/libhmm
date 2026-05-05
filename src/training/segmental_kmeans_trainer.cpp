#include "libhmm/training/segmental_kmeans_trainer.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <algorithm>
#include <iostream>

// Segmental K-means (hard-assignment EM) for discrete HMMs.
//
// Algorithm outline per iteration:
//   1. learnPi()   — estimate π from the first observation of each sequence
//   2. learnTrans()— estimate A from consecutive cluster-transition counts
//   3. learnEmis() — estimate B via MLE from hard cluster assignments
//   4. optimizeCluster() — re-run Viterbi; move observations to the decoded
//                          state; return true if any assignment changed
// Convergence: optimizeCluster() returning false (no movement) terminates
// train().
//
// Restriction: all HMM states must use DiscreteDistribution.
// For continuous data, use BaumWelchTrainer instead.

namespace libhmm {

/// Partition observations into k clusters by index position.
/// Observation i is assigned to cluster floor(i * k / N), giving an
/// approximately uniform initial partition without requiring data statistics.
Clusters::Clusters(std::size_t k, const ObservationSet &observations) {
    if (k == 0) {
        throw std::invalid_argument("Number of clusters must be greater than zero");
    }
    if (observations.size() == 0) {
        throw std::invalid_argument("Observations cannot be empty");
    }

    clusters_.resize(k);

    const auto obsSize = observations.size();
    for (std::size_t i = 0; i < obsSize; ++i) {
        const auto clusterIdx = (i * k) / obsSize;
        const auto obsValue = static_cast<std::size_t>(observations(i));

        clusters_[clusterIdx].push_back(obsValue);
        clustersHash_[obsValue] = Value(clusterIdx);
    }
}

std::size_t Clusters::clusterNumber(std::size_t observation) const {
    auto it = clustersHash_.find(observation);
    if (it == clustersHash_.end()) {
        throw std::out_of_range("Observation not found in clusters");
    }
    return it->second.getClusterNb();
}

const std::vector<std::size_t> &Clusters::cluster(std::size_t clusterNb) const {
    if (clusterNb >= clusters_.size()) {
        throw std::out_of_range("Invalid cluster number");
    }
    return clusters_[clusterNb];
}

/// Remove an observation from its current cluster.
/// The clustersHash_ entry is set to SIZE_MAX as a "no cluster" sentinel;
/// the observation is re-assigned via put() before any further access.
void Clusters::remove(std::size_t observation, std::size_t clusterNb) {
    if (clusterNb >= clusters_.size()) {
        throw std::out_of_range("Invalid cluster number");
    }

    auto &clusterVec = clusters_[clusterNb];
    auto it = std::find(clusterVec.begin(), clusterVec.end(), observation);
    if (it == clusterVec.end()) {
        throw std::runtime_error("Observation not found in expected cluster");
    }

    clusterVec.erase(it);
    clustersHash_[observation].setClusterNb(static_cast<std::size_t>(-1));
}

void Clusters::put(std::size_t observation, std::size_t clusterNb) {
    if (clusterNb >= clusters_.size()) {
        throw std::out_of_range("Invalid cluster number");
    }

    clustersHash_[observation].setClusterNb(clusterNb);
    clusters_[clusterNb].push_back(observation);
}

void SegmentalKMeansTrainer::validateDiscreteDistributions() const {
    Hmm &hmm = hmm_ref_.get();
    const auto numStates = static_cast<std::size_t>(hmm.getNumStates());
    for (std::size_t i = 0; i < numStates; ++i) {
        const auto *dist = dynamic_cast<const DiscreteDistribution *>(&hmm.getDistribution(i));
        if (!dist) {
            throw std::runtime_error(
                "SegmentalKMeansTrainer requires all states to have DiscreteDistribution");
        }
    }
}

void SegmentalKMeansTrainer::train() {
    ObservationSet observations = flattenObservationLists(obsLists_);
    const auto numStates = static_cast<std::size_t>(hmm_ref_.get().getNumStates());
    clusters_ = Clusters(numStates, observations);

    do {
        iterate();
    } while (!isTerminated());
}

void SegmentalKMeansTrainer::iterate() {
    learnPi();
    learnTrans();
    learnEmis();

    terminated_ = !optimizeCluster();
}

/// Estimate π: each observation sequence contributes one vote to the cluster
/// that its first observation belongs to. pi[j] = count(first obs in cluster j)
/// / total sequences.
void SegmentalKMeansTrainer::learnPi() {
    Hmm &hmm = hmm_ref_.get();
    const auto numStates = static_cast<std::size_t>(hmm.getNumStates());
    Vector pi(numStates);

    clear_vector(pi);

    for (const auto &obsList : obsLists_) {
        if (!obsList.empty()) {
            const auto firstObs = static_cast<std::size_t>(obsList(0));
            const auto clusterNum = clusters_.clusterNumber(firstObs);
            pi(clusterNum)++;
        }
    }

    const auto totalLists = static_cast<double>(obsLists_.size());
    for (std::size_t j = 0; j < numStates; ++j) {
        pi(j) = pi(j) / totalLists;
    }

    hmm.setPi(pi);
}

/// Estimate A: count consecutive (from_cluster → to_cluster) transitions
/// across all sequences, then row-normalise. Rows with zero counts are set
/// to uniform to avoid a degenerate transition matrix.
void SegmentalKMeansTrainer::learnTrans() {
    Hmm &hmm = hmm_ref_.get();
    const auto numStates = static_cast<std::size_t>(hmm.getNumStates());
    Matrix trans(numStates, numStates);

    clear_matrix(trans);

    for (const auto &obsList : obsLists_) {
        if (obsList.size() < 2)
            continue;

        auto firstState = clusters_.clusterNumber(static_cast<std::size_t>(obsList(0)));

        for (std::size_t j = 1; j < obsList.size(); ++j) {
            const auto secondState = clusters_.clusterNumber(static_cast<std::size_t>(obsList(j)));
            trans(firstState, secondState) += 1.0;
            firstState = secondState;
        }
    }

    for (std::size_t j = 0; j < numStates; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < numStates; ++k)
            sum += trans(j, k);

        if (sum == 0.0) {
            const double uniform = 1.0 / static_cast<double>(numStates);
            for (std::size_t k = 0; k < numStates; ++k)
                trans(j, k) = uniform;
        } else {
            for (std::size_t k = 0; k < numStates; ++k)
                trans(j, k) /= sum;
        }
    }

    hmm.setTrans(trans);
}

/// Estimate B: for each cluster/state, count how often each symbol appears
/// in its observations and divide by the cluster size (MLE). Empty clusters
/// fall back to uniform. A 1e-10 floor prevents exact zeros, which would
/// cause -inf log-probabilities during subsequent Viterbi decoding.
void SegmentalKMeansTrainer::learnEmis() {
    Hmm &hmm = hmm_ref_.get();
    const auto numStates = static_cast<std::size_t>(hmm.getNumStates());

    for (std::size_t i = 0; i < numStates; ++i) {
        auto *dist = dynamic_cast<DiscreteDistribution *>(&hmm.getDistribution(i));
        if (!dist)
            continue;

        const auto &clusterObservations = clusters_.cluster(i);
        const auto numSymbols = dist->getNumSymbols();

        if (clusterObservations.empty()) {
            const double uniform = 1.0 / static_cast<double>(numSymbols);
            for (std::size_t j = 0; j < numSymbols; ++j) {
                dist->setProbability(static_cast<int>(j), uniform);
            }
        } else {
            std::vector<std::size_t> counts(numSymbols, 0);
            for (const auto &obs : clusterObservations) {
                if (obs < numSymbols) {
                    counts[obs]++;
                }
            }

            const double totalObs = static_cast<double>(clusterObservations.size());
            for (std::size_t j = 0; j < numSymbols; ++j) {
                const double prob = static_cast<double>(counts[j]) / totalObs;
                dist->setProbability(static_cast<int>(j), prob > 0.0 ? prob : 1e-10);
            }
        }
    }
}

/// Re-run Viterbi on every sequence and reassign observations to the decoded
/// state. Returns true if at least one observation moved to a different
/// cluster (i.e., training has not yet converged).
bool SegmentalKMeansTrainer::optimizeCluster() {
    bool modified = false;

    for (const auto &obsList : obsLists_) {
        ViterbiCalculator vc(hmm_ref_.get(), obsList);
        StateSequence states = vc.decode();

        for (std::size_t j = 0; j < obsList.size(); ++j) {
            const auto observation = static_cast<std::size_t>(obsList(j));
            const auto currentCluster = clusters_.clusterNumber(observation);
            const auto newCluster = static_cast<std::size_t>(states(j));

            if (currentCluster != newCluster) {
                modified = true;
                clusters_.remove(observation, currentCluster);
                clusters_.put(observation, newCluster);
            }
        }
    }

    return modified;
}

/// Flatten multiple observation sequences into one contiguous ObservationSet.
/// Used to initialise Clusters, which requires a single ordered set so that
/// the index-based partitioning assigns contiguous observations to the same
/// initial cluster.
ObservationSet
SegmentalKMeansTrainer::flattenObservationLists(const ObservationLists &observationLists) {
    std::size_t totalObservations = 0;

    for (const auto &obsList : observationLists) {
        totalObservations += obsList.size();
    }

    ObservationSet flattened(totalObservations);
    std::size_t k = 0;

    for (const auto &obsList : observationLists) {
        for (std::size_t j = 0; j < obsList.size(); ++j) {
            flattened(k) = obsList(j);
            ++k;
        }
    }

    return flattened;
}

} // namespace libhmm
