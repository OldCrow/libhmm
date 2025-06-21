#include "libhmm/training/segmented_kmeans_trainer.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <algorithm>
#include <iostream>

namespace libhmm
{

// Clusters implementation
Clusters::Clusters(std::size_t k, const ObservationSet& observations) {
    if (k == 0) {
        throw std::invalid_argument("Number of clusters must be greater than zero");
    }
    if (observations.size() == 0) {
        throw std::invalid_argument("Observations cannot be empty");
    }
    
    clusters_.resize(k);
    
    // Simple initialization: divide observations into k roughly equal groups
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

const std::vector<std::size_t>& Clusters::cluster(std::size_t clusterNb) const {
    if (clusterNb >= clusters_.size()) {
        throw std::out_of_range("Invalid cluster number");
    }
    return clusters_[clusterNb];
}

void Clusters::remove(std::size_t observation, std::size_t clusterNb) {
    if (clusterNb >= clusters_.size()) {
        throw std::out_of_range("Invalid cluster number");
    }
    
    auto& clusterVec = clusters_[clusterNb];
    auto it = std::find(clusterVec.begin(), clusterVec.end(), observation);
    if (it == clusterVec.end()) {
        throw std::runtime_error("Observation not found in expected cluster");
    }
    
    clusterVec.erase(it);
    clustersHash_[observation].setClusterNb(static_cast<std::size_t>(-1)); // Mark as removed
}

void Clusters::put(std::size_t observation, std::size_t clusterNb) {
    if (clusterNb >= clusters_.size()) {
        throw std::out_of_range("Invalid cluster number");
    }
    
    clustersHash_[observation].setClusterNb(clusterNb);
    clusters_[clusterNb].push_back(observation);
}

// SegmentedKMeansTrainer implementation
void SegmentedKMeansTrainer::validateDiscreteDistributions() const {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    for (std::size_t i = 0; i < numStates; ++i) {
        const auto* dist = dynamic_cast<const DiscreteDistribution*>(hmm_->getProbabilityDistribution(static_cast<int>(i)));
        if (!dist) {
            throw std::runtime_error("SegmentedKMeansTrainer requires all states to have DiscreteDistribution");
        }
    }
}

void SegmentedKMeansTrainer::train() {
    ObservationSet observations = flattenObservationLists(obsLists_);
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    clusters_ = Clusters(numStates, observations);

    do {
        iterate();
    } while (!isTerminated());
}

void SegmentedKMeansTrainer::iterate() {
    learnPi();
    learnTrans();
    learnEmis();

    terminated_ = !optimizeCluster(); // optimizeCluster returns true if modified
}

void SegmentedKMeansTrainer::learnPi() {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    Vector pi(numStates);

    clear_vector(pi);
    
    for (const auto& obsList : obsLists_) {
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
    
    hmm_->setPi(pi);
}
void SegmentedKMeansTrainer::learnTrans() {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    Matrix trans(numStates, numStates);

    clear_matrix(trans);

    for (const auto& obsList : obsLists_) {
        // List is too short
        if (obsList.size() < 2) continue;

        auto firstState = clusters_.clusterNumber(static_cast<std::size_t>(obsList(0)));
        
        for (std::size_t j = 1; j < obsList.size(); ++j) {
            const auto secondState = clusters_.clusterNumber(static_cast<std::size_t>(obsList(j)));
            trans(firstState, secondState) += 1.0;
            firstState = secondState;
        }
    }

    // Normalize transition matrix
    for (std::size_t j = 0; j < numStates; ++j) {
        double sum = 0.0;
        
        for (std::size_t k = 0; k < numStates; ++k) {
            sum += trans(j, k);
        }

        if (sum == 0.0) {
            // Uniform distribution if no transitions observed
            const double uniform = 1.0 / static_cast<double>(numStates);
            for (std::size_t k = 0; k < numStates; ++k) {
                trans(j, k) = uniform;
            }
        } else {
            for (std::size_t k = 0; k < numStates; ++k) {
                trans(j, k) = trans(j, k) / sum;
            }
        }
    }

    hmm_->setTrans(trans);
}

void SegmentedKMeansTrainer::learnEmis() {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    
    // Update emission probabilities for discrete distributions
    for (std::size_t i = 0; i < numStates; ++i) {
        auto* dist = dynamic_cast<DiscreteDistribution*>(hmm_->getProbabilityDistribution(static_cast<int>(i)));
        if (!dist) continue;
        
        const auto& clusterObservations = clusters_.cluster(i);
        const auto numSymbols = dist->getNumSymbols();
        
        if (clusterObservations.empty()) {
            // Use uniform distribution if no observations
            const double uniform = 1.0 / static_cast<double>(numSymbols);
            for (std::size_t j = 0; j < numSymbols; ++j) {
                dist->setProbability(static_cast<int>(j), uniform);
            }
        } else {
            // Count occurrences of each symbol
            std::vector<std::size_t> counts(numSymbols, 0);
            for (const auto& obs : clusterObservations) {
                if (obs < numSymbols) {
                    counts[obs]++;
                }
            }
            
            // Set probabilities based on counts
            const double totalObs = static_cast<double>(clusterObservations.size());
            for (std::size_t j = 0; j < numSymbols; ++j) {
                const double prob = static_cast<double>(counts[j]) / totalObs;
                dist->setProbability(static_cast<int>(j), prob > 0.0 ? prob : 1e-10); // Avoid zero probabilities
            }
        }
    }
}

bool SegmentedKMeansTrainer::optimizeCluster() {
    bool modified = false;

    for (const auto& obsList : obsLists_) {
        ViterbiCalculator vc(hmm_, obsList);
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

ObservationSet SegmentedKMeansTrainer::flattenObservationLists(const ObservationLists& observationLists) {
    std::size_t totalObservations = 0;

    // Calculate total number of observations
    for (const auto& obsList : observationLists) {
        totalObservations += obsList.size();
    }
    
    ObservationSet flattened(totalObservations);
    std::size_t k = 0;
    
    // Flatten all observation lists into one
    for (const auto& obsList : observationLists) {
        for (std::size_t j = 0; j < obsList.size(); ++j) {
            flattened(k) = obsList(j);
            ++k;
        }
    }
    
    return flattened;
}

}
