#include "libhmm/training/viterbi_trainer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <cstdlib>

namespace libhmm
{

ObservationSet ViterbiTrainer::flattenObservationLists() {
    std::size_t totalObservations = 0;

    for (const auto& obsSet : obsLists_) {
        totalObservations += obsSet.size();
    }

    ObservationSet s(totalObservations);
    std::size_t k = 0; // counter through the flattened list
    
    for (const auto& obsSet : obsLists_) {
        for (std::size_t j = 0; j < obsSet.size(); ++j, ++k) {
            s(k) = obsSet(j);
        }
    }

    assert(k == totalObservations);
    return s;
}

/*
 * Implements Segmented k-Means.
 */
void ViterbiTrainer::train() {
    // Step 1 - Initialize
    ObservationSet s = flattenObservationLists();
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    std::size_t iCounter = 0;

    // Maximum average log probability
    double maxLogP = -DBL_MAX;

    // How many times we've seen the max avg log probability
    std::size_t maxLogPCounter = 0;

    // Seed the RNG here
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::cout << "Training for " << s.size() << " observations" << std::endl;

    // Step 2 - Initialize clusters
    std::cout << "Step 2 " << std::endl;
    std::cout << "  Beginning quicksort..." << std::endl;

    // Enable this sort if you want a REPEATABLE assignment of cluster values.
    quickSort(s);
    
    std::cout << "  Initializing clusters..." << std::endl;
    for (std::size_t i = 0; i < numStates; ++i) {
        // Non-random way: divide into numStates+1 chunks and take the dividers
        const auto j = ((i + 1) * (s.size() - 1)) / (numStates + 1);
        
        // Create and add cluster
        clusters_[i].init(s(j));
        
        std::cout << std::endl << j << " " << clusters_[i].size() 
                  << " " << clusters_[i].getCentroidValue() << std::endl;
    }

    // Step 3: Assign each observation to a cluster
    std::cout << "Step 3" << std::endl;
    auto start_time = std::time(nullptr);
    
    for (std::size_t i = 0; i < obsLists_.size(); ++i) {
        const ObservationSet& os = obsLists_.at(i);

        // Progress bar
        auto current_time = std::time(nullptr);
        std::cout << "\r  Batch add " << i+1 << " out of " << obsLists_.size()
                  << " windows Elapsed time: " << (current_time - start_time) / 60.0 
                  << " ETA: ";
        
        if (i > 0) {
            double eta = ((current_time - start_time) * obsLists_.size()) / (60.0 * i);
            std::cout << std::setprecision(2) << eta << " mins";
        }
        std::cout.flush();
        
        for (std::size_t l = 0; l < os.size(); ++l) {
            // find and add to closest cluster
            auto clustersIndex = findClosestCluster(os(l));
            clusters_[clustersIndex].batchAdd(os(l));

            // record the association
            associateObservation(i, l, clustersIndex);
        }
    }
    std::cout << std::endl;
    
    std::cout << "  Recalculating centroids" << std::endl;
    for (std::size_t i = 0; i < numStates; ++i) {
        clusters_[i].recalculateCentroid();
    }

    do{

        // Steps 4 and 5: Calculate pi vector, transmission matrix;
        std::cout << "Step 4: pi matrix" << std::endl; 
        calculatePi( );
        
        std::cout << "Step 5: A matrix" << std::endl;
        calculateTrans( );

        // Step 6: Fit each cluster's observations to the ProbabilityDistribution
        std::cout << "Step 6 " << std::endl;
        for (std::size_t i = 0; i < numStates; ++i) {
            std::cout << "\r  Curve fitting cluster " << i+1 << "/" << numStates;
            std::cout.flush();
            
            std::vector<Observation> clusterObservations = clusters_[i].getObservations();

            // Fix: Modify the existing distribution in place instead of reassigning
            // The HMM already owns this distribution, so we don't need to transfer ownership
            ProbabilityDistribution* pdist = hmm_->getProbabilityDistribution(static_cast<int>(i));
            pdist->fit(clusterObservations);
            // Note: No need to call setProbabilityDistribution - the object is already owned by HMM
        }
        std::cout << std::endl;

        // Step 7: Viterbi decode each ObservationSet and check for assignment changes
        std::cout << "Step 7" << std::endl;
        numChanges_ = 0;
        std::vector<double> probabilities;
        
        for (std::size_t i = 0; i < obsLists_.size(); ++i) {
            std::cout << "\r  Viterbi decode and Reassignment " << i+1 << "/" << obsLists_.size();
            std::cout.flush();
            
            const ObservationSet& os = obsLists_.at(i);
            ViterbiCalculator vc(hmm_, os);
            StateSequence sequence = vc.decode();
        
            compareSequence(sequence, i);

            ScaledForwardBackwardCalculator sfbc(hmm_, os);
            double logProbability = sfbc.logProbability();
            probabilities.push_back(logProbability);
        }
        std::cout << std::endl;

        GaussianDistribution gd;
        gd.fit(probabilities);

        if (gd.getMean() > maxLogP) {
            maxLogP = gd.getMean();
            maxLogPCounter = 0;
        } else if (gd.getMean() == maxLogP) {
            maxLogPCounter++;
        }
        
        std::cout << "Iteration: " << std::setw(5) << iCounter 
                  << " Total changes this pass: " << std::setw(6) << numChanges_  
                  << " Average log P: " << std::setw(8) << gd.getMean()  
                  << " Standard dev: " << std::setw(8) << gd.getStandardDeviation()
                  << " Max (avg) log P: " << std::setw(8) << maxLogP << std::endl;
        
        terminated_ = (numChanges_ == 0);
        iCounter++;

        // Step 8: redo steps 4 - 7 if there are any changes in the assignments
    } while (!terminated_ && iCounter < MAX_VITERBI_ITERATIONS && maxLogPCounter != 5);
}

void ViterbiTrainer::compareSequence(const StateSequence& sequence, std::size_t setIndex) {
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        auto associationIndex = findAssociation(setIndex, i);

        if (associationIndex != static_cast<std::size_t>(sequence(i))) {
            // association has changed, so update clusters
            numChanges_++;

            // Remove from the old cluster
            clusters_[associationIndex].remove(obsLists_[setIndex](i));

            // Add to the new cluster
            clusters_[static_cast<std::size_t>(sequence(i))].onlineAdd(obsLists_[setIndex](i));

            // Reassociate
            associateObservation(setIndex, i, static_cast<std::size_t>(sequence(i)));
        }
    }
}
   
std::size_t ViterbiTrainer::findClosestCluster(Observation o) {
    double minDistance = DBL_MAX;
    std::size_t index = 0;
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());

    for (std::size_t i = 0; i < numStates; ++i) {
        // Get distance from observation to cluster
        const Cluster& cluster = clusters_[i];
        double tempDistance = cluster.getDistance(o);

        // Smallest distance?
        if (minDistance > tempDistance) {
            minDistance = tempDistance;
            index = i;
        }
    }
    
    return index;   
}

/*
 * Association is different.
 *
 * I'm looking for simple, so the std::map that's in use has a key, value pair.
 * The key in this case is a std::string which is shows that Observation o is
 * the jth observation in the ith ObservationSet in obsLists, so the key is
 * "i-j" and the value is the the index of the Cluster in clusters that 
 * it is associated to.
 */
void ViterbiTrainer::associateObservation(std::size_t i, std::size_t j, std::size_t c) {
    std::ostringstream keyStream;
    keyStream << i << "-" << j;
    std::string key = keyStream.str();
    associations_[key] = static_cast<int>(c);
}

/*
 * Returns clusterIndex that the jth Observation in the ith ObservationSet is
 * associated with
 */
std::size_t ViterbiTrainer::findAssociation(std::size_t i, std::size_t j) {
    std::ostringstream keyStream;
    keyStream << i << "-" << j;
    std::string key = keyStream.str();
    
    auto it = associations_.find(key);
    if (it != associations_.end()) {
        return static_cast<std::size_t>(it->second);
    }
    
    throw std::runtime_error("Association not found for observation");
}

/*
 * Pi, according to Dugad and Desai(1996), is equal to
 *
 *    number of times O_1 of an ObservationSet is associated with cluster i
 *   ------------------------------------------------------------------------
 *                          number of ObservationSets
 *
 * The easiest way I know of to get this information is to go through all the
 * ObservationSets, check which cluster the first Observation is associated
 * with, and count.
 */
void ViterbiTrainer::calculatePi() {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    Vector pi(numStates);

    clear_vector(pi);

    for (std::size_t i = 0; i < obsLists_.size(); ++i) {
        auto clusterIndex = findAssociation(i, 0);

        // The Observation is associated with the clusterIndex-th Cluster.
        // clusterIndex is analogous to a state index, so...
        pi(clusterIndex)++;
    }

    for (std::size_t i = 0; i < numStates; ++i) {
        pi(i) /= static_cast<double>(obsLists_.size());
        if (pi(i) == 0) pi(i) = ZERO;
        assert(pi(i) <= 1);
    }

    // Assign pi to the Hmm
    hmm_->setPi(pi);
}

/*
 * A(i, j), as defined by Dugad and Desai(1996) is 
 *
 *  number of times O_t is in cluster i and O_(t+1) is in cluster j
 *  ---------------------------------------------------------------
 *      number of times that O_t is associated with state i
 *
 *  which may be rewritten as 
 *
 *  number of times O_t is in cluster i and O_(t+1) is in cluster j
 *  ---------------------------------------------------------------
 *          number of Observations associated with state i
 *
 * To compute this, we need to go through each ObservationSet that is longer
 * than 2 and keep two iterators, one behind the other by an index of one.  We
 * check the associations of both and get cluster indices x and y.  A( x, y )
 * gets increased monotonically until we go through all ObservationSets.  
 * Finally, divide each A( x, y ) by size of Cluster x.
 */
void ViterbiTrainer::calculateTrans() {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    Matrix trans(numStates, numStates);
    StateIndex x, y;

    clear_matrix(trans);

    for (std::size_t k = 0; k < obsLists_.size(); ++k) {
        const ObservationSet& os = obsLists_.at(k);

        // Is the ObservationSet too short?
        if (os.size() < 2) continue;

        for (std::size_t i = 0, j = 1; j < os.size(); ++j, ++i) {
            // Find associations
            x = static_cast<StateIndex>(findAssociation(k, i));
            y = static_cast<StateIndex>(findAssociation(k, j));

            assert(x < static_cast<StateIndex>(numStates));
            assert(y < static_cast<StateIndex>(numStates));

            // Set trans(x, y)
            trans(x, y)++;
        }
    }

    // Normalize each state's transition probabilities
    for (x = 0; x < static_cast<StateIndex>(numStates); ++x) {
        double sum = 0;
        for (y = 0; y < static_cast<StateIndex>(numStates); ++y) {
            sum += trans(x, y);
        }

        for (y = 0; y < static_cast<StateIndex>(numStates); ++y) {
            if (sum == 0) {
                trans(x, y) = ZERO;
            } else {
                trans(x, y) /= sum;
            }
            assert(trans(x, y) <= 1);
        }
    }

    // Assign trans to the HMM
    hmm_->setTrans(trans);
}

void ViterbiTrainer::quickSort(ObservationSet& set) {
    quickSort(set, 0, static_cast<int>(set.size()) - 1);
}

void ViterbiTrainer::quickSort(ObservationSet& set, int left, int right) {
    if (left >= right) return;
    
    int lHold = left;
    int rHold = right;
    double pivot = set(left);
    
    while (left < right) {
        while ((set(right) >= pivot) && (left < right))
            right--;
        if (left != right) {
            set(left) = set(right);
            left++;
        }
        
        while ((set(left) <= pivot) && (left < right))
            left++;
        if (left != right) {
            set(right) = set(left);
            right--;
        }
    }
    set(left) = pivot;
    pivot = left;
    left = lHold;
    right = rHold;
    if (left < pivot) {
        quickSort(set, left, static_cast<int>(pivot) - 1);
    }
    if (right > pivot) {
        quickSort(set, static_cast<int>(pivot) + 1, right);
    }
}

} //namespace
