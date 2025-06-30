#include <iostream>
#include <vector>
#include <memory>
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"

using namespace std;

int main() {
    cout << "Testing basic libhmm functionality..." << endl;
    
    try {
        // Create a simple 2-state HMM
        auto hmm = make_unique<libhmm::Hmm>(2);
        
        // Set transition matrix
        libhmm::Matrix trans_matrix(2, 2);
        trans_matrix(0, 0) = 0.7; trans_matrix(0, 1) = 0.3;
        trans_matrix(1, 0) = 0.4; trans_matrix(1, 1) = 0.6;
        hmm->setTrans(trans_matrix);
        
        // Set initial probabilities  
        libhmm::Vector pi_vector(2);
        pi_vector(0) = 0.6; pi_vector(1) = 0.4;
        hmm->setPi(pi_vector);
        
        // Set discrete emission distributions
        auto dist0 = make_unique<libhmm::DiscreteDistribution>(2);
        dist0->setProbability(0, 0.8);
        dist0->setProbability(1, 0.2);
        hmm->setProbabilityDistribution(0, dist0.release());
        
        auto dist1 = make_unique<libhmm::DiscreteDistribution>(2);
        dist1->setProbability(0, 0.3);
        dist1->setProbability(1, 0.7);
        hmm->setProbabilityDistribution(1, dist1.release());
        
        // Test observation sequence
        libhmm::ObservationSet obs(3);
        obs(0) = 0; obs(1) = 1; obs(2) = 0;
        
        cout << "HMM created successfully!" << endl;
        cout << "Number of states: " << hmm->getNumStates() << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
        return 1;
    }
}
