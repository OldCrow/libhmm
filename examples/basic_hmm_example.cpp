/**
 * basic_hmm_example — introductory libhmm example.
 *
 * 1. The occasionally dishonest casino (Durbin et al. 1998)
 *    Demonstrates ForwardBackward probability evaluation.
 * 2. Distribution showcase — PDF evaluation and MLE fitting.
 * 3. Viterbi training on a 3-state Gaussian HMM.
 * 4. XML file round-trip.
 */
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>
#include "libhmm/libhmm.h"
#include "two_state_hmm.h"

using libhmm::Hmm;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::ForwardBackwardCalculator;
using libhmm::ViterbiCalculator;
using libhmm::GaussianDistribution;
using libhmm::GammaDistribution;
using libhmm::LogNormalDistribution;
using libhmm::ExponentialDistribution;
using libhmm::PoissonDistribution;
using libhmm::ViterbiTrainer;
using libhmm::Vector;
using libhmm::Matrix;
using libhmm::Observation;

int main() {

    // =========================================================================
    // 1. The occasionally dishonest casino (Durbin et al. 1998)
    // =========================================================================
    {
        auto hmm = std::make_unique<Hmm>(2);
        libhmm::examples::prepare_two_state_hmm(*hmm);

        std::cout << "Occasionally Dishonest Casino\n";
        std::cout << "-----------------------------\n";
        std::cout << *hmm << "\n";
    }
    {
        auto hmm = std::make_unique<Hmm>(2);
        libhmm::examples::prepare_two_state_hmm(*hmm);

        // Face 5 (0-indexed) = rolling a 6
        ObservationSet set1(1); set1(0) = 5;
        ObservationSet set2(2); set2(0) = 5; set2(1) = 4;
        ObservationSet set3(3); set3(0) = 5; set3(1) = 4; set3(2) = 3;

        std::cout << "ForwardBackward (canonical log-space calculator)\n";
        std::cout << "------------------------------------------------\n";
        for (const ObservationSet* obs : {&set1, &set2, &set3}) {
            ForwardBackwardCalculator fbc(hmm.get(), *obs);
            std::cout << "  Observations: " << *obs
                      << "  P(O|\u03bb)=" << fbc.probability()
                      << "  log P(O|\u03bb)=" << fbc.getLogProbability() << "\n";
        }
        std::cout << "\n";
    }

    std::vector<Observation> trainvector;
    trainvector.push_back( 5 );
    trainvector.push_back( 4 );
    trainvector.push_back( 3 );
    
    {
        // Test Gaussian Distribution
        std::cout << std::endl;
        std::cout << "Test GaussianDistribution" << std::endl;
        std::cout << "-------------------------" << std::endl;
        GaussianDistribution gd;
        std::cout << gd << std::endl;
        std::cout << "p( 0 ) \t\t p( 1 ) \t p( 2 ) " << std::endl;
        std::cout << gd.getProbability( 0 ) << "\t"
            << gd.getProbability( 1 ) << "\t"
            << gd.getProbability( 2 ) << std::endl;
        std::cout << "Fitting to [5,4,3] (expect mean=4)" << std::endl;
        gd.fit( trainvector );
        std::cout << gd << std::endl;
    
        // Test Gamma Distribution
        std::cout << std::endl;
        std::cout << "Test GammaDistribution" << std::endl;
        std::cout << "----------------------" << std::endl;
        GammaDistribution gamdist;
        std::cout << gamdist << std::endl;
        std::cout << "p( 0 ) \t\t p( 1 ) \t\t p( 2 ) " << std::endl;
        std::cout << gamdist.getProbability( 0 ) << "\t\t"
            << gamdist.getProbability( 1 ) << "\t\t"
            << gamdist.getProbability( 2 ) << std::endl;
        std::cout << "Fitting to [5,4,3]" << std::endl;
        gamdist.fit( trainvector );
        std::cout << gamdist << std::endl;

        // Test LogNormal Distribution
        std::cout << std::endl;
        std::cout << "Test LogNormalDistribution" << std::endl;
        std::cout << "----------------------" << std::endl;
        LogNormalDistribution logndist;
        std::cout << logndist << std::endl;
        std::cout << "p( 0 ) \t\t p( 1 ) \t\t p( 2 ) " << std::endl;
        std::cout << logndist.getProbability( 0 ) << "\t\t"
            << logndist.getProbability( 1 ) << "\t\t"
            << logndist.getProbability( 2 ) << std::endl;
        std::cout << "Fitting to [5,4,3]" << std::endl;
        logndist.fit( trainvector );
        std::cout << logndist << std::endl;

        // Test Exponential Distribution
        std::cout << std::endl;
        std::cout << "Test ExponentialDistribution" << std::endl;
        std::cout << "----------------------------" << std::endl;
        ExponentialDistribution edist;
        std::cout << edist << std::endl;
        std::cout << "p( 0 ) \t\t p( 1 ) \t\t p( 2 ) " << std::endl;
        std::cout << edist.getProbability( 0 ) << "\t\t"
            << edist.getProbability( 1 ) << "\t\t"
            << edist.getProbability( 2 ) << std::endl;
        std::cout << "Fitting to [5,4,3] (expect rate=0.25)" << std::endl;
        edist.fit( trainvector );
        std::cout << edist << std::endl;

        // Test Poisson Distribution (NEW!)
        std::cout << std::endl;
        std::cout << "Test PoissonDistribution" << std::endl;
        std::cout << "------------------------" << std::endl;
        PoissonDistribution poissonDist(2.0);
        std::cout << poissonDist << std::endl;
        std::cout << "p( 0 ) \t\t p( 1 ) \t\t p( 2 ) \t\t p( 3 )" << std::endl;
        std::cout << poissonDist.getProbability( 0 ) << "\t\t"
            << poissonDist.getProbability( 1 ) << "\t\t"
            << poissonDist.getProbability( 2 ) << "\t\t"
            << poissonDist.getProbability( 3 ) << std::endl;
        std::cout << "Expected: 0.1353\t0.2707\t\t0.2707\t\t0.1804" << std::endl;
        std::cout << "Fitting to count data [1, 2, 2, 3, 3, 3, 4, 2, 1, 4] (mean=2.5)" << std::endl;
        std::vector<Observation> countData = {1, 2, 2, 3, 3, 3, 4, 2, 1, 4};
        poissonDist.fit( countData );
        std::cout << poissonDist << std::endl;

    }
    {
        std::cout << "Test Viterbi Training" << std::endl;
        std::cout << "---------------------" << std::endl;
        Hmm hmm(3);

        ObservationSet set1(8);
        ObservationSet set2(8);
        ObservationSet set3(8);

        hmm.setDistribution(0, std::make_unique<GaussianDistribution>());
        hmm.setDistribution(1, std::make_unique<GaussianDistribution>());
        hmm.setDistribution(2, std::make_unique<GaussianDistribution>());

        std::cout << hmm << std::endl;

        // Modern C++ loop with bounds checking
        for(auto i = 0u; i < set1.size(); ++i) {
            set1(i) = static_cast<Observation>(i);
            set2(i) = static_cast<Observation>(i + 10);
            set3(i) = static_cast<Observation>(i + 20);
        }
        
        ObservationLists lists;
        // Use smaller dataset for faster testing with the fixed library
        lists.reserve(30);
        for(auto i = 0u; i < 10; ++i) {
            lists.push_back(set1);
            lists.push_back(set2);
            lists.push_back(set3);
        }

        std::cout << "Training with fixed ViterbiTrainer (30 observation sets)" << std::endl;
        
        ViterbiTrainer vt(&hmm, lists);
        vt.train();

        std::cout << hmm << std::endl;
    }
    {
        std::cout << "Testing File Read/Write" << std::endl;
        Hmm hmm( 2 );
        Vector pi( 2 );
        Matrix trans( 2, 2 );

        pi( 0 ) = 0.75;
        pi( 1 ) = 0.25;
        trans( 0, 0 ) = 0.5;
        trans( 0, 1 ) = 0.5;
        trans( 1, 0 ) = 0.2;
        trans( 1, 1 ) = 0.8;
        hmm.setPi( pi );
        hmm.setTrans( trans );
        hmm.setDistribution(0, std::make_unique<GaussianDistribution>());
        hmm.setDistribution(1, std::make_unique<GaussianDistribution>(2.0, 2.0));
        std::ofstream of( "testrw", std::ios::out );

        std::cout << hmm << std::endl;
        of << hmm << std::endl;
        of.close( );

        Hmm hmm1( 2 );
        std::ifstream inf( "testrw", std::ios::in );
        inf >> hmm1;
//        inf.close( );

        std::cout << hmm1 << std::endl;
    }
}
