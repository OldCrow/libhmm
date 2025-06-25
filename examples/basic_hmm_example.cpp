#include <iostream>
#include <fstream>
#include <cassert>
#include <memory>
#include <vector>
#include <cmath>
#include "libhmm/libhmm.h"
#include "libhmm/two_state_hmm.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"

// Avoid global using namespace - use specific imports
using libhmm::Hmm;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::ForwardBackwardCalculator;
using libhmm::ScaledSIMDForwardBackwardCalculator;
using libhmm::LogSIMDForwardBackwardCalculator;
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

    {
        auto hmm = std::make_unique<Hmm>(2);  // Initialize with 2 states
        prepareTwoStateHmm(hmm.get());

        std::cout << "Occasionally Dishonest Casino" << std::endl;
        std::cout << "-----------------------------" << std::endl;
        std::cout << *hmm << std::endl;
    }
    {
        auto hmm = std::make_unique<Hmm>(2);  // Initialize with 2 states
        prepareTwoStateHmm(hmm.get());
        // A '5' is when the dice rolls a '6'
        ObservationSet set1( 1 );
        set1( 0 ) = 5;
        ObservationSet set2( 2 );
        set2( 0 ) = 5;
        set2( 1 ) = 4;
        ObservationSet set3( 3 );
        set3( 0 ) = 5;
        set3( 1 ) = 4;
        set3( 2 ) = 3;  

        std::cout << "Test ForwardBackward and ScaledForwardBackward" 
            << std::endl;
        std::cout << "----------------------------------------------" 
            << std::endl;
        ForwardBackwardCalculator fbc1(hmm.get(), set1);
        ScaledSIMDForwardBackwardCalculator sfbc1(hmm.get(), set1);
        LogSIMDForwardBackwardCalculator lfbc1(hmm.get(), set1);
        std::cout << "Observation Sequence: " << set1 << std::endl;
        std::cout << "Probability: "<< fbc1.probability( ) << "\t"
            << sfbc1.getProbability( ) << "\t" 
            << sfbc1.getLogProbability( ) << "\t"
            << lfbc1.getLogProbability( ) << std::endl;
        // Note: Due to SIMD optimizations, exact equality may not hold, so we use approximate comparison
        assert( std::abs(fbc1.probability( ) - sfbc1.getProbability( )) < 1e-10 );

        ForwardBackwardCalculator fbc2(hmm.get(), set2);
        ScaledSIMDForwardBackwardCalculator sfbc2(hmm.get(), set2);
        LogSIMDForwardBackwardCalculator lfbc2(hmm.get(), set2);
        std::cout << "Observation Sequence: " << set2 << std::endl;
        std::cout << "Probability: " << fbc2.probability() << "\t"
            << sfbc2.getProbability() << "\t" 
            << sfbc2.getLogProbability() << "\t"
            << lfbc2.getLogProbability() << std::endl;

        ForwardBackwardCalculator fbc3(hmm.get(), set3);
        ScaledSIMDForwardBackwardCalculator sfbc3(hmm.get(), set3);
        LogSIMDForwardBackwardCalculator lfbc3(hmm.get(), set3);
        std::cout << "Observation Sequence: " << set3 << std::endl;
        std::cout << "Probability: " << fbc3.probability() << "\t"
            << sfbc3.getProbability() << "\t" 
            << sfbc3.getLogProbability() << "\t" 
            << lfbc3.getLogProbability() << std::endl;
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
        std::cout << "3.9894e-07\t\t2.4197e-07\t\t5.3991e-08" << std::endl;
        std::cout << "Fitting: (4,1)" << std::endl;
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
        std::cout << "0\t\t3.6788e-07\t\t1.3534e-07" << std::endl;
        std::cout << "Fitting: (23.4063, 0.170894)" << std::endl;
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
        std::cout << "0\t\t3.9894e-07\t\t1.5687e-07" << std::endl;
        std::cout << "Fitting: (1.36748, .256091)" << std::endl;
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
        std::cout << "0\t\t3.6788e-07\t\t1.3534e-07" << std::endl;
        std::cout << "Fitting: (4)" << std::endl;
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

        // Use smart pointers for memory safety
        hmm.setProbabilityDistribution(0, std::make_unique<GaussianDistribution>());
        hmm.setProbabilityDistribution(1, std::make_unique<GaussianDistribution>());
        hmm.setProbabilityDistribution(2, std::make_unique<GaussianDistribution>());

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
        hmm.setProbabilityDistribution(0, std::make_unique<GaussianDistribution>());
        hmm.setProbabilityDistribution(1, std::make_unique<GaussianDistribution>(2, 2));
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
