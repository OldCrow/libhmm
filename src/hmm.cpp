#include "libhmm/hmm.h"
#include <iomanip>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>

namespace libhmm{

// All the constructors and core methods are now implemented in the header
// using modern C++17 features. Only the I/O operators need implementation here.

std::ostream& operator<<( std::ostream& os, const libhmm::Hmm& h ){
    const auto numStates = h.getNumStatesModern();
    
    os << "Hidden Markov Model parameters" << std::endl;
    os << "  States: " << numStates << std::endl;
    
    os << "  Pi: ["; 
    for( std::size_t i = 0; i < numStates; ++i ){
        os << " " << std::setw( 12 ) << h.getPi()( i );
    }
    os << " ]" << std::endl;

    os << "  Transmission matrix: " << std::endl;
    for( std::size_t i = 0; i < numStates; ++i ){
        os << "   [";
        for( std::size_t j = 0; j < numStates; ++j ){
            os << std::setw( 12 ) << h.getTrans()( i, j );
        }
        os << " ]" << std::endl;
    }
    
    os << "  Emissions: " << std::endl;
    for( std::size_t i = 0; i < numStates; ++i ){
        const ProbabilityDistribution* pd = h.getProbabilityDistribution(i);
        os << "   State " << i << ": ";
        os << pd->toString() << std::endl;
    }
    
    return os;
} //operator<<()

std::istream& operator>>( std::istream& is, libhmm::Hmm& hmm ){
    std::string s, t;
    std::size_t states;
    
    // Parse header
    is >> s >> s >> s >> s; // "Hidden Markov Model parameters"
    is >> s >> s; // "States:" 
    states = std::stoull(s);
    
    if (states == 0) {
        throw std::runtime_error("Invalid number of states in HMM input");
    }
    
    // Create new HMM with proper number of states
    hmm = Hmm(states);
    
    Vector pi(states);
    Matrix trans(states, states);

    // Parse Pi vector
    is >> s >> s; // "Pi:" "["
    for(std::size_t i = 0; i < states; ++i){
        is >> t;
        pi(i) = std::stod(t);
    }
    is >> s; // "]"

    // Parse transition matrix
    is >> s >> s; // "Transmission" "matrix:"
    for(std::size_t i = 0; i < states; ++i){
        is >> s; // "["
        for(std::size_t j = 0; j < states; ++j){
            is >> t;
            trans(i, j) = std::stod(t);
        }
        is >> s; // "]"
    }

    // Parse emissions
    is >> s; // "Emissions:"
    for(std::size_t i = 0; i < states; ++i){
        is >> s >> s >> t; // "State" "i:" "DistributionType"

        if(t == "Gaussian"){
            is >> s >> s >> s >> t; // "Distribution" "Mean" "=" value
            double mean = std::stod(t);
            is >> s >> s >> s >> t; // "Standard" "Deviation" "=" value
            double sd = std::stod(t);
            
            auto gaussDist = std::make_unique<GaussianDistribution>(mean, sd);
            hmm.setProbabilityDistribution(i, std::move(gaussDist));
        }
        else if(t == "Discrete"){
            is >> s; // "Distribution"

            // For now, assume a fixed number of symbols for simplicity
            // In a real implementation, this should be more flexible
            constexpr std::size_t MAX_SYMBOLS = 11;
            std::vector<double> symbols(MAX_SYMBOLS);

            for(std::size_t symIndex = 0; symIndex < MAX_SYMBOLS; ++symIndex) {
                is >> t;
                symbols[symIndex] = std::stod(t);
            }

            auto discreteDist = std::make_unique<DiscreteDistribution>(MAX_SYMBOLS);
            for(std::size_t symIndex = 0; symIndex < MAX_SYMBOLS; ++symIndex) {
                discreteDist->setProbability(static_cast<int>(symIndex), symbols[symIndex]);
            }
            
            hmm.setProbabilityDistribution(i, std::move(discreteDist));
        }
        else {
            throw std::runtime_error("Unknown distribution type: " + t);
        }
    }

    // Set the parsed parameters
    hmm.setPi(pi);
    hmm.setTrans(trans);

    return is;
}//operator>>()

}//namespace
