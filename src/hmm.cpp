#include "libhmm/hmm.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <functional>
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

        // Modern C++17 approach: Hash-based dispatch for cleaner code
        using DistributionParser = std::function<std::unique_ptr<ProbabilityDistribution>(std::istream&)>;
        
        static const std::unordered_map<std::string, DistributionParser> parsers = {
            {"Gaussian", [](std::istream& is) {
                std::string s, t;
                is >> s >> s >> s >> t; // "Distribution" "Mean" "=" value
                double mean = std::stod(t);
                is >> s >> s >> s >> t; // "Standard" "Deviation" "=" value
                double sd = std::stod(t);
                return std::make_unique<GaussianDistribution>(mean, sd);
            }},
            
            {"Discrete", [](std::istream& is) {
                std::string s, t;
                is >> s; // "Distribution"
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
                return discreteDist;
            }},
            
            {"Gamma", [](std::istream& is) {
                std::string s, t;
                is >> s >> s >> s >> t; // "Distribution:" "k" "=" value
                double k = std::stod(t);
                is >> s >> s >> t; // "theta" "=" value
                double theta = std::stod(t);
                return std::make_unique<GammaDistribution>(k, theta);
            }},
            
            {"Exponential", [](std::istream& is) {
                std::string s, t;
                is >> s >> s >> s >> s >> t; // "Distribution:" "Rate" "parameter" "=" value
                double lambda = std::stod(t);
                return std::make_unique<ExponentialDistribution>(lambda);
            }},
            
            {"LogNormal", [](std::istream& is) {
                std::string s, t;
                is >> s >> s >> s >> t; // "Distribution:" "Mean" "=" value
                double mean = std::stod(t);
                is >> s >> s >> s >> t; // "Standard" "Deviation" "=" value
                double sd = std::stod(t);
                return std::make_unique<LogNormalDistribution>(mean, sd);
            }},
            
            {"Pareto", [](std::istream& is) {
                std::string s, t;
                is >> s >> s >> s >> t; // "Distribution:" "k" "=" value
                double k = std::stod(t);
                is >> s >> s >> t; // "xm" "=" value
                double xm = std::stod(t);
                return std::make_unique<ParetoDistribution>(k, xm);
            }},
            
            {"Poisson", [](std::istream& is) {
                std::string s, t;
                is >> s >> s >> s >> t; // "Distribution:" "λ" "=" value
                double lambda = std::stod(t);
                return std::make_unique<PoissonDistribution>(lambda);
            }},
            
            {"Beta", [](std::istream& is) {
                std::string s, t;
                is >> s >> s >> s >> s >> t; // "Distribution:" "α" "(alpha)" "=" value
                double alpha = std::stod(t);
                is >> s >> s >> s >> t; // "β" "(beta)" "=" value
                double beta = std::stod(t);
                return std::make_unique<BetaDistribution>(alpha, beta);
            }}
        };
        
        // Execute the appropriate parser
        auto parser_it = parsers.find(t);
        if (parser_it != parsers.end()) {
            auto distribution = parser_it->second(is);
            hmm.setProbabilityDistribution(i, std::move(distribution));
        } else {
            throw std::runtime_error("Unknown distribution type: " + t);
        }
    }

    // Set the parsed parameters
    hmm.setPi(pi);
    hmm.setTrans(trans);

    return is;
}//operator>>()

}//namespace
