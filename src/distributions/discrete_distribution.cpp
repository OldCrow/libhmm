#include "libhmm/distributions/discrete_distribution.h"
#include <iostream>

namespace libhmm
{

/*
 * Returns the value of the PDF of the discrete distribution.
 */            
double DiscreteDistribution::getProbability(double x) {
    // Validate input
    if (std::isnan(x) || std::isinf(x) || x < 0) {
        return 0.0;
    }
    
    const auto index = static_cast<std::size_t>(x);
    if (!isValidIndex(index)) {
        return 0.0;
    }
    
    const double p = pdf_(index); 
    assert(p <= 1.0 && p >= 0.0);
    return p;
}

/*
 * Sets the value of the PDF of the discrete distribution to the given value.
 */
void DiscreteDistribution::setProbability(Observation o, double value) {
    if (std::isnan(value) || std::isinf(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument("Probability value must be between 0 and 1");
    }
    
    const auto index = static_cast<std::size_t>(o);
    if (!isValidIndex(index)) {
        throw std::out_of_range("Observation index out of range");
    }
    
    pdf_(index) = value;
}

/*
 * Sets probabilities such that the resulting distribution fits the data.
 *
 * Note that we're just using a 'uniform' distribution here
 *
 */
void DiscreteDistribution::fit(const std::vector<Observation>& values) {
    const auto N = values.size();

    /* Empty cluster - use uniform distribution */
    if(N == 0) {
        reset();
        return;
    }

    // Zero out the PDF
    for(std::size_t i = 0; i < pdf_.size(); i++) {
        pdf_(i) = 0;
    }

    // Count the values with bounds checking
    for(const auto& val : values) {
        if (val >= 0) {
            const auto index = static_cast<std::size_t>(val);
            if (isValidIndex(index)) {
                pdf_(index)++;
            }
        }
    }

    // Normalize
    for(std::size_t i = 0; i < numSymbols_; i++) {
        pdf_(i) /= static_cast<double>(values.size());
    }
}

/*
 * Resets the the distribution to some default value.  Creates uniform distribution.
 */
void DiscreteDistribution::reset() noexcept {
    const double uniformProb = 1.0 / static_cast<double>(numSymbols_);
    for(std::size_t i = 0; i < numSymbols_; i++) {
        pdf_(i) = uniformProb;
    }
}

std::string DiscreteDistribution::toString() const {
    std::stringstream os;
    os << "Discrete Distribution:\n"; 
    for(std::size_t i = 0; i < numSymbols_; i++) {
        os << pdf_(i) << "\t";
    }
    os << "\n";
    return os.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::DiscreteDistribution& distribution ){
    
    os << distribution.toString( );
    return os;
}


}//namespace
