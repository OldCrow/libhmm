#include "libhmm/distributions/discrete_distribution.h"
#include <iostream>

namespace libhmm
{

/**
 * Gets the probability mass function value for a discrete observation.
 * 
 * @param x The discrete value (will be cast to integer index)
 * @return Probability mass for the given value, 0.0 if out of range
 */            
double DiscreteDistribution::getProbability(double x) {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(x) || std::isinf(x) || x < 0.0) {
        return 0.0;
    }
    
    // Convert to integer index
    const auto index = static_cast<std::size_t>(x);
    if (!isValidIndex(index)) {
        return 0.0;
    }
    
    const double p = pdf_[index]; 
    assert(p <= 1.0 && p >= 0.0);
    return p;
}

/**
 * Sets the probability for a specific discrete observation.
 * 
 * @param o The discrete observation (symbol index)
 * @param value The probability value (must be in [0,1])
 * @throws std::invalid_argument if value is not a valid probability
 * @throws std::out_of_range if observation index is out of range
 */
void DiscreteDistribution::setProbability(Observation o, double value) {
    if (std::isnan(value) || std::isinf(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument("Probability value must be between 0 and 1");
    }
    
    const auto index = static_cast<std::size_t>(o);
    if (!isValidIndex(index)) {
        throw std::out_of_range("Observation index out of range");
    }
    
    pdf_[index] = value;
    cacheValid_ = false; // Invalidate cache since probabilities changed
}

/**
 * Fits the distribution to observed data using maximum likelihood estimation.
 * Computes empirical probabilities: P(X = k) = count(k) / total_count
 * 
 * @param values Vector of observed discrete values
 */
void DiscreteDistribution::fit(const std::vector<Observation>& values) {
    const auto N = values.size();

    // Handle empty data - use uniform distribution
    if (N == 0) {
        reset();
        return;
    }

    // Initialize counts to zero
    std::fill(pdf_.begin(), pdf_.end(), 0.0);

    // Count valid observations
    std::size_t validCount = 0;
    for (const auto& val : values) {
        if (val >= 0.0) {
            const auto index = static_cast<std::size_t>(val);
            if (isValidIndex(index)) {
                pdf_[index]++;
                validCount++;
            }
        }
    }

    // If no valid observations, fall back to uniform distribution
    if (validCount == 0) {
        reset();
        return;
    }

    // Normalize by total valid count to get probabilities
    const double normalizationFactor = 1.0 / static_cast<double>(validCount);
    for (double& p : pdf_) {
        p *= normalizationFactor;
    }
    
    cacheValid_ = false; // Invalidate cache since probabilities changed
}

/**
 * Resets the distribution to uniform probabilities.
 * Each symbol gets probability 1/numSymbols
 */
void DiscreteDistribution::reset() noexcept {
    const double uniformProb = 1.0 / static_cast<double>(numSymbols_);
    std::fill(pdf_.begin(), pdf_.end(), uniformProb);
    cacheValid_ = false; // Invalidate cache since probabilities changed
}

/**
 * Returns a string representation of the distribution.
 * 
 * @return String showing all symbol probabilities
 */
std::string DiscreteDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Discrete Distribution:\n";
    oss << "      Number of symbols = " << numSymbols_ << "\n";
    for (std::size_t i = 0; i < numSymbols_; ++i) {
        oss << "      P(" << i << ") = " << pdf_[i] << "\n";
    }
    return oss.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::DiscreteDistribution& distribution ){
    
    os << distribution.toString( );
    return os;
}


}//namespace
