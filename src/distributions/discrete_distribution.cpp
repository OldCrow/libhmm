#include "libhmm/distributions/discrete_distribution.h"
#include <span>

using namespace libhmm::constants;

namespace libhmm
{

/**
 * Gets the probability mass function value for a discrete observation.
 * 
 * @param x The discrete value (will be cast to integer index)
 * @return Probability mass for the given value, 0.0 if out of range
 */            
double DiscreteDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x) || x < math::ZERO_DOUBLE) return math::ZERO_DOUBLE;
    const auto index = static_cast<std::size_t>(x);
    if (!isValidIndex(index)) return math::ZERO_DOUBLE;
    assert(pdf_[index] <= 1.0 && pdf_[index] >= 0.0);
    return pdf_[index];
}
// setProbability is now inline in the header.

/**
 * Fits the distribution to observed data using maximum likelihood estimation.
 * Computes empirical probabilities: P(X = k) = count(k) / total_count
 * 
 * @param values Vector of observed discrete values
 */
void DiscreteDistribution::fit(std::span<const double> data) {
    if (data.empty()) { reset(); return; }

    std::fill(pdf_.begin(), pdf_.end(), 0.0);
    std::size_t validCount = 0;
    for (const double val : data) {
        if (val >= 0.0) {
            const auto index = static_cast<std::size_t>(val);
            if (isValidIndex(index)) { pdf_[index]++; ++validCount; }
        }
    }
    if (validCount == 0) { reset(); return; }
    const double inv = 1.0 / static_cast<double>(validCount);
    for (double& p : pdf_) p *= inv;
    invalidateCache();
}

void DiscreteDistribution::fit(std::span<const double> data,
                               std::span<const double> weights) {
    // Weighted empirical probabilities: P(X=k) = Σ(w_i for x_i=k) / Σ(w_i)
    double sumW = 0.0;
    for (const double w : weights) sumW += w;
    if (sumW < precision::ZERO || std::isnan(sumW)) { reset(); return; }

    std::fill(pdf_.begin(), pdf_.end(), 0.0);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] >= 0.0) {
            const auto index = static_cast<std::size_t>(data[i]);
            if (isValidIndex(index)) pdf_[index] += weights[i];
        }
    }
    for (double& p : pdf_) p /= sumW;
    invalidateCache();
}

/**
 * Resets the distribution to uniform probabilities.
 * Each symbol gets probability 1/numSymbols
 */
void DiscreteDistribution::reset() noexcept {
    const double uniformProb = math::ONE / static_cast<double>(numSymbols_);
    std::fill(pdf_.begin(), pdf_.end(), uniformProb);
    invalidateCache();
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

/**
 * Evaluates the logarithm of the probability mass function
 * Uses cached log probabilities for maximum performance
 */
double DiscreteDistribution::getLogProbability(double value) const noexcept {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value) || value < math::ZERO_DOUBLE) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Convert to integer index
    const auto index = static_cast<std::size_t>(value);
    if (!isValidIndex(index)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    if (!isCacheValid()) updateCache();
    return cachedLogProbs_[index];
}

/**
 * Evaluates the CDF at k using pre-computed cached values
 * O(1) lookup for maximum performance
 */
double DiscreteDistribution::getCumulativeProbability(double value) const noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }
    
    if (value < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }
    
    const auto k = static_cast<std::size_t>(std::floor(value));
    
    // If k is beyond our range, CDF = 1.0
    if (k >= numSymbols_) {
        return math::ONE;
    }
    
    if (!isCacheValid()) updateCache();
    return cachedCDF_[k];
}

/**
 * Equality comparison operator with numerical tolerance
 */
bool DiscreteDistribution::operator==(const DiscreteDistribution& other) const {
    if (numSymbols_ != other.numSymbols_) {
        return false;
    }
    
    const double tolerance = 1e-10;
    for (std::size_t i = 0; i < numSymbols_; ++i) {
        if (std::abs(pdf_[i] - other.pdf_[i]) > tolerance) {
            return false;
        }
    }
    
    return true;
}

/**
 * Stream input operator implementation
 * Expects format with number of symbols followed by probabilities
 */
std::istream& operator>>(std::istream& is, libhmm::DiscreteDistribution& distribution) {
    std::size_t numSymbols = 0;
    
    if (!(is >> numSymbols)) {
        is.setstate(std::ios::failbit);
        return is;
    }
    
    // Create new distribution with the specified number of symbols
    try {
        DiscreteDistribution newDist(numSymbols);
        
        // Read probabilities
        for (std::size_t i = 0; i < numSymbols; ++i) {
            double prob = 0.0;
            if (!(is >> prob)) {
                is.setstate(std::ios::failbit);
                return is;
            }
            newDist.setProbability(static_cast<double>(i), prob);
        }
        
        // If successful, update the distribution
        distribution = std::move(newDist);
        
    } catch (const std::exception&) {
        is.setstate(std::ios::failbit);
    }
    
    return is;
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::DiscreteDistribution& distribution ){
    
    os << distribution.toString( );
    return os;
}


}//namespace
