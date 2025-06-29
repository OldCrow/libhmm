#include "libhmm/distributions/binomial_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>     // For std::accumulate (not in common.h)
#include <algorithm>   // For std::for_each, std::max_element (exists in common.h, included for clarity)

using namespace libhmm::constants;

namespace libhmm
{

/**
 * Computes the probability mass function for the Binomial distribution.
 * 
 * For discrete distributions, this returns the exact probability mass
 * P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
 * 
 * @param value The value at which to evaluate the PMF (rounded to nearest integer)
 * @return Probability mass for the given value
 */            
double BinomialDistribution::getProbability(double value) {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }
    
    // Round to nearest integer and check if it's in valid range
    int k = static_cast<int>(std::round(value));
    if (k < 0 || k > n_) {
        return math::ZERO_DOUBLE;
    }
    
    // Handle edge cases
    if (p_ == math::ZERO_DOUBLE) {
        return (k == 0) ? math::ONE : math::ZERO_DOUBLE;
    }
    if (p_ == math::ONE) {
        return (k == n_) ? math::ONE : math::ZERO_DOUBLE;
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // Compute log probability for numerical stability
    // log P(X = k) = log C(n,k) + k*log(p) + (n-k)*log(1-p)
    const double logCoeff = logBinomialCoefficient(n_, k);
    const double logProb = logCoeff + k * logP_ + (n_ - k) * log1MinusP_;
    
    const double prob = std::exp(logProb);
    
    // Ensure numerical stability
    if (std::isnan(prob) || prob < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }
    
    assert(prob <= math::ONE);
    return prob;
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 * 
 * For Binomial distribution with known n, the MLE of p is:
 * p̂ = sample_mean / n
 * 
 * If n is unknown, we estimate it as the maximum observed value, then fit p.
 * This is a common approach when the number of trials is not known a priori.
 * 
 * @param values Vector of observed data points
 */                   
void BinomialDistribution::fit(const std::vector<Observation>& values) {
    // Handle edge case: empty data
    if (values.empty()) {
        reset();
        return;
    }
    
    // Single-pass algorithm: compute max, sum, and count simultaneously
    int maxObs = 0;
    double sum = math::ZERO_DOUBLE;
    std::size_t validCount = 0;
    
    for (const auto& val : values) {
        // Validate: finite non-negative values only
        if (val >= math::ZERO_DOUBLE && std::isfinite(val)) {
            const int intVal = static_cast<int>(std::round(val));
            maxObs = std::max(maxObs, intVal);
            sum += static_cast<double>(intVal);
            ++validCount;
        }
    }
    
    // Handle edge cases: no valid data or single data point
    if (validCount == 0) {
        reset(); // No valid data
        return;
    }
    
    if (validCount == 1) {
        // For single point, estimate n as that value and p = 1 (degenerate case)
        if (maxObs >= 0) {
            n_ = std::max(1, maxObs);
            p_ = (maxObs == 0) ? math::ZERO_DOUBLE : math::ONE;
            cacheValid_ = false;
        } else {
            reset(); // Invalid data
        }
        return;
    }

    // Estimate parameters using maximum likelihood
    if (maxObs == 0) {
        // All observations are 0
        n_ = 1;
        p_ = math::ZERO_DOUBLE;
    } else {
        // Estimate n as the maximum observed value (common approach when n is unknown)
        n_ = maxObs;
        
        // Calculate sample mean efficiently
        const double sampleMean = sum / static_cast<double>(validCount);
        
        // MLE estimate: p = sample_mean / n
        p_ = sampleMean / static_cast<double>(n_);
        
        // Ensure p is in valid range [0,1]
        p_ = std::max(math::ZERO_DOUBLE, std::min(math::ONE, p_));
    }
    
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (n = 10, p = 0.5).
 * This corresponds to a balanced binomial distribution with moderate number of trials.
 */
void BinomialDistribution::reset() noexcept {
    n_ = 10;
    p_ = math::HALF;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Returns a string representation of the distribution following the standardized format.
 * 
 * @return String describing the distribution parameters and statistics
 */
std::string BinomialDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Binomial Distribution:\n";
    oss << "      n (trials) = " << n_ << "\n";
    oss << "      p (success probability) = " << p_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

double BinomialDistribution::getLogProbability(double value) const noexcept {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Round to nearest integer and check if it's in valid range
    int k = static_cast<int>(std::round(value));
    if (k < 0 || k > n_) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Handle edge cases
    if (p_ == math::ZERO_DOUBLE) {
        return (k == 0) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }
    if (p_ == math::ONE) {
        return (k == n_) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // Compute log probability for numerical stability
    // log P(X = k) = log C(n,k) + k*log(p) + (n-k)*log(1-p)
    const double logCoeff = logBinomialCoefficient(n_, k);
    const double logProb = logCoeff + k * logP_ + (n_ - k) * log1MinusP_;
    
    return logProb;
}

double BinomialDistribution::getCumulativeProbability(double value) noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }
    
    int k = static_cast<int>(std::floor(value));
    
    // Handle boundary cases
    if (k < 0) {
        return math::ZERO_DOUBLE;
    }
    if (k >= n_) {
        return math::ONE;
    }
    
    // Compute CDF as cumulative sum: P(X <= k) = sum_{i=0}^{k} P(X = i)
    double cdf = math::ZERO_DOUBLE;
    for (int i = 0; i <= k; ++i) {
        cdf += getProbability(static_cast<double>(i));
    }
    
    return std::min(math::ONE, cdf);
}

bool BinomialDistribution::operator==(const BinomialDistribution& other) const {
    const double tolerance = 1e-10;
    return (n_ == other.n_) && 
           (std::abs(p_ - other.p_) < tolerance);
}

std::istream& operator>>(std::istream& is, libhmm::BinomialDistribution& distribution) {
    std::string token;
    int n = 0;
    double p = 0.0;
    
    // Expected format: "Binomial(n,p)" or "n p"
    if (is >> token) {
        if (token.find("Binomial") != std::string::npos) {
            // Skip to parameters
            char ch = '\0';
            is >> ch >> n >> ch >> p >> ch; // Read (n,p)
        } else {
            // Assume first token is n
            n = std::stoi(token);
            is >> p;
        }
        
        try {
            distribution.setParameters(n, p);
        } catch (const std::exception&) {
            is.setstate(std::ios::failbit);
        }
    }
    
    return is;
}

std::ostream& operator<<(std::ostream& os, 
        const libhmm::BinomialDistribution& distribution) {
    os << "Binomial Distribution:" << std::endl;
    os << "    n = " << distribution.getN() << std::endl;
    os << "    p = " << distribution.getP() << std::endl;
    os << std::endl;
    
    return os;
}

} // namespace libhmm
