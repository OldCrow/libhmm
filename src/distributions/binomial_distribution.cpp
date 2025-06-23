#include "libhmm/distributions/binomial_distribution.h"
#include <iostream>
#include <numeric>
#include <algorithm>

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
        return 0.0;
    }
    
    // Round to nearest integer and check if it's in valid range
    int k = static_cast<int>(std::round(value));
    if (k < 0 || k > n_) {
        return 0.0;
    }
    
    // Handle edge cases
    if (p_ == 0.0) {
        return (k == 0) ? 1.0 : 0.0;
    }
    if (p_ == 1.0) {
        return (k == n_) ? 1.0 : 0.0;
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
    if (std::isnan(prob) || prob < 0.0) {
        return 0.0;
    }
    
    assert(prob <= 1.0);
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
    const auto N = values.size();

    // Handle edge cases: empty data or single data point
    if (N == 0) {
        reset();
        return;
    }
    
    if (N == 1) {
        // For single point, estimate n as that value and p = 1 (degenerate case)
        const int observedValue = static_cast<int>(std::round(values[0]));
        if (observedValue >= 0) {
            n_ = std::max(1, observedValue);
            p_ = (observedValue == 0) ? 0.0 : 1.0;
            cacheValid_ = false;
        } else {
            reset(); // Invalid data
        }
        return;
    }

    // Filter valid integer observations
    std::vector<int> validObs;
    for (const auto& val : values) {
        if (val >= 0.0 && !std::isnan(val) && !std::isinf(val)) {
            validObs.push_back(static_cast<int>(std::round(val)));
        }
    }
    
    if (validObs.empty()) {
        reset(); // No valid data
        return;
    }

    // Estimate n as the maximum observed value (common approach when n is unknown)
    const int maxObs = *std::max_element(validObs.begin(), validObs.end());
    if (maxObs == 0) {
        // All observations are 0
        n_ = 1;
        p_ = 0.0;
    } else {
        n_ = maxObs;
        
        // Calculate sample mean
        const double sum = std::accumulate(validObs.begin(), validObs.end(), 0.0);
        const double sampleMean = sum / static_cast<double>(validObs.size());
        
        // MLE estimate: p = sample_mean / n
        p_ = sampleMean / static_cast<double>(n_);
        
        // Ensure p is in valid range [0,1]
        p_ = std::max(0.0, std::min(1.0, p_));
    }
    
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (n = 10, p = 0.5).
 * This corresponds to a balanced binomial distribution with moderate number of trials.
 */
void BinomialDistribution::reset() noexcept {
    n_ = 10;
    p_ = 0.5;
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

std::ostream& operator<<(std::ostream& os, 
        const libhmm::BinomialDistribution& distribution) {
    os << "Binomial Distribution:" << std::endl;
    os << "    n = " << distribution.getN() << std::endl;
    os << "    p = " << distribution.getP() << std::endl;
    os << std::endl;
    
    return os;
}

} // namespace libhmm
