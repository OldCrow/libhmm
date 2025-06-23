#include "libhmm/distributions/negative_binomial_distribution.h"
#include <iostream>
#include <numeric>
#include <algorithm>

namespace libhmm
{

/**
 * Computes the probability mass function for the Negative Binomial distribution.
 * 
 * For discrete distributions, this returns the exact probability mass
 * P(X = k) = C(k+r-1, k) * p^r * (1-p)^k
 * 
 * @param value The value at which to evaluate the PMF (rounded to nearest integer)
 * @return Probability mass for the given value
 */            
double NegativeBinomialDistribution::getProbability(double value) {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return 0.0;
    }
    
    // Round to nearest integer and check if it's in valid range
    int k = static_cast<int>(std::round(value));
    if (k < 0) {
        return 0.0;
    }
    
    // Handle edge cases
    if (p_ == 1.0) {
        return (k == 0) ? 1.0 : 0.0;
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // Compute log probability for numerical stability
    // log P(X = k) = log C(k+r-1, k) + r*log(p) + k*log(1-p)
    const double logCoeff = logGeneralizedBinomialCoefficient(k);
    const double logProb = logCoeff + r_ * logP_ + k * log1MinusP_;
    
    const double prob = std::exp(logProb);
    
    // Ensure numerical stability
    if (std::isnan(prob) || prob < 0.0) {
        return 0.0;
    }
    
    assert(prob <= 1.0);
    return prob;
}

/**
 * Fits the distribution parameters to the given data using method of moments.
 * 
 * For Negative Binomial distribution, the method of moments estimators are:
 * p̂ = mean / variance (if variance > mean)
 * r̂ = mean² / (variance - mean) (if variance > mean)
 * 
 * If variance ≤ mean, the negative binomial model is not appropriate 
 * (indicates under-dispersion), so we fall back to default parameters.
 * 
 * @param values Vector of observed data points
 */                   
void NegativeBinomialDistribution::fit(const std::vector<Observation>& values) {
    const auto N = values.size();

    // Handle edge cases: empty data or single data point
    if (N == 0) {
        reset();
        return;
    }
    
    if (N == 1) {
        // For single point, use default parameters as estimation is not meaningful
        reset();
        return;
    }

    // Filter valid non-negative observations
    std::vector<double> validObs;
    for (const auto& val : values) {
        if (val >= 0.0 && !std::isnan(val) && !std::isinf(val)) {
            validObs.push_back(val);
        }
    }
    
    if (validObs.size() < 2) {
        reset(); // Need at least 2 points for variance estimation
        return;
    }

    // Calculate sample mean
    const double sum = std::accumulate(validObs.begin(), validObs.end(), 0.0);
    const double sampleMean = sum / static_cast<double>(validObs.size());
    
    // Calculate sample variance (using sample variance with N-1 denominator)
    double sumSquaredDiffs = 0.0;
    for (const auto& val : validObs) {
        const double diff = val - sampleMean;
        sumSquaredDiffs += diff * diff;
    }
    const double sampleVariance = sumSquaredDiffs / static_cast<double>(validObs.size() - 1);
    
    // Check if negative binomial is appropriate (requires variance > mean for over-dispersion)
    if (sampleVariance <= sampleMean || sampleMean <= 0.0) {
        reset(); // Fall back to default parameters
        return;
    }
    
    // Method of moments estimators
    const double pHat = sampleMean / sampleVariance;
    const double rHat = (sampleMean * sampleMean) / (sampleVariance - sampleMean);
    
    // Validate estimated parameters
    if (std::isnan(pHat) || std::isinf(pHat) || pHat <= 0.0 || pHat > 1.0 ||
        std::isnan(rHat) || std::isinf(rHat) || rHat <= 0.0) {
        reset(); // Fall back to default parameters
        return;
    }
    
    // Set the estimated parameters
    p_ = pHat;
    r_ = rHat;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (r = 5.0, p = 0.5).
 * This corresponds to a moderate negative binomial distribution.
 */
void NegativeBinomialDistribution::reset() noexcept {
    r_ = 5.0;
    p_ = 0.5;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Returns a string representation of the distribution following the standardized format.
 * 
 * @return String describing the distribution parameters and statistics
 */
std::string NegativeBinomialDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Negative Binomial Distribution:\n";
    oss << "      r (successes) = " << r_ << "\n";
    oss << "      p (success probability) = " << p_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, 
        const libhmm::NegativeBinomialDistribution& distribution) {
    os << "Negative Binomial Distribution:" << std::endl;
    os << "    r = " << distribution.getR() << std::endl;
    os << "    p = " << distribution.getP() << std::endl;
    os << std::endl;
    
    return os;
}

} // namespace libhmm
