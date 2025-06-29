#include "libhmm/distributions/negative_binomial_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>     // For std::accumulate (not in common.h)
#include <algorithm>   // For std::for_each (exists in common.h, included for clarity)

using namespace libhmm::constants;

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
        return math::ZERO_DOUBLE;
    }
    
    // Round to nearest integer and check if it's in valid range
    auto k = static_cast<int>(std::round(value));
    if (k < 0) {
        return math::ZERO_DOUBLE;
    }
    
    // Handle edge cases
    if (p_ == math::ONE) {
        return (k == 0) ? math::ONE : math::ZERO_DOUBLE;
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
    if (std::isnan(prob) || prob < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }
    
    assert(prob <= math::ONE);
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
    // Handle edge case: empty data
    if (values.empty()) {
        reset();
        return;
    }
    
    // Single-pass Welford's algorithm for mean and variance calculation
    double mean = math::ZERO_DOUBLE;
    double m2 = math::ZERO_DOUBLE;  // Sum of squared differences from current mean
    std::size_t validCount = 0;
    
    for (const auto& val : values) {
        // Validate: finite non-negative values only
        if (val >= math::ZERO_DOUBLE && std::isfinite(val)) {
            ++validCount;
            const double delta = val - mean;
            mean += delta / static_cast<double>(validCount);
            const double delta2 = val - mean;
            m2 += delta * delta2;
        }
    }
    
    // Handle edge cases: insufficient valid data
    if (validCount < 2) {
        reset(); // Need at least 2 points for variance estimation
        return;
    }
    
    // Calculate sample variance using Bessel's correction (N-1)
    const double sampleMean = mean;
    const double sampleVariance = m2 / static_cast<double>(validCount - 1);
    
    // Check if negative binomial is appropriate (requires variance > mean for over-dispersion)
    if (sampleVariance <= sampleMean || sampleMean <= math::ZERO_DOUBLE) {
        reset(); // Fall back to default parameters
        return;
    }
    
    // Method of moments estimators
    const double pHat = sampleMean / sampleVariance;
    const double rHat = (sampleMean * sampleMean) / (sampleVariance - sampleMean);
    
    // Validate estimated parameters
    if (std::isnan(pHat) || std::isinf(pHat) || pHat <= math::ZERO_DOUBLE || pHat > math::ONE ||
        std::isnan(rHat) || std::isinf(rHat) || rHat <= math::ZERO_DOUBLE) {
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
    p_ = math::HALF;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Returns a string representation of the distribution following the standardized format.
 * 
 * @return String describing the distribution parameters and statistics
 */
std::string NegativeBinomialDistribution::toString() const {
    std::ostringstream oss{};
    oss << std::fixed << std::setprecision(6);
    oss << "Negative Binomial Distribution:\n";
    oss << "      r (successes) = " << r_ << "\n";
    oss << "      p (success probability) = " << p_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

double NegativeBinomialDistribution::getLogProbability(double value) const noexcept {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Round to nearest integer and check if it's in valid range
    auto k = static_cast<int>(std::round(value));
    if (k < 0) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Handle edge cases
    if (p_ == math::ONE) {
        return (k == 0) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // Compute log probability for numerical stability
    // log P(X = k) = log C(k+r-1, k) + r*log(p) + k*log(1-p)
    const double logCoeff = logGeneralizedBinomialCoefficient(k);
    const double logProb = logCoeff + r_ * logP_ + k * log1MinusP_;
    
    return logProb;
}

double NegativeBinomialDistribution::CDF(double value) noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }
    
    auto k = static_cast<int>(std::floor(value));
    
    // Handle boundary cases
    if (k < 0) {
        return math::ZERO_DOUBLE;
    }
    
    // Compute CDF as cumulative sum: P(X <= k) = sum_{i=0}^{k} P(X = i)
    // For efficiency, we limit computation to reasonable range
    const int maxK = std::min(k, 1000); // Practical upper limit for computation
    
    double cdf = math::ZERO_DOUBLE;
    for (int i = 0; i <= maxK; ++i) {
        cdf += getProbability(static_cast<double>(i));
    }
    
    return std::min(math::ONE, cdf);
}

bool NegativeBinomialDistribution::operator==(const NegativeBinomialDistribution& other) const {
    const double tolerance = 1e-10;
    return (std::abs(r_ - other.r_) < tolerance) && 
           (std::abs(p_ - other.p_) < tolerance);
}

std::istream& operator>>(std::istream& is, libhmm::NegativeBinomialDistribution& distribution) {
    std::string token;
    double r, p;
    
    // Expected format: "NegativeBinomial(r,p)" or "r p"
    if (is >> token) {
        if (token.find("NegativeBinomial") != std::string::npos) {
            // Parse formatted input: NegativeBinomial(r,p)
            std::string fullInput = token;
            std::string remaining;
            std::getline(is, remaining);
            fullInput += remaining;
            
            // Find the opening and closing parentheses
            size_t openParen = fullInput.find('(');
            size_t closeParen = fullInput.find(')');
            size_t comma = fullInput.find(',');
            
            if (openParen != std::string::npos && closeParen != std::string::npos && comma != std::string::npos) {
                std::string rStr = fullInput.substr(openParen + 1, comma - openParen - 1);
                std::string pStr = fullInput.substr(comma + 1, closeParen - comma - 1);
                
                try {
                    r = std::stod(rStr);
                    p = std::stod(pStr);
                } catch (const std::exception&) {
                    is.setstate(std::ios::failbit);
                    return is;
                }
            } else {
                is.setstate(std::ios::failbit);
                return is;
            }
        } else {
            // Assume first token is r
            try {
                r = std::stod(token);
                is >> p;
            } catch (const std::exception&) {
                is.setstate(std::ios::failbit);
                return is;
            }
        }
        
        try {
            distribution.setParameters(r, p);
        } catch (const std::exception&) {
            is.setstate(std::ios::failbit);
        }
    }
    
    return is;
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
