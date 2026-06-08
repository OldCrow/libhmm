#include "libhmm/distributions/independent_components_distribution.h"

#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "libhmm/distributions/gaussian_distribution.h"

namespace libhmm {

// =============================================================================
// Construction
// =============================================================================

IndependentComponentsDistribution::IndependentComponentsDistribution(std::size_t dim)
    : dim_{dim}
{
    if (dim == 0) {
        throw std::invalid_argument("IndependentComponentsDistribution: dim must be > 0");
    }
    components_.reserve(dim);
    for (std::size_t d = 0; d < dim; ++d) {
        components_.push_back(std::make_unique<GaussianDistribution>());
    }
}

IndependentComponentsDistribution::IndependentComponentsDistribution(
    std::vector<std::unique_ptr<EmissionDistribution>> components)
    : dim_{components.size()}, components_{std::move(components)}
{
    if (dim_ == 0) {
        throw std::invalid_argument(
            "IndependentComponentsDistribution: components must be non-empty");
    }
    for (std::size_t d = 0; d < dim_; ++d) {
        if (!components_[d]) {
            throw std::invalid_argument(
                "IndependentComponentsDistribution: component " +
                std::to_string(d) + " is null");
        }
    }
}

IndependentComponentsDistribution::IndependentComponentsDistribution(
    const IndependentComponentsDistribution& other)
    : dim_{other.dim_}
{
    components_.reserve(dim_);
    for (const auto& c : other.components_) {
        components_.push_back(c->clone());
    }
}

// =============================================================================
// Evaluation
// =============================================================================

double IndependentComponentsDistribution::getLogProbability(
    const ObservationVectorView& x) const noexcept
{
    if (x.size() != dim_) return -std::numeric_limits<double>::infinity();
    double logp = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) {
        logp += components_[d]->getLogProbability(x[d]);
    }
    return logp;
}

double IndependentComponentsDistribution::getProbability(
    const ObservationVectorView& x) const
{
    return std::exp(getLogProbability(x));
}

// =============================================================================
// Fitting
// =============================================================================

void IndependentComponentsDistribution::fit(
    std::span<const ObservationVectorView> data)
{
    if (data.empty()) { reset(); return; }
    const std::size_t n = data.size();
    std::vector<double> dim_data(n);
    for (std::size_t d = 0; d < dim_; ++d) {
        for (std::size_t i = 0; i < n; ++i) {
            dim_data[i] = data[i][d];
        }
        components_[d]->fit(dim_data);
    }
}

void IndependentComponentsDistribution::fit(
    std::span<const ObservationVectorView> data,
    std::span<const double> weights)
{
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW <= 0.0 || data.empty()) return;

    const std::size_t n = data.size();
    std::vector<double> dim_data(n);
    std::vector<double> w_vec(weights.begin(), weights.end());

    for (std::size_t d = 0; d < dim_; ++d) {
        for (std::size_t i = 0; i < n; ++i) {
            dim_data[i] = data[i][d];
        }
        components_[d]->fit(dim_data, w_vec);
    }
}

void IndependentComponentsDistribution::reset() noexcept
{
    for (auto& c : components_) { c->reset(); }
}

// =============================================================================
// Sampling
// =============================================================================

double IndependentComponentsDistribution::sample(std::mt19937_64&) const
{
    throw std::logic_error(
        "IndependentComponentsDistribution::sample() returns double which is "
        "not meaningful for D-dimensional distributions. Use sample_mv().");
}

std::vector<double> IndependentComponentsDistribution::sample_mv(
    std::mt19937_64& rng) const
{
    std::vector<double> result(dim_);
    for (std::size_t d = 0; d < dim_; ++d) {
        result[d] = components_[d]->sample(rng);
    }
    return result;
}

// =============================================================================
// Metadata
// =============================================================================

bool IndependentComponentsDistribution::isDiscrete() const noexcept
{
    for (const auto& c : components_) {
        if (!c->isDiscrete()) return false;
    }
    return true;
}

std::size_t IndependentComponentsDistribution::getNumParameters() const noexcept
{
    std::size_t total = 0;
    for (const auto& c : components_) { total += c->getNumParameters(); }
    return total;
}

std::string IndependentComponentsDistribution::to_json() const
{
    // Phase I will define the full multivariate JSON schema.
    // For now emit a minimal JSON object that round-trips the type tag.
    std::ostringstream oss;
    oss << "{\"type\":\"IndependentComponents\",\"dim\":" << dim_ << "}";
    return oss.str();
}

std::string IndependentComponentsDistribution::toString() const
{
    std::ostringstream oss;
    oss << "IndependentComponents Distribution (D=" << dim_ << "):\n";
    for (std::size_t d = 0; d < dim_; ++d) {
        oss << "  [" << d << "] " << components_[d]->toString() << "\n";
    }
    return oss.str();
}

} // namespace libhmm
