#include "libhmm/io/hmm_json.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/distributions.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"
#include "libhmm/distributions/independent_components_distribution.h"
#include "libhmm/io/file_io_manager.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {

// =============================================================================
// Deserialization limits
// =============================================================================
//
// These constants bound attacker-controlled allocation sizes so that malformed
// or malicious JSON cannot trigger OOM, integer overflow, or UB casts.

namespace {

// Maximum number of HMM states accepted during deserialization.
constexpr std::size_t kMaxHmmStates = 4096;
// Maximum JSON input size accepted by from_json / from_json_mv.
constexpr std::size_t kMaxJsonInputBytes = 10UL * 1024UL * 1024UL; // 10 MB

/// Validate a JSON-read double as a bounded positive integer.
/// Shared by both scalar and MV deserializers.
/// @throws std::runtime_error with @p err_msg if the value is out of [lo, hi].
std::size_t checked_size(double raw, double lo, double hi, const char* err_msg) {
    if (!std::isfinite(raw) || raw < lo || raw > hi)
        throw std::runtime_error(err_msg);
    return static_cast<std::size_t>(raw);
}

// =============================================================================
// Scalar distribution factory

using FactoryFn = std::unique_ptr<EmissionDistribution> (*)(json::Reader&);

const std::unordered_map<std::string, FactoryFn> kFactory = {
    {"Gaussian",         &GaussianDistribution::from_json},
    {"Exponential",      &ExponentialDistribution::from_json},
    {"Gamma",            &GammaDistribution::from_json},
    {"Beta",             &BetaDistribution::from_json},
    {"Weibull",          &WeibullDistribution::from_json},
    {"LogNormal",        &LogNormalDistribution::from_json},
    {"Pareto",           &ParetoDistribution::from_json},
    {"NegativeBinomial", &NegativeBinomialDistribution::from_json},
    {"ChiSquared",       &ChiSquaredDistribution::from_json},
    {"StudentT",         &StudentTDistribution::from_json},
    {"Poisson",          &PoissonDistribution::from_json},
    {"Binomial",         &BinomialDistribution::from_json},
    {"Discrete",         &DiscreteDistribution::from_json},
    {"Uniform",          &UniformDistribution::from_json},
    {"Rayleigh",         &RayleighDistribution::from_json},
    {"VonMises",         &VonMisesDistribution::from_json},
};

/// Parse one scalar distribution object.  Reader is positioned before '{'.
std::unique_ptr<EmissionDistribution> read_distribution(json::Reader& r) {
    r.consume('{');
    r.read_key();
    const std::string type = r.read_string();
    const auto it = kFactory.find(type);
    if (it == kFactory.end())
        throw std::runtime_error("HMM JSON: unknown distribution type \"" + type + "\"");
    return it->second(r);
}

/// Read the scalar distribution array.  Reader positioned before '['.
std::vector<std::unique_ptr<EmissionDistribution>>
read_distribution_array(json::Reader& r, std::size_t N) {
    std::vector<std::unique_ptr<EmissionDistribution>> emis;
    emis.reserve(N);
    r.consume('[');
    if (!r.at(']')) {
        emis.push_back(read_distribution(r));
        while (r.at(',')) { r.consume(','); emis.push_back(read_distribution(r)); }
    }
    r.consume(']');
    return emis;
}

/// Validate parsed arrays and construct scalar Hmm.
Hmm build_hmm(const std::vector<double>& pi_data,
               const std::vector<std::vector<double>>& trans_rows,
               std::vector<std::unique_ptr<EmissionDistribution>> emis,
               std::size_t N)
{
    if (pi_data.size() != N)
        throw std::runtime_error("HMM JSON: pi size mismatch");
    if (trans_rows.size() != N)
        throw std::runtime_error("HMM JSON: trans row count mismatch");
    if (emis.size() != N)
        throw std::runtime_error("HMM JSON: distribution count mismatch");
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i) pi[i] = pi_data[i];
    Matrix trans(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        if (trans_rows[i].size() != N)
            throw std::runtime_error("HMM JSON: trans row length mismatch at row " +
                                     std::to_string(i));
        for (std::size_t j = 0; j < N; ++j) trans(i, j) = trans_rows[i][j];
    }
    return Hmm(std::move(trans), std::move(emis), std::move(pi));
}

} // anonymous namespace

// =============================================================================
// to_json
// =============================================================================

std::string to_json(const Hmm &hmm) {
    const std::size_t N = hmm.getNumStatesModern();
    const auto &pi = hmm.getPi();
    const auto &trans = hmm.getTrans();

    std::string s;
    s.reserve(256 + N * N * 20); // rough pre-allocation

    s += "{\"states\":";
    s += json::write_double(static_cast<double>(N));

    s += ",\"pi\":";
    s += json::write_array(std::span<const double>(pi.data(), N));

    s += ",\"trans\":";
    s += json::write_matrix(N, N, std::span<const double>(trans.data(), N * N));

    s += ",\"distributions\":[";
    for (std::size_t i = 0; i < N; ++i) {
        if (i)
            s += ',';
        s += hmm.getDistribution(i).to_json();
    }
    s += "]}";
    return s;
}

// =============================================================================
// from_json
// =============================================================================

Hmm from_json(std::string_view src) {
    if (src.size() > kMaxJsonInputBytes)
        throw std::runtime_error(
            "HMM JSON: input exceeds maximum allowed size (" +
            std::to_string(kMaxJsonInputBytes / (1024 * 1024)) + " MB)");

    json::Reader r(src);
    r.consume('{');

    // states: validated against [1, kMaxHmmStates] before any heap allocation.
    r.read_key();
    const std::size_t N = checked_size(r.read_double(), 1.0,
                                        static_cast<double>(kMaxHmmStates),
                                        "HMM JSON: states must be a positive integer");

    r.read_key(); // "pi"
    const auto pi_data = r.read_double_array(N);

    r.read_key(); // "trans"
    const auto trans_rows = r.read_double_matrix(N, N);

    r.read_key(); // "distributions"
    auto emis = read_distribution_array(r, N);
    r.consume('}');

    return build_hmm(pi_data, trans_rows, std::move(emis), N);
}

// =============================================================================
// File I/O wrappers
// =============================================================================

void save_json(const Hmm &hmm, const std::filesystem::path &filepath) {
    FileIOManager::writeTextFile(filepath, to_json(hmm));
}

Hmm load_json(const std::filesystem::path &filepath) {
    return from_json(FileIOManager::readTextFile(filepath));
}

// =============================================================================
// Multivariate IO
// =============================================================================

namespace {

// Maximum dimension accepted for MV distributions during deserialization.
// D=1024 means a 1024×1024 covariance matrix (~8 MB of doubles in JSON),
// which covers every realistic research scenario.
constexpr std::size_t kMaxMvDimensions = 1024;

/// Parse one MV distribution object.  Reader is positioned before the '{'.
/// Must be defined before read_mv_distribution_array which calls it.
std::unique_ptr<BasicEmissionDistribution<ObservationVectorView>>
read_mv_distribution(json::Reader& r) {
    r.consume('{');
    r.read_key(); // "type"
    const std::string type = r.read_string();

    if (type == "DiagonalGaussian")
        return DiagonalGaussianDistribution::from_json(r);
    if (type == "FullCovarianceGaussian")
        return FullCovarianceGaussianDistribution::from_json(r);
    if (type == "IndependentComponents") {
        // Reads the component array using the scalar kFactory defined above.
        r.read_key(); // "dim"
        const std::size_t D = checked_size(
            r.read_double(), 1.0, static_cast<double>(kMaxMvDimensions),
            "IndependentComponents JSON: invalid dim");
        r.read_key(); // "components"
        r.consume('[');
        std::vector<std::unique_ptr<EmissionDistribution>> comps;
        comps.reserve(D);
        if (!r.at(']')) {
            comps.push_back(read_distribution(r));
            while (r.at(',')) { r.consume(','); comps.push_back(read_distribution(r)); }
        }
        r.consume(']');
        r.consume('}');
        if (comps.size() != D)
            throw std::runtime_error("IndependentComponents JSON: component count mismatch");
        return std::make_unique<IndependentComponentsDistribution>(std::move(comps));
    }
    throw std::runtime_error("HMM MV JSON: unknown distribution type \"" + type + "\"");
}

/// Read the full MV distribution array.  Reader positioned before '['.
std::vector<std::unique_ptr<BasicEmissionDistribution<ObservationVectorView>>>
read_mv_distribution_array(json::Reader& r, std::size_t N) {
    std::vector<std::unique_ptr<BasicEmissionDistribution<ObservationVectorView>>> emis;
    emis.reserve(N);
    r.consume('[');
    if (!r.at(']')) {
        emis.push_back(read_mv_distribution(r));
        while (r.at(',')) { r.consume(','); emis.push_back(read_mv_distribution(r)); }
    }
    r.consume(']');
    return emis;
}

/// Validate parsed field arrays and construct HmmMV.
HmmMV build_hmm_mv(const std::vector<double>& pi_data,
                    const std::vector<std::vector<double>>& trans_rows,
                    std::vector<std::unique_ptr<BasicEmissionDistribution<ObservationVectorView>>> emis,
                    std::size_t N)
{
    if (pi_data.size() != N)
        throw std::runtime_error("HMM MV JSON: pi size mismatch");
    if (trans_rows.size() != N)
        throw std::runtime_error("HMM MV JSON: trans row count mismatch");
    if (emis.size() != N)
        throw std::runtime_error("HMM MV JSON: distribution count mismatch");

    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i) pi[i] = pi_data[i];

    Matrix trans(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        if (trans_rows[i].size() != N)
            throw std::runtime_error("HMM MV JSON: trans row length mismatch at row " +
                                     std::to_string(i));
        for (std::size_t j = 0; j < N; ++j)
            trans(i, j) = trans_rows[i][j];
    }
    return HmmMV(std::move(trans), std::move(emis), std::move(pi));
}

} // anonymous namespace (MV)

// -----------------------------------------------------------------------------
// to_json(HmmMV)
// -----------------------------------------------------------------------------

std::string to_json(const HmmMV& hmm) {
    const std::size_t N = hmm.getNumStatesModern();
    const auto& pi    = hmm.getPi();
    const auto& trans = hmm.getTrans();

    // Query D from the first distribution (getDimension() is now virtual).
    const std::size_t D = (N > 0) ? hmm.getDistribution(0).getDimension() : 0;

    std::string s;
    s.reserve(256 + N * N * 20);

    s += "{\"libhmm_version\":\"4\"";
    s += ",\"obs_type\":\"multivariate\"";
    s += ",\"dimensions\":";
    s += json::write_double(static_cast<double>(D));
    s += ",\"states\":";
    s += json::write_double(static_cast<double>(N));
    s += ",\"pi\":";
    s += json::write_array(std::span<const double>(pi.data(), N));
    s += ",\"trans\":";
    s += json::write_matrix(N, N, std::span<const double>(trans.data(), N * N));
    s += ",\"distributions\":[";
    for (std::size_t i = 0; i < N; ++i) {
        if (i) s += ',';
        s += hmm.getDistribution(i).to_json();
    }
    s += "]}";  // close distributions array and root object
    return s;
}

// -----------------------------------------------------------------------------
// from_json_mv
// -----------------------------------------------------------------------------

HmmMV from_json_mv(std::string_view src) {
    if (src.size() > kMaxJsonInputBytes)
        throw std::runtime_error(
            "HMM MV JSON: input exceeds maximum allowed size (" +
            std::to_string(kMaxJsonInputBytes / (1024 * 1024)) + " MB)");

    json::Reader r(src);
    r.consume('{');

    r.read_key(); // "libhmm_version"
    if (r.read_string() != "4")
        throw std::runtime_error("HMM MV JSON: unsupported libhmm_version");

    r.read_key(); // "obs_type"
    if (r.read_string() != "multivariate")
        throw std::runtime_error("HMM MV JSON: expected obs_type \"multivariate\"");

    r.read_key(); // "dimensions" (stored in JSON for reference; each distribution validates its own dim)
    checked_size(r.read_double(), 1.0, static_cast<double>(kMaxMvDimensions),
                 "HMM MV JSON: invalid dimensions value");

    r.read_key(); // "states"
    const std::size_t N = checked_size(r.read_double(), 1.0,
                                        static_cast<double>(kMaxHmmStates),
                                        "HMM MV JSON: states out of range");

    r.read_key(); // "pi"
    const auto pi_data = r.read_double_array(N);

    r.read_key(); // "trans"
    const auto trans_rows = r.read_double_matrix(N, N);

    r.read_key(); // "distributions"
    auto emis = read_mv_distribution_array(r, N);
    r.consume('}');

    return build_hmm_mv(pi_data, trans_rows, std::move(emis), N);
}

// -----------------------------------------------------------------------------
// File I/O wrappers (MV)
// -----------------------------------------------------------------------------

void save_json_mv(const HmmMV& hmm, const std::filesystem::path& filepath) {
    FileIOManager::writeTextFile(filepath, to_json(hmm));
}

HmmMV load_json_mv(const std::filesystem::path& filepath) {
    return from_json_mv(FileIOManager::readTextFile(filepath));
}

} // namespace libhmm
