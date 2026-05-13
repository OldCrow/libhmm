#include "libhmm/io/hmm_json.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "libhmm/distributions/distributions.h"
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
// A 4096-state model requires a 4096×4096 transition matrix (~128 MB of
// doubles), which covers every realistic research use case. Values above this
// almost certainly indicate corrupted or adversarial input.
constexpr std::size_t kMaxHmmStates = 4096;

// Maximum JSON input size accepted by from_json(string_view).
// A fully-saturated 4096-state HMM with max-precision doubles is well under
// 10 MB; anything larger is rejected before any parsing begins.
constexpr std::size_t kMaxJsonInputBytes = 10UL * 1024UL * 1024UL; // 10 MB

// =============================================================================
// Distribution factory — CC 2 dispatch

using FactoryFn = std::unique_ptr<EmissionDistribution> (*)(json::Reader &);

// Keyed on the "type" string written by each distribution's to_json().
const std::unordered_map<std::string, FactoryFn> kFactory = {
    {"Gaussian", &GaussianDistribution::from_json},
    {"Exponential", &ExponentialDistribution::from_json},
    {"Gamma", &GammaDistribution::from_json},
    {"Beta", &BetaDistribution::from_json},
    {"Weibull", &WeibullDistribution::from_json},
    {"LogNormal", &LogNormalDistribution::from_json},
    {"Pareto", &ParetoDistribution::from_json},
    {"NegativeBinomial", &NegativeBinomialDistribution::from_json},
    {"ChiSquared", &ChiSquaredDistribution::from_json},
    {"StudentT", &StudentTDistribution::from_json},
    {"Poisson", &PoissonDistribution::from_json},
    {"Binomial", &BinomialDistribution::from_json},
    {"Discrete", &DiscreteDistribution::from_json},
    {"Uniform", &UniformDistribution::from_json},
    {"Rayleigh", &RayleighDistribution::from_json},
    {"VonMises", &VonMisesDistribution::from_json},
};

/// Parse one distribution object: reader is positioned before the '{'.
std::unique_ptr<EmissionDistribution> read_distribution(json::Reader &r) {
    r.consume('{');
    r.read_key(); // "type"
    const std::string type = r.read_string();
    const auto it = kFactory.find(type);
    if (it == kFactory.end())
        throw std::runtime_error("HMM JSON: unknown distribution type \"" + type + "\"");
    return it->second(r); // reads remaining fields + closing '}'
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
    // Reject oversized inputs before any parsing to avoid slow traversal of
    // huge buffers. The limit is intentionally generous (see kMaxJsonInputBytes).
    if (src.size() > kMaxJsonInputBytes)
        throw std::runtime_error("HMM JSON: input exceeds maximum allowed size (" +
                                 std::to_string(kMaxJsonInputBytes / (1024 * 1024)) + " MB)");

    json::Reader r(src);
    r.consume('{');

    r.read_key(); // "states"
    const double N_raw = r.read_double();
    // Guard against non-finite values: static_cast<size_t>(inf) is undefined behaviour.
    // Guard against values > kMaxHmmStates: an N×N matrix of doubles requires N² × 8 bytes;
    // at N=4096 that is ~128 MB, covering every realistic research use case.
    if (!std::isfinite(N_raw) || N_raw < 1.0 || N_raw > static_cast<double>(kMaxHmmStates))
        throw std::runtime_error("HMM JSON: states must be an integer in [1, " +
                                 std::to_string(kMaxHmmStates) + "]");
    const std::size_t N = static_cast<std::size_t>(N_raw);

    // Pass N as the element cap: any pi or trans array longer than N is malformed
    // and must not be allowed to grow the heap before the post-hoc size checks fire.
    r.read_key(); // "pi"
    const auto pi_data = r.read_double_array(N);

    r.read_key(); // "trans"
    const auto trans_rows = r.read_double_matrix(N, N);

    r.read_key(); // "distributions"
    std::vector<std::unique_ptr<EmissionDistribution>> emis;
    emis.reserve(N);

    r.consume('[');
    if (!r.at(']')) {
        emis.push_back(read_distribution(r));
        while (r.at(',')) {
            r.consume(',');
            emis.push_back(read_distribution(r));
        }
    }
    r.consume(']');

    r.consume('}');

    // Validate dimensions before constructing
    if (pi_data.size() != N)
        throw std::runtime_error("HMM JSON: pi size mismatch");
    if (trans_rows.size() != N)
        throw std::runtime_error("HMM JSON: trans row count mismatch");
    if (emis.size() != N)
        throw std::runtime_error("HMM JSON: distribution count mismatch");

    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i)
        pi[i] = pi_data[i];

    Matrix trans(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        if (trans_rows[i].size() != N)
            throw std::runtime_error("HMM JSON: trans row length mismatch at row " +
                                     std::to_string(i));
        for (std::size_t j = 0; j < N; ++j)
            trans(i, j) = trans_rows[i][j];
    }

    return Hmm(std::move(trans), std::move(emis), std::move(pi));
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

} // namespace libhmm
