#pragma once

// HMM JSON serialization/deserialization.
//
// Scalar API (unchanged from v3):
//   to_json(Hmm)   — serialize scalar HMM → JSON string
//   from_json      — deserialize JSON string → Hmm
//   save_json      — write scalar JSON to file
//   load_json      — read scalar JSON from file
//
// Multivariate API (v4):
//   to_json(HmmMV) — serialize MV HMM → JSON string
//   from_json_mv   — deserialize MV JSON string → HmmMV
//   save_json_mv   — write MV JSON to file
//   load_json_mv   — read MV JSON from file
//
// Scalar schema (backward-compatible with v3):
//   {"states":N, "pi":[...], "trans":[[...]], "distributions":[...]}
//
// Multivariate schema (v4):
//   {"libhmm_version":"4", "obs_type":"multivariate", "dimensions":D,
//    "states":N, "pi":[...], "trans":[[...]], "distributions":[...]}
//
// All doubles use max_digits10 precision for exact round-trip.

#include <filesystem>
#include <string>
#include <string_view>

#include "libhmm/hmm.h"

namespace libhmm {

// =============================================================================
// Scalar HMM (Obs=double, v3-compatible)
// =============================================================================

/// Serialize a scalar HMM to a compact JSON string.
[[nodiscard]] std::string to_json(const Hmm& hmm);

/// Deserialize a scalar HMM from a JSON string produced by to_json(Hmm).
/// @throws std::runtime_error on malformed input.
[[nodiscard]] Hmm from_json(std::string_view src);

/// Write scalar HMM as JSON to @p filepath (creates parent directories).
/// @throws std::runtime_error on I/O failure.
void save_json(const Hmm& hmm, const std::filesystem::path& filepath);

/// Read and deserialize a scalar HMM from a JSON file.
/// @throws std::runtime_error on I/O or parse failure.
[[nodiscard]] Hmm load_json(const std::filesystem::path& filepath);

// =============================================================================
// Multivariate HMM (Obs=ObservationVectorView, v4)
// =============================================================================

/// Serialize a multivariate HMM to a compact JSON string.
/// Writes the v4 schema including obs_type, dimensions, and full distribution
/// parameters (means, variances/covariances, component distributions).
[[nodiscard]] std::string to_json(const HmmMV& hmm);

/**
 * @brief Deserialize a multivariate HMM from a v4 JSON string.
 *
 * Expects the v4 schema written by to_json(HmmMV).  Validates obs_type and
 * dimensions; throws if the input is a scalar schema or is malformed.
 *
 * @throws std::runtime_error on malformed input, unknown distribution type,
 *         or mismatched dimensions.
 */
[[nodiscard]] HmmMV from_json_mv(std::string_view src);

/// Write multivariate HMM as JSON to @p filepath (creates parent directories).
/// @throws std::runtime_error on I/O failure.
void save_json_mv(const HmmMV& hmm, const std::filesystem::path& filepath);

/// Read and deserialize a multivariate HMM from a JSON file.
/// @throws std::runtime_error on I/O or parse failure.
[[nodiscard]] HmmMV load_json_mv(const std::filesystem::path& filepath);

} // namespace libhmm
