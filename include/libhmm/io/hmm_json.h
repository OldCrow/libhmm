#pragma once

// HMM JSON serialization/deserialization.
//
// Provides four free functions:
//   to_json   — serialize Hmm → JSON string
//   from_json — deserialize JSON string → Hmm
//   save_json — write JSON string to a file
//   load_json — read JSON string from a file and deserialize
//
// The JSON schema is:
//   {
//     "states": <N>,
//     "pi":    [<p0>, ..., <pN-1>],
//     "trans": [[<row0>], ..., [<rowN-1>]],
//     "distributions": [
//         {"type":"<TypeName>", ...params...},
//         ...
//     ]
//   }
//
// All doubles are serialized with max_digits10 precision for exact round-trip.

#include <filesystem>
#include <string>
#include <string_view>

#include "libhmm/hmm.h"

namespace libhmm {

/// Serialize an HMM to a compact JSON string.
[[nodiscard]] std::string to_json(const Hmm &hmm);

/// Deserialize an HMM from a JSON string produced by to_json().
/// Throws std::runtime_error on malformed input.
Hmm from_json(std::string_view src);

/// Write hmm as JSON to filepath.
/// Creates parent directories as needed.
/// Throws std::runtime_error on I/O failure.
void save_json(const Hmm &hmm, const std::filesystem::path &filepath);

/// Read and deserialize an HMM from a JSON file at filepath.
/// Throws std::runtime_error on I/O or parse failure.
Hmm load_json(const std::filesystem::path &filepath);

} // namespace libhmm
