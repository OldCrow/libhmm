#pragma once

// Internal JSON utilities for the libhmm serializer/deserializer.
//
// This is NOT a general-purpose JSON library.  It handles exactly the
// schema used by libhmm's HMM files:
//   - Objects with string and double scalar fields
//   - Arrays of doubles (pi, distribution parameters)
//   - 2-D arrays of doubles (transition matrices)
//
// Do not include this header from distribution or training code;
// it is an implementation detail of src/io/.

#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace libhmm {
namespace json {

// =============================================================================
// Write helpers — produce JSON text fragments
// =============================================================================

/// Double with enough precision for exact round-trip (numeric_limits max_digits10).
[[nodiscard]] std::string write_double(double v);

/// JSON array of doubles: [1.0, 2.0, 3.0]
[[nodiscard]] std::string write_array(std::span<const double> v);

/// JSON 2-D array from a row-major flat buffer (rows × cols elements):
/// [[r0c0, r0c1], [r1c0, r1c1], ...]
[[nodiscard]] std::string write_matrix(std::size_t rows, std::size_t cols,
                                       std::span<const double> data);

/// JSON object for a distribution with only scalar (double) fields.
/// Example: write_distribution("Gaussian", {{"mu", 0.0}, {"sigma", 1.0}})
/// → {"type":"Gaussian","mu":0.0,"sigma":1.0}
[[nodiscard]] std::string
write_distribution(std::string_view type,
                   std::initializer_list<std::pair<std::string_view, double>> fields);

/// JSON object for a distribution that has one array field (e.g. Discrete).
/// Scalar fields appear before the array field.
[[nodiscard]] std::string
write_distribution_with_array(std::string_view type,
                              std::initializer_list<std::pair<std::string_view, double>> scalars,
                              std::string_view array_key, std::span<const double> array_val);

// =============================================================================
// Reader — schema-aware tokenizer over a JSON string
// =============================================================================

/// Lightweight schema-aware tokenizer.  Operates against a fixed, known schema;
/// does not build a generic parse tree.
///
/// Typical call sequence when reading a distribution object:
///   r.consume('{');
///   r.read_key();           // "type"
///   std::string t = r.read_string();
///   // dispatch on t, then call distribution-specific from_json(r)
///   // which reads remaining fields and the closing '}'
class Reader {
public:
    explicit Reader(std::string_view src) noexcept : src_(src) {}

    // ---- Low-level ----

    /// Skip JSON whitespace (space, tab, CR, LF).
    void skip_ws() noexcept;

    /// Assert the next non-whitespace character is c and advance past it.
    /// Throws std::runtime_error if the character does not match.
    void consume(char c);

    /// Peek at the next non-whitespace character without advancing.
    [[nodiscard]] char peek();

    /// Return true if the next non-whitespace character is c.
    [[nodiscard]] bool at(char c);

    // ---- Value readers ----

    /// Read a JSON string literal ("...") and return its contents (no escape processing).
    [[nodiscard]] std::string read_string();

    /// Read a JSON number and return it as double.
    [[nodiscard]] double read_double();

    /// Read a JSON array of numbers: [d0, d1, ...] → vector<double>.
    /// Throws std::runtime_error if the array contains more than max_elements entries.
    /// Default is effectively unlimited; callers with a known expected size should pass it.
    [[nodiscard]] std::vector<double>
    read_double_array(std::size_t max_elements = std::numeric_limits<std::size_t>::max());

    /// Read a JSON 2-D array: [[r0c0,...], [r1c0,...], ...] → vector<vector<double>>.
    /// Throws if the matrix exceeds max_rows rows or any row exceeds max_cols_per_row entries.
    [[nodiscard]] std::vector<std::vector<double>>
    read_double_matrix(std::size_t max_rows = std::numeric_limits<std::size_t>::max(),
                       std::size_t max_cols_per_row = std::numeric_limits<std::size_t>::max());

    // ---- Object helpers ----

    /// Read a JSON object key including the trailing colon: "key" → "key".
    /// Optionally preceded by a comma if not the first key.
    std::string read_key();

private:
    std::string_view src_;
    std::size_t pos_{0};
};

} // namespace json
} // namespace libhmm
