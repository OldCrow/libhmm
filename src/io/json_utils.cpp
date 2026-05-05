#include "libhmm/io/json_utils.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace libhmm {
namespace json {

// =============================================================================
// Write helpers
// =============================================================================

std::string write_double(double v) {
    // Use max_digits10 and the classic "C" locale to guarantee an exact
    // round-trip with '.' as the decimal separator on all platforms.
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss.precision(std::numeric_limits<double>::max_digits10);
    oss << v;
    return oss.str();
}

std::string write_array(std::span<const double> v) {
    std::string s;
    s += '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i)
            s += ',';
        s += write_double(v[i]);
    }
    s += ']';
    return s;
}

std::string write_matrix(std::size_t rows, std::size_t cols, std::span<const double> data) {
    std::string s;
    s += '[';
    for (std::size_t i = 0; i < rows; ++i) {
        if (i)
            s += ',';
        s += write_array(data.subspan(i * cols, cols));
    }
    s += ']';
    return s;
}

std::string write_distribution(std::string_view type,
                               std::initializer_list<std::pair<std::string_view, double>> fields) {
    std::string s;
    s += "{\"type\":\"";
    s += type;
    s += '"';
    for (const auto &[k, v] : fields) {
        s += ",\"";
        s += k;
        s += "\":";
        s += write_double(v);
    }
    s += '}';
    return s;
}

std::string
write_distribution_with_array(std::string_view type,
                              std::initializer_list<std::pair<std::string_view, double>> scalars,
                              std::string_view array_key, std::span<const double> array_val) {
    std::string s;
    s += "{\"type\":\"";
    s += type;
    s += '"';
    for (const auto &[k, v] : scalars) {
        s += ",\"";
        s += k;
        s += "\":";
        s += write_double(v);
    }
    s += ",\"";
    s += array_key;
    s += "\":";
    s += write_array(array_val);
    s += '}';
    return s;
}

// =============================================================================
// Reader implementation
// =============================================================================

void Reader::skip_ws() noexcept {
    while (pos_ < src_.size() &&
           (src_[pos_] == ' ' || src_[pos_] == '\t' || src_[pos_] == '\r' || src_[pos_] == '\n'))
        ++pos_;
}

void Reader::consume(char c) {
    skip_ws();
    if (pos_ >= src_.size() || src_[pos_] != c) {
        std::string msg = "json::Reader: expected '";
        msg += c;
        msg += '\'';
        if (pos_ < src_.size()) {
            msg += ", got '";
            msg += src_[pos_];
            msg += '\'';
        } else {
            msg += " but reached end of input";
        }
        throw std::runtime_error(msg);
    }
    ++pos_;
}

char Reader::peek() {
    skip_ws();
    if (pos_ >= src_.size())
        throw std::runtime_error("json::Reader: unexpected end of input");
    return src_[pos_];
}

bool Reader::at(char c) {
    skip_ws();
    return pos_ < src_.size() && src_[pos_] == c;
}

std::string Reader::read_string() {
    consume('"');
    const std::size_t start = pos_;
    while (pos_ < src_.size() && src_[pos_] != '"')
        ++pos_;
    if (pos_ >= src_.size())
        throw std::runtime_error("json::Reader: unterminated string");
    std::string result(src_.substr(start, pos_ - start));
    ++pos_; // consume closing '"'
    return result;
}

double Reader::read_double() {
    skip_ws();
    if (pos_ >= src_.size())
        throw std::runtime_error("json::Reader: unexpected end of input");
    // std::from_chars for floating-point is not available on AppleClang / libc++.
    // std::strtod provides the same consumed-position semantics and is portable.
    // write_double() imbues std::locale::classic() so the decimal separator is
    // always '.' — strtod uses the same convention under the default C locale.
    const char *begin = src_.data() + pos_;
    char *end_ptr = nullptr;
    errno = 0;
    const double value = std::strtod(begin, &end_ptr);
    if (end_ptr == begin)
        throw std::runtime_error("json::Reader: failed to parse number");
    if (errno == ERANGE)
        throw std::runtime_error("json::Reader: number out of range");
    pos_ = static_cast<std::size_t>(end_ptr - src_.data());
    return value;
}

std::vector<double> Reader::read_double_array(std::size_t max_elements) {
    consume('[');
    std::vector<double> result;
    if (!at(']')) {
        result.push_back(read_double());
        while (at(',')) {
            // Check before consuming the next element so we never read beyond the limit.
            if (result.size() >= max_elements)
                throw std::runtime_error("json::Reader: array exceeds maximum allowed size (" +
                                         std::to_string(max_elements) + " elements)");
            consume(',');
            result.push_back(read_double());
        }
    }
    consume(']');
    return result;
}

std::vector<std::vector<double>> Reader::read_double_matrix(std::size_t max_rows,
                                                            std::size_t max_cols_per_row) {
    consume('[');
    std::vector<std::vector<double>> result;
    if (!at(']')) {
        result.push_back(read_double_array(max_cols_per_row));
        while (at(',')) {
            if (result.size() >= max_rows)
                throw std::runtime_error("json::Reader: matrix exceeds maximum allowed rows (" +
                                         std::to_string(max_rows) + ")");
            consume(',');
            result.push_back(read_double_array(max_cols_per_row));
        }
    }
    consume(']');
    return result;
}

std::string Reader::read_key() {
    skip_ws();
    // Consume leading comma between key-value pairs if present.
    if (pos_ < src_.size() && src_[pos_] == ',') {
        ++pos_;
        skip_ws();
    }
    std::string key = read_string();
    consume(':');
    return key;
}

} // namespace json
} // namespace libhmm
