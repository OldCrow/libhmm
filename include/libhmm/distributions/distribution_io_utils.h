#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace libhmm::detail {

/// Splits a "name=value" token at the first '=', strips whitespace from both
/// parts, and returns {name, value}.
/// Throws std::invalid_argument (with context in the message) if no '=' found.
[[nodiscard]] inline std::pair<std::string, std::string>
parse_named_param(const std::string &param, const std::string &context) {
    const auto eq = param.find('=');
    if (eq == std::string::npos)
        throw std::invalid_argument("Invalid " + context + " parameter format");
    std::string name  = param.substr(0, eq);
    std::string value = param.substr(eq + 1);
    const auto trim = [](std::string &s) {
        s.erase(std::remove_if(s.begin(), s.end(),
                               [](unsigned char c) { return std::isspace(c); }),
                s.end());
    };
    trim(name);
    trim(value);
    return {std::move(name), std::move(value)};
}

} // namespace libhmm::detail
