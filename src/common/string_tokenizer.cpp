#include "libhmm/common/string_tokenizer.h"

namespace libhmm{

std::size_t StringTokenizer::countTokens() {
    if (count_ >= 0) {
        return static_cast<std::size_t>(count_);
    }

    std::string::size_type n = 0;
    std::string::size_type i = 0;

    while (true) {
        // advance to the first token
        i = str_.find_first_not_of(delim_, i);
        if (i == std::string::npos) {
            break;
        }

        // advance to the next delimiter
        i = str_.find_first_of(delim_, i + 1);
        ++n;
        if (i == std::string::npos) {
            break;
        }
    }
    count_ = static_cast<int>(n);
    return n;
}

void StringTokenizer::nextToken(std::string& s) {
    if (begin_ == std::string::npos) {
        throw std::runtime_error("No more tokens available");
    }

    if (end_ != std::string::npos) {
        s = str_.substr(begin_, end_ - begin_);
        begin_ = str_.find_first_not_of(delim_, end_);
        end_ = str_.find_first_of(delim_, begin_);
    } else {
        s = str_.substr(begin_);
        begin_ = std::string::npos; // No more tokens
    }
}

std::string StringTokenizer::nextToken() {
    std::string token;
    nextToken(token);
    return token;
}

}//namespace
