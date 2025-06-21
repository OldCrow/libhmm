#ifndef STRINGTOKENIZER_H_
#define STRINGTOKENIZER_H_

#include <string>
#include <iostream>
#include <stdexcept>

namespace libhmm{

class StringTokenizer{

public: 
    /// Constructor with string and optional delimiter
    /// @param s The string to tokenize
    /// @param delim The delimiter characters (default is whitespace)
    explicit StringTokenizer(const std::string& s, const char* delim = nullptr):
        str_(s), count_(-1), begin_(0), end_(0) {

        if (!delim) {
            delim_ = " \f\n\r\t\v"; // default to whitespace
        } else {
            delim_ = delim;
        }

        // Point to first token
        begin_ = str_.find_first_not_of(delim_);
        end_ = str_.find_first_of(delim_, begin_);
    }

    /// Count the total number of tokens
    /// @return The number of tokens in the string
    std::size_t countTokens();

    /// Check if there are more tokens available
    /// @return true if more tokens exist, false otherwise
    bool hasMoreTokens() const noexcept { 
        return begin_ != std::string::npos; 
    }

    /// Get the next token
    /// @param s Reference to string where token will be stored
    /// @throws std::runtime_error if no more tokens are available
    void nextToken(std::string& s);

    /// Get the next token as return value
    /// @return The next token as a string
    /// @throws std::runtime_error if no more tokens are available
    std::string nextToken();

private:
    // Prevent default construction
    StringTokenizer() = delete;
    
    std::string delim_;
    std::string str_;
    mutable int count_; // mutable because countTokens() can be called on const objects
    std::string::size_type begin_;
    std::string::size_type end_;
};

} //namespace

#endif
