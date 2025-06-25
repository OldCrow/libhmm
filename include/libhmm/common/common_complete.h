#ifndef COMMON_H_
#define COMMON_H_

/*
 * Standard Library includes - C++17 only, no external dependencies
 */
#include <vector>
#include <cfloat>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace libhmm
{

/* 
 * If we make all observations as double, we can truncate as needed and still
 * retain tons of precision for most of the operations.
 */
typedef double Observation;

/* 
 * We know that a state is an integral value, so...
 */
typedef int StateIndex;

}//namespace

/*
 * Custom libhmm Matrix and Vector classes (C++17 standard library only)
 * Must be included after basic typedefs to avoid naming conflicts
 */
#include "libhmm/common/matrix.h"
#include "libhmm/common/vector.h"

namespace libhmm
{

/*
 * Type aliases - Replace boost::numeric::ublas types with custom implementations
 * These provide drop-in replacement for existing Boost uBLAS usage
 */
using Matrix = BasicMatrix<Observation>;
using Vector = BasicVector<Observation>;
using ObservationSet = BasicVector<Observation>;

/*
 * Viterbi decode requires a fixed size vector for state sequences
 */
using StateSequence = BasicVector<StateIndex>;

/*
 * Training requires creating a list of all the observation sets.
 * We can't make a Vector of Vectors, but we can use std::list or std::vector
 * for things like this.
 */
typedef std::vector<ObservationSet> ObservationLists;

// Tolerance for both forms of Baum-Welch training and iterations of gamma
// functions
inline constexpr double BW_TOLERANCE = 3.0e-7;    

// Minimum possible double value that I want
// The C standard defines this as DBL_MIN, we can redefine it to be bigger.
// Note that this value essentially defines zero for the purposes of libhmm.
inline constexpr double ZERO = 1.0e-30;

/* 
 * Used in continuous distributions as a range over which a probability is valid
 */
inline constexpr double LIMIT_TOLERANCE = 1.0e-6;

/*
 * Max ViterbiTrainer iterations.
 */
inline constexpr std::size_t MAX_VITERBI_ITERATIONS = 500;

/*
 * Max iterations for gamma related functions.
 */
inline constexpr std::size_t ITMAX = 10000;

/*
 * Value of Pi.
 */
inline constexpr double PI = 3.141592653589793238462643383279502884;
 
// Custom Matrix and Vector classes initialize to zero on construction,
// but provide explicit clear functions for API compatibility
void clear_matrix( Matrix& m );
void clear_vector( Vector& v );
void clear_vector( StateSequence& v );

}//namespace

/*
 * Custom C++17 serialization functions for matrices and vectors.
 * These replace the Boost serialization functionality with lightweight,
 * standards-compliant serialization using simple XML format.
 * 
 * Design Goals:
 * - Zero external dependencies (pure C++17)
 * - Backward compatibility with existing XML format where possible
 * - Simple, human-readable XML output
 * - Efficient parsing for HMM-specific use cases
 */
namespace libhmm {
namespace serialization {

/**
 * Simple XML serialization for BasicMatrix objects
 * Replaces boost::serialization with lightweight implementation
 */
template<typename T>
class MatrixSerializer {
public:
    // Save matrix to XML format
    static void save(std::ostream& os, const BasicMatrix<T>& matrix, const std::string& name = "matrix") {
        os << "<" << name << ">\n";
        os << "  <rows>" << matrix.size1() << "</rows>\n";
        os << "  <cols>" << matrix.size2() << "</cols>\n";
        os << "  <data>\n";
        
        for (std::size_t i = 0; i < matrix.size1(); ++i) {
            os << "    <row>";
            for (std::size_t j = 0; j < matrix.size2(); ++j) {
                os << matrix(i, j);
                if (j < matrix.size2() - 1) os << " ";
            }
            os << "</row>\n";
        }
        
        os << "  </data>\n";
        os << "</" << name << ">\n";
    }
    
    // Load matrix from XML format
    static void load(std::istream& is, BasicMatrix<T>& matrix, const std::string& name = "matrix") {
        std::string line;
        std::size_t rows = 0, cols = 0;
        
        // Simple XML parsing - find opening tag
        while (std::getline(is, line)) {
            if (line.find("<" + name + ">") != std::string::npos) {
                break;
            }
        }
        
        // Read dimensions
        if (std::getline(is, line)) {
            std::size_t start = line.find("<rows>") + 6;
            std::size_t end = line.find("</rows>");
            if (start != std::string::npos && end != std::string::npos) {
                rows = std::stoull(line.substr(start, end - start));
            }
        }
        
        if (std::getline(is, line)) {
            std::size_t start = line.find("<cols>") + 6;
            std::size_t end = line.find("</cols>");
            if (start != std::string::npos && end != std::string::npos) {
                cols = std::stoull(line.substr(start, end - start));
            }
        }
        
        // Resize matrix
        matrix.resize(rows, cols);
        
        // Skip <data> tag
        std::getline(is, line);
        
        // Read matrix data
        for (std::size_t i = 0; i < rows; ++i) {
            if (std::getline(is, line)) {
                std::size_t start = line.find("<row>") + 5;
                std::size_t end = line.find("</row>");
                if (start != std::string::npos && end != std::string::npos) {
                    std::string data_str = line.substr(start, end - start);
                    std::istringstream row_stream(data_str);
                    
                    for (std::size_t j = 0; j < cols; ++j) {
                        T value;
                        row_stream >> value;
                        matrix(i, j) = value;
                    }
                }
            }
        }
    }
};

/**
 * Simple XML serialization for BasicVector objects
 * Replaces boost::serialization with lightweight implementation
 */
template<typename T>
class VectorSerializer {
public:
    // Save vector to XML format
    static void save(std::ostream& os, const BasicVector<T>& vector, const std::string& name = "vector") {
        os << "<" << name << ">\n";
        os << "  <size>" << vector.size() << "</size>\n";
        os << "  <data>";
        
        for (std::size_t i = 0; i < vector.size(); ++i) {
            os << vector[i];
            if (i < vector.size() - 1) os << " ";
        }
        
        os << "</data>\n";
        os << "</" << name << ">\n";
    }
    
    // Load vector from XML format  
    static void load(std::istream& is, BasicVector<T>& vector, const std::string& name = "vector") {
        std::string line;
        std::size_t size = 0;
        
        // Simple XML parsing - find opening tag
        while (std::getline(is, line)) {
            if (line.find("<" + name + ">") != std::string::npos) {
                break;
            }
        }
        
        // Read size
        if (std::getline(is, line)) {
            std::size_t start = line.find("<size>") + 6;
            std::size_t end = line.find("</size>");
            if (start != std::string::npos && end != std::string::npos) {
                size = std::stoull(line.substr(start, end - start));
            }
        }
        
        // Resize vector
        vector.resize(size);
        
        // Read data
        if (std::getline(is, line)) {
            std::size_t start = line.find("<data>") + 6;
            std::size_t end = line.find("</data>");
            if (start != std::string::npos && end != std::string::npos) {
                std::string data_str = line.substr(start, end - start);
                std::istringstream data_stream(data_str);
                
                for (std::size_t i = 0; i < size; ++i) {
                    T value;
                    data_stream >> value;
                    vector[i] = value;
                }
            }
        }
    }
};

/**
 * Convenience functions to match the old boost::serialization style
 */

// Matrix serialization convenience functions
template<typename Archive, typename T>
void save(Archive& ar, const BasicMatrix<T>& matrix, const std::string& name = "matrix") {
    MatrixSerializer<T>::save(ar, matrix, name);
}

template<typename Archive, typename T>
void load(Archive& ar, BasicMatrix<T>& matrix, const std::string& name = "matrix") {
    MatrixSerializer<T>::load(ar, matrix, name);
}

// Vector serialization convenience functions
template<typename Archive, typename T>
void save(Archive& ar, const BasicVector<T>& vector, const std::string& name = "vector") {
    VectorSerializer<T>::save(ar, vector, name);
}

template<typename Archive, typename T>
void load(Archive& ar, BasicVector<T>& vector, const std::string& name = "vector") {
    VectorSerializer<T>::load(ar, vector, name);
}

} // namespace serialization
} // namespace libhmm

#endif
