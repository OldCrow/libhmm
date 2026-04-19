#pragma once

/*
 * Standard Library includes - C++17 only, no external dependencies
 * Consolidated to reduce duplicate includes across the project
 */

// Core C++ standard library headers
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <utility>
#include <type_traits>

// Mathematical and numerical headers
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <limits>
#include <cassert>

// Platform/SIMD headers are in libhmm/platform/ (moved from performance/ in Phase 1 refactor)
// Include libhmm/platform/simd_platform.h if you need SIMD intrinsics

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

// linalg types (headers forward to libhmm/linalg/ as of Phase 1 refactor)
// Must be included after basic typedefs to avoid naming conflicts
#include "libhmm/common/basic_matrix.h"
#include "libhmm/common/basic_vector.h"
#include "libhmm/common/basic_matrix3d.h"

// Constants extracted to their own header in Phase 1 refactor
#include "libhmm/math/constants.h"

namespace libhmm
{

// Type aliases
using Matrix            = BasicMatrix<Observation>;
using Vector            = BasicVector<Observation>;
using ObservationSet    = BasicVector<Observation>;
using StateSequence     = BasicVector<StateIndex>;
using ObservationMatrix3D = BasicMatrix3D<Observation>;

template<typename T>
using Matrix3DTemplate  = BasicMatrix3D<T>;

typedef std::vector<ObservationSet> ObservationLists;


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

