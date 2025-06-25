#ifndef COMMON_H_
#define COMMON_H_

/*
 * Custom libhmm Matrix and Vector classes (C++17 standard library only)
 */
#include "libhmm/common/matrix.h"
#include "libhmm/common/vector.h"

/*
 * Standard Library includes for serialization replacement
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <vector>
#include <cfloat>
#include <cstddef>

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


/*
 * Used for Transmission matrices and pi vector.  Also, some of the operations
 * on HMMs require extra 2D matrices.  Observation lists will be of type Vector
 * as well...its just not obvious.
 */
typedef libhmm::Matrix<Observation> Matrix;
typedef libhmm::Vector<Observation> Vector;
typedef libhmm::Vector<Observation> ObservationSet;

/*
 * Viterbi decode requires a fixed size vector for state sequences
 */
typedef libhmm::Vector<StateIndex> StateSequence;

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
 
// Custom Matrix and Vector classes zero themselves on construction,
// but provide explicit clear functions for consistency
void clear_matrix( Matrix& m );
void clear_vector( Vector& v );
void clear_vector( StateSequence& v );

}//namespace

/*
 * Custom serialization functions for matrices and vectors using standard C++17.
 * These replace the Boost serialization functionality with lightweight,
 * standards-compliant serialization.
 */
namespace libhmm {
namespace serialization {

/**
 * Simple XML serialization for Matrix objects
 * Replaces boost::serialization with lightweight implementation
 */
class MatrixXMLSerializer {
public:
    // Save matrix to XML format
    template<typename T>
    static void save(std::ostream& os, const libhmm::Matrix<T>& matrix, const std::string& name) {
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
    template<typename T>
    static void load(std::istream& is, libhmm::Matrix<T>& matrix, const std::string& name) {
        std::string line, tag;
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
 * Simple XML serialization for Vector objects
 * Replaces boost::serialization with lightweight implementation
 */
class VectorXMLSerializer {
public:
    // Save vector to XML format
    template<typename T>
    static void save(std::ostream& os, const libhmm::Vector<T>& vector, const std::string& name) {
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
    template<typename T>
    static void load(std::istream& is, libhmm::Vector<T>& vector, const std::string& name) {
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

} // namespace serialization
} // namespace libhmm

#endif
