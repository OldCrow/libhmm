#pragma once

// XML serialization helpers for libhmm linalg types.
//
// Provides MatrixSerializer<T> and VectorSerializer<T> used by the HMM XML
// I/O layer (src/hmm.cpp, src/io/).  Extracted from common/common.h to keep
// that header lean and avoid pulling serialization templates into every
// translation unit that only needs basic types.

#include <sstream>
#include <string>

#include "libhmm/linalg/linalg_types.h"

namespace libhmm {
namespace serialization {

/**
 * Simple XML serialization for BasicMatrix objects.
 * Replaces boost::serialization with a lightweight C++17 implementation.
 */
template <typename T>
class MatrixSerializer {
public:
    /// Save matrix to XML format
    static void save(std::ostream &os, const BasicMatrix<T> &matrix,
                     const std::string &name = "matrix") {
        os << "<" << name << ">\n";
        os << "  <rows>" << matrix.size1() << "</rows>\n";
        os << "  <cols>" << matrix.size2() << "</cols>\n";
        os << "  <data>\n";

        for (std::size_t i = 0; i < matrix.size1(); ++i) {
            os << "    <row>";
            for (std::size_t j = 0; j < matrix.size2(); ++j) {
                os << matrix(i, j);
                if (j < matrix.size2() - 1)
                    os << " ";
            }
            os << "</row>\n";
        }

        os << "  </data>\n";
        os << "</" << name << ">\n";
    }

    /// Load matrix from XML format
    static void load(std::istream &is, BasicMatrix<T> &matrix, const std::string &name = "matrix") {
        std::string line;
        std::size_t rows = 0, cols = 0;

        while (std::getline(is, line)) {
            if (line.find("<" + name + ">") != std::string::npos)
                break;
        }

        if (std::getline(is, line)) {
            std::size_t start = line.find("<rows>") + 6;
            std::size_t end = line.find("</rows>");
            if (start != std::string::npos && end != std::string::npos)
                rows = std::stoull(line.substr(start, end - start));
        }

        if (std::getline(is, line)) {
            std::size_t start = line.find("<cols>") + 6;
            std::size_t end = line.find("</cols>");
            if (start != std::string::npos && end != std::string::npos)
                cols = std::stoull(line.substr(start, end - start));
        }

        matrix.resize(rows, cols);
        std::getline(is, line); // skip <data>

        for (std::size_t i = 0; i < rows; ++i) {
            if (std::getline(is, line)) {
                std::size_t start = line.find("<row>") + 5;
                std::size_t end = line.find("</row>");
                if (start != std::string::npos && end != std::string::npos) {
                    std::istringstream row_stream(line.substr(start, end - start));
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
 * Simple XML serialization for BasicVector objects.
 * Replaces boost::serialization with a lightweight C++17 implementation.
 */
template <typename T>
class VectorSerializer {
public:
    /// Save vector to XML format
    static void save(std::ostream &os, const BasicVector<T> &vector,
                     const std::string &name = "vector") {
        os << "<" << name << ">\n";
        os << "  <size>" << vector.size() << "</size>\n";
        os << "  <data>";
        for (std::size_t i = 0; i < vector.size(); ++i) {
            os << vector[i];
            if (i < vector.size() - 1)
                os << " ";
        }
        os << "</data>\n";
        os << "</" << name << ">\n";
    }

    /// Load vector from XML format
    static void load(std::istream &is, BasicVector<T> &vector, const std::string &name = "vector") {
        std::string line;
        std::size_t size = 0;

        while (std::getline(is, line)) {
            if (line.find("<" + name + ">") != std::string::npos)
                break;
        }

        if (std::getline(is, line)) {
            std::size_t start = line.find("<size>") + 6;
            std::size_t end = line.find("</size>");
            if (start != std::string::npos && end != std::string::npos)
                size = std::stoull(line.substr(start, end - start));
        }

        vector.resize(size);

        if (std::getline(is, line)) {
            std::size_t start = line.find("<data>") + 6;
            std::size_t end = line.find("</data>");
            if (start != std::string::npos && end != std::string::npos) {
                std::istringstream data_stream(line.substr(start, end - start));
                for (std::size_t i = 0; i < size; ++i) {
                    T value;
                    data_stream >> value;
                    vector[i] = value;
                }
            }
        }
    }
};

// Convenience wrappers matching the old boost::serialization style
template <typename Archive, typename T>
void save(Archive &ar, const BasicMatrix<T> &matrix, const std::string &name = "matrix") {
    MatrixSerializer<T>::save(ar, matrix, name);
}

template <typename Archive, typename T>
void load(Archive &ar, BasicMatrix<T> &matrix, const std::string &name = "matrix") {
    MatrixSerializer<T>::load(ar, matrix, name);
}

template <typename Archive, typename T>
void save(Archive &ar, const BasicVector<T> &vector, const std::string &name = "vector") {
    VectorSerializer<T>::save(ar, vector, name);
}

template <typename Archive, typename T>
void load(Archive &ar, BasicVector<T> &vector, const std::string &name = "vector") {
    VectorSerializer<T>::load(ar, vector, name);
}

} // namespace serialization
} // namespace libhmm
