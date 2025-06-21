#ifndef COMMON_H_
#define COMMON_H_

/*
 * Boost uBLAS
 */
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/io.hpp>

/*
 * Boost Serialization
 */
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

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
typedef boost::numeric::ublas::matrix<Observation> Matrix;
typedef boost::numeric::ublas::vector<Observation> Vector;
typedef boost::numeric::ublas::vector<Observation> ObservationSet;

/*
 * Viterbi decode requires a fixed size vector for state sequences
 */
typedef boost::numeric::ublas::vector<StateIndex> StateSequence;

/*
 * Training requires creating a list of all the observation sets.
 * We can't make a Vector of Vectors, but we can use std::list or std::vector
 * for things like this.
 */
typedef std::vector<ObservationSet> ObservationLists;

/*
 * Simplified naming for the STL vector class.
 * Template syntax may not be legal with GCC.
 */
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
 
// Boost uBLAS does not zero matrices when they are allocated.
// This provides a way to clear it out.
void clear_matrix( Matrix& m );
void clear_vector( Vector& v );
void clear_vector( StateSequence& v );

}//namespace

/*
 * Serialization functions for matrices and vectors.
 * Vector code is from an email found at 
 * http://archives.free.net.ph/message/20060821.155858e0fc8417.en.html
 *
 * Matrix code follows the same idea.
 */
namespace boost{ 
namespace serialization{

template<class T>
struct implementation_level<boost::numeric::ublas::vector<T> >{
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<object_serializable> type;
    BOOST_STATIC_CONSTANT( int, value = implementation_level::type::value );
};

template<class T>
struct implementation_level<boost::numeric::ublas::matrix<T> >{
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<object_serializable> type;
    BOOST_STATIC_CONSTANT( int, value = implementation_level::type::value );
};

template<class Archive, class U>
void save( Archive& ar, const boost::numeric::ublas::vector<U> &v, const unsigned int ){
    const std::size_t count = v.size();
    ar << BOOST_SERIALIZATION_NVP( count );
    for(const auto& item : v) {
        ar << boost::serialization::make_nvp( "item", item );
    }
}

template<class Archive, class U>
void save( Archive& ar, const boost::numeric::ublas::matrix<U>& m, const unsigned int ){
    const std::size_t rows = m.size1();
    const std::size_t cols = m.size2();
    ar << BOOST_SERIALIZATION_NVP( rows );
    ar << BOOST_SERIALIZATION_NVP( cols );
    for( std::size_t i = 0; i < rows; i++ ){
        for( std::size_t j = 0; j < cols; j++ ){
            ar << boost::serialization::make_nvp( "item", m( i, j ) );
        }
    }
}

template<class Archive, class U>
void load( Archive& ar, boost::numeric::ublas::vector<U>& v, const unsigned int ){
    std::size_t count;
    ar >> BOOST_SERIALIZATION_NVP( count );
    v.resize( count );
    for(auto& item : v) {
        ar >> boost::serialization::make_nvp( "item", item );
    }
}

template<class Archive, class U>
void load( Archive& ar, boost::numeric::ublas::matrix<U>& m, const unsigned int ){
    std::size_t rows, cols;
    ar >> BOOST_SERIALIZATION_NVP( rows );
    ar >> BOOST_SERIALIZATION_NVP( cols );
    m.resize( rows, cols );
    for( std::size_t i = 0; i < rows; i++ ){
        for( std::size_t j = 0; j < cols; j++ ){
            ar >> boost::serialization::make_nvp( "item", m( i, j ) );
        }
    }
}

template<class Archive, class U>
inline void serialize( Archive& ar, boost::numeric::ublas::vector<U>& v, const unsigned int file_version ){
    boost::serialization::split_free( ar, v, file_version );
}

template<class Archive, class U>
inline void serialize( Archive& ar, boost::numeric::ublas::matrix<U>& m, const unsigned int file_version ){
    boost::serialization::split_free( ar, m, file_version );
}

}//namespace serialization
}//namespace boost
#endif
