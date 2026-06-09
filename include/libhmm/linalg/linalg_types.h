#pragma once

// Linalg types for libhmm.
//
// Provides the concrete matrix/vector types and the type aliases that the HMM
// engine, calculators, trainers, and IO layer depend on.  Distribution headers
// do NOT need to include this file — they work with std::span<double> and do
// not expose or store linalg types in their public signatures.
//
// Include this header from any file that uses Matrix, Vector, ObservationSet,
// ObservationLists, StateSequence, MultiObservationLists, or the clear_* helpers.

#include <cstddef>
#include <span>
#include <vector>

#include "libhmm/common/common.h" // Observation, StateIndex, STL headers
#include "libhmm/linalg/basic_matrix.h"
#include "libhmm/linalg/basic_matrix3d.h"
#include "libhmm/linalg/basic_vector.h"

namespace libhmm {

// Core linalg type aliases
using Matrix = BasicMatrix<Observation>;
using Vector = BasicVector<Observation>;
using ObservationSet = BasicVector<Observation>;
using StateSequence = BasicVector<StateIndex>;
using ObservationMatrix3D = BasicMatrix3D<Observation>;

template <typename T>
using Matrix3DTemplate = BasicMatrix3D<T>;

/// Sequence of observation vectors, one per training sequence.
typedef std::vector<ObservationSet> ObservationLists;

// Utility helpers — implementations in src/common/common.cpp
void clear_matrix(Matrix &m);
void clear_vector(Vector &v);
void clear_vector(StateSequence &v);

// ===========================================================================
// Multivariate observation types (Phase E)
//
// These extend the scalar path (Obs = double) to D-dimensional observations.
// ObservationMatrix stores an entire sequence as a row-major T×D matrix;
// ObservationVectorView is a lightweight non-owning view of one row, used as
// the Obs template parameter for BasicHmm<ObservationVectorView> = HmmMV.
// ===========================================================================

/// Non-owning view of one D-dimensional observation (one row of an
/// ObservationMatrix).  As a std::span it is zero-cost: one pointer + one
/// size_t.  The viewed data must outlive the span.
using ObservationVectorView = std::span<const double>;

/// Row-major sequence matrix: row t is the D-dimensional observation at time t.
///   size1() = T (sequence length)
///   size2() = D (observation dimensionality)
using ObservationMatrix = BasicMatrix<double>;

/// One ObservationMatrix per training sequence.
using MultiObservationLists = std::vector<ObservationMatrix>;

/// Return a non-owning view of row t of an ObservationMatrix.
/// The returned span is valid as long as mat is alive.
[[nodiscard]] inline ObservationVectorView
row_view(const ObservationMatrix& mat, std::size_t t) noexcept {
    return ObservationVectorView(mat.data() + t * mat.cols(), mat.cols());
}

// ===========================================================================
// ObsSeqTraits<Obs>
//
// Maps an observation type to its sequence and list container types, and
// provides sequence_length() for uniform length extraction.
// Two specialisations: Obs=double (scalar) and Obs=ObservationVectorView (MV).
// ===========================================================================

template<typename Obs>
struct ObsSeqTraits;  ///< Primary; only the two specialisations below are valid.

template<>
struct ObsSeqTraits<double> {
    using SeqType  = ObservationSet;    ///< BasicVector<double>
    using ListType = ObservationLists;  ///< vector<ObservationSet>

    /// Length of a single observation sequence.
    static std::size_t sequence_length(const SeqType& s) noexcept { return s.size(); }
};

template<>
struct ObsSeqTraits<ObservationVectorView> {
    using SeqType  = ObservationMatrix;          ///< BasicMatrix<double>, T rows x D cols
    using ListType = MultiObservationLists;      ///< vector<ObservationMatrix>

    /// Length of a single observation sequence (number of time steps = rows).
    static std::size_t sequence_length(const SeqType& s) noexcept { return s.size1(); }
};

} // namespace libhmm
