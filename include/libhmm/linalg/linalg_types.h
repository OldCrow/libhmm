#pragma once

// Linalg types for libhmm.
//
// Provides the concrete matrix/vector types and the type aliases that the HMM
// engine, calculators, trainers, and IO layer depend on.  Distribution headers
// do NOT need to include this file — they work with std::span<double> and do
// not expose or store linalg types in their public signatures.
//
// Include this header from any file that uses Matrix, Vector, ObservationSet,
// ObservationLists, StateSequence, or the clear_* helpers.

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

} // namespace libhmm
