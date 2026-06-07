#pragma once

// v4 compatibility header.
//
// Hmm is now a type alias for BasicHmm<double>.  All v3 code that uses
// Hmm continues to compile unchanged: constructors, setters, getters,
// validate(), JSON I/O, and stream operators are all unaffected.

#include "libhmm/basic_hmm.h"
#include <iostream>  // for stream operator signatures

namespace libhmm {

/// @brief Scalar HMM type alias (v3 compatibility).
///
/// Hmm is BasicHmm<double>. All v3 code that constructs, trains, scores,
/// or serialises an Hmm continues to compile unchanged.
using Hmm = BasicHmm<double>;

/// @brief Multivariate HMM alias.
///
/// Each observation is a non-owning row view (ObservationVectorView =
/// std::span<const double>) into an ObservationMatrix sequence.
/// Emission distributions must be set explicitly via setDistribution();
/// the default constructor leaves emission slots null for non-scalar Obs.
using HmmMV = BasicHmm<ObservationVectorView>;

/// Legacy stream I/O operators (scalar HMM only).
/// Prefer JSON I/O (hmm_json.h) for new code.
std::ostream& operator<<(std::ostream&, const Hmm&);
std::istream& operator>>(std::istream&, Hmm&);

} // namespace libhmm
