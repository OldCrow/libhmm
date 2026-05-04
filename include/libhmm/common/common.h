#pragma once

// Core C++ standard library headers used throughout the library.
// Lean version: linalg types (Matrix, Vector, ObservationSet, etc.) and XML
// serialization helpers now live in separate headers:
//   libhmm/linalg/linalg_types.h  — type aliases and clear_* helpers
//   libhmm/common/serialization.h — MatrixSerializer / VectorSerializer
// Include those headers from files that need them.

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip> // std::setprecision, std::fixed, std::put_time — used in toString()
                   // and file I/O throughout the library. Must be explicit: MSVC and
                   // libstdc++ do not include <iomanip> transitively through <iostream>.
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <utility>
#include <type_traits>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <limits>
#include <cassert>

// Constants (includes M_PI guard)
#include "libhmm/math/constants.h"

namespace libhmm {

/// Scalar observation type. Using double gives sufficient precision for all
/// currently supported distributions while keeping the linalg types concrete.
typedef double Observation;

/// State index type.
typedef int StateIndex;

} // namespace libhmm
