#include "libhmm/common/matrix.h"
#include "libhmm/common/vector.h"
#include <iostream>

// Test the type aliases approach
namespace libhmm {
    // Alias the template instantiations to match the old typedef names  
    using Matrix = BasicMatrix<double>;
    using Vector = BasicVector<double>;
    using StateSequence = BasicVector<int>;
}

int main() {
    // Test basic functionality with the aliased names
    libhmm::Matrix m(3, 3);
    libhmm::Vector v(3);
    libhmm::StateSequence seq(5);
    
    // Test uBLAS-compatible API
    m(0, 0) = 1.0;
    v(0) = 2.0;
    seq(0) = 1;
    
    std::cout << "Matrix size: " << m.size1() << "x" << m.size2() << std::endl;
    std::cout << "Vector size: " << v.size() << std::endl;
    std::cout << "StateSequence size: " << seq.size() << std::endl;
    
    std::cout << "Simple replacement test: SUCCESS" << std::endl;
    
    return 0;
}
