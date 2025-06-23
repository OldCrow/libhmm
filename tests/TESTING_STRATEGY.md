# libhmm Testing Strategy

This document explains the two-tier testing approach used in libhmm for comprehensive and efficient testing.

## 🎯 Testing Philosophy

We use a **two-tier testing strategy** that balances thorough validation with development efficiency:

1. **Integration Tests** - Comprehensive GoogleTest suite
2. **Unit Tests** - Fast, focused standalone tests

## 📁 Directory Structure

```
tests/
├── test_distributions.cpp        # Integration: All distributions together
├── test_hmm_core.cpp             # Integration: Core HMM functionality
├── test_calculators.cpp          # Integration: All calculators
├── test_training.cpp             # Integration: Training algorithms
├── test_io.cpp                   # Integration: File I/O operations
├── test_common.cpp               # Integration: Common utilities
├── test_type_safety.cpp          # Legacy: Type safety validation
├── unit/                         # Unit tests directory
│   └── test_poisson_distribution.cpp  # Unit: Poisson-specific tests
└── TESTING_STRATEGY.md           # This documentation
```

## 🚀 Quick Testing Commands

### **Integration Tests (GoogleTest)**
```bash
# Run all integration tests
ctest

# Run all distribution tests (38 tests covering 7 distributions)
./tests/test_distributions

# Run only Poisson distribution tests within integration suite
./tests/test_distributions --gtest_filter="*Poisson*"

# Run with verbose output
./tests/test_distributions --gtest_filter="*Poisson*" --gtest_brief=1
```

### **Unit Tests (Standalone)**
```bash
# Run Poisson unit tests (fast, focused)
./tests/test_poisson_distribution

# Run all unit tests via CTest
ctest -R "unit_"

# Run specific unit test via CTest
ctest -R "unit_test_poisson"
```

## 📊 When to Use Each Type

### **Use Integration Tests When:**
- ✅ **Pre-commit validation** - Ensure everything works together
- ✅ **CI/CD pipelines** - Comprehensive validation
- ✅ **Release testing** - Full functionality verification
- ✅ **Polymorphic interface testing** - Common interface compliance
- ✅ **Cross-distribution consistency** - Behavior validation across all distributions

### **Use Unit Tests When:**
- ✅ **Active development** - Quick feedback during feature development
- ✅ **Debugging** - Isolate issues in specific components
- ✅ **Performance profiling** - Measure individual component performance
- ✅ **Documentation** - Examples of component usage
- ✅ **Regression testing** - Verify specific bug fixes

## ⚡ Performance Comparison

| Test Type | Command | Duration | Tests | Coverage |
|-----------|---------|----------|-------|----------|
| Unit (Poisson) | `./test_poisson_distribution` | ~0.005s | 7 tests | Single distribution |
| Integration (Poisson) | `./test_distributions --gtest_filter="*Poisson*"` | ~0.007s | 10 tests | Poisson + common interface |
| Integration (All) | `./test_distributions` | ~0.007s | 38 tests | All 7 distributions |

## 🔧 Adding New Distribution Tests

### **Step 1: Add Unit Test**
1. Create `tests/unit/test_[distribution]_distribution.cpp`
2. Use the standalone test framework (see `test_poisson_distribution.cpp` as example)
3. Add to `UNIT_TEST_SOURCES` in `tests/CMakeLists.txt`

### **Step 2: Add Integration Test**
1. Add test class to `tests/test_distributions.cpp`
2. Follow the GoogleTest pattern (see `PoissonDistributionTest` as example)
3. Add to `CommonDistributionTest` setup

### **Example: Adding Gamma Unit Tests**
```cmake
# In tests/CMakeLists.txt
set(UNIT_TEST_SOURCES
    unit/test_poisson_distribution.cpp
    unit/test_gamma_distribution.cpp    # Add this line
)
```

## 🎨 Test Framework Differences

### **Integration Tests (GoogleTest)**
```cpp
class PoissonDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        dist_ = std::make_unique<PoissonDistribution>(2.0);
    }
    std::unique_ptr<PoissonDistribution> dist_;
};

TEST_F(PoissonDistributionTest, ConstructorValidation) {
    EXPECT_NO_THROW(PoissonDistribution(1.0));
    EXPECT_THROW(PoissonDistribution(0.0), std::invalid_argument);
}
```

### **Unit Tests (Standalone)**
```cpp
void testConstructorValidation() {
    std::cout << "Testing constructor validation..." << std::endl;
    
    try {
        PoissonDistribution valid(1.0);
        std::cout << "✓ Valid construction passed" << std::endl;
    } catch (...) {
        assert(false);
    }
    
    try {
        PoissonDistribution invalid(0.0);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        std::cout << "✓ Invalid construction correctly rejected" << std::endl;
    }
}
```

## 📈 Benefits of This Approach

### **For Developers:**
- 🚀 **Fast feedback** during development (unit tests)
- 🔍 **Focused debugging** capabilities
- 📚 **Clear examples** of component usage
- ⚡ **Quick verification** of changes

### **For Maintainers:**
- 🛡️ **Comprehensive validation** (integration tests)
- 🔄 **Consistent interface testing** across all components
- 📊 **Performance monitoring** capabilities
- 🎯 **Targeted testing** options

### **For CI/CD:**
- 📦 **Flexible test execution** strategies
- ⏱️ **Parallel test execution** possibilities
- 🎛️ **Granular failure reporting**
- 🔧 **Easy maintenance** and updates

## 🏆 Best Practices

1. **Keep unit tests fast** - Focus on core functionality only
2. **Keep integration tests comprehensive** - Test interactions and edge cases
3. **Use appropriate assertions** - GoogleTest for integration, simple asserts for unit
4. **Document test purposes** - Clear comments explaining what each test validates
5. **Maintain both tiers** - Don't let one approach dominate over the other

## 🔮 Future Enhancements

Potential improvements to consider:

- **Benchmark tests** - Performance regression detection
- **Fuzzing tests** - Random input validation
- **Property-based tests** - Mathematical property verification
- **Mock objects** - For testing with external dependencies
- **Test data generators** - Automated test case generation

---

*This testing strategy ensures both rapid development cycles and robust validation, providing the best of both worlds for the libhmm project.*
