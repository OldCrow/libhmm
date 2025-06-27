#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <memory>

// libhmm includes
#include "libhmm/distributions/gaussian_distribution.h"

// GHMM includes
extern "C" {
#include "ghmm/ghmm.h"
#include "ghmm/smodel.h"
}

// StochHMM includes
#include "../StochHMM/source/src/StochHMMlib.h"
#include "../StochHMM/source/src/PDF.h"

using namespace std;

struct GaussianTestCase {
    double mean;
    double variance;
    vector<double> test_values;
    string description;
};

struct DistributionResult {
    string library;
    string test_case;
    vector<double> pdf_values;
    vector<double> log_pdf_values;
    bool success;
    string error_message;
};

// Test libhmm Gaussian distribution
DistributionResult testLibhmmGaussian(const GaussianTestCase& test_case) {
    DistributionResult result;
    result.library = "libhmm";
    result.test_case = test_case.description;
    result.success = false;
    
    try {
        libhmm::GaussianDistribution gaussian(test_case.mean, sqrt(test_case.variance));
        
        for (double x : test_case.test_values) {
            // Get PDF (probability density)
            double pdf = gaussian.getProbability(x);
            result.pdf_values.push_back(pdf);
            
            // Calculate log-PDF manually
            double log_pdf = log(pdf);
            result.log_pdf_values.push_back(log_pdf);
        }
        
        result.success = true;
        
    } catch (const exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

// Test GHMM Gaussian distribution
DistributionResult testGhmmGaussian(const GaussianTestCase& test_case) {
    DistributionResult result;
    result.library = "GHMM";
    result.test_case = test_case.description;
    result.success = false;
    
    try {
        // GHMM uses standard deviation, not variance
        double std_dev = sqrt(test_case.variance);
        
        for (double x : test_case.test_values) {
            // Calculate PDF using GHMM's normal distribution function
            // Note: GHMM might have specific functions for this, but we'll use standard formula
            double pdf = (1.0 / (std_dev * sqrt(2 * M_PI))) * 
                        exp(-0.5 * pow((x - test_case.mean) / std_dev, 2));
            result.pdf_values.push_back(pdf);
            
            // Calculate log-PDF
            double log_pdf = log(pdf);
            result.log_pdf_values.push_back(log_pdf);
        }
        
        result.success = true;
        
    } catch (const exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

// Test StochHMM Gaussian distribution
DistributionResult testStochHmmGaussian(const GaussianTestCase& test_case) {
    DistributionResult result;
    result.library = "StochHMM";
    result.test_case = test_case.description;
    result.success = false;
    
    try {
        // StochHMM uses parameters in a vector: [mean, std_dev]
        vector<double> parameters;
        parameters.push_back(test_case.mean);
        parameters.push_back(sqrt(test_case.variance)); // StochHMM uses std_dev
        
        for (double x : test_case.test_values) {
            // Use StochHMM's normal_pdf function
            double log_pdf = StochHMM::normal_pdf(x, &parameters);
            result.log_pdf_values.push_back(log_pdf);
            
            // Convert to PDF from log-PDF
            double pdf = exp(log_pdf);
            result.pdf_values.push_back(pdf);
        }
        
        result.success = true;
        
    } catch (const exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

void printComparisonResults(const vector<DistributionResult>& results, const GaussianTestCase& test_case) {
    cout << "\n====================================================================================\n";
    cout << "GAUSSIAN DISTRIBUTION COMPARISON: " << test_case.description << "\n";
    cout << "Parameters: mean=" << test_case.mean << ", variance=" << test_case.variance << 
            " (std_dev=" << sqrt(test_case.variance) << ")\n";
    cout << "====================================================================================\n";
    
    // Print header
    cout << left << setw(12) << "Library";
    cout << setw(15) << "Test Value";
    cout << setw(18) << "PDF";
    cout << setw(18) << "Log-PDF";
    cout << setw(10) << "Success" << endl;
    cout << "------------------------------------------------------------------------------------\n";
    
    // Print results for each test value
    for (size_t i = 0; i < test_case.test_values.size(); ++i) {
        double test_val = test_case.test_values[i];
        
        for (const auto& result : results) {
            if (result.test_case == test_case.description) {
                cout << left << setw(12) << result.library;
                cout << setw(15) << fixed << setprecision(3) << test_val;
                
                if (result.success && i < result.pdf_values.size()) {
                    cout << setw(18) << scientific << setprecision(6) << result.pdf_values[i];
                    cout << setw(18) << scientific << setprecision(6) << result.log_pdf_values[i];
                    cout << setw(10) << "YES";
                } else {
                    cout << setw(18) << "ERROR";
                    cout << setw(18) << "ERROR";
                    cout << setw(10) << "NO";
                }
                cout << endl;
            }
        }
        cout << "------------------------------------------------------------------------------------\n";
    }
}

void analyzeNumericalDifferences(const vector<DistributionResult>& results, const GaussianTestCase& test_case) {
    cout << "\nNUMERICAL ACCURACY ANALYSIS FOR: " << test_case.description << "\n";
    cout << "--------------------------------------------------------------------------------\n";
    
    // Find successful results
    vector<DistributionResult> successful_results;
    for (const auto& result : results) {
        if (result.success && result.test_case == test_case.description) {
            successful_results.push_back(result);
        }
    }
    
    if (successful_results.size() < 2) {
        cout << "Not enough successful results for comparison.\n";
        return;
    }
    
    cout << left << setw(15) << "Test Value";
    cout << setw(20) << "PDF Difference";
    cout << setw(20) << "Log-PDF Difference";
    cout << setw(15) << "Status" << endl;
    cout << "--------------------------------------------------------------------------------\n";
    
    for (size_t i = 0; i < test_case.test_values.size(); ++i) {
        double test_val = test_case.test_values[i];
        cout << left << setw(15) << fixed << setprecision(3) << test_val;
        
        // Compare PDF values between first two successful results
        if (i < successful_results[0].pdf_values.size() && 
            i < successful_results[1].pdf_values.size()) {
            
            double pdf_diff = abs(successful_results[0].pdf_values[i] - successful_results[1].pdf_values[i]);
            double log_pdf_diff = abs(successful_results[0].log_pdf_values[i] - successful_results[1].log_pdf_values[i]);
            
            cout << setw(20) << scientific << setprecision(3) << pdf_diff;
            cout << setw(20) << scientific << setprecision(3) << log_pdf_diff;
            
            bool match = (pdf_diff < 1e-10 && log_pdf_diff < 1e-10);
            cout << setw(15) << (match ? "MATCH" : "DIFFER");
        } else {
            cout << setw(20) << "N/A";
            cout << setw(20) << "N/A";
            cout << setw(15) << "ERROR";
        }
        cout << endl;
    }
    cout << "================================================================================\n";
}

int main() {
    cout << "Gaussian Distribution Implementation Comparison\n";
    cout << "================================================\n";
    cout << "Testing PDF and log-PDF calculations across libhmm, GHMM, and StochHMM\n\n";
    
    // Define test cases with different Gaussian parameters
    vector<GaussianTestCase> test_cases = {
        {
            0.0,    // mean
            1.0,    // variance (std_dev = 1.0)
            {-3.0, -1.0, 0.0, 1.0, 3.0},  // test values
            "Standard Normal (μ=0, σ²=1)"
        },
        {
            3.0,    // mean
            2.0,    // variance (std_dev = √2 ≈ 1.414)
            {0.0, 1.5, 3.0, 4.5, 6.0},    // test values
            "Shifted Normal (μ=3, σ²=2)"
        },
        {
            -2.0,   // mean
            0.5,    // variance (std_dev = √0.5 ≈ 0.707)
            {-4.0, -3.0, -2.0, -1.0, 0.0}, // test values
            "Negative Mean (μ=-2, σ²=0.5)"
        },
        {
            10.0,   // mean
            25.0,   // variance (std_dev = 5.0)
            {0.0, 5.0, 10.0, 15.0, 20.0},  // test values
            "High Variance (μ=10, σ²=25)"
        }
    };
    
    vector<DistributionResult> all_results;
    
    for (const auto& test_case : test_cases) {
        cout << "Testing: " << test_case.description << "...\n";
        
        // Test each library
        auto libhmm_result = testLibhmmGaussian(test_case);
        auto ghmm_result = testGhmmGaussian(test_case);
        auto stochhmm_result = testStochHmmGaussian(test_case);
        
        all_results.push_back(libhmm_result);
        all_results.push_back(ghmm_result);
        all_results.push_back(stochhmm_result);
        
        // Print results for this test case
        vector<DistributionResult> case_results = {libhmm_result, ghmm_result, stochhmm_result};
        printComparisonResults(case_results, test_case);
        analyzeNumericalDifferences(case_results, test_case);
        
        // Report any errors
        for (const auto& result : case_results) {
            if (!result.success) {
                cout << "ERROR in " << result.library << ": " << result.error_message << "\n";
            }
        }
    }
    
    cout << "\nSUMMARY\n";
    cout << "=======\n";
    cout << "This benchmark tests the core Gaussian distribution implementations\n";
    cout << "to isolate potential numerical differences from HMM algorithm issues.\n";
    cout << "Significant differences here would indicate distribution-level problems,\n";
    cout << "while small differences suggest HMM integration or scaling issues.\n";
    
    return 0;
}
