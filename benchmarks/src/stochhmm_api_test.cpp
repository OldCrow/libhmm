#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "StochHMM/source/src/StochHMMlib.h"

using namespace std;
using namespace StochHMM;

int main() {
    cout << "StochHMM Continuous API Test" << endl;
    cout << "=============================" << endl;
    
    try {
        // Test 1: Direct PDF calculations
        cout << "Testing direct PDF calculations..." << endl;
        vector<double> params = {0.0, 1.0};  // mean=0, std_dev=1
        
        for (double x : {-1.0, 0.0, 1.0}) {
            double log_pdf = normal_pdf(x, &params);
            double pdf = exp(log_pdf);
            cout << "normal_pdf(" << x << ", μ=0, σ=1): PDF=" << pdf << ", log-PDF=" << log_pdf << endl;
        }
        
        // Test 2: Try to load the model
        string model_file = "/Users/wolfman/libhmm/benchmarks/simple_continuous.hmm";
        cout << "\nLoading model from: " << model_file << endl;
        
        // Check if file exists
        ifstream file_check(model_file);
        if (!file_check.good()) {
            cerr << "Model file does not exist or cannot be opened" << endl;
            return 1;
        }
        file_check.close();
        
        cout << "File exists, attempting to create model object..." << endl;
        model* hmm_model = new model();
        
        cout << "Model object created, attempting import..." << endl;
        bool import_result = hmm_model->import(model_file);
        
        if (!import_result) {
            cerr << "Failed to load model" << endl;
            delete hmm_model;
            return 1;
        }
        
        cout << "Model loaded successfully!" << endl;
        cout << "Model name: " << hmm_model->getName() << endl;
        cout << "Number of states: " << hmm_model->state_size() << endl;
        cout << "Number of tracks: " << hmm_model->track_size() << endl;
        
        // Try to get track information
        tracks* model_tracks = hmm_model->getTracks();
        track* real_track = model_tracks->getTrack("REAL_TRACK");
        if (!real_track) {
            cerr << "Could not find REAL_TRACK in model" << endl;
            delete hmm_model;
            return 1;
        }
        
        cout << "Found real track: " << real_track->getName() << endl;
        cout << "Track found and ready for use" << endl;
        
        // Cleanup
        delete hmm_model;
        
        cout << "\nBasic test completed successfully!" << endl;
        
    } catch (const exception& e) {
        cerr << "Exception caught: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
