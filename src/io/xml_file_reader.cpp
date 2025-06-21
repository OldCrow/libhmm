#include "libhmm/io/xml_file_reader.h"
#include <iostream>
#include <sstream>

namespace libhmm{

Hmm XMLFileReader::read(const std::string& filename) {
    if (filename.empty()) {
        throw std::invalid_argument("Filename cannot be empty");
    }
    
    return read(std::filesystem::path{filename});
}

Hmm XMLFileReader::read(const std::filesystem::path& filepath) {
    if (filepath.empty()) {
        throw std::invalid_argument("Filepath cannot be empty");
    }
    
    // Validate that we can read from this path
    if (!canReadFromPath(filepath)) {
        throw std::runtime_error("Cannot read from path: " + filepath.string());
    }
    
    // Check if it appears to be a valid XML file
    if (!isValidXMLFile(filepath)) {
        throw std::runtime_error("File does not appear to be a valid XML file: " + filepath.string());
    }
    
    try {
        std::ifstream ifs(filepath, std::ios::in);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());
        }
        
        auto hmm = readFromStream(ifs);
        
        ifs.close();
        if (ifs.fail()) {
            // Note: This might not be critical for reading, but worth logging
            std::cerr << "Warning: Failed to properly close file: " << filepath.string() << std::endl;
        }
        
        return hmm;
        
    // TODO: Uncomment when boost serialization is implemented
    // } catch (const boost::archive::archive_exception& e) {
    //     throw std::runtime_error("XML deserialization failed: " + std::string(e.what()));
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("I/O operation failed: " + std::string(e.what()));
    }
}

bool XMLFileReader::canReadFromPath(const std::filesystem::path& filepath) noexcept {
    try {
        // Check if the file exists and is readable
        if (!std::filesystem::exists(filepath)) {
            return false;
        }
        
        // Check if it's a regular file (not a directory)
        if (!std::filesystem::is_regular_file(filepath)) {
            return false;
        }
        
        // Check read permissions
        auto perms = std::filesystem::status(filepath).permissions();
        return (perms & std::filesystem::perms::owner_read) != std::filesystem::perms::none;
        
    } catch (...) {
        return false;
    }
}

bool XMLFileReader::isValidXMLFile(const std::filesystem::path& filepath) noexcept {
    try {
        if (!canReadFromPath(filepath)) {
            return false;
        }
        
        // Check file size - empty files or extremely large files are suspicious
        const auto fileSize = std::filesystem::file_size(filepath);
        if (fileSize == 0 || fileSize > 100 * 1024 * 1024) { // 100MB limit
            return false;
        }
        
        // Try to open and read first few bytes to check for XML header
        std::ifstream file(filepath, std::ios::in);
        if (!file.is_open()) {
            return false;
        }
        
        std::string firstLine;
        if (!std::getline(file, firstLine)) {
            return false;
        }
        
        // Basic XML validation - look for XML declaration or root element
        return firstLine.find("<?xml") != std::string::npos || 
               firstLine.find("<boost_serialization>") != std::string::npos ||
               firstLine.find("<") != std::string::npos;
        
    } catch (...) {
        return false;
    }
}

Hmm XMLFileReader::readFromStream(std::ifstream& stream) {
    if (!stream.good()) {
        throw std::runtime_error("Stream is not in a good state for reading");
    }
    
    try {
        // TODO: Implement boost serialization when Hmm class supports it
        // boost::archive::xml_iarchive ia(stream);
        // Hmm hmm(1); // Start with 1 state, will be resized during deserialization
        // ia & BOOST_SERIALIZATION_NVP(hmm);
        
        // For now, skip XML wrapper and read HMM using stream operator
        std::string line;
        
        // Skip XML header and opening tag
        while (std::getline(stream, line)) {
            if (line.find("<![CDATA[") != std::string::npos) {
                break;
            }
        }
        
        // Create default HMM and read using stream operator
        Hmm hmm(2); // Default 2-state HMM
        stream >> hmm;
        
        if (stream.fail()) {
            throw std::runtime_error("Stream error occurred during reading");
        }
        
        return hmm;
        
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("Stream I/O error: " + std::string(e.what()));
    }
    // TODO: Uncomment when boost serialization is implemented
    // catch (const boost::archive::archive_exception& e) {
    //     throw std::runtime_error("XML archive error: " + std::string(e.what()));
    // }
}

}
