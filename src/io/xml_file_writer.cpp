#include "libhmm/io/xml_file_writer.h"
#include <iostream>
#include <sstream>

namespace libhmm{

void XMLFileWriter::write(const Hmm& hmm, const std::string& filename) {
    if (filename.empty()) {
        throw std::invalid_argument("Filename cannot be empty");
    }
    
    write(hmm, std::filesystem::path{filename});
}

void XMLFileWriter::write(const Hmm& hmm, const std::filesystem::path& filepath) {
    if (filepath.empty()) {
        throw std::invalid_argument("Filepath cannot be empty");
    }
    
    // Validate that we can write to this path
    if (!canWriteToPath(filepath)) {
        throw std::runtime_error("Cannot write to path: " + filepath.string());
    }
    
    // Create directory if it doesn't exist
    if (filepath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(filepath.parent_path(), ec);
        if (ec) {
            throw std::runtime_error("Failed to create directory: " + ec.message());
        }
    }
    
    try {
        std::ofstream ofs(filepath, std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());
        }
        
        writeToStream(hmm, ofs);
        
        ofs.close();
        if (ofs.fail()) {
            throw std::runtime_error("Failed to properly close file: " + filepath.string());
        }
        
    // TODO: Uncomment when boost serialization is implemented
    // } catch (const boost::archive::archive_exception& e) {
    //     throw std::runtime_error("XML serialization failed: " + std::string(e.what()));
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("I/O operation failed: " + std::string(e.what()));
    }
}

bool XMLFileWriter::canWriteToPath(const std::filesystem::path& filepath) noexcept {
    try {
        // Check if the file already exists and is writable
        if (std::filesystem::exists(filepath)) {
            auto perms = std::filesystem::status(filepath).permissions();
            return (perms & std::filesystem::perms::owner_write) != std::filesystem::perms::none;
        }
        
        // Check if the parent directory exists and is writable
        auto parent = filepath.parent_path();
        if (parent.empty()) {
            parent = std::filesystem::current_path();
        }
        
        if (std::filesystem::exists(parent)) {
            auto perms = std::filesystem::status(parent).permissions();
            return (perms & std::filesystem::perms::owner_write) != std::filesystem::perms::none;
        }
        
        // Parent doesn't exist, but we might be able to create it
        return true;
        
    } catch (...) {
        return false;
    }
}

void XMLFileWriter::writeToStream(const Hmm& hmm, std::ofstream& stream) {
    if (!stream.good()) {
        throw std::runtime_error("Stream is not in a good state for writing");
    }
    
    try {
        // TODO: Implement boost serialization when Hmm class supports it
        // boost::archive::xml_oarchive oa(stream);
        // oa & BOOST_SERIALIZATION_NVP(hmm);
        
        // For now, use the HMM's stream operator with XML wrapper
        stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
        stream << "<libhmm_model>" << std::endl;
        stream << "<![CDATA[" << std::endl;
        stream << hmm << std::endl;
        stream << "]]>" << std::endl;
        stream << "</libhmm_model>" << std::endl;
        
        stream.flush();
        if (stream.fail()) {
            throw std::runtime_error("Failed to flush stream after writing");
        }
        
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("Stream I/O error: " + std::string(e.what()));
    }
    // TODO: Uncomment when boost serialization is implemented
    // catch (const boost::archive::archive_exception& e) {
    //     throw std::runtime_error("XML archive error: " + std::string(e.what()));
    // }
}

}
