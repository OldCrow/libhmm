#ifndef XMLFILEWRITER_H_
#define XMLFILEWRITER_H_

#include <cassert>
#include <string>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <memory>
#include <boost/archive/xml_oarchive.hpp>

#include "libhmm/common/common.h"
#include "libhmm/hmm.h"

namespace libhmm{

/**
 * Modern XML file writer for HMM serialization with C++17 features.
 * Provides safe XML serialization with proper error handling and validation.
 */
class XMLFileWriter{
public:    
    XMLFileWriter() = default;
    ~XMLFileWriter() = default;
    
    // Non-copyable but movable
    XMLFileWriter(const XMLFileWriter&) = delete;
    XMLFileWriter& operator=(const XMLFileWriter&) = delete;
    XMLFileWriter(XMLFileWriter&&) = default;
    XMLFileWriter& operator=(XMLFileWriter&&) = default;

    /**
     * Writes an HMM to an XML file with comprehensive error handling.
     * 
     * @param hmm The HMM to serialize
     * @param filename Path to the output XML file
     * @throws std::invalid_argument if filename is empty
     * @throws std::runtime_error if file cannot be created or written
     */
    void write(const Hmm& hmm, const std::string& filename);
    
    /**
     * Writes an HMM to an XML file with filesystem path.
     * 
     * @param hmm The HMM to serialize
     * @param filepath Path to the output XML file
     * @throws std::invalid_argument if filepath is empty
     * @throws std::runtime_error if file cannot be created or written
     */
    void write(const Hmm& hmm, const std::filesystem::path& filepath);
    
    /**
     * Validates that a file can be written to the given path.
     * 
     * @param filepath Path to validate
     * @return true if the file can be written, false otherwise
     */
    static bool canWriteToPath(const std::filesystem::path& filepath) noexcept;
    
private:
    /**
     * Internal implementation for writing HMM to stream.
     * 
     * @param hmm The HMM to serialize
     * @param stream Output stream
     * @throws std::runtime_error if serialization fails
     */
    void writeToStream(const Hmm& hmm, std::ofstream& stream);};

}

#endif
