#pragma once

#include <cassert>
#include <string>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <memory>
// Removed boost include - using custom C++17 XML serialization

#include "libhmm/hmm.h"

namespace libhmm {

/**
 * XML file writer for HMM serialization.
 *
 * Writes a CDATA-wrapped text format:
 *   <?xml version="1.0" encoding="UTF-8"?><libhmm_model><![CDATA[ <HMM text> ]]></libhmm_model>
 *
 * @deprecated Prefer save_json() / load_json() from hmm_json.h for new code.
 *             XMLFileWriter is retained for producing legacy .xml files only.
 */
class XMLFileWriter {
public:
    XMLFileWriter() = default;
    ~XMLFileWriter() = default;

    // Non-copyable but movable
    XMLFileWriter(const XMLFileWriter &) = delete;
    XMLFileWriter &operator=(const XMLFileWriter &) = delete;
    XMLFileWriter(XMLFileWriter &&) = default;
    XMLFileWriter &operator=(XMLFileWriter &&) = default;

    /**
     * Writes an HMM to an XML file with comprehensive error handling.
     *
     * @param hmm The HMM to serialize
     * @param filename Path to the output XML file
     * @throws std::invalid_argument if filename is empty
     * @throws std::runtime_error if file cannot be created or written
     */
    void write(const Hmm &hmm, const std::string &filename);

    /**
     * Writes an HMM to an XML file with filesystem path.
     *
     * @param hmm The HMM to serialize
     * @param filepath Path to the output XML file
     * @throws std::invalid_argument if filepath is empty
     * @throws std::runtime_error if file cannot be created or written
     */
    void write(const Hmm &hmm, const std::filesystem::path &filepath);

    /**
     * Validates that a file can be written to the given path.
     *
     * @param filepath Path to validate
     * @return true if the file can be written, false otherwise
     */
    static bool canWriteToPath(const std::filesystem::path &filepath) noexcept;

private:
    /**
     * Internal implementation for writing HMM to stream.
     *
     * @param hmm The HMM to serialize
     * @param stream Output stream
     * @throws std::runtime_error if serialization fails
     */
    void writeToStream(const Hmm &hmm, std::ofstream &stream);
};

} // namespace libhmm
