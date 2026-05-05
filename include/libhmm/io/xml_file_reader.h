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
 * XML file reader for HMM deserialization.
 *
 * Reads the CDATA-wrapped text format written by XMLFileWriter:
 *   <?xml ...?><libhmm_model><![CDATA[ <HMM text> ]]></libhmm_model>
 *
 * @deprecated Prefer load_json() / save_json() from hmm_json.h for new code.
 *             XMLFileReader is retained for reading legacy .xml files only.
 */
class XMLFileReader {
public:
    XMLFileReader() = default;
    ~XMLFileReader() = default;

    // Non-copyable but movable
    XMLFileReader(const XMLFileReader &) = delete;
    XMLFileReader &operator=(const XMLFileReader &) = delete;
    XMLFileReader(XMLFileReader &&) = default;
    XMLFileReader &operator=(XMLFileReader &&) = default;

    /**
     * Reads an HMM from an XML file with comprehensive error handling.
     *
     * @param filename Path to the input XML file
     * @return Loaded HMM object
     * @throws std::invalid_argument if filename is empty
     * @throws std::runtime_error if file cannot be read or parsed
     */
    Hmm read(const std::string &filename);

    /**
     * Reads an HMM from an XML file with filesystem path.
     *
     * @param filepath Path to the input XML file
     * @return Loaded HMM object
     * @throws std::invalid_argument if filepath is empty
     * @throws std::runtime_error if file cannot be read or parsed
     */
    Hmm read(const std::filesystem::path &filepath);

    /**
     * Validates that a file can be read from the given path.
     *
     * @param filepath Path to validate
     * @return true if the file can be read, false otherwise
     */
    static bool canReadFromPath(const std::filesystem::path &filepath) noexcept;

    /**
     * Checks if a file exists and looks like a libhmm XML file.
     *
     * @param filepath Path to check
     * @return true if file is readable, non-empty, and starts with an XML declaration
     */
    static bool canParseAsHmm(const std::filesystem::path &filepath) noexcept;

private:
    /**
     * Internal implementation for reading HMM from stream.
     *
     * @param stream Input stream
     * @return Loaded HMM object
     * @throws std::runtime_error if deserialization fails
     */
    Hmm readFromStream(std::ifstream &stream);
};

} // namespace libhmm
