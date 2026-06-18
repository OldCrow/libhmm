#include "libhmm/io/xml_file_reader.h"
#include <iostream>
#include <sstream>

namespace libhmm {

// cppcheck-suppress functionStatic
Hmm XMLFileReader::read(std::string_view filename) {
    if (filename.empty()) {
        throw std::invalid_argument("Filename cannot be empty");
    }

    return read(std::filesystem::path{filename});
}

// cppcheck-suppress functionStatic
Hmm XMLFileReader::read(const std::filesystem::path &filepath) {
    if (filepath.empty()) {
        throw std::invalid_argument("Filepath cannot be empty");
    }

    // Validate that we can read from this path
    if (!canReadFromPath(filepath)) {
        throw std::runtime_error("Cannot read from path: " + filepath.string());
    }

    // Check if the file looks like a libhmm XML file before opening
    if (!canParseAsHmm(filepath)) {
        throw std::runtime_error("File does not appear to be a libhmm XML file: " +
                                 filepath.string());
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
            std::cerr << "Warning: Failed to properly close file: " << filepath.string()
                      << std::endl;
        }

        return hmm;

    } catch (const std::ios_base::failure &e) {
        throw std::runtime_error("I/O operation failed: " + std::string(e.what()));
    }
}

bool XMLFileReader::canReadFromPath(const std::filesystem::path &filepath) noexcept {
    try {
        // Reject non-regular files (directories, devices, sockets, etc.).
        if (!std::filesystem::is_regular_file(filepath)) {
            return false;
        }
        // Probe with an actual open rather than checking permission bits: the
        // permission-bit approach only tests owner_read and misses group-readable
        // or world-readable files not owned by the current user (e.g. shared
        // model files on NFS, or files with ACLs).  The OS open() is definitive.
        std::ifstream probe(filepath, std::ios::in | std::ios::binary);
        return probe.is_open();
    } catch (...) {
        return false;
    }
}

bool XMLFileReader::canParseAsHmm(const std::filesystem::path &filepath) noexcept {
    try {
        if (!canReadFromPath(filepath)) {
            return false;
        }

        // Reject empty files and implausibly large ones (100 MB).
        const auto fileSize = std::filesystem::file_size(filepath);
        if (fileSize == 0 || fileSize > 100UL * 1024UL * 1024UL) {
            return false;
        }

        // Check that the file begins with an XML declaration — the only marker
        // written by XMLFileWriter::writeToStream.
        std::ifstream file(filepath, std::ios::in);
        if (!file.is_open()) {
            return false;
        }

        std::string firstLine;
        if (!std::getline(file, firstLine)) {
            return false;
        }

        return firstLine.find("<?xml") != std::string::npos;

    } catch (...) {
        return false;
    }
}

Hmm XMLFileReader::readFromStream(std::ifstream &stream) {
    if (!stream.good()) {
        throw std::runtime_error("Stream is not in a good state for reading");
    }

    // Format: <?xml ...?><libhmm_model><![CDATA[ <HMM text via operator<<> ]]></libhmm_model>
    // Skip lines until the CDATA section begins, then hand off to operator>>.
    try {
        std::string line;
        while (std::getline(stream, line)) {
            if (line.find("<![CDATA[") != std::string::npos) {
                break;
            }
        }

        Hmm hmm(2); // initial size; operator>> resizes via state-count parsing
        stream >> hmm;

        if (stream.fail()) {
            throw std::runtime_error("Stream error occurred during reading");
        }

        return hmm;

    } catch (const std::ios_base::failure &e) {
        throw std::runtime_error("Stream I/O error: " + std::string(e.what()));
    }
}

} // namespace libhmm
