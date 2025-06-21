#ifndef FILEIOMANAGER_H_
#define FILEIOMANAGER_H_

#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>
#include <memory>

namespace libhmm {

/**
 * Modern file I/O manager with C++17 features and comprehensive error handling.
 * Provides safe file operations with automatic resource management and validation.
 */
class FileIOManager {
public:
    FileIOManager() = default;
    ~FileIOManager() = default;
    
    // Non-copyable but movable
    FileIOManager(const FileIOManager&) = delete;
    FileIOManager& operator=(const FileIOManager&) = delete;
    FileIOManager(FileIOManager&&) = default;
    FileIOManager& operator=(FileIOManager&&) = default;
    
    /**
     * Reads entire file content as a string.
     * 
     * @param filepath Path to the file
     * @return File content as string
     * @throws std::runtime_error if file cannot be read
     */
    static std::string readTextFile(const std::filesystem::path& filepath);
    
    /**
     * Writes string content to a file.
     * 
     * @param filepath Path to the file
     * @param content Content to write
     * @param append If true, append to file; if false, overwrite
     * @throws std::runtime_error if file cannot be written
     */
    static void writeTextFile(const std::filesystem::path& filepath, 
                             const std::string& content, 
                             bool append = false);
    
    /**
     * Reads file content as lines.
     * 
     * @param filepath Path to the file
     * @return Vector of lines
     * @throws std::runtime_error if file cannot be read
     */
    static std::vector<std::string> readLines(const std::filesystem::path& filepath);
    
    /**
     * Writes lines to a file.
     * 
     * @param filepath Path to the file
     * @param lines Lines to write
     * @param append If true, append to file; if false, overwrite
     * @throws std::runtime_error if file cannot be written
     */
    static void writeLines(const std::filesystem::path& filepath, 
                          const std::vector<std::string>& lines, 
                          bool append = false);
    
    /**
     * Safely copies a file with error handling.
     * 
     * @param source Source file path
     * @param destination Destination file path
     * @param overwrite If true, overwrite existing file
     * @throws std::runtime_error if copy fails
     */
    static void copyFile(const std::filesystem::path& source,
                        const std::filesystem::path& destination,
                        bool overwrite = false);
    
    /**
     * Creates a backup of a file with timestamp.
     * 
     * @param filepath Path to the file to backup
     * @return Path to the backup file
     * @throws std::runtime_error if backup fails
     */
    static std::filesystem::path createBackup(const std::filesystem::path& filepath);
    
    /**
     * Validates file path and permissions.
     * 
     * @param filepath Path to validate
     * @param checkRead Check read permissions
     * @param checkWrite Check write permissions
     * @return true if path is valid and has required permissions
     */
    static bool validatePath(const std::filesystem::path& filepath,
                           bool checkRead = false,
                           bool checkWrite = false) noexcept;
    
    /**
     * Gets file size safely.
     * 
     * @param filepath Path to the file
     * @return File size in bytes, or nullopt if file doesn't exist
     */
    static std::optional<std::uintmax_t> getFileSize(const std::filesystem::path& filepath) noexcept;
    
    /**
     * Checks if file has expected extension.
     * 
     * @param filepath Path to check
     * @param expectedExtension Expected file extension (with or without dot)
     * @return true if file has the expected extension
     */
    static bool hasExtension(const std::filesystem::path& filepath, 
                           const std::string& expectedExtension) noexcept;
    
    /**
     * Creates directory structure if it doesn't exist.
     * 
     * @param dirpath Directory path to create
     * @throws std::runtime_error if directory creation fails
     */
    static void ensureDirectoryExists(const std::filesystem::path& dirpath);
    
    /**
     * Gets file modification time.
     * 
     * @param filepath Path to the file
     * @return File modification time, or nullopt if file doesn't exist
     */
    static std::optional<std::filesystem::file_time_type> getModificationTime(
        const std::filesystem::path& filepath) noexcept;

private:
    /**
     * Internal helper to validate file operations.
     */
    static void validateFileOperation(const std::filesystem::path& filepath, 
                                    const std::string& operation);
};

}

#endif
