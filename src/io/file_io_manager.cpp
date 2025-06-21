#include "libhmm/io/file_io_manager.h"
#include <sstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>

namespace libhmm {

std::string FileIOManager::readTextFile(const std::filesystem::path& filepath) {
    validateFileOperation(filepath, "read");
    
    try {
        std::ifstream file(filepath, std::ios::in);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());
        }
        
        // Use efficient string stream reading
        std::stringstream buffer;
        buffer << file.rdbuf();
        
        if (file.bad()) {
            throw std::runtime_error("Error occurred while reading file: " + filepath.string());
        }
        
        return buffer.str();
        
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("I/O error while reading file " + filepath.string() + ": " + e.what());
    }
}

void FileIOManager::writeTextFile(const std::filesystem::path& filepath, 
                                 const std::string& content, 
                                 bool append) {
    try {
        // Ensure directory exists
        if (filepath.has_parent_path()) {
            ensureDirectoryExists(filepath.parent_path());
        }
        
        auto mode = append ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc);
        std::ofstream file(filepath, mode);
        
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());
        }
        
        file << content;
        
        if (file.bad()) {
            throw std::runtime_error("Error occurred while writing to file: " + filepath.string());
        }
        
        file.close();
        if (file.fail()) {
            throw std::runtime_error("Failed to properly close file: " + filepath.string());
        }
        
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("I/O error while writing file " + filepath.string() + ": " + e.what());
    }
}

std::vector<std::string> FileIOManager::readLines(const std::filesystem::path& filepath) {
    validateFileOperation(filepath, "read");
    
    try {
        std::ifstream file(filepath, std::ios::in);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());
        }
        
        std::vector<std::string> lines;
        std::string line;
        
        while (std::getline(file, line)) {
            lines.push_back(std::move(line));
        }
        
        if (file.bad()) {
            throw std::runtime_error("Error occurred while reading file: " + filepath.string());
        }
        
        return lines;
        
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("I/O error while reading file " + filepath.string() + ": " + e.what());
    }
}

void FileIOManager::writeLines(const std::filesystem::path& filepath, 
                              const std::vector<std::string>& lines, 
                              bool append) {
    try {
        // Ensure directory exists
        if (filepath.has_parent_path()) {
            ensureDirectoryExists(filepath.parent_path());
        }
        
        auto mode = append ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc);
        std::ofstream file(filepath, mode);
        
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());
        }
        
        for (const auto& line : lines) {
            file << line << '\n';
        }
        
        if (file.bad()) {
            throw std::runtime_error("Error occurred while writing to file: " + filepath.string());
        }
        
        file.close();
        if (file.fail()) {
            throw std::runtime_error("Failed to properly close file: " + filepath.string());
        }
        
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("I/O error while writing file " + filepath.string() + ": " + e.what());
    }
}

void FileIOManager::copyFile(const std::filesystem::path& source,
                            const std::filesystem::path& destination,
                            bool overwrite) {
    validateFileOperation(source, "read");
    
    try {
        if (std::filesystem::exists(destination) && !overwrite) {
            throw std::runtime_error("Destination file already exists: " + destination.string());
        }
        
        // Ensure destination directory exists
        if (destination.has_parent_path()) {
            ensureDirectoryExists(destination.parent_path());
        }
        
        std::error_code ec;
        if (overwrite) {
            std::filesystem::copy_file(source, destination, 
                                     std::filesystem::copy_options::overwrite_existing, ec);
        } else {
            std::filesystem::copy_file(source, destination, ec);
        }
        
        if (ec) {
            throw std::runtime_error("Failed to copy file from " + source.string() + 
                                   " to " + destination.string() + ": " + ec.message());
        }
        
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Filesystem error during copy: " + std::string(e.what()));
    }
}

std::filesystem::path FileIOManager::createBackup(const std::filesystem::path& filepath) {
    validateFileOperation(filepath, "read");
    
    // Generate backup filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << filepath.stem().string() << "_backup_" 
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
       << filepath.extension().string();
    
    auto backupPath = filepath.parent_path() / ss.str();
    
    copyFile(filepath, backupPath, false);
    return backupPath;
}

bool FileIOManager::validatePath(const std::filesystem::path& filepath,
                                bool checkRead,
                                bool checkWrite) noexcept {
    try {
        if (!std::filesystem::exists(filepath)) {
            return false;
        }
        
        if (!std::filesystem::is_regular_file(filepath)) {
            return false;
        }
        
        auto perms = std::filesystem::status(filepath).permissions();
        
        if (checkRead && (perms & std::filesystem::perms::owner_read) == std::filesystem::perms::none) {
            return false;
        }
        
        if (checkWrite && (perms & std::filesystem::perms::owner_write) == std::filesystem::perms::none) {
            return false;
        }
        
        return true;
        
    } catch (...) {
        return false;
    }
}

std::optional<std::uintmax_t> FileIOManager::getFileSize(const std::filesystem::path& filepath) noexcept {
    try {
        if (!std::filesystem::exists(filepath)) {
            return std::nullopt;
        }
        
        return std::filesystem::file_size(filepath);
        
    } catch (...) {
        return std::nullopt;
    }
}

bool FileIOManager::hasExtension(const std::filesystem::path& filepath, 
                                const std::string& expectedExtension) noexcept {
    try {
        auto ext = filepath.extension().string();
        auto expected = expectedExtension;
        
        // Ensure both extensions start with '.'
        if (!expected.empty() && expected[0] != '.') {
            expected = "." + expected;
        }
        
        // Case-insensitive comparison
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        std::transform(expected.begin(), expected.end(), expected.begin(), ::tolower);
        
        return ext == expected;
        
    } catch (...) {
        return false;
    }
}

void FileIOManager::ensureDirectoryExists(const std::filesystem::path& dirpath) {
    try {
        if (!std::filesystem::exists(dirpath)) {
            std::error_code ec;
            std::filesystem::create_directories(dirpath, ec);
            if (ec) {
                throw std::runtime_error("Failed to create directory " + dirpath.string() + ": " + ec.message());
            }
        }
        
        if (!std::filesystem::is_directory(dirpath)) {
            throw std::runtime_error("Path exists but is not a directory: " + dirpath.string());
        }
        
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Filesystem error: " + std::string(e.what()));
    }
}

std::optional<std::filesystem::file_time_type> FileIOManager::getModificationTime(
    const std::filesystem::path& filepath) noexcept {
    try {
        if (!std::filesystem::exists(filepath)) {
            return std::nullopt;
        }
        
        return std::filesystem::last_write_time(filepath);
        
    } catch (...) {
        return std::nullopt;
    }
}

void FileIOManager::validateFileOperation(const std::filesystem::path& filepath, 
                                         const std::string& operation) {
    if (filepath.empty()) {
        throw std::invalid_argument("Filepath cannot be empty");
    }
    
    if (operation == "read") {
        if (!std::filesystem::exists(filepath)) {
            throw std::runtime_error("File does not exist: " + filepath.string());
        }
        
        if (!std::filesystem::is_regular_file(filepath)) {
            throw std::runtime_error("Path is not a regular file: " + filepath.string());
        }
        
        auto perms = std::filesystem::status(filepath).permissions();
        if ((perms & std::filesystem::perms::owner_read) == std::filesystem::perms::none) {
            throw std::runtime_error("No read permission for file: " + filepath.string());
        }
    }
}

}
