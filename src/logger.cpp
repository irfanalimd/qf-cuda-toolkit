#define _CRT_SECURE_NO_WARNINGS

#include "logger.h"
#include <iostream>
#include <chrono>
#include <iomanip>

std::ofstream Logger::log_file;

void Logger::init(const std::string& filename) {
    log_file.open(filename, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << filename << std::endl;
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    if (!log_file.is_open()) return;

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    
    log_file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << " ";

    switch (level) {
        case LogLevel::INFO:    log_file << "[INFO] "; break;
        case LogLevel::WARNING: log_file << "[WARNING] "; break;
        case LogLevel::ERROR:   log_file << "[ERROR] "; break;
    }

    log_file << message << std::endl;
}

void Logger::cleanup() {
    if (log_file.is_open()) {
        log_file.close();
    }
}