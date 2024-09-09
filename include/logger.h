#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>

enum class LogLevel {
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static void init(const std::string& filename);
    static void log(LogLevel level, const std::string& message);
    static void cleanup();

private:
    static std::ofstream log_file;
};

#endif // LOGGER_H