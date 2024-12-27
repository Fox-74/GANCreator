#ifndef LOGGER_H
#define LOGGER_H

#include <spdlog/spdlog.h>
#include <memory>

class Logger {
public:
    // Инициализация логгера
    static void init() {
        if (!logger) {
            logger = spdlog::stdout_color_mt("console");
            spdlog::set_level(spdlog::level::info); // Устанавливаем уровень логирования
        }
    }

    // Получить логгер
    static std::shared_ptr<spdlog::logger> getLogger() {
        if (!logger) {
            init();
        }
        return logger;
    }

private:
    static std::shared_ptr<spdlog::logger> logger;
};

#endif // LOGGER_H
