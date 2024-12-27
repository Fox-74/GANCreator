#include "gan/Gan.h"
#include "data/DataLoader.h"
#include "utils/Logger.h"
#include <nlohmann/json.hpp>
#include <fstream>

int main() {
    // Инициализация логгера
    Logger::init();
    spdlog::info("Starting GAN training...");

    // Загрузка конфигурации
    std::ifstream configFile("config.json");
    nlohmann::json config;
    configFile >> config;

    int noise_dim = config["noise_dim"];
    int epochs = config["epochs"];
    double lr = config["learning_rate"];
    std::string inputFile = config["input_file"];
    std::string outputFile = config["output_file"];

    // Загрузка данных
    spdlog::info("Loading data from: {}", inputFile);
    DataLoader loader;
    auto data = loader.loadCSV(inputFile);
    int data_dim = data[0].size();

    // Создание GAN
    GAN gan(noise_dim, data_dim, epochs, lr);

    // Обучение GAN
    spdlog::info("Training GAN...");
    gan.train(data);

    // Генерация данных
    spdlog::info("Generating synthetic data...");
    auto syntheticData = gan.generate(1000);
    loader.saveCSV(outputFile, syntheticData);

    // Сохранение модели
    spdlog::info("Saving GAN model...");
    gan.saveModel("gan_model");

    spdlog::info("Training complete. Synthetic data saved to: {}", outputFile);
    return 0;
}
