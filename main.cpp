#include "gan/Gan.h"
#include "data/DataLoader.h"
#include <nlohmann/json.hpp>
#include <fstream>

int main() {
    // Load configuration
    std::ifstream configFile("config.json");
    nlohmann::json config;
    configFile >> config;

    int noise_dim = config["noise_dim"];
    int epochs = config["epochs"];
    double lr = config["learning_rate"];
    std::string inputFile = config["input_file"];
    std::string outputFile = config["output_file"];

    DataLoader loader;
    auto data = loader.loadCSV(inputFile);
    int data_dim = data[0].size();

    GAN gan(noise_dim, data_dim, epochs, lr);
    gan.train(data);

    auto syntheticData = gan.generate(1000);
    loader.saveCSV(outputFile, syntheticData);

    gan.saveModel("gan_model");

    return 0;
}
