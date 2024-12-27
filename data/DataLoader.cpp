#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<std::vector<double>> DataLoader::loadCSV(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        std::string value;

        while (std::getline(iss, value, ',')) {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
    }

    return data;
}

void DataLoader::saveCSV(const std::string& filePath, const std::vector<std::vector<double>>& data) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i != row.size() - 1) file << ",";
        }
        file << "\n";
    }
}
