#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>

class DataLoader {
public:
    std::vector<std::vector<double>> loadCSV(const std::string& filePath);
    void saveCSV(const std::string& filePath, const std::vector<std::vector<double>>& data);
};

#endif // DATALOADER_H
