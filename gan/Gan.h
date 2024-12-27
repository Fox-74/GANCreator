#ifndef GAN_H
#define GAN_H

#include <torch/torch.h>
#include <vector>
#include <string>

class Generator : public torch::nn::Module {
public:
    Generator(int input_size, int output_size);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential network;
};

class Discriminator : public torch::nn::Module {
public:
    Discriminator(int input_size);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential network;
};

class GAN {
public:
    GAN(int noise_dim, int data_dim, int epochs, double lr);
    void train(const std::vector<std::vector<double>>& data);
    std::vector<std::vector<double>> generate(int num_samples);
    void saveModel(const std::string& path);
    void loadModel(const std::string& path);

private:
    int noise_dim;
    int data_dim;
    int epochs;
    double lr;

    Generator generator;
    Discriminator discriminator;

    torch::optim::Adam gen_optimizer;
    torch::optim::Adam disc_optimizer;

    void logMetrics(int epoch, float d_loss, float g_loss);
};

#endif // GAN_H
