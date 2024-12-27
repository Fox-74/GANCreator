#include "Gan.h"
#include "utils/Logger.h"
#include <iostream>
#include <spdlog/spdlog.h>

// Реализация генератора
Generator::Generator(int input_size, int output_size) {
    network = torch::nn::Sequential(
        torch::nn::Linear(input_size, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, output_size),
        torch::nn::Tanh()
    );
    register_module("network", network);
}

torch::Tensor Generator::forward(torch::Tensor x) {
    return network->forward(x);
}

// Реализация дискриминатора
Discriminator::Discriminator(int input_size) {
    network = torch::nn::Sequential(
        torch::nn::Linear(input_size, 256),
        torch::nn::LeakyReLU(0.2),
        torch::nn::Linear(256, 128),
        torch::nn::LeakyReLU(0.2),
        torch::nn::Linear(128, 1),
        torch::nn::Sigmoid()
    );
    register_module("network", network);
}

torch::Tensor Discriminator::forward(torch::Tensor x) {
    return network->forward(x);
}

// Реализация GAN
GAN::GAN(int noise_dim, int data_dim, int epochs, double lr)
    : noise_dim(noise_dim), data_dim(data_dim), epochs(epochs), lr(lr),
      generator(noise_dim, data_dim), discriminator(data_dim),
      gen_optimizer(generator.parameters(), lr), disc_optimizer(discriminator.parameters(), lr) {
    Logger::init(); // Инициализация логгера
}

void GAN::train(const std::vector<std::vector<double>>& data) {
    auto real_data = torch::tensor(data).to(torch::kFloat32);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Обучение дискриминатора
        auto noise = torch::randn({real_data.size(0), noise_dim});
        auto fake_data = generator.forward(noise);
        auto real_output = discriminator.forward(real_data);
        auto fake_output = discriminator.forward(fake_data.detach());

        auto d_loss = -torch::mean(torch::log(real_output) + torch::log(1 - fake_output));
        disc_optimizer.zero_grad();
        d_loss.backward();
        disc_optimizer.step();

        // Обучение генератора
        auto fake_output_gen = discriminator.forward(fake_data);
        auto g_loss = -torch::mean(torch::log(fake_output_gen));
        gen_optimizer.zero_grad();
        g_loss.backward();
        gen_optimizer.step();

        if (epoch % 100 == 0) {
            logMetrics(epoch, d_loss.item<float>(), g_loss.item<float>());
        }
    }
}

std::vector<std::vector<double>> GAN::generate(int num_samples) {
    auto noise = torch::randn({num_samples, noise_dim});
    auto generated = generator.forward(noise).to(torch::kCPU);

    std::vector<std::vector<double>> result;
    for (int i = 0; i < num_samples; i++) {
        result.push_back(std::vector<double>(generated[i].data_ptr<float>(), 
                                             generated[i].data_ptr<float>() + data_dim));
    }
    return result;
}

void GAN::saveModel(const std::string& path) {
    torch::save(generator, path + "_generator.pt");
    torch::save(discriminator, path + "_discriminator.pt");
}

void GAN::loadModel(const std::string& path) {
    torch::load(generator, path + "_generator.pt");
    torch::load(discriminator, path + "_discriminator.pt");
}

void GAN::logMetrics(int epoch, float d_loss, float g_loss) {
    auto logger = Logger::getLogger();
    logger->info("Epoch [{}]: D Loss = {:.4f}, G Loss = {:.4f}", epoch, d_loss, g_loss);
}
