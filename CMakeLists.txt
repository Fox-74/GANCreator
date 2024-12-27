cmake_minimum_required(VERSION 3.10)
project(GAN_CPP)

find_package(Torch REQUIRED)
find_package(spdlog REQUIRED)

add_executable(gan_app main.cpp gan/Gan.cpp data/DataLoader.cpp)
target_link_libraries(gan_app "${TORCH_LIBRARIES}" spdlog::spdlog)
set_property(TARGET gan_app PROPERTY CXX_STANDARD 17)
