#include <iostream>
#include "TensorRT_Loader.h"

int main() {
    std::cout << "Hello, TensorRT!" << std::endl;

    // Example usage of a function from TensorRT_Loader.cpp
    if (loadTensorRTModel()) {
        std::cout << "Model loaded successfully!" << std::endl;
    }
    else {
        std::cout << "Failed to load the model." << std::endl;
    }

    return 0;
}
