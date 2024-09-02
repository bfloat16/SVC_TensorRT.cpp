#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <iostream> 
#include <fstream>
#include <vector>
#include <random>
#include <cassert>

class TRTLogger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} logger;

nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& engineFilePath, nvinfer1::IRuntime* runtime) {
    std::ifstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error opening engine file!" << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    size_t modelSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> modelData(modelSize);
    engineFile.read(modelData.data(), modelSize);
    engineFile.close();

    return runtime->deserializeCudaEngine(modelData.data(), modelSize);
}

bool build_model(const std::string& onnx_filename) {
    TRTLogger logger;

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(onnx_filename.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    int device;
    cudaGetDevice(&device);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t memory_pool_limit = static_cast<size_t>(free_mem * 0.9);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, memory_pool_limit);

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

    nvinfer1::Dims mel_dims = network->getInput(0)->getDimensions();
    nvinfer1::Dims f0_dims = network->getInput(1)->getDimensions();

    // Set the dimensions for the 'mel' input
    profile->setDimensions("mel", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 2656, 128));
    profile->setDimensions("mel", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 2656, 128));
    profile->setDimensions("mel", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 2656, 128));

    // Set the dimensions for the 'f0' input
    profile->setDimensions("f0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 2656));
    profile->setDimensions("f0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 2656));
    profile->setDimensions("f0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 2656));

    config->addOptimizationProfile(profile);
    config->setBuilderOptimizationLevel(3);

    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (serializedModel == nullptr) {
        std::cout << "Build serialized network failed." << std::endl;
        return false;
    }

    FILE* f;
    errno_t err = fopen_s(&f, "nsf_hifigan.engine", "wb");
    if (err != 0 || !f) {
        std::cout << "Failed to open file for writing." << std::endl;
        delete serializedModel;
        return false;
    }
    fwrite(serializedModel->data(), 1, serializedModel->size(), f);
    fclose(f);

    // Clean up resources
    delete serializedModel;
    delete parser;
    delete network;
    delete config;
    delete builder;

    std::cout << "Done." << std::endl;
    return true;
}

std::vector<unsigned char> load_file(const std::string& file) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

bool performInference(const std::string& engineFilePath) {
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = loadEngineFromFile(engineFilePath, runtime);
    if (!engine) {
        std::cerr << "Failed to load engine!" << std::endl;
        delete runtime;
        return false;
    }

    // Create execution context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // Random number generation setup
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Determine size for input buffers
    int melSize = 1 * 2656 * 128;
    int f0Size = 1 * 2656;

    // GPU memory allocation for input buffers
    float* d_inputMel;
    float* d_inputF0;
    cudaMalloc((void**)&d_inputMel, melSize * sizeof(float));
    cudaMalloc((void**)&d_inputF0, f0Size * sizeof(float));

    // Populate input buffers with random data
    std::vector<float> h_inputMel(melSize), h_inputF0(f0Size);
    std::generate(h_inputMel.begin(), h_inputMel.end(), [&]() { return dist(rng); });
    std::generate(h_inputF0.begin(), h_inputF0.end(), [&]() { return dist(rng); });

    // Copy data from host to device
    cudaMemcpy(d_inputMel, h_inputMel.data(), melSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputF0, h_inputF0.data(), f0Size * sizeof(float), cudaMemcpyHostToDevice);

    // Set input shapes
    context->setInputShape("mel", nvinfer1::Dims3(1, 2656, 128));
    context->setInputShape("f0", nvinfer1::Dims2(1, 2656));

    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "Error, not all required dimensions specified." << std::endl;
        return false;
    }

    // Get output dimensions after setting the input shapes
    nvinfer1::Dims outputDims = context->getTensorShape("waveform");
    int totalOutputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) {
        totalOutputSize *= outputDims.d[i];
    }

    // GPU memory allocation for output buffer
    float* d_outputWaveform;
    cudaMalloc((void**)&d_outputWaveform, totalOutputSize * sizeof(float));

    // Set tensor addresses
    context->setTensorAddress("mel", d_inputMel);
    context->setTensorAddress("f0", d_inputF0);
    context->setTensorAddress("waveform", d_outputWaveform);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    std::vector<float> h_outputWaveform(totalOutputSize);
    cudaMemcpy(h_outputWaveform.data(), d_outputWaveform, totalOutputSize * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream outputFile("output.bin", std::ios::binary);
    outputFile.write(reinterpret_cast<char*>(h_outputWaveform.data()), totalOutputSize * sizeof(float));
    outputFile.close();

    cudaFree(d_inputMel);
    cudaFree(d_inputF0);
    cudaFree(d_outputWaveform);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;

    return true;
}
int main() {
    /*
        std::string onnx_filename = "C:/Users/OOPPEENN/Desktop/SVC_TensorRT.cpp/build/x64-debug/nsf_hifigan.onnx";
        if (!build_model(onnx_filename)) {
        return -1;
    }
    */
    std::string engine_filename = "C:/Users/OOPPEENN/Desktop/SVC_TensorRT.cpp/build/x64-debug/nsf_hifigan.engine";
        if (!performInference(engine_filename)) {
        std::cerr << "Inference failed." << std::endl;
        return 1;
    }
}