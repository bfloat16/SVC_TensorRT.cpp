#ifndef TENSORRT_LOADER_H
#define TENSORRT_LOADER_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>

std::unique_ptr<char[]> loadEngine(const std::string& filename, size_t& engineSize);

void createContext(const std::unique_ptr<char[]>& engineData, const size_t& engineSize, nvinfer1::IRuntime*& runtime, nvinfer1::ICudaEngine*& engine, nvinfer1::IExecutionContext*& context);

#endif