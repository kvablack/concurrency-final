#pragma once

#include <stdint.h>
#include <memory>
#include <string>
#include <chrono>

using namespace std;

class Matrix {
    struct GPUData;
    static const char MAGIC[];

    shared_ptr<float> cpuData;
    shared_ptr<GPUData> gpuData;

public:
    static chrono::duration<float> allocTime;
    static chrono::duration<float> saveTime;
    static chrono::duration<float> loadTime;
    static chrono::duration<float> copyTime;

    uint32_t height;
    uint32_t width;
    bool isAllocated = false;
    bool isGpu;

    Matrix(uint32_t height, uint32_t width);

    void allocate(bool gpu = false);
    float* get();
    void toGpu();
    void toCpu();

    static Matrix load(string path);
    void save(string path);
};
