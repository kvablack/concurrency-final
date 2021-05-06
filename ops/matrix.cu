/* Data for this project is required to be stored in a special binary file format
 * that is easy to read. It must be 2-dimensional and be represented as float32.
 *
 * First 8 bytes: a magic string, "\x93CS378H\0".
 * The next 4 bytes: an unsigned integer giving the first dimension, n.
 * The next 4 bytes: an unsigned integer giving the second dimension, m.
 *
 * The rest of the file is the data in row-major order, float32 format,
 * which should be 4 * n * m bytes.
 */
#include <fstream>
#include <assert.h>
#include <iostream>
#include <string.h>
#include "matrix.h"

const char Matrix::MAGIC[] = "\x93" "CS378H";
chrono::duration<float> Matrix::allocTime;
chrono::duration<float> Matrix::saveTime;
chrono::duration<float> Matrix::loadTime;
chrono::duration<float> Matrix::copyTime;

struct Matrix::GPUData {
    float* ptr;

    GPUData(uint32_t size) {
        cudaMalloc(&ptr, size * 4);
        cudaDeviceSynchronize();
    }

    ~GPUData() {
        cudaFree(ptr);
    }
};

Matrix::Matrix(uint32_t height, uint32_t width): height(height), width(width) {
    assert(height > 0);
    assert(width > 0);
}

void Matrix::allocate(bool gpu) {
    if (this->isAllocated) {
        assert(gpu == this->isGpu);
        return;
    }

    this->isAllocated = true;
    this->isGpu = gpu;
    auto start = chrono::steady_clock::now();
    if (gpu) {
        this->gpuData = shared_ptr<GPUData>(new GPUData(height * width));
    } else {
        this->cpuData = shared_ptr<float>(new float[height * width], default_delete<float[]>());
    }
    this->allocTime += chrono::steady_clock::now() - start;
}

float* Matrix::get() {
    // get host or device pointer, if allocated, depending on matrix type
    assert(this->isAllocated);
    if (this->isGpu) {
        return this->gpuData->ptr;
    } else {
        return this->cpuData.get();
    }
}

void Matrix::toGpu() {
    // move data to gpu, turn this into GPU matrix
    assert(this->isAllocated && !this->isGpu);
    this->isGpu = true;
    if (!this->gpuData) {
        auto start = chrono::steady_clock::now();
        this->gpuData = shared_ptr<GPUData>(new GPUData(height * width));
        this->allocTime += chrono::steady_clock::now() - start;
    }
    auto start = chrono::steady_clock::now();
    cudaMemcpy(this->gpuData->ptr, this->cpuData.get(), height * width * 4, cudaMemcpyHostToDevice);
    this->copyTime += chrono::steady_clock::now() - start;
}

void Matrix::toCpu() {
    // move data to cpu, turn this into CPU matrix
    assert(this->isAllocated && this->isGpu);
    this->isGpu = false;
    if (!this->cpuData) {
        auto start = chrono::steady_clock::now();
        this->cpuData = shared_ptr<float>(new float[height * width], default_delete<float[]>());
        this->allocTime += chrono::steady_clock::now() - start;
    }
    auto start = chrono::steady_clock::now();
    cudaMemcpy(this->cpuData.get(), this->gpuData->ptr, height * width * 4, cudaMemcpyDeviceToHost);
    this->copyTime += chrono::steady_clock::now() - start;
}

Matrix Matrix::load(string path) {
    auto start = chrono::steady_clock::now();
    ifstream in(path);
    assert(in.is_open());

    in.seekg(0);
    char magic[8];
    in.read(magic, 8);
    assert(strcmp(magic, MAGIC) == 0);

    uint32_t n;
    uint32_t m;
    in.read(reinterpret_cast<char*>(&n), 4);
    in.read(reinterpret_cast<char*>(&m), 4);

    in.seekg(0, ios::end);
    size_t size = in.tellg();
    assert(size == 16 + 4 * n * m);

    Matrix mat = Matrix(n, m);
    mat.allocate(); // on CPU
    in.seekg(16);
    in.read(reinterpret_cast<char*>(mat.get()), size);

    in.close();
    Matrix::loadTime += chrono::steady_clock::now() - start;
    return mat;
}

void Matrix::save(string path) {
    auto start = chrono::steady_clock::now();
    assert(this->isAllocated);
    // if GPU matrix, get newest data off GPU first
    if (this->isGpu) {
        this->toCpu();
        this->isGpu = true;
    }
    ofstream out(path);
    assert(out.is_open());

    out.write(MAGIC, 8);
    out.write(reinterpret_cast<char*>(&this->height), 4);
    out.write(reinterpret_cast<char*>(&this->width), 4);
    out.write(reinterpret_cast<char*>(this->cpuData.get()), 4 * height * width);

    out.close();
    this->saveTime += chrono::steady_clock::now() - start;
}

