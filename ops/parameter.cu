#include "ops.h"
#include <assert.h>
#include <iostream>
#include <random>

using namespace Ops;

static const uint32_t STEP_TILE_SIZE = 32;

chrono::duration<float> Parameter::initTime;
chrono::duration<float> Parameter::stepTime;

Parameter::Parameter(uint32_t height, uint32_t width, string name): Op(height, width), name(name) {}

__global__ void stepKernel(
        float* data, float* gradData,
        uint32_t height, uint32_t width,
        float lr
) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        data[i * width + j] -= lr * gradData[i * width + j];
    }
}

void Parameter::init() {
    this->out.allocate(); // on CPU

    auto start = chrono::steady_clock::now();

    if (this->height == 1) {
        // zero initialization for bias
        memset(this->out.get(), 0, this->height * this->width * 4);
    } else {
        // He weight initialization, which is best for ReLU
        default_random_engine generator;
        normal_distribution<float> dist(0, sqrt(2 / this->width));
        float* data = this->out.get();
        for (uint32_t i = 0; i < this->height; i++) {
            for (uint32_t j = 0; j < this->width; j++) {
                data[i * this->width + j] = dist(generator);
            }
        }
    }

    this->initTime += chrono::steady_clock::now() - start;
}

void Parameter::step(float lr) {
    assert(this->grad.isAllocated);

    auto start = chrono::steady_clock::now();

    float* gradData = this->grad.get();
    float* data = this->out.get();
    if (!this->grad.isGpu) {
        for (uint32_t i = 0; i < this->height; i++) {
            for (uint32_t j = 0; j < this->width; j++) {
                data[i * this->width + j] -= lr * gradData[i * this->width + j];
            }
        }
    } else {
        dim3 blocks((this->width + STEP_TILE_SIZE - 1) / STEP_TILE_SIZE, (this->height + STEP_TILE_SIZE - 1) / STEP_TILE_SIZE);
        dim3 threads(STEP_TILE_SIZE, STEP_TILE_SIZE);
        stepKernel<<<blocks, threads>>>(
                data, gradData,
                this->height, this->width,
                lr
        );
        cudaDeviceSynchronize();
    }

    this->stepTime += chrono::steady_clock::now() - start;
}

void Parameter::load(string dir) {
    string path = dir + "/" + this->name + ".data";
    this->out = Matrix::load(path);
    assert(this->out.height == this->height && this->out.width == this->width);
}

void Parameter::save(string dir) {
    string path = dir + "/" + this->name + ".data";
    this->out.save(path);
}

void Parameter::forward(bool gpu) {
    if (gpu && !this->out.isGpu) {
        this->out.toGpu();
    }
}

void Parameter::backward() {}
