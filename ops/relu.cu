#include "ops.h"
#include <assert.h>

using namespace Ops;

static const uint32_t RELU_TILE_SIZE = 32;

chrono::duration<float> Relu::forwardTime;
chrono::duration<float> Relu::backwardTime;

Relu::Relu(shared_ptr<Op> a): Op(a->height, a->width), a(a) {}

__global__ void reluKernel(
        float* a, float* c, uint32_t height, uint32_t width
) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        c[i * width + j] = max(0.0f, a[i * width + j]);
    }
}

__global__ void aGradKernel(
        float* aData, float* outGrad, float* aGrad, uint32_t height, uint32_t width
) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        aGrad[i * width + j] = aData[i * width + j] <= 0 ? 0 : outGrad[i * width + j];
    }
}

void Relu::forward(bool gpu) {
    this->out.allocate(gpu);
    this->a->forward(gpu);

    auto start = chrono::steady_clock::now();

    float* aData = this->a->out.get();
    float* outData = this->out.get();
    if (!gpu) {
        for (uint32_t i = 0; i < this->height; i++) {
            for (uint32_t j = 0; j < this->width; j++) {
                outData[i * this->width + j] = max(0.0f, aData[i * this->width + j]);
            }
        }
    } else {
        dim3 blocks((this->width + RELU_TILE_SIZE - 1) / RELU_TILE_SIZE, (this->height + RELU_TILE_SIZE - 1) / RELU_TILE_SIZE);
        dim3 threads(RELU_TILE_SIZE, RELU_TILE_SIZE);
        reluKernel<<<blocks, threads>>>(
                aData, outData, this->height, this->width
        );
        cudaDeviceSynchronize();
    }
    this->forwardTime += chrono::steady_clock::now() - start;
}

void Relu::backward() {
    assert(this->out.isAllocated && this->grad.isAllocated);
    this->a->grad.allocate(this->out.isGpu);

    auto start = chrono::steady_clock::now();

    float* aData = this->a->out.get();
    float* outGrad = this->grad.get();
    float* aGrad = this->a->grad.get();

    if (!this->out.isGpu) {
        for (uint32_t i = 0; i < this->height; i++) {
            for (uint32_t j = 0; j < this->width; j++) {
                aGrad[i * this->width + j] = aData[i * this->width + j] < 0 ? 0 : outGrad[i * this->width + j];
                // printf("%f\n", aGrad[i * this->width + j]);
            }
        }
    } else {
        dim3 blocks((this->width + RELU_TILE_SIZE - 1) / RELU_TILE_SIZE, (this->height + RELU_TILE_SIZE - 1) / RELU_TILE_SIZE);
        dim3 threads(RELU_TILE_SIZE, RELU_TILE_SIZE);
        aGradKernel<<<blocks, threads>>>(
                aData, outGrad, aGrad, this->height, this->width
        );
        cudaDeviceSynchronize();
    }

    this->backwardTime += chrono::steady_clock::now() - start;

    this->a->backward();
}
