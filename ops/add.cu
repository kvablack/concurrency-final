/* Performs batched vector addition. The first argument must have a height of 1.
 * The second argument is assumed to be a batch of data with the batch dimension first.
 */
#include "ops.h"
#include <iostream>
#include <assert.h>

using namespace Ops;

static const uint32_t ADD_TILE_SIZE = 32;

chrono::duration<float> Add::forwardTime;
chrono::duration<float> Add::backwardTime;

Add::Add(shared_ptr<Op> a, shared_ptr<Op> b): Op(b->height, b->width), a(a), b(b) {
    assert(a->width == b->width);
    assert(a->height == 1);
}

__global__ void addKernel(
        float* a, float* b, float* c,
        uint32_t a_width, uint32_t b_width,
        uint32_t c_height, uint32_t c_width
) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < c_height && j < c_width) {
        c[i * c_width + j] = a[j] + b[i * b_width + j];
    }
}

__global__ void aGradKernel(
        float* outGrad, float* aGrad, uint32_t height, uint32_t width
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < width) {
        float sum = 0;
        for (uint32_t i = 0; i < height; i++) {
            sum += outGrad[i * width + j];
        }
        aGrad[j] = sum;
    }
}

void Add::forward(bool gpu) {
    this->out.allocate(gpu);
    this->a->forward(gpu);
    this->b->forward(gpu);

    auto start = chrono::steady_clock::now();

    float* aData = this->a->out.get();
    float* bData = this->b->out.get();
    float* outData = this->out.get();
    if (!gpu) {
        for (uint32_t i = 0; i < this->height; i++) {
            for (uint32_t j = 0; j < this->width; j++) {
                outData[i * this->width + j] = aData[j] + bData[i * this->width + j];
            }
        }
    } else {
        dim3 blocks((this->width + ADD_TILE_SIZE - 1) / ADD_TILE_SIZE, (this->height + ADD_TILE_SIZE - 1) / ADD_TILE_SIZE);
        dim3 threads(ADD_TILE_SIZE, ADD_TILE_SIZE);
        addKernel<<<blocks, threads>>>(
                aData, bData, outData,
                this->a->width, this->b->width,
                this->height, this->width
        );
        cudaDeviceSynchronize();
    }
    this->forwardTime += chrono::steady_clock::now() - start;
}

void Add::backward() {
    assert(this->out.isAllocated && this->grad.isAllocated);
    this->a->grad.allocate(this->out.isGpu);
    this->b->grad.allocate(this->out.isGpu);

    auto start = chrono::steady_clock::now();

    float* outGrad = this->grad.get();
    float* aGrad = this->a->grad.get();
    float* bGrad = this->b->grad.get();
    if (!this->out.isGpu) {
        for (uint32_t j = 0; j < this->width; j++) {
            float sum = 0;
            for (uint32_t i = 0; i < this->height; i++) {
                sum += outGrad[i * this->width + j];
            }
            aGrad[j] = sum;
        }
        memcpy(bGrad, outGrad, this->height * this->width * 4);
    } else {
        aGradKernel<<<(this->width + ADD_TILE_SIZE - 1) / ADD_TILE_SIZE, ADD_TILE_SIZE>>>(
                outGrad, aGrad, this->height, this->width
        );
        cudaMemcpy(bGrad, outGrad, this->height * this->width * 4, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }

    this->backwardTime += chrono::steady_clock::now() - start;

    this->a->backward();
    this->b->backward();
}
