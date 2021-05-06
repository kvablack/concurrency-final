/* Calculates the mean cross entropy loss for a batch of raw predictions.
 * The first argument should be the raw output, shape (batch_size, num_labels).
 * The second argument should be the labels, shape (1, batch_size).
 */
#include "ops.h"
#include <math.h>
#include <iostream>
#include <assert.h>

using namespace Ops;

static const uint32_t CROSSENTROPY_TILE_SIZE = 32;

chrono::duration<float> CrossEntropy::forwardTime;
chrono::duration<float> CrossEntropy::backwardTime;

CrossEntropy::CrossEntropy(shared_ptr<Op> a, shared_ptr<Op> b): Op(1, 1), a(a), b(b), expSums(1, a->height) {
    assert(a->height == b->width);
    assert(b->height == 1);
}

__global__ void crossEntropyKernel(
        float* a, float* b, float* c, float* expSums,
        uint32_t a_height, uint32_t a_width
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < a_height) {
        float expSum = 0;
        for (uint32_t j = 0; j < a_width; j++) {
            expSum += exp(a[i * a_width + j]);
        }
        atomicAdd(c, (log(expSum) - a[i * a_width + __float2uint_rd(b[i])]) / a_height);
        expSums[i] = expSum;
    }
}

__global__ void aGradKernel(
        float* aData, float* bData, float* expSums, float* aGrad,
        uint32_t a_height, uint32_t a_width

) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < a_height && j < a_width) {
        float shift = j == __float2uint_rd(bData[i]);
        aGrad[i * a_width + j] = exp(aData[i * a_width + j]) / expSums[i] - shift;
    }
}

void CrossEntropy::forward(bool gpu) {
    this->out.allocate(gpu);
    this->expSums.allocate(gpu);
    this->a->forward(gpu);
    this->b->forward(gpu);

    auto start = chrono::steady_clock::now();
    if (!gpu) {
        float* aData = this->a->out.get();
        float* bData = this->b->out.get();
        float* expSumData = this->expSums.get();
        float loss = 0;
        for (uint32_t i = 0; i < this->a->height; i++) {
            float expSum = 0;
            for (uint32_t j = 0; j < this->a->width; j++) {
                expSum += exp(aData[i * this->a->width + j]);
            }
            expSumData[i] = expSum;
            loss += log(expSum) - aData[i * this->a->width + static_cast<uint32_t>(bData[i])];
        }
        *out.get() = loss / this->a->height;
    } else {
        cudaMemset(this->out.get(), 0, 4);
        crossEntropyKernel<<<(this->a->height + CROSSENTROPY_TILE_SIZE - 1) / CROSSENTROPY_TILE_SIZE, CROSSENTROPY_TILE_SIZE>>>(
                this->a->out.get(), this->b->out.get(), this->out.get(), this->expSums.get(),
                this->a->height, this->a->width
        );
        cudaDeviceSynchronize();
    }
    this->forwardTime += chrono::steady_clock::now() - start;
}

void CrossEntropy::backward() {
    assert(this->out.isAllocated);
    this->a->grad.allocate(this->out.isGpu);

    auto start = chrono::steady_clock::now();

    float* aData = this->a->out.get();
    float* bData = this->b->out.get();
    float* expSumData = this->expSums.get();
    float* aGrad = this->a->grad.get();

    if (!this->out.isGpu) {
        for (uint32_t i = 0; i < this->a->height; i++) {
            for (uint32_t j = 0; j < this->a->width; j++) {
                aGrad[i * this->a->width + j] = exp(aData[i * this->a->width + j]) / expSumData[i]; // / this->a->height;
            }
            aGrad[i * this->a->width + static_cast<uint32_t>(bData[i])] -= 1; // / this->a->height;
        }
    } else {
        dim3 blocks((this->a->width + CROSSENTROPY_TILE_SIZE - 1) / CROSSENTROPY_TILE_SIZE, (this->a->height + CROSSENTROPY_TILE_SIZE - 1) / CROSSENTROPY_TILE_SIZE);
        dim3 threads(CROSSENTROPY_TILE_SIZE, CROSSENTROPY_TILE_SIZE);
        aGradKernel<<<blocks, threads>>>(
                aData, bData, expSumData, aGrad,
                this->a->height, this->a->width
        );
        cudaDeviceSynchronize();
    }
    this->backwardTime += chrono::steady_clock::now() - start;

    this->a->backward();
    this->b->backward();
}
