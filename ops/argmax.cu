#include "ops.h"
#include <iostream>
#include <assert.h>

using namespace Ops;

static const uint32_t ARGMAX_TILE_SIZE = 32;

chrono::duration<float> Argmax::forwardTime;

Argmax::Argmax(shared_ptr<Op> a): Op(1, a->height), a(a) {}

__global__ void argmaxKernel(
        float* a, float* c,
        uint32_t a_height, uint32_t a_width
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < a_height) {
        float max = a[i * a_width];
        uint32_t ind = 0;
        for (uint32_t j = 1; j < a_width; j++) {
            float val = a[i * a_width + j];
            if (val > max) {
                max = val;
                ind = j;
            }
        }
        c[i] = ind;
    }
}

void Argmax::forward(bool gpu) {
    this->out.allocate(gpu);
    this->a->forward(gpu);

    auto start = chrono::steady_clock::now();
    if (!gpu) {
        float* aData = this->a->out.get();
        float* outData = this->out.get();
        for (uint32_t i = 0; i < this->width; i++) {
            float max = aData[i * this->a->width];
            uint32_t ind = 0;
            for (uint32_t j = 1; j < this->a->width; j++) {
                if (aData[i * this->a->width + j] > max) {
                    max = aData[i * this->a->width + j];
                    ind = j;
                }
            }
            outData[i] = ind;
        }
    } else {
        argmaxKernel<<<(this->width + ARGMAX_TILE_SIZE - 1) / ARGMAX_TILE_SIZE, ARGMAX_TILE_SIZE>>>(
                this->a->out.get(), this->out.get(),
                this->a->height, this->a->width
        );
        cudaDeviceSynchronize();
    }
    this->forwardTime += chrono::steady_clock::now() - start;
}

void Argmax::backward() {
    assert(false);
}
