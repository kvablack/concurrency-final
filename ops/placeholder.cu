#include "ops.h"
#include <iostream>
#include <assert.h>

using namespace Ops;

Placeholder::Placeholder(uint32_t height, uint32_t width): Op(height, width) {}

void Placeholder::forward(bool gpu) {
    assert(this->out.isAllocated);
    if (gpu && !this->out.isGpu) {
        this->out.toGpu();
    }
}

void Placeholder::backward() {}

void Placeholder::fill(float* data) {
    if (!this->out.isAllocated) {
        this->out.allocate(); // allocate on CPU
    }
    
    auto start = chrono::steady_clock::now();
    if (this->out.isGpu) {
        cudaMemcpy(this->out.get(), data, this->height * this->width * 4, cudaMemcpyHostToDevice);
    } else {
        memcpy(this->out.get(), data, this->height * this->width * 4);
    }
    Matrix::copyTime += chrono::steady_clock::now() - start;
}
