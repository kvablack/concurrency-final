/* Actually performs a transposed matrix multiplication, i.e. (a @ b.T).T
 * This is for convenience, since the batch dimension will always be the first dimension,
 * so this will properly matrix-multiply a weight matrix and a batch of data while keeping the batch
 * dimension first.
 */
#include "ops.h"
#include <iostream>
#include <assert.h>

using namespace Ops;

static const uint32_t MATMUL_TILE_SIZE = 8;

chrono::duration<float> Matmul::forwardTime;
chrono::duration<float> Matmul::backwardTime;

Matmul::Matmul(shared_ptr<Op> a, shared_ptr<Op> b): Op(b->height, a->height), a(a), b(b) {
    assert(a->width == b->width);
}

/* An overly complicated dot product macro to take the dot product between a row/column of one matrix
   and a row/column of another matrix. Allows for such generalized dot products that are expanded at
   compile time without adding any more runtime conditionals.

    Arguments:
        sum (float): the identifier to store the result in, should be initialized to 0
        a, b (float*): the two matrices
        aWidth, bWidth: the widths of the two matrices
        i, j: indices into matrices a and b, respectively
        iRow, jRow (bool): If iRow is true, then i is treated as a row index into a. Otherwise,
            i is treated as a column index into a. Same for jRow, except for j and b.
 */
#define DOTPROD(sum, a, b, aWidth, bWidth, i, j, iRow, jRow, dim) \
    for (uint32_t k = 0; k < dim; k++) {\
        sum += DOTPROD_##iRow ## _##jRow(a, b, aWidth, bWidth, i, j);\
    }

#define DOTPROD_true_true(a, b, aWidth, bWidth, i, j) a[i * aWidth + k] * b[j * bWidth + k]
#define DOTPROD_true_false(a, b, aWidth, bWidth, i, j) a[i * aWidth + k] * b[k * bWidth + j]
#define DOTPROD_false_true(a, b, aWidth, bWidth, i, j) a[k * aWidth + i] * b[j * bWidth + k]
#define DOTPROD_false_false(a, b, aWidth, bWidth, i, j) a[k * aWidth + i] * b[k * bWidth + j]

__global__ void matmulKernel(
        float* a, float* b, float* c,
        uint32_t a_width, uint32_t b_width,
        uint32_t c_height, uint32_t c_width
) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < c_height && j < c_width) {
        float sum = 0;
        DOTPROD(sum, a, b, a_width, b_width, j, i, true, true, a_width)
        c[i * c_width + j] = sum;
    }
}

__global__ void aGradKernel(
        float* a, float* b, float* c,
        uint32_t a_width, uint32_t b_width,
        uint32_t c_height, uint32_t c_width,
        uint32_t b_height
) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < c_height && j < c_width) {
        float sum = 0;
        DOTPROD(sum, a, b, a_width, b_width, j, i, false, false, b_height)
        c[i * c_width + j] = sum;
    }
}

__global__ void bGradKernel(
        float* a, float* b, float* c,
        uint32_t a_width, uint32_t b_width,
        uint32_t c_height, uint32_t c_width
) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < c_height && j < c_width) {
        float sum = 0;
        DOTPROD(sum, a, b, a_width, b_width, j, i, false, true, b_width)
        c[i * c_width + j] = sum;
    }
}

void Matmul::forward(bool gpu) {
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
                float sum = 0;
                /*for (uint32_t k = 0; k < this->a->width; k++) {
                    sum += aData[j * this->a->width + k] * bData[i * this->b->width + k];
                }*/
                DOTPROD(sum, aData, bData, this->a->width, this->b->width, j, i, true, true, this->a->width)
                outData[i * this->width + j] = sum;
            }
        }
    } else {
        dim3 blocks((this->width + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE, (this->height + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE);
        dim3 threads(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE);
        matmulKernel<<<blocks, threads>>>(
                aData, bData, outData,
                this->a->width, this->b->width,
                this->height, this->width
        );
        cudaDeviceSynchronize();
    }
    this->forwardTime += chrono::steady_clock::now() - start;
}

void Matmul::backward() {
    assert(this->out.isAllocated && this->grad.isAllocated);
    this->a->grad.allocate(this->out.isGpu);
    this->b->grad.allocate(this->out.isGpu);

    auto start = chrono::steady_clock::now();

    float* aData = this->a->out.get();
    float* bData = this->b->out.get();
    float* outGrad = this->grad.get();
    float* aGrad = this->a->grad.get();
    float* bGrad = this->b->grad.get();
    if (!this->out.isGpu) {
        for (uint32_t i = 0; i < this->a->height; i++) {
            for (uint32_t j = 0; j < this->a->width; j++) {
                float sum = 0;
                /*for (uint32_t k = 0; k < this->height; k++) {
                    sum += bData[k * this->b->width + j] * outGrad[k * this->width + i];
                }*/
                DOTPROD(sum, bData, outGrad, this->b->width, this->width, j, i, false, false, this->height)
                aGrad[i * this->a->width + j] = sum;
            }
        }
        for (uint32_t i = 0; i < this->b->height; i++) {
            for (uint32_t j = 0; j < this->b->width; j++) {
                float sum = 0;
                /*for (uint32_t k = 0; k < this->width; k++) {
                    sum += aData[k * this->a->width + j] * outGrad[i * this->width + k];
                }*/
                DOTPROD(sum, aData, outGrad, this->a->width, this->width, j, i, false, true, this->width)
                bGrad[i * this->b->width + j] = sum;
            }
        }
    } else {
        dim3 threads(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE);

        dim3 aBlocks((this->a->width + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE, (this->a->height + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE);
        aGradKernel<<<aBlocks, threads>>>(
                bData, outGrad, aGrad,
                this->b->width, this->width,
                this->a->height, this->a->width,
                this->height
        );

        dim3 bBlocks((this->b->width + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE, (this->b->height + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE);
        bGradKernel<<<bBlocks, threads>>>(
                aData, outGrad, bGrad,
                this->a->width, this->width,
                this->b->height, this->b->width
        );
        cudaDeviceSynchronize();
    }

    this->backwardTime += chrono::steady_clock::now() - start;

    this->a->backward();
    this->b->backward();
}
