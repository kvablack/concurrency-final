#include <iostream>
#include <chrono>
#include <assert.h>
#include <sys/stat.h>
#include "model.h"

float cudaInit() {
    auto start = chrono::steady_clock::now();
    cudaFree(0);
    chrono::duration<double> duration = chrono::steady_clock::now() - start;
    return duration.count();
}


Model::Model(char* checkpointDir, uint32_t batchSize): checkpointDir(checkpointDir) {
    this->data = make_shared<Placeholder>(batchSize, 784);
    this->labels = make_shared<Placeholder>(1, batchSize);
    auto w1 = make_shared<Parameter>(256, 784, "w1");
    auto b1 = make_shared<Parameter>(1, 256, "b1");
    auto w2 = make_shared<Parameter>(10, 256, "w2");
    auto b2 = make_shared<Parameter>(1, 10, "b2");

    auto r1 = make_shared<Matmul>(w1, this->data);
    auto r2 = make_shared<Add>(b1, r1);
    auto r3 = make_shared<Relu>(r2);
    auto r4 = make_shared<Matmul>(w2, r3);
    auto result = make_shared<Add>(b2, r4);
    auto max = make_shared<Argmax>(result);

    this->out = max;
    this->loss = make_shared<CrossEntropy>(result, this->labels);

    this->parameters.push_back(w1);
    this->parameters.push_back(b1);
    this->parameters.push_back(w2);
    this->parameters.push_back(b2);

    struct stat sb;
    if (checkpointDir && stat(checkpointDir, &sb) == 0 && S_ISDIR(sb.st_mode)) {
        for (auto p : parameters) {
            p->load(checkpointDir);
        }
    } else {
        for (auto p : parameters) {
            p->init();
        }
    }
}

float Model::evaluate(bool useGpu) {
    this->out->forward(useGpu);
    
    if (useGpu) {
        this->out->out.toCpu();
    }
    int sum = 0;
    float* outData = this->out->out.get();
    float* labelsData = this->labels->out.get();
    for (uint32_t i = 0; i < this->out->out.width; i++) {
        sum += (outData[i] == labelsData[i]);
    }

    if (useGpu) {
        this->out->out.toGpu();
    }

    return static_cast<float>(sum) / this->labels->width;
}

float Model::trainStep(float lr, bool useGpu) {
    this->loss->forward(useGpu);
    this->loss->backward();

    for (auto p : parameters) {
        // must divide LR by batch size because gradients are summed accross batch
        p->step(lr / this->labels->width);
    }

    float loss;
    if (useGpu) {
        this->loss->out.toCpu();
        loss = *this->loss->out.get();
        this->loss->out.toGpu();
    } else {
        loss = *this->loss->out.get();
    }
    return loss;
}

void Model::save() {
    if (this->checkpointDir) {
        struct stat sb;
        if (stat(checkpointDir, &sb) != 0) {
            assert(!mkdir(checkpointDir, 0777));
        }
        for (auto p : parameters) {
            p->save(this->checkpointDir);
        }
    }
}
