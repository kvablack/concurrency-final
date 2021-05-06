#include <unistd.h>
#include <stdint.h>
#include <iostream>
#include <assert.h>
#include "ops/ops.h"
#include "model.h"

using namespace std;

int main(int argc, char* argv[]) {
    // argument parsing
    int opt;

    char* checkpointDir = nullptr; // checkpoint dir
    char* dataPath = nullptr;
    char* labelsPath = nullptr;
    uint32_t train = 0;
    int32_t batchSize = -1;
    bool useGpu = false;

    while ((opt = getopt(argc, argv, "c:b:d:l:t:g")) != -1) {
        switch(opt) {
            case 'c':
                checkpointDir = optarg;
                break;
            case 'b':
                batchSize = atoi(optarg);
                break;
            case 'd':
                dataPath = optarg;
                break;
            case 'l':
                labelsPath = optarg;
                break;
            case 't':
                train = atoi(optarg);
                break;
            case 'g':
                useGpu = true;
                break;
            case '?':
                cerr << "Illegal argument\n";
                exit(1);
                break;
        }
    }
    assert(dataPath && labelsPath);

    Matrix data = Matrix::load(dataPath);
    Matrix labels = Matrix::load(labelsPath);

    if (batchSize == -1) {
        batchSize = data.height;
    }

    if (useGpu) {
        printf("CUDA init time: %f\n", cudaInit());
    }

    uint32_t numBatches = data.height / batchSize;
    Model model(checkpointDir, batchSize);
    auto start = chrono::steady_clock::now();
    if (!train) {
        float accuracy = 0;
        for (uint32_t i = 0; i < numBatches; i++) {
            model.data->fill(data.get() + i * data.width);
            model.labels->fill(labels.get() + i);
            accuracy += model.evaluate(useGpu);
        }
        accuracy /= numBatches;
        printf("Accuracy: %f\n", accuracy);
    } else {
        for (uint32_t i = 0; i < train; i++) {
            float loss = 0;
            for (uint32_t j = 0; j < numBatches; j++) {
                model.data->fill(&data.get()[j * data.width * batchSize]);
                model.labels->fill(&labels.get()[j * batchSize]);
                loss += model.trainStep(0.01, useGpu);
            }
            printf("Epoch: %d, loss: %f\n", i, loss / numBatches);
            model.save();
        }
    }
    chrono::duration<double> duration = chrono::steady_clock::now() - start;

    printf("Primary execution time: %f\n\n", duration.count());

    printf("Allocation time: %f\n", Matrix::allocTime.count());
    printf("Copy time: %f\n", Matrix::copyTime.count());
    printf("Save time: %f\n", Matrix::saveTime.count());
    printf("Load time: %f\n\n", Matrix::loadTime.count());
    printf("Matmul forward time: %f\n", Matmul::forwardTime.count());
    printf("Add forward time: %f\n", Add::forwardTime.count());
    printf("Relu forward time: %f\n", Relu::forwardTime.count());
    printf("Argmax forward time: %f\n", Argmax::forwardTime.count());
    printf("CrossEntropy forward time: %f\n\n", CrossEntropy::forwardTime.count());
    printf("Matmul backward time: %f\n", Matmul::backwardTime.count());
    printf("Add backward time: %f\n", Add::backwardTime.count());
    printf("Relu backward time: %f\n", Relu::backwardTime.count());
    printf("CrossEntropy backward time: %f\n\n", CrossEntropy::backwardTime.count());
    printf("Parameter init time: %f\n", Parameter::initTime.count());
    printf("Parameter step time: %f\n", Parameter::stepTime.count());
}
