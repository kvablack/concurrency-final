#pragma once

#include <vector>
#include <memory>
#include "ops/ops.h"

using namespace std;
using namespace Ops;

float cudaInit();

struct Model {
    shared_ptr<Placeholder> data;
    shared_ptr<Placeholder> labels;
    shared_ptr<Op> out;
    shared_ptr<Op> loss;
    vector<shared_ptr<Parameter>> parameters;
    char* checkpointDir;

    Model(char* checkpointDir, uint32_t batchSize);

    float trainStep(float lr, bool useGpu = false);

    float evaluate(bool useGpu = false);

    void save();
};
