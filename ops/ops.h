#pragma once

#include "matrix.h"
#include <chrono>
#include <memory>
#include <vector>

namespace Ops {
    class Op {
    public:
        Matrix out;
        Matrix grad;

        uint32_t height;
        uint32_t width;

        Op(uint32_t height, uint32_t width);
        virtual void forward(bool gpu = false) = 0;
        virtual void backward() = 0;
    };


    class Placeholder: public Op {
    public:
        Placeholder(uint32_t height, uint32_t width);

        void forward(bool gpu) override;
        void backward() override;

        void fill(float* data);
    };

    class Parameter: public Op {
        string name;

    public:
        static chrono::duration<float> initTime;
        static chrono::duration<float> stepTime;

        Parameter(uint32_t height, uint32_t width, string name);

        void forward(bool gpu) override;
        void backward() override;

        // perform a step of gradient descent using accumulated gradients
        void step(float lr);

        // If this is a vector (height 1), use zero initialization. Otherwise, use He random initialization.
        void init();

        // Save/load weights from a checkpoint directory. Uses the naming scheme "{dir}/{name}.data".
        void load(string dir);
        void save(string dir);
    };

    class Matmul: public Op {
        shared_ptr<Op> a;
        shared_ptr<Op> b;

    public:
        static chrono::duration<float> forwardTime;
        static chrono::duration<float> backwardTime;

        Matmul(shared_ptr<Op> a, shared_ptr<Op> b);

        void forward(bool gpu) override;
        void backward() override;
    };

    class Add: public Op {
        shared_ptr<Op> a;
        shared_ptr<Op> b;

    public:
        static chrono::duration<float> forwardTime;
        static chrono::duration<float> backwardTime;

        Add(shared_ptr<Op> a, shared_ptr<Op> b);

        void forward(bool gpu) override;
        void backward() override;
    };

    class Relu: public Op {
        shared_ptr<Op> a;

    public:
        static chrono::duration<float> forwardTime;
        static chrono::duration<float> backwardTime;

        Relu(shared_ptr<Op> a);

        void forward(bool gpu) override;
        void backward() override;
    };

    class Argmax: public Op {
        shared_ptr<Op> a;

    public:
        static chrono::duration<float> forwardTime;

        Argmax(shared_ptr<Op> a);

        void forward(bool gpu) override;
        void backward() override;
    };

    class CrossEntropy: public Op {
        shared_ptr<Op> a;
        shared_ptr<Op> b;

    public:
        static chrono::duration<float> forwardTime;
        static chrono::duration<float> backwardTime;

        Matrix expSums;

        CrossEntropy(shared_ptr<Op> a, shared_ptr<Op> b);

        void forward(bool gpu) override;
        void backward() override;
    };
}
