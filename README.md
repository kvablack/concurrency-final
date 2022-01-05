# Concurrency Final
Final project for Honors Concurrency (Spring 2021). I built a rudimentary GPU-accelerated machine learning library à la PyTorch or TensorFlow. It supports matrix multiplication, vector addition, ReLU, arg max, and cross-entropy loss: enough to implement a simple feedforward neural network. It also includes an automatic backpropagation engine that computes gradients.

I used it to train a small neural network on the Fashion-MNIST dataset and verified its correctness by comparing against PyTorch. I then did timing comparisons between my sequential implementation, my parallel implementation, and PyTorch. See [report.pdf](report.pdf) for more details.

The entire project is written in CUDA C++ for NVIDIA GPUs and uses no external libraries (aside from the C++ standard library).

## Usage

```
$ make
$ ./main -d DATA_PATH -l LABELS_PATH [other options]

DATA_PATH and LABELS_PATH must be in my custom data format (run export.py to generate).

Other options:
    -c CHECKPOINT_DIR: Specify a directory from which to load parameters. If the dir is empty or does not exist,
        then the parameters will be initialized randomly. If training is enabled, then parameters will be saved
        there after every epoch. Each parameter matrix is saved in a separate file called "{NAME}.data",
        where NAME is the parameter name passed to the Parameter constructor.

    -b BATCH_SIZE: Specify batch size for training or evaluation. If omitted, the batch size will be set to the size
        of the entire dataset.

    -t EPOCHS: Train for EPOCHS epochs. If omitted, will run in evaluation mode.

    -g: Enable GPU.


Source code layout:
    - ops/: All of the core machine learning library API and implementation.
    - model.h and model.cu: The "user" of the machine learning library, which uses the ops API to construct and train/evaluate a 2-layer
        neural network.
    - main.cpp: Driver.
    
    - export.py: Utilities for translating between my custom on-disk data format and NumPy arrays. Run this script
        directly to write the Fashion-MNIST dataset to disk in the data/ directory.
    - mnist.py: The PyTorch code I used to run the comparison and check correctness. Not organized.
 ```
