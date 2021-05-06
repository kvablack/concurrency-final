"""
This file provides utilities to translate between my custom data format and NumPy arrays.
"""
import numpy as np

def save(array, path):
    assert(len(array.shape) == 2)
    assert(array.dtype == np.float32)

    b = array.tobytes()
    s = np.array(array.shape, dtype=np.uint32).tobytes()

    with open(path, 'wb') as f:
        f.write(b'\x93CS378H\0')
        f.write(s)
        f.write(b)


def load(path):
    with open(path, 'rb') as f:
        magic = f.read(8)
        assert(magic == b'\x93CS378H\0')
        height = int.from_bytes(f.read(4), byteorder="little")
        width = int.from_bytes(f.read(4), byteorder="little")
        dtype = np.dtype([('data', np.float32, [height, width])])
        array = np.fromfile(f, dtype=dtype)['data'][0]
    return array



if __name__ == "__main__":
    from torchvision.datasets import FashionMNIST

    train = FashionMNIST("data", train=True, download=True)
    test = FashionMNIST("data", train=False, download=True)

    train_data = train.data.numpy().reshape(-1, 784).astype(np.float32) / 127.5 - 1
    train_labels = train.targets.numpy()[None, :].astype(np.float32)

    test_data = test.data.numpy().reshape(-1, 784).astype(np.float32) / 127.5 - 1
    test_labels = test.targets.numpy()[None, :].astype(np.float32)

    save(train_data, "data/train_data.data")
    save(train_labels, "data/train_labels.data")

    save(test_data, "data/test_data.data")
    save(test_labels, "data/test_labels.data")
