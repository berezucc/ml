"""MNIST loader: torchvision for the bytes, numpy for everything else."""

import numpy as np
from torchvision.datasets import MNIST

DATA_ROOT = "./mnist-data"


def load_mnist():
    """Return (X_train, y_train, X_test, y_test) as float32/int64 numpy arrays.

    Pixels are flattened to 784 and scaled to [0, 1]. Labels are int64 in [0, 9].
    """
    train = MNIST(DATA_ROOT, train=True, download=True)
    test = MNIST(DATA_ROOT, train=False, download=True)

    X_train = train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"train: {X_tr.shape} {y_tr.shape}  test: {X_te.shape} {y_te.shape}")
    print(f"pixel range: [{X_tr.min():.2f}, {X_tr.max():.2f}]  labels: {sorted(set(y_tr.tolist()))}")
