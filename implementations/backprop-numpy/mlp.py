"""2-layer MLP for MNIST. Forward, backward, SGD+momentum, all by hand."""

import time

import numpy as np

from data import load_mnist

SEED = 0
HIDDEN = 256
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He init: keeps activations from collapsing under ReLU
        self.W1 = rng.standard_normal((in_dim, hidden)).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden, out_dim)).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        # cached activations for the backward pass
        self._cache = None

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        z2 = h @ self.W2 + self.b2
        p = softmax(z2)
        self._cache = (X, z1, h, p)
        return p

    def backward(self, y):
        X, z1, h, p = self._cache
        n = X.shape[0]
        # softmax + cross-entropy collapses to (p - onehot(y)) / N
        dz2 = p.copy()
        dz2[np.arange(n), y] -= 1.0
        dz2 /= n
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ self.W2.T
        dz1 = dh * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2

    def step(self, grads, lr: float, momentum: float):
        dW1, db1, dW2, db2 = grads
        self.vW1 = momentum * self.vW1 - lr * dW1
        self.vb1 = momentum * self.vb1 - lr * db1
        self.vW2 = momentum * self.vW2 - lr * dW2
        self.vb2 = momentum * self.vb2 - lr * db2
        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def cross_entropy(p, y):
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()


def accuracy(model, X, y, batch=1000):
    correct = 0
    for i in range(0, len(X), batch):
        correct += (model.predict(X[i : i + batch]) == y[i : i + batch]).sum()
    return correct / len(X)


def train(model, X, y, X_test, y_test, epochs: int, lr: float):
    rng = np.random.default_rng(SEED)
    n = len(X)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        loss_sum = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            p = model.forward(xb)
            loss_sum += cross_entropy(p, yb)
            n_batches += 1
            grads = model.backward(yb)
            model.step(grads, lr, MOMENTUM)
        train_loss = loss_sum / n_batches
        test_acc = accuracy(model, X_test, y_test)
        print(f"epoch {epoch + 1:2d}/{epochs}  loss {train_loss:.4f}  test_acc {test_acc:.4f}")
    return test_acc


if __name__ == "__main__":
    np.random.seed(SEED)
    print(f"loading MNIST (seed={SEED}, hidden={HIDDEN}, epochs={EPOCHS}, lr={LR}, momentum={MOMENTUM})")
    X_train, y_train, X_test, y_test = load_mnist()

    model = MLP(in_dim=784, hidden=HIDDEN, out_dim=10, seed=SEED)

    t0 = time.time()
    final_acc = train(model, X_train, y_train, X_test, y_test, EPOCHS, LR)
    elapsed = time.time() - t0
    print(f"\nfinal test accuracy: {final_acc:.4f}  ({elapsed:.1f}s)")

    assert final_acc > 0.97, f"expected >97% test accuracy, got {final_acc:.4f}"
    assert elapsed < 120, f"training took {elapsed:.1f}s, expected under 2 minutes"
