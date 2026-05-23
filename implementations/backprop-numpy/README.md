# Backprop in pure NumPy

A 2-layer MLP for MNIST, gradients derived by hand. No autograd.

## Result

98.33% test accuracy after 20 epochs (12.4 seconds on a single Apple M-series
CPU core). Peak test accuracy of 98.40% at epoch 15.

Configuration: 784 -> 256 ReLU -> 10 softmax, SGD with momentum 0.9, learning
rate 0.1, batch size 128.

## Run

    pip install numpy torchvision
    python mlp.py

`data.py` downloads MNIST to `./mnist-data/` on first run (~11 MB).

## The gradient math

Forward (per batch of size N):

    z1 = X W1 + b1,   h = relu(z1)
    z2 = h W2 + b2,   p = softmax(z2)
    L  = -mean(log p[y])

Backward (softmax + cross-entropy collapses to a one-liner):

    dz2 = (p - onehot(y)) / N
    dW2 = h.T @ dz2,   db2 = dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (z1 > 0)
    dW1 = X.T @ dz1,   db1 = dz1.sum(0)

Update with momentum: `v = mu*v - lr*g`, `W += v`.

## What this shows

Cross-entropy gradient through softmax, hidden-layer chain rule, weight updates
with momentum, all written out in ~150 lines. The point is that nothing magic
happens inside `loss.backward()`.

## References

- Goodfellow, Bengio, Courville. *Deep Learning*, chapter 6
- Karpathy, *Neural Networks: Zero to Hero*, lecture 1
- Nielsen, *Neural Networks and Deep Learning*, chapter 2
