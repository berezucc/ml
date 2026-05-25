"""Train the LSTM on char-level tinyshakespeare. Truncated BPTT, Adagrad."""

import os
import time
import urllib.request

import numpy as np

from lstm import LSTM

SEED = 0
HIDDEN = 128
SEQ_LEN = 25
BATCH_SIZE = 32
LR = 0.1
ITERS = 5000
SAMPLE_EVERY = 500
SAMPLE_LEN = 300

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
LOCAL_DATA = "../bpe-tokenizer/data/tinyshakespeare.txt"
CACHE_DATA = "tinyshakespeare.txt"


def load_text() -> str:
    if os.path.exists(LOCAL_DATA):
        with open(LOCAL_DATA) as f:
            return f.read()
    if not os.path.exists(CACHE_DATA):
        print(f"downloading tinyshakespeare to {CACHE_DATA}...")
        urllib.request.urlretrieve(DATA_URL, CACHE_DATA)
    with open(CACHE_DATA) as f:
        return f.read()


def sample_batch(data: np.ndarray, vocab_size: int, batch_size: int, seq_len: int, rng):
    """Returns one-hot X (T, N, V) and target y (T, N) where y[t] is the next char."""
    N = batch_size
    T = seq_len
    starts = rng.integers(0, len(data) - T - 1, size=N)
    X = np.zeros((T, N, vocab_size), dtype=np.float32)
    y = np.zeros((T, N), dtype=np.int64)
    for n, s in enumerate(starts):
        for t in range(T):
            X[t, n, data[s + t]] = 1.0
            y[t, n] = data[s + t + 1]
    return X, y


def main():
    text = load_text()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    print(f"corpus: {len(data):,} chars, vocab {V}")

    model = LSTM(V, HIDDEN, seed=SEED)
    rng = np.random.default_rng(SEED)

    # Smoothed loss for printing: random model has loss ~ log(V).
    smooth = float(np.log(V))
    t0 = time.time()
    history = []
    for it in range(1, ITERS + 1):
        X, y = sample_batch(data, V, BATCH_SIZE, SEQ_LEN, rng)
        loss, cache, _, _ = model.forward(X, y)
        grads = model.backward(cache)
        model.step(grads, LR)
        smooth = 0.999 * smooth + 0.001 * float(loss)
        if it % 100 == 0:
            history.append((it, smooth))
        if it % SAMPLE_EVERY == 0:
            seed_ix = stoi["\n"] if "\n" in stoi else 0
            ixs = model.sample(seed_ix, SAMPLE_LEN, rng=rng)
            text_out = "".join(itos[i] for i in ixs)
            elapsed = time.time() - t0
            print(f"\n[iter {it:5d}  loss {smooth:.4f}  {elapsed:5.1f}s]")
            print(text_out.replace("\n", "\\n")[:200])

    elapsed = time.time() - t0
    print(f"\nfinal smoothed loss: {smooth:.4f}  ({elapsed:.1f}s, {ITERS} iters)")
    return smooth, elapsed, history


if __name__ == "__main__":
    final_loss, elapsed, _ = main()
    assert final_loss < 2.0, f"expected smoothed loss < 2.0, got {final_loss:.4f}"
