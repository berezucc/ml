"""Single-layer LSTM language model in pure NumPy. Forward, backward, BPTT — all by hand.

Run this file directly to grad-check the implementation against numerical gradients.
"""

import numpy as np


def sigmoid(x):
    # Branchless stable sigmoid: avoids overflow for large |x|.
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class LSTM:
    """Char-level LSTM: one-hot input -> LSTM cell -> linear projection -> softmax.

    Shapes used everywhere:
        T = sequence length, N = batch size, V = vocab size, H = hidden size.
        X is (T, N, V) one-hot. Targets y is (T, N) int indices.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = hidden_dim
        D = vocab_size
        # Combined gate matrix: [W_x; W_h] @ [x; h_prev] gives all 4 gates in one matmul.
        scale = np.sqrt(1.0 / (D + H))
        self.W = (rng.standard_normal((D + H, 4 * H)).astype(np.float32) * scale)
        self.b = np.zeros(4 * H, dtype=np.float32)
        # Forget-gate bias = 1: model defaults to "remember" until it learns to forget.
        self.b[H:2 * H] = 1.0
        # Output projection hidden -> vocab.
        self.Wy = (rng.standard_normal((H, D)).astype(np.float32) * np.sqrt(1.0 / H))
        self.by = np.zeros(D, dtype=np.float32)
        self.H = H
        self.D = D
        # Adagrad accumulators.
        self.mW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.mWy = np.zeros_like(self.Wy)
        self.mby = np.zeros_like(self.by)

    def forward(self, X, y, h0=None, c0=None):
        """Forward + loss. Returns (loss, cache, h_T, c_T)."""
        T, N, D = X.shape
        H = self.H
        dtype = self.W.dtype
        if h0 is None:
            h0 = np.zeros((N, H), dtype=dtype)
        if c0 is None:
            c0 = np.zeros((N, H), dtype=dtype)

        zs = np.zeros((T, N, D + H), dtype=dtype)
        igates = np.zeros((T, N, H), dtype=dtype)
        fgates = np.zeros((T, N, H), dtype=dtype)
        ggates = np.zeros((T, N, H), dtype=dtype)
        ogates = np.zeros((T, N, H), dtype=dtype)
        cs = np.zeros((T, N, H), dtype=dtype)
        c_tanhs = np.zeros((T, N, H), dtype=dtype)
        hs = np.zeros((T, N, H), dtype=dtype)

        h_prev, c_prev = h0, c0
        for t in range(T):
            z = np.concatenate([X[t], h_prev], axis=1)
            a = z @ self.W + self.b
            i = sigmoid(a[:, :H])
            f = sigmoid(a[:, H:2 * H])
            g = np.tanh(a[:, 2 * H:3 * H])
            o = sigmoid(a[:, 3 * H:])
            c = f * c_prev + i * g
            ct = np.tanh(c)
            h = o * ct

            zs[t], igates[t], fgates[t], ggates[t], ogates[t] = z, i, f, g, o
            cs[t], c_tanhs[t], hs[t] = c, ct, h
            h_prev, c_prev = h, c

        # Logits and softmax cross-entropy over all timesteps.
        logits = hs @ self.Wy + self.by  # (T, N, D)
        logits_flat = logits.reshape(T * N, D)
        logits_flat = logits_flat - logits_flat.max(axis=1, keepdims=True)
        e = np.exp(logits_flat)
        p = e / e.sum(axis=1, keepdims=True)
        y_flat = y.reshape(T * N)
        loss = -np.log(p[np.arange(T * N), y_flat] + 1e-12).mean()

        cache = (X, zs, igates, fgates, ggates, ogates, cs, c_tanhs, hs, h0, c0, p, y_flat)
        return loss, cache, hs[-1], cs[-1]

    def backward(self, cache):
        """Returns dict of param grads + dh0, dc0."""
        X, zs, igates, fgates, ggates, ogates, cs, c_tanhs, hs, h0, c0, p, y_flat = cache
        T, N, D = X.shape
        H = self.H

        # d logits via softmax + cross-entropy.
        dlogits_flat = p.copy()
        dlogits_flat[np.arange(T * N), y_flat] -= 1.0
        dlogits_flat /= (T * N)
        dlogits = dlogits_flat.reshape(T, N, D)

        # Output projection.
        hs_flat = hs.reshape(T * N, H)
        dWy = hs_flat.T @ dlogits_flat
        dby = dlogits_flat.sum(axis=0)
        dhs_from_out = dlogits @ self.Wy.T  # (T, N, H)

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(X)
        dh_next = np.zeros((N, H), dtype=np.float32)
        dc_next = np.zeros((N, H), dtype=np.float32)

        for t in reversed(range(T)):
            i, f, g, o = igates[t], fgates[t], ggates[t], ogates[t]
            ct = c_tanhs[t]
            c_prev = cs[t - 1] if t > 0 else c0
            z = zs[t]

            dh = dhs_from_out[t] + dh_next
            do = dh * ct
            # Gradient through tanh(c) AND carry from c_{t+1}: this is the LSTM gradient highway.
            dc = dh * o * (1.0 - ct * ct) + dc_next
            di = dc * g
            df = dc * c_prev
            dg = dc * i
            dc_prev = dc * f

            da_i = di * i * (1.0 - i)
            da_f = df * f * (1.0 - f)
            da_g = dg * (1.0 - g * g)
            da_o = do * o * (1.0 - o)
            da = np.concatenate([da_i, da_f, da_g, da_o], axis=1)

            dW += z.T @ da
            db += da.sum(axis=0)
            dz = da @ self.W.T
            dX[t] = dz[:, :D]
            dh_next = dz[:, D:]
            dc_next = dc_prev

        return {"W": dW, "b": db, "Wy": dWy, "by": dby, "dh0": dh_next, "dc0": dc_next, "dX": dX}

    def step(self, grads, lr: float, clip: float = 5.0):
        # Per-tensor norm clip. Without this, BPTT blows up to NaN within ~100 steps.
        for k in ("W", "b", "Wy", "by"):
            n = np.linalg.norm(grads[k])
            if n > clip:
                grads[k] *= clip / n
        # Adagrad: per-param adaptive LR, robust on RNNs without much tuning.
        eps = 1e-8
        self.mW += grads["W"] ** 2
        self.mb += grads["b"] ** 2
        self.mWy += grads["Wy"] ** 2
        self.mby += grads["by"] ** 2
        self.W -= lr * grads["W"] / np.sqrt(self.mW + eps)
        self.b -= lr * grads["b"] / np.sqrt(self.mb + eps)
        self.Wy -= lr * grads["Wy"] / np.sqrt(self.mWy + eps)
        self.by -= lr * grads["by"] / np.sqrt(self.mby + eps)

    def sample(self, seed_ix: int, n: int, rng=None):
        """Greedy-sample n chars starting from seed_ix. Returns list of indices."""
        if rng is None:
            rng = np.random.default_rng(0)
        H = self.H
        D = self.D
        h = np.zeros((1, H), dtype=np.float32)
        c = np.zeros((1, H), dtype=np.float32)
        ix = seed_ix
        out = []
        for _ in range(n):
            x = np.zeros((1, D), dtype=np.float32)
            x[0, ix] = 1.0
            z = np.concatenate([x, h], axis=1)
            a = z @ self.W + self.b
            i = sigmoid(a[:, :H])
            f = sigmoid(a[:, H:2 * H])
            g = np.tanh(a[:, 2 * H:3 * H])
            o = sigmoid(a[:, 3 * H:])
            c = f * c + i * g
            h = o * np.tanh(c)
            logits = h @ self.Wy + self.by
            logits = logits - logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            ix = int(rng.choice(D, p=probs.ravel()))
            out.append(ix)
        return out


def _grad_check():
    """Numerical gradient check on a tiny instance (float64 for finite-diff precision)."""
    rng = np.random.default_rng(0)
    V, H, T, N = 5, 4, 3, 2
    model = LSTM(V, H, seed=0)
    # Upcast all params to float64 to avoid finite-diff precision floor at eps=1e-5.
    for name in ("W", "b", "Wy", "by"):
        setattr(model, name, getattr(model, name).astype(np.float64))
    X = np.zeros((T, N, V), dtype=np.float64)
    ix = rng.integers(0, V, size=(T, N))
    for t in range(T):
        for n in range(N):
            X[t, n, ix[t, n]] = 1.0
    y = rng.integers(0, V, size=(T, N))

    loss, cache, _, _ = model.forward(X, y)
    grads = model.backward(cache)

    eps = 1e-6
    rel_errs = {}
    for name in ("W", "b", "Wy", "by"):
        P = getattr(model, name)
        G = grads[name]
        # Sweep ALL entries for small params; spot-check 16 for large W.
        if P.size <= 80:
            idxs = [np.unravel_index(i, P.shape) for i in range(P.size)]
        else:
            idxs = [tuple(int(rng.integers(0, s)) for s in P.shape) for _ in range(16)]
        worst = 0.0
        worst_idx = None
        worst_num = 0.0
        worst_ana = 0.0
        for idx in idxs:
            orig = float(P[idx])
            P[idx] = orig + eps
            lp, _, _, _ = model.forward(X, y)
            P[idx] = orig - eps
            lm, _, _, _ = model.forward(X, y)
            P[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(G[idx])
            denom = max(abs(num) + abs(ana), 1e-8)
            rel = abs(num - ana) / denom
            if rel > worst:
                worst = rel
                worst_idx = idx
                worst_num = num
                worst_ana = ana
        rel_errs[name] = (worst, worst_idx, worst_num, worst_ana)
    return rel_errs


if __name__ == "__main__":
    print("running gradient check on LSTM(V=5, H=4, T=3, N=2)...")
    errs = _grad_check()
    for k, (rel, idx, num, ana) in errs.items():
        print(f"  d{k:>3}  worst rel err = {rel:.2e}  at {idx}  num={num:+.4e}  ana={ana:+.4e}")
    worst = max(rel for rel, *_ in errs.values())
    assert worst < 1e-3, f"gradient check failed: worst rel err = {worst:.2e}"
    print(f"OK (worst {worst:.2e} < 1e-3)")
