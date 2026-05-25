# LSTM from scratch

A single-layer character-level LSTM language model in pure NumPy. Forward,
backward, BPTT, Adagrad — all by hand. No autograd, no `nn.LSTM`.

## Result

Smoothed cross-entropy loss drops from `log(65) ≈ 4.17` (random) to **1.76**
in **417 s** (7 min) on a single Apple M2 Max CPU core. Configuration:
`hidden=128`, `seq_len=25`, `batch=32`, Adagrad `lr=0.1`, grad-clip `5.0`.

| iter | smoothed loss |
|---:|---:|
| 500  | 3.47 |
| 1000 | 2.90 |
| 2000 | 2.26 |
| 3000 | 1.97 |
| 4000 | 1.83 |
| 5000 | **1.76** |

Sample after 5000 iters (greedy multinomial sampling, no temperature):

```
KING RYCHARD II:
We the eep tingle speaks
For'up!
O those viclack that this heart' thaigble Is eye with,
Of that from a sancoun, than here comply of 'tis fist? and grewestood with
hingst will thi
```

Not Shakespeare, but the model has learned line breaks, ALL-CAPS character
names with a trailing colon, the rough rhythm of dialogue, apostrophes
inside words, and a small Shakespeare-tinted vocabulary — entirely from
character-level statistics, with zero word knowledge.

## The cell

An LSTM cell takes `(x_t, h_{t-1}, c_{t-1})` and produces `(h_t, c_t)`. The
trick that makes it learn long dependencies is the **cell-state highway** `c`:
gradients flow through `c` with only an elementwise multiply by the forget
gate, so they don't vanish the way they do in a vanilla RNN.

```
   x_t       h_{t-1}                                    c_{t-1}
     \         /                                            |
      \       /                                             |
       concat                                               |
         |                                                  |
   z = [x_t ; h_{t-1}]   ── (D+H,)                          |
         |                                                  |
   a = z·W + b           ── (4H,)   one matmul, four gates  |
         |                                                  |
   split into 4 chunks: a_i | a_f | a_g | a_o               |
         |     |      |      |                              |
         σ     σ    tanh     σ                              |
         |     |      |      |                              |
         i     f      g      o                              |
               |      |                                     |
               └──────┼──────────×─────────────────────►── + ── c_t  ← cell-state highway
                      └──×                                       |
                                                               tanh
                                                                 |
                                                                 × ◄── o
                                                                 |
                                                                h_t
```

**One matmul, four gates.** Concatenating `x_t` and `h_{t-1}` and using a
single `(D+H) × 4H` weight matrix is faster *and* far less error-prone than
maintaining eight separate matrices.

## Forward (per timestep)

    z   = [x_t ; h_{t-1}]                      shape (N, D+H)
    a   = z · W + b                            shape (N, 4H)
    i   = σ(a[:, 0 :  H])                      input gate     (write strength)
    f   = σ(a[:, H : 2H])                      forget gate    (keep strength)
    g   = tanh(a[:, 2H: 3H])                   candidate      (what to write)
    o   = σ(a[:, 3H: 4H])                      output gate    (read strength)
    c_t = f * c_{t-1} + i * g                  cell state
    h_t = o * tanh(c_t)                        hidden state

Then a linear head: `logits = h · W_y + b_y`, softmax + cross-entropy loss.

## Backward (per timestep, reversed)

The whole point of BPTT is that each timestep receives gradients from **two**
sources: the loss at this step, and gradients flowing back from step `t+1`.

    dh_t = dh_from_output_t  +  dh_from_next_step
    do   = dh_t * tanh(c_t)
    dc_t = dh_t * o * (1 - tanh(c_t)^2)  +  dc_from_next_step    ← the highway
    di   = dc_t * g
    df   = dc_t * c_{t-1}
    dg   = dc_t * i
    dc_{t-1} (carry) = dc_t * f                                  ← the highway

Backprop through gate activations:

    da_i = di * i * (1 - i)
    da_f = df * f * (1 - f)
    da_g = dg * (1 - g^2)
    da_o = do * o * (1 - o)
    da   = concat([da_i, da_f, da_g, da_o])

Accumulate parameter grads, then pull `dh_{t-1}` for the next step backward:

    dW += z.T · da
    db += da.sum(axis=0)
    dz  = da · W.T
    dx_t      = dz[:, :D]
    dh_{t-1}  = dz[:, D:]   ← carries back to the next iteration of the loop

## Run

    pip install numpy
    python lstm.py     # numerical gradient check, must pass before training
    python train.py    # ~5-6 min on a single Apple M-series CPU core

`train.py` reads tinyshakespeare from `../../implementations/bpe-tokenizer/data/`
if it exists, otherwise downloads it on first run.

## Pitfalls (learned the hard way while writing this)

- **Initialize forget-gate bias to 1.0.** Otherwise the model takes a very
  long time to learn to remember anything (Jozefowicz et al. 2015).
- **Clip gradients by norm.** Without it, BPTT blows up to NaN within ~100
  steps. Threshold 5.0 is fine.
- **Truncate BPTT** to ~25 timesteps. Full-sequence BPTT is slow and
  unstable; truncation is the standard cheat.
- **The cell-state highway is two terms.** `dc_t` gets contributions from
  *both* `tanh(c_t)` (the current output path) *and* `dc_{t+1} * f_{t+1}`
  (the carry from the next step). Miss the second term and you've built a
  vanilla RNN with extra steps.
- **Float32 cache with float64 params silently downcasts everything.** Make
  intermediate buffers inherit `self.W.dtype`. The gradient check will catch
  this — that's what `python lstm.py` is for.
- **Concatenate inputs once, use one big weight matrix.** Eight separate
  matrices is eight times more places to get a shape wrong.

## Gradient check

`python lstm.py` runs central finite differences against the analytic
gradients on a tiny `(V=5, H=4, T=3, N=2)` instance with float64 params.
Asserts worst relative error `< 1e-3`. With the dtype fix, actual worst is
~`4e-7` (machine precision for float64 with `eps=1e-6`).

If you're modifying the backward pass, run this *first*. A 1e-1 rel error
means a real bug; a 1e-4 rel error usually means dtype precision noise.

## References

- Hochreiter & Schmidhuber, [*Long Short-Term Memory*](https://www.bioinf.jku.at/publications/older/2604.pdf) (1997) — the original paper.
- Olah, [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (2015) — the canonical visual explanation.
- Karpathy, [*The Unreasonable Effectiveness of Recurrent Neural Networks*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (2015) and his [min-char-rnn gist](https://gist.github.com/karpathy/d4dee566867f8291f086) — closest reference implementation in spirit (vanilla RNN; this repo upgrades it to LSTM).
- Jozefowicz, Zaremba, Sutskever, [*An Empirical Exploration of Recurrent Network Architectures*](https://proceedings.mlr.press/v37/jozefowicz15.pdf) (2015) — the forget-bias=1 trick lives here.
- Goodfellow, Bengio, Courville, *Deep Learning*, chapter 10.
