# implementations

Small from-scratch implementations of ML algorithms and primitives. One
to three files per project, real numbers in the README, no abstractions
past what the algorithm requires.

| Project | Result |
|---|---|
| [backprop-numpy](backprop-numpy/) | 2-layer MLP for MNIST, gradients by hand. **98.33%** test accuracy in **12.4 s** on a single M-series CPU core. |
| [bpe-tokenizer](bpe-tokenizer/) | Byte-level BPE, GPT-2 style. **0.3500** tok/byte vs tiktoken **0.3031** (vocab 2k vs 50k); pure-Python encoder **2.9x** slower than the Rust tiktoken. |
| [lora-from-scratch](lora-from-scratch/) | LoRA wrapping `nn.Linear` / HF `Conv1D`, fine-tuning `distilgpt2` on tiny-shakespeare. **0.49%** trainable params (405k of 82M), **115 s** on M2 Max MPS. |
| [lstm-from-scratch](lstm-from-scratch/) | Char-level LSTM language model in pure NumPy, BPTT by hand. Loss **4.17 → 1.76** on tiny-shakespeare in **417 s** (5000 iters) on a single M2 Max CPU core. Includes a finite-difference gradient check (worst rel err **4.1e-7**). |

See [`HANDOFF.md`](HANDOFF.md) for the build conventions (one to three
files per project, hardcoded constants, asserts in `__main__`, concrete
numbers in the README, two commits maximum per project).
