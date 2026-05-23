# LoRA from scratch

A ~40-line LoRA layer that wraps any `nn.Linear` (or HF `Conv1D`).
Fine-tune `distilgpt2` on tiny-shakespeare with 0.49% of params trainable.

## Result

- Trainable params: **405,504 of 82,318,080 (0.49%)**
- Wrapped: 18 layers (`c_attn` and `c_proj` in all 6 blocks), `r=8`, `alpha=16`
- 800 steps, batch 8, seq_len 128, first 1 MB of tiny-shakespeare (~300k tokens)
- Training time: **115 s on Apple M-series MPS** (~7 min on CPU)
- Loss: 4.84 (step 1) -> 4.23 (50) -> 4.00 (200) -> 3.93 (800)

### Before fine-tune

    ROMEO: I love thee not.
    R: But I also love them.
    T: Aye! No, I love thee not!
    Q: What will we do with the Holy Scriptures?

### After fine-tune

    ROMEO: I love thee not.

    BUDGET:
    How did that?

    BUDGET:
    No, I was the best.

    DUKE VINCENTIO:
    I am, for

Not Shakespeare, but the model now emits the right *shape*: all-caps named
speakers, blank-line stanza breaks, an actual Shakespeare character
("DUKE VINCENTIO") after touching half a percent of the weights.

## Run

    pip install torch transformers
    python train.py

## What this shows

The LoRA update is `B @ A` with `A: (r, in)` and `B: (out, r)`. Forward is
`base(x) + (x @ A.T @ B.T) * alpha/r`. `B` is init-zero so the wrapped
layer starts equal to the base. The "1000x cheaper" magic is two skinny
matmuls and an init trick.

## References

- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv 2106.09685)
- `microsoft/LoRA` reference; HuggingFace `peft` `LoraConfig` (read once)
