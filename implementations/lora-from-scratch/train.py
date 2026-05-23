"""Fine-tune distilgpt2 on tiny-shakespeare with LoRA on c_attn and c_proj."""

import os
import time
import urllib.request

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from lora import LoRALinear, apply_lora, trainable_param_counts

SEED = 0
MODEL_NAME = "distilgpt2"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "/tmp/tinyshakespeare.txt"
CORPUS_BYTES = 1_000_000
LORA_TARGETS = ("c_attn", "c_proj")
LORA_R = 8
LORA_ALPHA = 16
SEQ_LEN = 128
BATCH_SIZE = 8
STEPS = 800
LR = 3e-4
LOG_EVERY = 50
PROMPT = "ROMEO: I love thee not."
GEN_TOKENS = 40


def download_corpus():
    if not os.path.exists(DATA_PATH):
        print(f"downloading tiny-shakespeare -> {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()[:CORPUS_BYTES]


def make_batches(token_ids, seq_len, batch_size, device):
    # Drop the tail so the tensor reshapes cleanly into (N, seq_len).
    n = (len(token_ids) - 1) // seq_len * seq_len
    x = torch.tensor(token_ids[:n], dtype=torch.long).view(-1, seq_len)
    y = torch.tensor(token_ids[1 : n + 1], dtype=torch.long).view(-1, seq_len)
    perm = torch.randperm(x.size(0))
    x, y = x[perm], y[perm]
    for i in range(0, x.size(0) - batch_size + 1, batch_size):
        yield x[i : i + batch_size].to(device), y[i : i + batch_size].to(device)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, device):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )
    model.train()
    return tokenizer.decode(out[0], skip_special_tokens=True)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    print(f"loading {MODEL_NAME}")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Freeze every base parameter; LoRA layers will register their own trainable A, B.
    for p in model.parameters():
        p.requires_grad = False
    n_swapped = apply_lora(model, LORA_TARGETS, r=LORA_R, alpha=LORA_ALPHA)
    model.to(device)

    trainable, total = trainable_param_counts(model)
    print(f"wrapped {n_swapped} linears; trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    text = download_corpus()
    print(f"corpus: {len(text):,} chars")
    token_ids = tokenizer.encode(text)
    print(f"tokens: {len(token_ids):,}")

    print("\n--- sample BEFORE fine-tune ---")
    torch.manual_seed(SEED)
    print(generate(model, tokenizer, PROMPT, GEN_TOKENS, device))
    print("-------------------------------\n")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    print(f"training {STEPS} steps, batch={BATCH_SIZE}, seq_len={SEQ_LEN}, lr={LR}")
    t0 = time.time()
    model.train()
    step = 0
    losses = []
    while step < STEPS:
        for xb, yb in make_batches(token_ids, SEQ_LEN, BATCH_SIZE, device):
            logits = model(xb).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % LOG_EVERY == 0 or step == 1:
                print(f"  step {step:4d}  loss {loss.item():.3f}")
            if step >= STEPS:
                break
    dt = time.time() - t0
    print(f"trained {STEPS} steps in {dt:.1f}s ({dt/60:.2f} min); final loss {losses[-1]:.3f}")

    print("\n--- sample AFTER fine-tune ---")
    torch.manual_seed(SEED)
    print(generate(model, tokenizer, PROMPT, GEN_TOKENS, device))
    print("------------------------------")


if __name__ == "__main__":
    assert SEQ_LEN <= 1024, "distilgpt2 context is 1024"
    assert STEPS > 0
    main()
