"""Compare our byte-level BPE against tiktoken's gpt2 encoder.

Reports tokens/byte (compression) and tokens/sec (throughput). Decoded
round-trip is asserted for both so the comparison is apples-to-apples.
"""
import random
import time
from pathlib import Path

import tiktoken

from bpe import BPETokenizer, DATA_FILE, MERGES_FILE, VOCAB_FILE

SEED = 0
random.seed(SEED)


def timed_encode(encode_fn, text, repeats=3, clear_cache=None):
    # Best of N runs. If clear_cache is given, call it before each run so
    # the comparison measures cold-cache work, not memo lookups.
    best = float('inf')
    for _ in range(repeats):
        if clear_cache is not None:
            clear_cache()
        t0 = time.time()
        ids = encode_fn(text)
        best = min(best, time.time() - t0)
    return ids, best


def main():
    text = DATA_FILE.read_text(encoding='utf-8')
    n_bytes = len(text.encode('utf-8'))
    print(f'corpus: {DATA_FILE.name}  {n_bytes:,} bytes')

    ours = BPETokenizer.load(MERGES_FILE, VOCAB_FILE)
    gpt2 = tiktoken.get_encoding('gpt2')
    print(f'vocab sizes: ours={len(ours.vocab)}  tiktoken-gpt2={gpt2.n_vocab}')

    ours_ids, ours_t = timed_encode(ours.encode, text, clear_cache=ours._cache.clear)
    gpt2_ids, gpt2_t = timed_encode(gpt2.encode, text)

    # Sanity: both must round-trip on a 4KB slice. Our vocab is much smaller
    # than gpt2's, so we expect strictly more tokens per byte.
    sample = text[:4096]
    assert ours.decode(ours.encode(sample)) == sample
    assert gpt2.decode(gpt2.encode(sample)) == sample

    print(f'\n{"":<10}  {"tokens":>10}  {"tok/byte":>9}  {"sec":>6}  {"tok/sec":>10}')
    for name, ids, sec in [('ours', ours_ids, ours_t), ('tiktoken', gpt2_ids, gpt2_t)]:
        print(f'{name:<10}  {len(ids):>10,}  {len(ids) / n_bytes:>9.4f}  {sec:>6.2f}  {int(len(ids) / sec):>10,}')

    print(f'\nspeed ratio  tiktoken / ours = {(len(gpt2_ids) / gpt2_t) / (len(ours_ids) / ours_t):.1f}x')
    print(f'tok/byte ratio  ours / tiktoken = {(len(ours_ids) / n_bytes) / (len(gpt2_ids) / n_bytes):.2f}x')

    sample_str = 'The quick brown fox jumps over the lazy dog.'
    print(f'\nsample: {sample_str!r}')
    print(f'  ours    ({len(ours.encode(sample_str))} tok): {ours.encode(sample_str)}')
    print(f'  tiktoken({len(gpt2.encode(sample_str))} tok): {gpt2.encode(sample_str)}')


if __name__ == '__main__':
    main()
