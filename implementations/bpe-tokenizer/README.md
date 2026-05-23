# BPE Tokenizer

Byte-level Byte-Pair Encoding from scratch, GPT-2 style. Trains merges
on a corpus, saves them, encode/decode any UTF-8 text. Compared against
`tiktoken`'s `gpt2` encoder.

## Why

To see what's actually inside a tokenizer: the pre-tokenizer regex, the
byte-level base alphabet (so every byte round-trips, no `<unk>`), the
greedy lowest-rank merge loop at encode time, and the on-disk format.

## Results

Trained on `data/tinyshakespeare.txt` (1,115,394 bytes) with
`VOCAB_SIZE = 2000` (256 base bytes + 1744 merges) in 3.4 s on an M2
Max. Benchmark is best-of-3 encode of the full corpus with the per-word
cache cleared between runs.

|              | vocab  | tokens   | tok/byte | tok/sec       |
|--------------|-------:|---------:|---------:|--------------:|
| ours         |  2,000 | 390,417  | 0.3500   | 1,439,129     |
| tiktoken gpt2| 50,257 | 338,025  | 0.3031   | 4,104,140     |

Pure-Python encoder is **2.9x slower** than the Rust tiktoken and needs
**15% more tokens per byte** because our vocab is 25x smaller. Throughput
is high because pre-tokens repeat heavily on Shakespeare and the
per-word memo cache turns most lookups into dict hits; on more diverse
text the gap widens.

Sample `The quick brown fox jumps over the lazy dog.`:
- ours (17 tok): `[352, 1895, 268, 725, 691, 120, 561, 586, 938, 1406, 267, 279, 1800, 121, 383, 103, 46]`
- tiktoken (10 tok): `[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]`

## Run

```bash
pip install tiktoken regex   # Python 3.11+
python bpe.py        # trains, writes merges.txt + vocab.json, asserts round-trip
python benchmark.py  # prints the table above
```

Corpus comes from `raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`.

## References

- Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units* (2015).
- Radford et al. *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019).
- `openai/gpt-2/src/encoder.py` and `openai/tiktoken` for the byte-to-unicode trick and merges format.
