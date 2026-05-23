"""Byte-level BPE tokenizer in the style of GPT-2.

Trains merges over UTF-8 byte sequences, pre-tokenized with the GPT-2
regex so merges never cross word boundaries. Round-trips arbitrary
bytes since every input byte is in the base vocab.
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import regex as re

SEED = 0
random.seed(SEED)

# GPT-2's pre-tokenizer. Splits off contractions, runs of letters, runs of
# digits, runs of punctuation, runs of whitespace (with the trailing one
# absorbed into the next chunk so words keep their leading space).
GPT2_SPLIT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

DATA_FILE = Path('data/tinyshakespeare.txt')
MERGES_FILE = Path('merges.txt')
VOCAB_FILE = Path('vocab.json')
VOCAB_SIZE = 2000  # 256 base bytes + ~1744 merges; small enough to train in seconds


def bytes_to_unicode():
    """GPT-2's reversible map from bytes to printable unicode chars.

    Avoids putting control chars / whitespace into the merges file. Every
    byte 0..255 gets a unique printable codepoint.
    """
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # pair -> new token id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._cache: dict[str, list[int]] = {}

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256
        # Pre-tokenize, then count how often each pre-token (as a tuple of
        # byte ids) occurs. We merge inside pre-tokens only.
        word_freq: Counter[tuple[int, ...]] = Counter()
        for chunk in GPT2_SPLIT.findall(text):
            word_freq[tuple(chunk.encode('utf-8'))] += 1

        words = [list(w) for w in word_freq]
        freqs = [word_freq[w] for w in word_freq]
        n_merges = vocab_size - 256

        # Initial pair counts.
        pair_counts: Counter[tuple[int, int]] = Counter()
        for w, f in zip(words, freqs):
            for a, b in get_pairs(w):
                pair_counts[(a, b)] += f

        for i in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # Apply the merge to every word that contains it and incrementally
            # update pair_counts. Full recount each step is O(N) per merge and
            # turns training into minutes for VOCAB_SIZE=2000; incremental is
            # the standard trick.
            a, b = best
            new_words = []
            for w, f in zip(words, freqs):
                if a not in w or b not in w:
                    new_words.append(w)
                    continue
                merged = []
                j = 0
                while j < len(w):
                    if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                        # Remove the old neighbouring pairs, add the new ones.
                        if merged:
                            pair_counts[(merged[-1], a)] -= f
                            pair_counts[(merged[-1], new_id)] += f
                        if j + 2 < len(w):
                            pair_counts[(b, w[j + 2])] -= f
                            pair_counts[(new_id, w[j + 2])] += f
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(w[j])
                        j += 1
                new_words.append(merged)
            words = new_words
            del pair_counts[best]
            # Drop any non-positive counts so max() stays correct.
            for k in [k for k, v in pair_counts.items() if v <= 0]:
                del pair_counts[k]

            if (i + 1) % 200 == 0:
                print(f'  merge {i + 1}/{n_merges}  vocab={len(self.vocab)}  best={best}->{new_id}')

        self._cache.clear()

    def _bpe(self, token_bytes: bytes) -> list[int]:
        key = token_bytes.decode('latin-1')
        if key in self._cache:
            return self._cache[key]
        word = list(token_bytes)
        while True:
            pairs = get_pairs(word)
            if not pairs:
                break
            # Lowest merge rank wins. Unknown pairs get inf -> never picked.
            best = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if best not in self.merges:
                break
            a, b = best
            new_id = self.merges[best]
            merged = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                    merged.append(new_id)
                    j += 2
                else:
                    merged.append(word[j])
                    j += 1
            word = merged
        self._cache[key] = word
        return word

    def encode(self, text: str) -> list[int]:
        out = []
        for chunk in GPT2_SPLIT.findall(text):
            out.extend(self._bpe(chunk.encode('utf-8')))
        return out

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

    def save(self, merges_path: Path, vocab_path: Path) -> None:
        # GPT-2 merges file: one "tokA tokB" per line, in merge order, using
        # the byte->unicode mapping so the file is printable.
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write('#version: 0.2\n')
            for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1]):
                ta = ''.join(self.byte_encoder[c] for c in self.vocab[a])
                tb = ''.join(self.byte_encoder[c] for c in self.vocab[b])
                f.write(f'{ta} {tb}\n')
        vocab_strs = {''.join(self.byte_encoder[c] for c in v): k for k, v in self.vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_strs, f, ensure_ascii=False)

    @classmethod
    def load(cls, merges_path: Path, vocab_path: Path) -> 'BPETokenizer':
        tok = cls()
        with open(vocab_path, encoding='utf-8') as f:
            vocab_strs = json.load(f)
        # Invert byte_encoder to turn the printable token strings back into bytes.
        for s, idx in vocab_strs.items():
            tok.vocab[idx] = bytes(tok.byte_decoder[ch] for ch in s)
        str_to_id = vocab_strs
        with open(merges_path, encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f if ln and not ln.startswith('#')]
        for rank, line in enumerate(lines):
            a_str, b_str = line.split(' ')
            a_id = str_to_id[a_str]
            b_id = str_to_id[b_str]
            tok.merges[(a_id, b_id)] = 256 + rank
        return tok


if __name__ == '__main__':
    text = DATA_FILE.read_text(encoding='utf-8')
    print(f'corpus: {len(text):,} chars, {len(text.encode("utf-8")):,} bytes')

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(text, VOCAB_SIZE)
    print(f'trained vocab={len(tok.vocab)} in {time.time() - t0:.1f}s')
    tok.save(MERGES_FILE, VOCAB_FILE)
    print(f'saved {MERGES_FILE} and {VOCAB_FILE}')

    sample = "The quick brown fox jumps over the lazy dog.\n  Hello, world! 123 — café."
    ids = tok.encode(sample)
    round_trip = tok.decode(ids)
    print(f'sample: {sample!r}')
    print(f'ids ({len(ids)}): {ids[:24]}{"..." if len(ids) > 24 else ""}')
    print(f'decoded: {round_trip!r}')
    assert round_trip == sample, 'round-trip failed'

    reloaded = BPETokenizer.load(MERGES_FILE, VOCAB_FILE)
    assert reloaded.encode(sample) == ids, 'reloaded tokenizer differs'
    print('round-trip OK; reload OK')
