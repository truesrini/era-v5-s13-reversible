"""Download TinyStories, train a small byte-level BPE tokenizer and write uint16 token files.

Outputs (all under data/, ignored by git):
  data/raw/*.parquet        the downloaded TinyStories shards
  data/tokenizer.json       byte-level BPE, vocab 8192
  data/train.bin, val.bin   uint16 token ids, stories separated by <|endoftext|>
"""
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REPO = "roneneldan/TinyStories"
FILES = {
    "train": "data/train-00000-of-00004-2d5a1467fff1081b.parquet",  # ~249 MB, ~530k stories
    "val": "data/validation-00000-of-00001-869c898b519ad725.parquet",  # ~10 MB
}
VOCAB = 8192
EOT = "<|endoftext|>"


def log(*a):
    print(*a, flush=True)


def download():
    paths = {}
    for split, f in FILES.items():
        t = time.time()
        paths[split] = hf_hub_download(REPO, f, repo_type="dataset", local_dir=os.path.join(DATA, "raw"))
        log(f"[data] {split}: {paths[split]}  ({os.path.getsize(paths[split]) / 1e6:.1f} MB, {time.time() - t:.1f}s)")
    return paths


def load_texts(path):
    return [s for s in pq.read_table(path, columns=["text"]).column("text").to_pylist() if s and s.strip()]


def train_tokenizer(texts):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode(tok, texts, out_path):
    eot = tok.token_to_id(EOT)
    chunks, n = [], 0
    for i in range(0, len(texts), 20000):
        for enc in tok.encode_batch(texts[i:i + 20000]):
            chunks.append(np.asarray(enc.ids + [eot], dtype=np.uint16))
            n += len(enc.ids) + 1
    arr = np.concatenate(chunks)
    arr.tofile(out_path)
    return n


def main(force=False):
    os.makedirs(DATA, exist_ok=True)
    tr_bin, va_bin, tok_path = (os.path.join(DATA, x) for x in ("train.bin", "val.bin", "tokenizer.json"))
    if not force and all(os.path.exists(p) for p in (tr_bin, va_bin, tok_path)):
        log(f"[data] already prepared: train={os.path.getsize(tr_bin) // 2:,} tokens, "
            f"val={os.path.getsize(va_bin) // 2:,} tokens")
        return
    paths = download()
    train_texts, val_texts = load_texts(paths["train"]), load_texts(paths["val"])
    log(f"[data] stories: train={len(train_texts):,}  val={len(val_texts):,}")
    t = time.time()
    tok = train_tokenizer(train_texts[:200_000])
    tok.save(tok_path)
    log(f"[tok] byte-level BPE vocab={tok.get_vocab_size()} trained on 200k stories in {time.time() - t:.1f}s")
    t = time.time()
    n_tr = encode(tok, train_texts, tr_bin)
    n_va = encode(tok, val_texts, va_bin)
    log(f"[tok] train={n_tr:,} tokens  val={n_va:,} tokens  ({time.time() - t:.1f}s)")
    log(f"[tok] chars/token on val: {sum(map(len, val_texts)) / n_va:.2f}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
