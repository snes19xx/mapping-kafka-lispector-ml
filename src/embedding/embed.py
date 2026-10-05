"""Embeddings of every passage and twin per reading, 

    python -m src.embedding.embed                    every reading in config.READINGS
    python -m src.embedding.embed mpnet qwen-theme   only these

Writes data/emb_<reading>.npy in corpus order and data/twins_<reading>.npy in twins order.
"""
import sys
import time

import numpy as np

from src.core.cache import load_cache, save_cache
from src.core.common import read
from src.core.config import DATA, MODELS, READINGS, SHARD

sys.modules["torchvision"] = None


def encoder(reading):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    m = MODELS[READINGS[reading]["model"]]
    model = SentenceTransformer(m["name"], revision=m["revision"], device="cpu")
    model.max_seq_length = m["max_len"]
    return model


def run(reading):
    rows, twins = read("passages.json"), read("twins.json")
    texts = {r["uid"]: r["text"] for r in rows + twins}
    have = load_cache(reading)
    todo = [u for u in texts if u not in have]
    print(f"{reading}: {len(texts) - len(todo)} cached, {len(todo)} to encode", flush=True)
    if todo:
        model, t = encoder(reading), time.time()
        for k in range(0, len(todo), SHARD):
            batch = todo[k:k + SHARD]
            vec = model.encode([texts[u] for u in batch], batch_size=16, prompt=READINGS[reading]["prompt"],
                               normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            have.update(zip(batch, vec.astype(np.float32)))
            save_cache(reading, have)
            print(f"{reading}: {min(k + SHARD, len(todo))}/{len(todo)} at {time.time() - t:.0f} s", flush=True)
    np.save(DATA / f"emb_{reading}.npy", np.stack([have[r["uid"]] for r in rows]))
    if twins:
        np.save(DATA / f"twins_{reading}.npy", np.stack([have[r["uid"]] for r in twins]))
    print(f"{reading}: corpus {len(rows)}, twins {len(twins)}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or READINGS:
        run(name)
