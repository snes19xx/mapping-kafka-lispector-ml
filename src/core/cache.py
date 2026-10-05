"""The embedding cache: {passage uid: float32 vector} per reading, in data/cache/<reading>.npz."""
import os

import numpy as np

from src.core.config import CACHE


def load_cache(reading):
    path = CACHE / f"{reading}.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    return dict(zip(z["uids"].tolist(), z["vecs"]))


def save_cache(reading, have):
    """Written to a temporary file and renamed, so an interrupt doesnt leave a half-written cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"{reading}.tmp.npz"
    uids = sorted(have)
    np.savez(tmp, uids=np.array(uids), vecs=np.stack([have[u] for u in uids]))
    os.replace(tmp, CACHE / f"{reading}.npz")
