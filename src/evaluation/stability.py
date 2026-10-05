"""Whether constellations (ARI across UMAP seeds) and bridges (support across method variants) survive, to data/stability.json.

A shown bridge's support is the number of variants that find the same pair.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.core.common import bridges, emb, load, write
from src.core.config import (CORAL_DIMS_VARIANTS, CSLS_K_VARIANTS, DATA, FINAL, N_SHOWN,
                             READINGS, UMAP_SEEDS)
from src.erasure.erase import fit_eraser
from src.mapping.atlas import constellations


def variants(rows, y):
    theme, final = emb(FINAL), emb(FINAL, "erased")
    books = sorted({r["book"] for r in rows})
    onehot = np.eye(len(books))[[books.index(r["book"]) for r in rows]]
    out = {f"coral{d}": lambda d=d: bridges(fit_eraser(theme, y, dims=d)(theme, y), y) for d in CORAL_DIMS_VARIANTS}
    out.update({f"csls_k{k}": lambda k=k: bridges(final, y, k) for k in CSLS_K_VARIANTS})
    out["unerased"] = lambda: bridges(theme, y)
    # LEACE on the ten books (translator, genre, period), which contains the author contrast; CORAL by author as usual
    out["book_leace"] = lambda: bridges(fit_eraser(theme, y, concept=onehot)(theme, y), y)
    for r in READINGS:
        if r != FINAL and (DATA / f"emb_{r}.npy").exists():
            out[f"reading_{r}"] = lambda r=r: bridges(fit_eraser(emb(r), y)(emb(r), y), y)
    return out


def main():
    rows, y = load()
    labels = np.load(DATA / "atlas.npz")["labels"]
    final = emb(FINAL, "erased")
    ari = {seed: float(adjusted_rand_score(labels, constellations(final, seed)[0])) for seed in UMAP_SEEDS[1:]}
    print("constellation ARI by UMAP seed", {s: round(v, 3) for s, v in ari.items()}, flush=True)

    shown = bridges(final, y)[:N_SHOWN]
    key = lambda b: (rows[b["k"]]["uid"], rows[b["l"]]["uid"])
    top = {key(b) for b in shown}
    found, overlap = {}, {}
    for name, make in variants(rows, y).items():
        bs = make()
        found[name] = {key(b) for b in bs}
        overlap[name] = dict(bridges=len(bs), top_overlap=len(top & {key(b) for b in bs[:N_SHOWN]}),
                             shown_found=len(top & found[name]))
        print(f"{name:22s} {overlap[name]}", flush=True)
    support = [dict(rank=i, k=b["k"], l=b["l"], uid_k=key(b)[0], uid_l=key(b)[1], csls=b["csls"],
                    support=sum(key(b) in f for f in found.values()), of=len(found)) for i, b in enumerate(shown)]
    stable = sorted(support, key=lambda s: (-s["support"], s["rank"]))
    print("shown bridges found by every variant:", sum(s["support"] == s["of"] for s in support), flush=True)
    write("stability.json", dict(constellation_ari=ari, variants=overlap, bridges=stable), indent=1)


if __name__ == "__main__":
    main()
