"""Layouts, constellations, cross-author bridges, and the baseline pairings the blind reading compares them with.

Bridges are mutual nearest neighbours under CSLS (Conneau et al., 2018), which discounts hub passages that sit
close to everything by the other author.
"""
import re
from collections import Counter

import numpy as np
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.core.common import bridges, emb, load, mutual, write
from src.core.config import (DATA, FINAL, K_CONSTELLATIONS, N_SHOWN, RAW, SEED,
                             UMAP_2D, UMAP_10D)

WORD = re.compile(r"[a-zà-ÿ]+(?:’[a-z]+)?")
STOP = set(ENGLISH_STOP_WORDS) | {"said", "just", "like", "did", "know", "don’t", "it’s", "i’m", "one", "would", "could",
                                  "also", "even", "still", "yes", "oh", "ah", "mr", "mrs", "herr", "frau", "that’s"}


def layout(X, seed=SEED):
    return umap.UMAP(random_state=seed, **UMAP_2D).fit_transform(X)


def normalise(P):
    """Centre on the median and scale so 98% of points fall inside radius 1."""
    P = P - np.median(P, 0)
    return P / np.quantile(np.linalg.norm(P, axis=1), 0.98)


def procrustes(A, B):
    """Rotate/reflect B onto A (both centred), so a morph between layouts travels least."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    return B @ (U @ Vt)


def constellations(X, seed=SEED):
    """Ward clusters of a 10-d UMAP, and the embedding they were cut from."""
    z10 = umap.UMAP(random_state=seed, **UMAP_10D).fit_transform(X)
    return AgglomerativeClustering(n_clusters=K_CONSTELLATIONS, linkage="ward").fit_predict(z10), z10


def content_words(text):
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


def keywords(rows, labels, top=8):
    ids = sorted(set(labels))
    docs = [" ".join(rows[i]["text"] for i in np.where(labels == c)[0]) for c in ids]
    tf = TfidfVectorizer(stop_words=sorted(STOP), token_pattern=r"(?u)\b[a-zà-ÿ][a-zà-ÿ]{2,}\b",
                         sublinear_tf=True, min_df=2, max_df=0.6)
    M = tf.fit_transform(docs)
    vocab = np.array(tf.get_feature_names_out())
    return {c: list(vocab[np.argsort(-M[i].toarray()[0])[:top]]) for i, c in enumerate(ids)}


def tfidf(rows):
    """Word-level vectors (unit length), the baseline for "matches made on wording"."""
    return TfidfVectorizer(stop_words=sorted(STOP), sublinear_tf=True, min_df=2).fit_transform([r["text"] for r in rows])


def tfidf_pairs(rows, y):
    """Cross-author mutual nearest neighbours by TF-IDF cosine."""
    M = tfidf(rows)
    ki, li = np.where(y == 1)[0], np.where(y == 0)[0]
    S = (M[ki] @ M[li].T).toarray()
    out = [dict(k=int(ki[a]), l=int(li[b]), cos=float(S[a, b])) for a, b in mutual(S)]
    return sorted(out, key=lambda d: -d["cos"])


def lean(labels, y):
    return sorted(float(y[labels == c].mean()) for c in np.unique(labels))


def main():
    rows, y = load()
    raw, theme, final = emb(RAW), emb(FINAL), emb(FINAL, "erased")

    L_raw = normalise(layout(raw))
    # each stage laid out on its own, then turned to face the last, so a morph travels least
    L_theme = procrustes(L_raw, normalise(layout(theme)))
    L_final = procrustes(L_theme, normalise(layout(final)))

    labels, z10 = constellations(final)
    raw_labels, _ = constellations(raw)

    main_bridges = bridges(final, y)
    for b in main_bridges:
        wa, wb = content_words(rows[b["k"]]["text"]), content_words(rows[b["l"]]["text"])
        b["shared"] = sorted(wa & wb)
        b["jaccard"] = len(wa & wb) / max(1, len(wa | wb))
    words = bridges(raw, y, cosine=True)
    arms = dict(final=main_bridges, unerased=bridges(theme, y), cosine=bridges(final, y, cosine=True),
                words=words, tfidf=tfidf_pairs(rows, y))
    for pairs in arms.values():
        for b in pairs:
            b["uid_k"], b["uid_l"] = rows[b["k"]]["uid"], rows[b["l"]]["uid"]

    kw = keywords(rows, labels)
    consts = [dict(id=c, n=int((labels == c).sum()), kafka=float(y[labels == c].mean()),
                   books=Counter(rows[i]["book"] for i in np.where(labels == c)[0]).most_common(4), keywords=kw[c])
              for c in range(K_CONSTELLATIONS)]

    # bridges inside one constellation, against the share for cross-author pairs drawn at random
    nk, nl = np.bincount(labels[y == 1], minlength=K_CONSTELLATIONS), np.bincount(labels[y == 0], minlength=K_CONSTELLATIONS)
    top = main_bridges[:N_SHOWN]
    summary = dict(
        bridges=len(main_bridges), arms={a: len(p) for a, p in arms.items()},
        same_constellation=dict(bridges=sum(labels[b["k"]] == labels[b["l"]] for b in top) / len(top),
                                random_pairs=float((nk * nl).sum() / (nk.sum() * nl.sum()))),
        lean_final=lean(labels, y), lean_raw=lean(raw_labels, y),
    )

    np.savez(DATA / "atlas.npz", raw=L_raw, theme=L_theme, final=L_final, labels=labels, raw_labels=raw_labels, z10=z10)
    write("bridges.json", main_bridges)
    write("baseline_bridges.json", words)
    write("arms.json", arms)
    write("constellations.json", consts, indent=1)
    write("atlas_summary.json", summary, indent=1)
    print(f"bridges {len(main_bridges)}; arms {summary['arms']}")
    print("same constellation", summary["same_constellation"])
    print("Kafka share per constellation, raw words:", [round(v, 2) for v in summary["lean_raw"]])
    print("Kafka share per constellation, erased:  ", [round(v, 2) for v in summary["lean_final"]])
    for c in consts:
        print(c["id"], c["n"], round(c["kafka"], 2), c["books"][:3], c["keywords"])


if __name__ == "__main__":
    main()
