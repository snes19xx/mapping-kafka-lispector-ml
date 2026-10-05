"""Everything the piece draws, as one JS module: OUT/data/atlas.js.

Coordinates are in the atlas plane (radius 1 holds 98% of passages), rounded to 1e-4.
"""
import re
from collections import Counter

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.common import csls, emb, load, read, write_js
from src.core.config import DATA, FINAL, N_SHOWN, RAW, VIZ_DATA
from src.mapping.atlas import STOP

BOOKS = ["metamorphosis", "kstories", "amerika", "aphorisms", "milena", "gh", "lstories", "agua", "cronicas", "hour"]


def proper_nouns(rows):
    """Words that appear capitalised mid-sentence more often than not: names, places."""
    caps, low = Counter(), Counter()
    for r in rows:
        for m in re.finditer(r"(?<=[a-z,;] )([A-Za-zÀ-ÿ]+)", r["text"]):
            w = m.group(1)
            (caps if w[0].isupper() else low)[w.lower()] += 1
    return {w for w, c in caps.items() if c > low[w]}


def keywords(rows, labels, names, top=6):
    ids = sorted(set(labels))
    docs = [" ".join(rows[i]["text"] for i in np.where(labels == c)[0]) for c in ids]
    tf = TfidfVectorizer(stop_words=sorted(STOP | names), token_pattern=r"(?u)\b[a-zà-ÿ][a-zà-ÿ]{3,}\b",
                         sublinear_tf=True, min_df=3, max_df=0.5)
    M = tf.fit_transform(docs).toarray()
    vocab = np.array(tf.get_feature_names_out())
    return {c: list(vocab[np.argsort(-M[i])[:top]]) for i, c in enumerate(ids)}


def affinity(X, y):
    """Each passage's best CSLS score against the other author, which paragraph gives it, and their cosine."""
    ki, li, S, C = csls(X, y)
    a, partner, cos = np.zeros(len(y)), np.zeros(len(y), dtype=int), np.zeros(len(y))
    a[ki], partner[ki], cos[ki] = C.max(1), li[C.argmax(1)], S[np.arange(len(ki)), C.argmax(1)]
    a[li], partner[li], cos[li] = C.max(0), ki[C.argmax(0)], S[C.argmax(0), np.arange(len(li))]
    return a, partner, cos


def opening(text, words=10):
    """The first words, with a book's small-capital opening run set in sentence case."""
    w = text.split()[:words]
    caps = 0
    while caps < len(w) and w[caps].isupper() and len(w[caps].strip("“”‘’\"'.,;:!?")) > 1:
        caps += 1
    if caps:
        run = " ".join(w[:caps]).lower()
        w = (run[0].upper() + run[1:]).split() + w[caps:]
    return " ".join(w) + ("…" if len(text.split()) > words else "")


ROMAN = re.compile(r"^(?=[IVXLC]+$)", re.I)


def display_section(s):
    return s.upper() if s and ROMAN.match(s) else s


def reading_sequence(rows, X, y, book="metamorphosis", stations=16, top=4):
    """The book read in order: at evenly spaced paragraphs, its nearest Lispector paragraphs by CSLS."""
    ki, li, S, C = csls(X, y)
    pos = {int(v): j for j, v in enumerate(ki)}
    idx = sorted((i for i, r in enumerate(rows) if r["book"] == book), key=lambda i: rows[i]["index"])
    picks = [idx[round(j * (len(idx) - 1) / (stations - 1))] for j in range(stations)]
    out = []
    for i in picks:
        c = C[pos[i]]
        # the four best by CSLS, shown in order of the cosine the page draws
        best = sorted(np.argsort(-c)[:top], key=lambda b: -S[pos[i], b])
        out.append(dict(i=int(i), text=opening(rows[i]["text"]), cands=[
            dict(i=int(li[b]), csls=round(float(c[b]), 3), cos=round(float(S[pos[i], b]), 3),
                 text=opening(rows[li[b]]["text"], 9)) for b in best]))
    return out


def csls_null(X, y, n=6000, seed=0):
    """CSLS of random Kafka-Lispector pairs: the score distribution the bridges sit at the top of."""
    ki, li, _, C = csls(X, y)
    rng = np.random.default_rng(seed)
    return [round(float(v), 4) for v in C[rng.integers(0, len(ki), n), rng.integers(0, len(li), n)]]


def main():
    rows, y = load()
    A = np.load(DATA / "atlas.npz")
    labels = A["labels"]
    all_bridges = read("bridges.json")
    bridges = all_bridges[:N_SHOWN]
    kw = keywords(rows, labels, proper_nouns(rows))

    X = emb(FINAL, "erased")
    aff, partner, pcos = affinity(X, y)
    raw_aff = affinity(emb(RAW), y)[0]
    sections = sorted({r["section"] or "" for r in rows})
    lo, hi = np.quantile(np.r_[aff, raw_aff], [0.02, 0.995])
    mag = lambda a: np.clip((a - lo) / (hi - lo), 0, 1)

    P = A["final"]
    consts = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        pts = P[idx]
        # label at the densest member: most members within a small radius
        D = squareform(pdist(pts))
        dens = (D < 0.08).sum(1)
        core = idx[np.argsort(-dens)[: max(6, len(idx) // 3)]]
        stars = core[np.argsort(-aff[core])[:8]]
        T = minimum_spanning_tree(squareform(pdist(P[stars]))).tocoo()
        consts.append(dict(
            id=int(c), n=int(len(idx)), kafka=round(float(y[idx].mean()), 3),
            at=[round(float(v), 4) for v in pts[np.argmax(dens)]],
            books=Counter(rows[i]["book"] for i in idx).most_common(3),
            keywords=kw[c], figure=[[int(stars[a]), int(stars[b])] for a, b in zip(T.row, T.col)],
            bridges=int(sum(labels[b["k"]] == c or labels[b["l"]] == c for b in bridges)),
        ))

    q = lambda v: [round(float(t), 4) for t in v]
    atlas = dict(
        n=len(rows),
        author=[int(v) for v in y],
        book=[BOOKS.index(r["book"]) for r in rows],
        books=BOOKS,
        constellation=[int(v) for v in labels],
        raw=[q(p) for p in A["raw"]], theme=[q(p) for p in A["theme"]], final=[q(p) for p in A["final"]],
        mag_raw=q(mag(raw_aff)), mag_final=q(mag(aff)),
        # for the hover card: where each passage sits in its book, and its nearest by the other writer
        sections=[display_section(s) for s in sections], section=[sections.index(r["section"] or "") for r in rows],
        index=[r["index"] for r in rows], words=[r["words"] for r in rows],
        partner=[int(v) for v in partner], partner_cos=[round(float(v), 3) for v in pcos],
        bridges=[[b["k"], b["l"], round(b["csls"], 4), round(b["cos"], 4)] for b in bridges],
        sequence=reading_sequence(rows, X, y),
        csls_null=csls_null(X, y),
        bridgeCount=len(all_bridges),
        dims=int(np.load(DATA / f"emb_{FINAL}.npy", mmap_mode="r").shape[1]),
        erased_dims=int(X.shape[1]),
        constellations=consts,
        metrics=read("erasure_metrics.json"),
    )
    VIZ_DATA.mkdir(exist_ok=True)
    write_js(VIZ_DATA / "atlas.js", atlas)
    print(f"{(VIZ_DATA / 'atlas.js').stat().st_size / 1024:.0f} KB")
    for c in consts:
        print(c["id"], c["n"], c["kafka"], c["bridges"], c["books"][:2], c["keywords"])


if __name__ == "__main__":
    main()
