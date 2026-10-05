"""Benchmarks with intervals for the article, to data/benchmarks.json and OUT/data/bench.js.

    python -m src.evaluation.benchmarks          everything
    python -m src.evaluation.benchmarks export   rewrite bench.js from benchmarks.json
"""
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from src.core.common import (bridges, emb, expected_cross, load, neighbours, next_ranks, read, successors,
                             unit, write, write_js)
from src.core.config import (BOOT, BOOT_SEED, DATA, DUPLICATE_COS, FINAL, KS, N_SHOWN, NEIGHBOURS, PERMUTATIONS, RAW,
                             READINGS, SHUFFLES, VIZ_DATA)
from src.erasure.erase import fit_eraser, stages

RNG = np.random.default_rng(BOOT_SEED)
GRID = np.linspace(0, 1, 101)


def boot(stat, n, b=BOOT):
    """Percentile 95% interval of stat(idx) over b resamples of n items."""
    v = np.array([stat(RNG.integers(0, n, n)) for _ in range(b)])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def auc_ci(y, s):
    return boot(lambda i: roc_auc_score(y[i], s[i]) if 0 < y[i].sum() < len(i) else np.nan, len(y))


def oof_scores(X, y, stage, model):
    """Out-of-fold author scores: eraser on four folds, the probe cross-fitted inside the fifth."""
    s = np.zeros(len(y))
    outer = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in outer.split(X, y):
        Xe = unit(fit_eraser(X, y, tr, stage)(X, y)) if stage != "raw" else X
        inner = StratifiedKFold(5, shuffle=True, random_state=1)
        for itr, ite in inner.split(te, y[te]):
            m = model().fit(Xe[te[itr]], y[te[itr]])
            s[te[ite]] = m.predict_proba(Xe[te[ite]])[:, 1]
    return s


MODELS = {
    "linear": lambda: LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
    "knn": lambda: KNeighborsClassifier(NEIGHBOURS, metric="cosine", weights="distance"),
}


def leakage(X, y):
    out = {}
    for stage in ("raw", "leace", "moments"):
        for name, model in MODELS.items():
            s = oof_scores(X, y, stage, model)
            fpr, tpr, _ = roc_curve(y, s)
            auc = float(roc_auc_score(y, s))
            out[f"{stage}/{name}"] = dict(auc=auc, ci=auc_ci(y, s), roc=np.interp(GRID, fpr, tpr).round(4).tolist())
            print(" ", stage, name, round(auc, 3), [round(v, 3) for v in out[f"{stage}/{name}"]["ci"]], flush=True)
    return out


def retrieval(ranks, candidates):
    """Recall at k, MRR and median rank of the next paragraph; chance is k over the candidates per query."""
    rr = 1 / (ranks + 1)
    return dict(
        recall={k: float((ranks < k).mean()) for k in KS},
        chance={k: float(np.mean(k / candidates)) for k in KS},
        r10_ci=boot(lambda i: (ranks[i] < 10).mean(), len(ranks)),
        mrr=float(rr.mean()), mrr_ci=boot(lambda i: rr[i].mean(), len(ranks)),
        median_rank=float(np.median(ranks) + 1),
    )


def mixing_stats(X, y):
    nb = neighbours(X)
    cross = (y[nb] != y[:, None]).mean(1)
    expect = expected_cross(y)
    ci = boot(lambda i: cross[i].mean() / expect[i].mean(), len(y))
    # permutation null: shuffled authors on the same neighbour graph
    null = []
    for _ in range(PERMUTATIONS):
        yp = RNG.permutation(y)
        null.append((yp[nb] != yp[:, None]).mean(1).mean() / expect.mean())
    return dict(value=float(cross.mean() / expect.mean()), ci=ci, other_share=float(cross.mean()),
                chance_share=float(expect.mean()), null=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))])


def book_signal(X, y, rows):
    """How well a linear probe tells a passage's book within each author: book = translator, genre and period."""
    book = np.array([r["book"] for r in rows])
    out = {}
    for a, name in ((1, "Kafka"), (0, "Lispector")):
        idx = np.where(y == a)[0]
        acc = cross_val_score(LogisticRegression(max_iter=3000, C=1.0), unit(X[idx]), book[idx],
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy").mean()
        out[name] = dict(accuracy=float(acc), majority=max(Counter(book[idx]).values()) / len(idx))
    return out


def reading(name, rows, y):
    X = emb(name)
    print("==", name, flush=True)
    res = dict(leakage=leakage(X, y), retrieval={}, retrieval_book={}, mixing={}, book={})
    book = np.array([r["book"] for r in rows])
    succ = successors(rows)
    same_book = np.array([(book == book[i]).sum() - 1 for i in succ])
    for stage, Xs in stages(X, y).items():
        res["retrieval"][stage] = retrieval(next_ranks(Xs, rows), np.full(len(succ), len(y) - 1))
        res["retrieval_book"][stage] = retrieval(next_ranks(Xs, rows, within_book=True), same_book)
        res["mixing"][stage] = mixing_stats(Xs, y)
        if stage in ("raw", "moments"):
            res["book"][stage] = book_signal(Xs, y, rows)
        print(" ", stage, "R@10", round(res["retrieval"][stage]["recall"][10], 3),
              "within-book R@10", round(res["retrieval_book"][stage]["recall"][10], 3),
              "mix", round(res["mixing"][stage]["value"], 3), flush=True)
    return res


def shuffle_null(X, y):
    """Bridges found when the author labels are shuffled on the same erased geometry: what CSLS finds by construction."""
    real = bridges(X, y)
    runs = []
    for _ in range(SHUFFLES):
        b = bridges(X, RNG.permutation(y))
        runs.append(dict(n=len(b), weakest_shown=b[min(N_SHOWN, len(b)) - 1]["csls"]))
    return dict(real=dict(n=len(real), weakest_shown=real[N_SHOWN - 1]["csls"]), shuffled=runs)


def duplicates(rows, y):
    """Same-author passages in different books that are near-copies: one text counted twice."""
    X = emb(RAW)
    book = np.array([r["book"] for r in rows])
    out = []
    for a in (1, 0):
        idx = np.where(y == a)[0]
        S = X[idx] @ X[idx].T
        hit = np.argwhere(np.triu(S > DUPLICATE_COS, 1) & (book[idx][:, None] != book[idx][None, :]))
        for i, j in hit:
            r, s = rows[idx[i]], rows[idx[j]]
            out.append(dict(a=r["id"], b=s["id"], uid_a=r["uid"], uid_b=s["uid"], cos=round(float(S[i, j]), 3),
                            a_opening=" ".join(r["text"].split()[:10]), b_opening=" ".join(s["text"].split()[:10])))
    return out


def main():
    rows, y = load()
    names = [r for r in READINGS if (DATA / f"emb_{r}.npy").exists()]
    res = {"grid": GRID.round(2).tolist(), "ks": KS, "n": len(y), "leakage": {}, "retrieval": {}, "retrieval_book": {},
           "mixing": {}, "book": {}}
    for name in names:
        r = reading(name, rows, y)
        res["leakage"][name] = r["leakage"]
        for part in ("retrieval", "retrieval_book", "mixing", "book"):
            res[part].update({f"{name}/{stage}": v for stage, v in r[part].items()})
    words = np.array([r["words"] for r in rows])
    res["length"] = dict(author_auc=float(roc_auc_score(y, words)),
                         median={"Kafka": float(np.median(words[y == 1])), "Lispector": float(np.median(words[y == 0]))})
    res["shuffle_null"] = shuffle_null(emb(FINAL, "erased"), y)
    res["duplicates"] = duplicates(rows, y)
    blind2 = DATA / "blind2" / "result.json"
    if blind2.exists():
        res["blind2"] = read(blind2.name, blind2.parent)
    print("length AUC", round(res["length"]["author_auc"], 3), "| duplicates", len(res["duplicates"]),
          "| shuffle null", res["shuffle_null"]["real"], "vs",
          np.median([s["n"] for s in res["shuffle_null"]["shuffled"]]), flush=True)
    write("benchmarks.json", res)
    export()


def export():
    """The benchmarks as a JS module beside the atlas, for the article's figures."""
    write_js(VIZ_DATA / "bench.js", read("benchmarks.json"))


if __name__ == "__main__":
    export() if sys.argv[1:] == ["export"] else main()
