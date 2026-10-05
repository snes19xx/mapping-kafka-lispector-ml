"""The Metamorphosis in two translations, Bernofsky (2014, corpus) and the Muirs (1933, twins): a test with a known answer.

Ground truth comes from position in the text alone, never from embeddings; results go to data/translation.json.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src.core.common import (bridges, csls, emb, load, mutual, radii, read,
                             successors, unit, write, write_js)
from src.core.config import (CSLS_K, DATA, FINAL, N_SHOWN, RAW, READINGS,
                             VIZ_DATA)
from src.erasure.erase import fit_eraser

BOOK = "metamorphosis"
# a pair is the same stretch of German when their spans share half the shorter one
STRICT = 0.5
# CORAL needs fewer dimensions than passages per translator (about 120 each)
TRANSLATOR_DIMS = (16, 32, 64)


def spans(rows, by_part=True):
    """Each passage's part of the novella and its stretch of that part, as fractions of the part's words."""
    first_seen = {}
    part = np.array([first_seen.setdefault(r["section"], len(first_seen)) if by_part else 0 for r in rows])
    words = np.array([r["words"] for r in rows], dtype=float)
    span = np.zeros((len(rows), 2))
    for p in np.unique(part):
        w = words[part == p]
        end = np.cumsum(w) / w.sum()
        span[part == p] = np.c_[end - w / w.sum(), end]
    return part, span


def overlap(B_rows, M_rows):
    """For every Bernofsky x Muir pair, the share of the shorter passage's stretch that the other covers.

    Both translations follow the same German paragraphs in order, so position alone aligns them.
    """
    by_part = len({r["section"] for r in B_rows}) == len({r["section"] for r in M_rows})
    if not by_part:
        print("the translations have different numbers of parts: aligning on the whole text", flush=True)
    pb, sb = spans(B_rows, by_part)
    pm, sm = spans(M_rows, by_part)
    inter = np.clip(np.minimum(sb[:, None, 1], sm[None, :, 1]) - np.maximum(sb[:, None, 0], sm[None, :, 0]), 0, None)
    shorter = np.minimum((sb[:, 1] - sb[:, 0])[:, None], (sm[:, 1] - sm[:, 0])[None, :])
    return np.where(pb[:, None] == pm[None, :], inter / shorter, 0.0)


def ground_truth(B, M, O):
    """Bridges between the two translations (translator as the "author"), scored against the known alignment O.

    precision: bridges that join a paragraph to its own translation (near: to an overlapping stretch at all);
    recall: Bernofsky passages with a counterpart that end up in a correct bridge; top1/mrr: one-sided search.
    """
    T, near = O >= STRICT, O > 0
    has = T.any(1)
    S = B @ M.T
    rb, rm = radii(S, CSLS_K)
    out = {}
    for name, score in (("csls", 2 * S - rb[:, None] - rm[None, :]), ("cosine", S)):
        pairs = mutual(score)
        right = [a for a, b in pairs if T[a, b]]
        rank = np.array([(score[a] > score[a][T[a]].max()).sum() for a in np.where(has)[0]])
        out[name] = dict(
            bridges=len(pairs), precision=len(right) / max(1, len(pairs)),
            precision_near=float(np.mean([near[a, b] for a, b in pairs])) if pairs else None,
            recall=len(set(right)) / int(has.sum()),
            top1=float((rank == 0).mean()), mrr=float((1 / (rank + 1)).mean()),
            top1_chance=float((T[has].sum(1) / T.shape[1]).mean()),
        )
    return out


def in_corpus(M, K, O, bi, kafka, kbook):
    """Each Muir passage searched against every Kafka passage in the corpus, by cosine.

    Other Muir translations sit in the pool (most of The Complete Stories), so a voice-led space ranks them above the
    Bernofsky counterpart; a content-led space ranks the counterpart first.
    """
    pos = {int(g): j for j, g in enumerate(kafka)}
    S = M @ K.T
    counterpart_rank, best_book = [], []
    for m in np.where((O >= STRICT).any(0))[0]:
        true = [pos[int(bi[a])] for a in np.where(O[:, m] >= STRICT)[0]]
        counterpart_rank.append(int((S[m] > S[m, true].max()).sum()))
        best_book.append(kbook[S[m].argmax()])
    r, best_book = np.array(counterpart_rank), np.array(best_book)
    return dict(n=len(r), top1=float((r == 0).mean()), top10=float((r < 10).mean()), median_rank=float(np.median(r) + 1),
                best_is_metamorphosis=float((best_book == BOOK).mean()), best_is_kstories=float((best_book == "kstories").mean()),
                pool=len(kafka))


def groups(O, nb):
    """Fold groups for the translator probe: passages joined by the alignment share a group (union-find)."""
    parent = list(range(nb + O.shape[1]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in np.argwhere(O >= STRICT):
        parent[find(a)] = find(nb + b)
    return np.array([find(i) for i in range(len(parent))])


def translator_auc(B, M, O):
    """Held-out AUC of a Bernofsky-vs-Muir probe; aligned passages share a fold, so content cannot give the answer away."""
    X = np.r_[B, M]
    t = np.r_[np.ones(len(B)), np.zeros(len(M))].astype(int)
    s = cross_val_predict(LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"), X, t,
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), groups=groups(O, len(B)),
                          method="predict_proba")
    return float(roc_auc_score(t, s[:, 1]))


def median_cos(A, B, idx):
    return float(np.median([A[i] @ B[j] for i, j in idx])) if len(idx) else None


def translator_erased(B, M, stage="moments", dims=None):
    """The method with the translator as the concept: erasure fitted on the two translations together."""
    X = np.r_[B, M]
    t = np.r_[np.ones(len(B)), np.zeros(len(M))].astype(int)
    E = unit(fit_eraser(X, t, stage=stage, **({"dims": dims} if dims else {}))(X, t))
    return E[:len(B)], E[len(B):]


def page(rows, y, bi, twins, O, X, Mx, out):
    """What the piece draws of this test, in the map's own space (X, Mx), to OUT/data/translation.js.

    No passage text is written: both translations may be in copyright.
    """
    T = O >= STRICT
    q = lambda v: [round(float(t), 4) for t in v]
    pb, sb = spans([rows[i] for i in bi])
    pm, sm = spans(twins)
    B = X[bi]
    S = B @ Mx.T
    rb, rm = radii(S, CSLS_K)
    C = 2 * S - rb[:, None] - rm[None, :]
    links = [[int(a), int(b), int(T[a, b]), int(O[a, b] > 0)] for a, b in mutual(C)]

    Sc = Mx @ X.T
    kafka = np.where(y == 1)[0]
    Sk = Sc[:, kafka]
    muir = []
    for m in range(len(twins)):
        true = bi[T[:, m]]
        rank = int((Sk[m] > Sc[m, true].max()).sum()) if len(true) else None
        muir.append(dict(part=int(pm[m]), span=q(sm[m]), twin=int(O[:, m].argmax()),
                         best=int(kafka[Sk[m].argmax()]), rank=rank))

    ki, li, Sx, _ = csls(X, y)
    _, rl = radii(Sx)
    twin_of = {int(bi[a]): int(O[a].argmax()) for a in np.where(T.any(1))[0]}
    swap = []
    for j, b in enumerate(bridges(X, y)):
        if b["k"] in twin_of:
            m = twin_of[b["k"]]
            cos = Mx[m] @ X[li].T
            score = 2 * cos - np.sort(cos)[-CSLS_K:].mean() - rl
            swap.append(dict(bridge=j, k=b["k"], l=b["l"], muir=m, best=int(li[score.argmax()]),
                             rank=int((score > score[np.where(li == b["l"])[0][0]]).sum())))

    # the ladder's rungs as quartiles, on the pairs the readings' medians use
    rng = np.random.default_rng(0)
    # main draws random pairs for three readings; the map's is the third
    for _ in range(2):
        rng.choice(kafka, (2000, 2))
    succ = [i for i in successors(rows) if rows[i]["book"] == BOOK]
    quart = lambda v: [round(float(t), 3) for t in np.quantile(v, [0.25, 0.5, 0.75])]
    rungs = dict(
        random=quart([X[i] @ X[j] for i, j in rng.choice(kafka, (2000, 2)) if i != j]),
        adjacent=quart([X[i] @ X[i + 1] for i in succ]),
        bridge=quart([b["cos"] for b in bridges(X, y)[:N_SHOWN]]),
        twin=quart([X[bi[a]] @ Mx[O[a].argmax()] for a in np.where(T.any(1))[0]]),
    )

    g, c, r = out["ground_truth"][f"{FINAL}/map"]["csls"], out["in_corpus"][f"{FINAL}/map"], out["readings"][f"{FINAL} erased"]
    data = dict(
        bern=[dict(i=int(i), part=int(pb[a]), span=q(sb[a]), top=int(S[a].argmax())) for a, i in enumerate(bi)],
        muir=muir, links=links, swap=swap, shown=N_SHOWN,
        totals=dict(bridges=g["bridges"], precision=g["precision"], precision_near=g["precision_near"], top1=g["top1"],
                    chance=g["top1_chance"], in_corpus_top1=c["top1"], other_muir_first=round(c["best_is_kstories"] * c["n"]),
                    pool=c["pool"], translator_auc=r["translator_auc"],
                    raw_top1=out["ground_truth"][f"{FINAL}/raw"]["csls"]["top1"],
                    raw_in_corpus_top1=out["in_corpus"][f"{FINAL}/raw"]["top1"]),
        ladder=rungs,
    )
    write_js(VIZ_DATA / "translation.js", data)
    kept = sum(s["rank"] == 0 for s in swap)
    print(f"translation.js: {len(links)} links, {sum(l[2] for l in links)} to a counterpart; swap keeps {kept} of {len(swap)}; "
          f"{sum(s['bridge'] < N_SHOWN for s in swap)} of the {len(swap)} are among the {N_SHOWN} shown; ladder {rungs}", flush=True)


def main():
    rows, y = load()
    twins = read("twins.json")
    bi = np.array([i for i, r in enumerate(rows) if r["book"] == BOOK])
    O = overlap([rows[i] for i in bi], twins)
    best = {a: int(O[a].argmax()) for a in np.where((O >= STRICT).any(1))[0]}
    out = dict(n_bernofsky=len(bi), n_muir=len(twins), strict=STRICT,
               truth=dict(pairs=int((O >= STRICT).sum()), bernofsky_matched=len(best),
                          muir_matched=int((O >= STRICT).any(0).sum())),
               ground_truth={}, in_corpus={}, readings={})
    print(f"{len(bi)} Bernofsky and {len(twins)} Muir passages; {out['truth']}", flush=True)

    # the known-answer test: raw readings, then the method with the translator erased
    readings = [r for r in READINGS if (DATA / f"emb_{r}.npy").exists()]
    kafka = np.where(y == 1)[0]
    kbook = np.array([rows[i]["book"] for i in kafka])
    for name in readings:
        X, M = emb(name), emb(name, "twins")
        out["ground_truth"][f"{name}/raw"] = ground_truth(X[bi], M, O)
        out["in_corpus"][f"{name}/raw"] = in_corpus(M, X[kafka], O, bi, kafka, kbook)
    B, M = emb(FINAL)[bi], emb(FINAL, "twins")
    out["ground_truth"][f"{FINAL}/translator-leace"] = ground_truth(*translator_erased(B, M, "leace"), O)
    for d in TRANSLATOR_DIMS:
        out["ground_truth"][f"{FINAL}/translator-leace-coral{d}"] = ground_truth(*translator_erased(B, M, dims=d), O)
    # the map's own space: author erasure fitted on the corpus, twins transformed as Kafka
    apply = fit_eraser(emb(FINAL), y)
    X = unit(apply(emb(FINAL), y))
    Mx = unit(apply(emb(FINAL, "twins"), np.ones(len(twins), dtype=int)))
    out["ground_truth"][f"{FINAL}/map"] = ground_truth(X[bi], Mx, O)
    out["in_corpus"][f"{FINAL}/map"] = in_corpus(Mx, X[kafka], O, bi, kafka, kbook)
    for cond, r in out["ground_truth"].items():
        c = r["csls"]
        print(f"{cond:36s} bridges {c['bridges']:3d}  precision {c['precision']:.2f}  recall {c['recall']:.2f}  "
              f"top1 {c['top1']:.2f} (chance {c['top1_chance']:.3f})", flush=True)
    for cond, r in out["in_corpus"].items():
        print(f"{cond:36s} counterpart first {r['top1']:.2f} of {r['n']}; best is another Muir story "
              f"{r['best_is_kstories']:.2f}", flush=True)

    # distances: the same paragraph in two translations, adjacent paragraphs in one, random Kafka pairs, bridge ends
    rng = np.random.default_rng(0)
    succ = [i for i in successors(rows) if rows[i]["book"] == BOOK]
    for label, Xs, Ms in ((RAW, emb(RAW), emb(RAW, "twins")), (FINAL, emb(FINAL), emb(FINAL, "twins")),
                          (f"{FINAL} erased", X, Mx)):
        shown = bridges(Xs, y)[:N_SHOWN] if Xs is X else []
        out["readings"][label] = dict(
            twin_pair=median_cos(Xs[bi], Ms, list(best.items())),
            adjacent=median_cos(Xs, Xs, [(i, i + 1) for i in succ]),
            random_kafka=median_cos(Xs, Xs, [(i, j) for i, j in rng.choice(kafka, (2000, 2)) if i != j]),
            bridge_ends=float(np.median([b["cos"] for b in shown])) if shown else None,
            translator_auc=translator_auc(Xs[bi], Ms, O),
        )
        print(label, {k: round(v, 3) for k, v in out["readings"][label].items() if v is not None}, flush=True)

    # swap each bridged Bernofsky passage for its Muir counterpart: is the same Lispector passage still its best match?
    ki, li, S, _ = csls(X, y)
    _, rl = radii(S)
    twin_of = {int(bi[a]): m for a, m in best.items()}
    kept, ranks = [], []
    for b in bridges(X, y):
        if b["k"] in twin_of:
            cos = Mx[twin_of[b["k"]]] @ X[li].T
            score = 2 * cos - np.sort(cos)[-CSLS_K:].mean() - rl
            ranks.append(int((score > score[np.where(li == b["l"])[0][0]]).sum()))
            kept.append(ranks[-1] == 0)
    out["bridge_swap"] = dict(bridges_with_twin=len(kept), same_best_match=float(np.mean(kept)) if kept else None,
                              median_rank=float(np.median(ranks)) if ranks else None)
    print("bridge swap", out["bridge_swap"], flush=True)
    write("translation.json", out, indent=1)
    page(rows, y, bi, twins, O, X, Mx, out)


if __name__ == "__main__":
    main()
