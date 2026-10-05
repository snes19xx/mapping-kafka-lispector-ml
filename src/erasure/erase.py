"""Author erasure: LEACE equalises the writers' means, then CORAL their covariances in CORAL_DIMS dimensions.

    python -m src.erasure.erase [reading ...]   default: every reading with data/emb_<reading>.npy
"""
import sys

import numpy as np
from src.core.common import emb, load, mixing, next_ranks, read, unit, write
from src.core.config import CORAL_DIMS, DATA, READINGS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

STAGES = ("raw", "leace", "pca", "moments")


def leace(X, z, eps=1e-6):
    """Fit the LEACE eraser for concept labels z (n, k one-hot or n binary); returns x -> erased x."""
    Z = z.reshape(len(z), -1).astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    Zc = Z - Z.mean(0)
    cov = Xc.T @ Xc / len(X)
    L, V = np.linalg.eigh(cov)
    keep = L > eps * L.max()
    W = (V[:, keep] / np.sqrt(L[keep])) @ V[:, keep].T
    W_pinv = (V[:, keep] * np.sqrt(L[keep])) @ V[:, keep].T
    U, s, _ = np.linalg.svd(W @ (Xc.T @ Zc / len(X)), full_matrices=False)
    U = U[:, s > eps * max(s.max(), eps)]
    edit = W_pinv @ (U @ U.T) @ W

    def apply(A):
        return A - (A - mu) @ edit.T

    return apply


def principal(X, idx, k):
    """Centre and top-k principal directions of X[idx], and the share of variance they hold."""
    mu = X[idx].mean(0)
    _, S, Vt = np.linalg.svd(X[idx] - mu, full_matrices=False)
    return mu, Vt[:k].T, float((S[:k] ** 2).sum() / (S ** 2).sum())


def fit_coral(X, y, idx, k=CORAL_DIMS, ridge=1e-4):
    """Per author: centre, whiten, recolour to the pooled covariance, in the top-k subspace; applying it needs the author."""
    mu, P, kept = principal(X, idx, k)
    Z = (X[idx] - mu) @ P
    Lp, Vp = np.linalg.eigh(np.cov(Z.T))
    colour = (Vp * np.sqrt(np.clip(Lp, 1e-12, None))) @ Vp.T
    maps = {}
    for a in np.unique(y[idx]):
        Za = Z[y[idx] == a]
        m = Za.mean(0)
        L, V = np.linalg.eigh(np.cov((Za - m).T) + ridge * np.eye(k))
        maps[a] = (m, ((V / np.sqrt(L)) @ V.T) @ colour)

    def apply(A, labels):
        Za = (A - mu) @ P
        out = np.zeros((len(A), k))
        for a, (m, T) in maps.items():
            out[labels == a] = (Za[labels == a] - m) @ T
        return out

    return apply, kept


def fit_eraser(X, y, idx=None, stage="moments", dims=CORAL_DIMS, concept=None):
    """Fit on X[idx] (default: every passage); returns (A, author labels) -> erased A.

    stage is "leace", "pca" (LEACE then truncation only) or "moments" (LEACE then CORAL); concept overrides the author for LEACE.
    """
    idx = np.arange(len(y)) if idx is None else idx
    z = y if concept is None else concept
    lea = leace(X[idx], z[idx])
    if stage == "leace":
        return lambda A, labels: lea(A)
    Xl = lea(X)
    if stage == "pca":
        mu, P, _ = principal(Xl, idx, dims)
        return lambda A, labels: (lea(A) - mu) @ P
    cor, _ = fit_coral(Xl, y, idx, dims)
    return lambda A, labels: cor(lea(A), labels)


def stages(X, y):
    """Every stage fitted on the whole corpus: the representations the map and the bridges use."""
    out = {"raw": X}
    for s in STAGES[1:]:
        out[s] = fit_eraser(X, y, stage=s)(X, y)
    return out


def probe(X, y, seed=0):
    """Balanced accuracy of a logistic author probe, 5-fold."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    return float(cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy").mean())


def probe_heldout(X, y, stage, seed=0):
    """Eraser fit on four folds; the probe is trained and scored inside the fifth, so it never saw the fit."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y):
        Xe = unit(fit_eraser(X, y, tr, stage)(X, y))
        inner = StratifiedKFold(5, shuffle=True, random_state=seed + 1)
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        scores.append(float(cross_val_score(clf, Xe[te], y[te], cv=inner, scoring="balanced_accuracy").mean()))
    return float(np.mean(scores))


def report(name, rows, y):
    X = emb(name)
    S = stages(X, y)
    out = []
    for stage in ("raw", "leace", "moments"):
        Xs = S[stage]
        mix, same = mixing(Xs, y)
        row = dict(embedding=name, stage=stage, dims=int(Xs.shape[1]), mixing=mix, same_author=same,
                   continuity=float((next_ranks(Xs, rows) < 10).mean()),
                   probe=probe(unit(Xs), y) if stage == "raw" else probe_heldout(X, y, stage))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        out.append(row)
    out[-1]["coral_variance_kept"] = principal(S["leace"], np.arange(len(y)), CORAL_DIMS)[2]
    np.save(DATA / f"leace_{name}.npy", unit(S["leace"]).astype(np.float32))
    np.save(DATA / f"erased_{name}.npy", unit(S["moments"]).astype(np.float32))
    return out


def main(names):
    rows, y = load()
    path = DATA / "erasure_metrics.json"
    metrics = read(path.name) if path.exists() else []
    for name in names:
        metrics = [m for m in metrics if m["embedding"] != name] + report(name, rows, y)
    write(path.name, metrics, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or [r for r in READINGS if (DATA / f"emb_{r}.npy").exists()])
