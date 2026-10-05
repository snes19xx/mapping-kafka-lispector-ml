"""Loading, and the geometry every stage shares: neighbours, mixing, next-paragraph ranks, CSLS."""
import hashlib
import json

import numpy as np

from src.core.config import CSLS_K, DATA, NEIGHBOURS


def uid(text):
    """A short unique identifier for a passage, based on its text. The first 16 hex digits of the SHA1 hash."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def read(name, where=DATA):
    return json.loads((where / name).read_text(encoding="utf-8"))


def write(name, obj, where=DATA, indent=None):
    (where / name).write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def write_js(path, obj):
    """A JS module the June pieces import, so the bundler inlines the data."""
    path.write_text(f"export default {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))};\n", encoding="utf-8")


def load(where=DATA):
    """Passages and author labels (1 = Kafka)."""
    rows = read("passages.json", where)
    return rows, np.array([r["author"] == "Kafka" for r in rows], dtype=int)


def exact_ci(x, n, level=0.95):
    """Clopper-Pearson interval for x successes in n trials, at the given confidence level. Returns [lower, upper]."""
    from scipy.stats import beta
    a = (1 - level) / 2
    return [float(beta.ppf(a, x, n - x + 1)) if x else 0.0, float(beta.ppf(1 - a, x + 1, n - x)) if x < n else 1.0]


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def emb(name, prefix="emb"):
    return unit(np.load(DATA / f"{prefix}_{name}.npy").astype(np.float64))


def neighbours(X, k=NEIGHBOURS):
    Xn = unit(X)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def expected_cross(y):
    """Each passage's other-author share among its neighbours if authorship were random."""
    n1, n = y.sum(), len(y)
    return np.where(y == 1, (n - n1) / (n - 1), n1 / (n - 1))


def mixing(X, y, k=NEIGHBOURS):
    """Other-author share among k nearest neighbours over the random-authorship share (1 = mixed), and same-author share."""
    nb = neighbours(X, k)
    return float((y[nb] != y[:, None]).mean(1).mean() / expected_cross(y).mean()), float((y[nb] == y[:, None]).mean())


def successors(rows):
    """Passages followed by another in the same book and section, with no excluded passage between them."""
    return np.array([i for i in range(len(rows) - 1) if rows[i]["book"] == rows[i + 1]["book"]
                     and rows[i]["section"] == rows[i + 1]["section"] and not rows[i].get("gap_after")])


def next_ranks(X, rows, within_book=False):
    """Rank (0 = nearest) of each passage's next paragraph among all other passages, or among its own book's."""
    succ = successors(rows)
    Xn = unit(X)
    S = Xn[succ] @ Xn.T
    S[np.arange(len(succ)), succ] = -np.inf
    if within_book:
        book = np.array([r["book"] for r in rows])
        S[book[succ][:, None] != book[None, :]] = -np.inf
    target = S[np.arange(len(succ)), succ + 1]
    return (S > target[:, None]).sum(1)


def radii(S, k=CSLS_K):
    """Mean cosine of each row (Kafka) and each column (Lispector) to its k nearest on the other side."""
    return np.sort(S, 1)[:, -k:].mean(1), np.sort(S, 0)[-k:, :].mean(0)


def csls(X, y, k=CSLS_K):
    """Kafka rows ki, Lispector rows li, their cosines S and CSLS scores C = 2 cos - r_L(k) - r_K(l)."""
    Xn = unit(X)
    ki, li = np.where(y == 1)[0], np.where(y == 0)[0]
    S = Xn[ki] @ Xn[li].T
    rk, rl = radii(S, k)
    return ki, li, S, 2 * S - rk[:, None] - rl[None, :]


def mutual(C):
    """Row/column pairs that are each other's best match in the score matrix C."""
    best_l, best_k = C.argmax(1), C.argmax(0)
    return [(a, int(b)) for a, b in enumerate(best_l) if best_k[b] == a]


def bridges(X, y, k=CSLS_K, cosine=False):
    """Cross-author mutual best matches, strongest first, by CSLS (or plain cosine)."""
    ki, li, S, C = csls(X, y, k)
    M = S if cosine else C
    out = [dict(k=int(ki[a]), l=int(li[b]), csls=float(C[a, b]), cos=float(S[a, b])) for a, b in mutual(M)]
    out.sort(key=lambda d: -(d["cos"] if cosine else d["csls"]))
    return out
