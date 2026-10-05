"""The constellations as a garden, to OUT/data/tree.js: one plant per constellation, sorted by its lean.

A plant's branches are its nested Ward sub-clusters and its needles its passages; a bridge inside it is a flower.
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score

from src.core.common import load, read, write_js
from src.core.config import DATA, K_CONSTELLATIONS, N_SHOWN, VIZ_DATA

CUTS = [K_CONSTELLATIONS, 40, 90, 200, 420]
UP = [3, 7]
LEN = [0.55, 0.3, 0.2, 0.13, 0.08]
rng = np.random.default_rng(1883)


def curve(a, b, bend, steps=10):
    """Quadratic arc from a to b, its middle pushed sideways by bend times the length."""
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    c = (a + b) / 2 + np.array([-d[1], d[0]]) * bend
    t = np.linspace(0, 1, steps)[:, None]
    return ((1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b).round(4).tolist()


def main():
    rows, y = load()
    atlas = np.load(DATA / "atlas.npz")
    # the 10-d embedding the constellations were cut from, saved by atlas.py
    Z = linkage(atlas["z10"], "ward")
    cuts = [fcluster(Z, k, "maxclust") for k in CUTS]
    # the hierarchy above the constellations, for the one tree they join into
    above = [fcluster(Z, k, "maxclust") for k in UP]
    labels = atlas["labels"]
    print("constellations reproduced, ARI", round(adjusted_rand_score(cuts[0], labels), 4))
    mx = atlas["final"][:, 0]
    nmax = np.bincount(labels).max()
    bridges = read("bridges.json")[:N_SHOWN]

    plants = []
    for c in map(int, np.unique(labels)):
        members = np.where(labels == c)[0]
        scale = (len(members) / nmax) ** 0.3
        branches, leaves, twig_of, ends = [], [], {}, {}

        # the largest sub-cluster carries the axis on; the rest leave it as side shoots partway along
        def grow(idx, level, pos, th, path, t0):
            L = LEN[level] * scale * rng.uniform(0.85, 1.15)
            end = pos + L * np.array([np.cos(th), np.sin(th)])
            pts = curve(pos, end, rng.uniform(-0.15, 0.15))
            node = len(branches)
            branches.append(dict(pts=pts, n=int(len(idx)), t0=round(t0, 4), t1=round(t0 + L, 4)))
            path = path + [node]
            ends[node] = (end, path)
            groups = [idx[cuts[level + 1][idx] == g] for g in np.unique(cuts[level + 1][idx])] if level + 1 < len(CUTS) else []
            if len(groups) == 1:
                grow(groups[0], level + 1, end, th + rng.normal(0, 0.1), path, t0 + L)
            elif groups:
                groups.sort(key=len, reverse=True)
                main, sides = groups[0], groups[1:]
                a = th + rng.normal(0, 0.08)
                grow(main, level + 1, end, a + (-np.pi / 2 - a) * 0.2, path, t0 + L)
                for j, g in enumerate(sides):
                    sign = -1 if mx[g].mean() < mx[main].mean() else 1
                    t = 0.9 - 0.55 * (j + 1) / (len(sides) + 1)
                    at = np.asarray(pts[int(round(t * (len(pts) - 1)))])
                    a = th + sign * rng.uniform(0.5, 0.95)
                    grow(g, level + 1, at, a + (-np.pi / 2 - a) * 0.15, path, t0 + L * t)
            else:
                # the tuft: Kafka's needles fan to the left of the twig, Lispector's to the right
                order = idx[np.lexsort((mx[idx], -y[idx]))]
                m = len(order)
                for j, i in enumerate(order):
                    a = th + (j / max(1, m - 1) - 0.5) * 2.0 + rng.normal(0, 0.06)
                    r = rng.uniform(0.02, 0.028)
                    mid = end + r * np.array([np.cos(a), np.sin(a)])
                    leaves.append([*mid.round(4).tolist(), round(float(a), 3), int(y[i]), round(t0 + L + r, 4)])
                    twig_of[int(i)] = node

        grow(members, 0, np.zeros(2), -np.pi / 2, [], 0.0)

        # a bridge with both ends in this constellation flowers where its two paragraphs first share a branch
        flowers, inside = [], set(members.tolist())
        for rank, b in enumerate(bridges):
            if b["k"] in inside and b["l"] in inside:
                pa, pb = ends[twig_of[b["k"]]][1], ends[twig_of[b["l"]]][1]
                node = [u for u, v in zip(pa, pb) if u == v][-1]
                at = ends[node][0] + rng.normal(0, 0.012, 2)
                flowers.append([*at.round(4).tolist(), round(float(b["csls"]), 3), rank])
        touching = sum((b["k"] in inside) or (b["l"] in inside) for b in bridges)
        plants.append(dict(id=c, n=int(len(members)), kafka=round(float(y[members].mean()), 3), branches=branches,
                           up=[int(np.bincount(a[members]).argmax()) for a in above], mx=round(float(mx[members].mean()), 4),
                           leaves=leaves, flowers=flowers, touching=int(touching)))

    plants.sort(key=lambda p: -p["kafka"])
    for p in plants:
        print(p["id"], p["n"], p["kafka"], "flowers", len(p["flowers"]), "touching", p["touching"])
    print("bridges inside one constellation:", sum(len(p["flowers"]) for p in plants), "of", N_SHOWN)
    write_js(VIZ_DATA / "tree.js", dict(plants=plants, cuts=CUTS))


if __name__ == "__main__":
    main()
