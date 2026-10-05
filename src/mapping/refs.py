"""
Carry the curated readings (constellation names, close-up bridges) onto the current run, to OUT/data/refs.js.
"""
import sys

import numpy as np

from src.core.common import load, read, write, write_js
from src.core.config import CURATION, DATA, N_SHOWN, VIZ_DATA


def rename(cur, labels):
    """Constellation id -> (name, overlap); names are curated on the current run, so the overlap is 1."""
    names = cur["constellations"]["names"]
    return {int(c): (names[str(c)], 1.0) for c in np.unique(labels)}


def locate(cur, rows, bridges):
    """Close-up label -> rank among the shown bridges; problems as readable lines."""
    uid = [r["uid"] for r in rows]
    rank = {(uid[b["k"]], uid[b["l"]]): j for j, b in enumerate(bridges)}
    found, problems = {}, []
    for label, c in cur["closeups"].items():
        j = rank.get((c["kafka"], c["lispector"]))
        if j is not None and j < N_SHOWN:
            found[label] = j
            continue
        where = "is gone" if j is None else f"is now rank {j}, past the {N_SHOWN} shown"
        near = [f"rank {i}: {rows[b['k']]['text'][:50]}… / {rows[b['l']]['text'][:50]}…" for i, b in enumerate(bridges[:N_SHOWN])
                if uid[b["k"]] == c["kafka"] or uid[b["l"]] == c["lispector"]]
        problems.append(f"close-up '{label}' ({c['kafka_opening']}… / {c['lispector_opening']}…) {where}."
                        + ("\n    shown bridges sharing one of its passages:\n    " + "\n    ".join(near) if near else ""))
    return found, problems


def main():
    cur = read(CURATION.name, CURATION.parent)
    rows, y = load()
    labels = np.load(DATA / "atlas.npz")["labels"]
    names = rename(cur, labels)
    for c, (name, j) in sorted(names.items()):
        print(f"{c:2d} {name:18s} overlap {j:.2f}{'   <- check this name by reading the constellation' if j < 0.5 else ''}")
    closeups, problems = locate(cur, rows, read("bridges.json"))
    if problems:
        print("\n".join(problems))
        sys.exit("refs.js not written: choose new close-ups (curation.json, and the quotes in OUT/reading.js) and rerun refs")
    consts = read("constellations.json")
    solitary = [c["id"] for c in sorted(consts, key=lambda c: -abs(c["kafka"] - 0.5))[:3]]
    refs = dict(names={c: dict(name=n) for c, (n, _) in names.items()}, closeups=closeups, solitary=solitary,
                overlap={c: round(j, 3) for c, (_, j) in names.items()})
    write_js(VIZ_DATA / "refs.js", refs)
    write("refs.json", refs, indent=1)
    print("close-ups at ranks", closeups, "| most lopsided", solitary)


if __name__ == "__main__":
    main()
