"""Blind reading for human readers: scored pairs across arms, and triads that pit a bridge against a baseline match.

    python -m src.evaluation.blind deal  --key ~/blind-key.json       data/blind2/: sheets, score template, preregistration
    python -m src.evaluation.blind score --key ~/blind-key.json       data/blind2/result.json from every scores_<reader>.json
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import cohen_kappa_score

from src.core.common import csls, emb, exact_ci, load, read, write, write_js
from src.core.config import (BLIND_ARMS, BLIND_PER_ARM, BLIND_SEED, BLIND_TRIADS, DATA, FINAL, HERE, N_SHOWN, RAW,
                             VIZ_DATA)
from src.mapping.atlas import STOP, tfidf
from src.mapping.export import proper_nouns

OUT = DATA / "blind2"
RUBRIC = HERE / "rubric.md"
KEEP = {"god", "lord", "christ", "i"}


def masker(rows):
    names = sorted(proper_nouns(rows) - KEEP - STOP, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I)
    # a name is masked only where it is capitalised, so "rose" the flower survives "Rose" the woman
    return lambda text: pattern.sub(lambda m: "[NAME]" if m.group(0)[0].isupper() else m.group(0), text)


def pairs_from(arms, rows, y, rng):
    kafka, lisp = np.where(y == 1)[0], np.where(y == 0)[0]
    items = []
    for arm in BLIND_ARMS:
        if arm == "random":
            chosen = [(int(rng.choice(kafka)), int(rng.choice(lisp))) for _ in range(BLIND_PER_ARM)]
        else:
            top = arms[arm][:N_SHOWN]
            chosen = [(top[j]["k"], top[j]["l"]) for j in rng.choice(len(top), BLIND_PER_ARM, replace=False)]
        items += [dict(format="pair", arm=arm, k=k, l=l) for k, l in chosen]
    return items


def best_matches(rows, y):
    """For every passage, its best match by the other writer under each baseline, keyed by arm name."""
    ki, li, _, Cu = csls(emb(FINAL), y)
    _, _, Sw, _ = csls(emb(RAW), y)
    M = tfidf(rows)
    St = (M[ki] @ M[li].T).toarray()
    out = {}
    for arm, C in (("unerased", Cu), ("words", Sw), ("tfidf", St)):
        best = {int(k): int(li[j]) for k, j in zip(ki, C.argmax(1))}
        best.update({int(l): int(ki[j]) for l, j in zip(li, C.argmax(0))})
        out[arm] = best
    return out


def triads_from(arms, rows, y, rng):
    """Anchors from the shown bridges, half Kafka and half Lispector; the rival candidate rotates over the baselines."""
    rivals = best_matches(rows, y)
    shown = arms["final"][:N_SHOWN]
    order = rng.permutation(len(shown))
    items, names = [], ["unerased", "words", "tfidf"]
    for j in order:
        if len(items) == BLIND_TRIADS:
            break
        b = shown[j]
        anchor, partner = (b["k"], b["l"]) if len(items) % 2 == 0 else (b["l"], b["k"])
        rival_arm = names[len(items) % len(names)]
        rival = rivals[rival_arm][anchor]
        if rival != partner:
            items.append(dict(format="triad", anchor=anchor, bridge=partner, rival=rival, rival_arm=rival_arm))
    return items


def deal(key_path):
    if OUT.resolve() in key_path.resolve().parents:
        raise SystemExit(f"--key must lie outside {OUT}: readers must never have it")
    rubric = RUBRIC.read_text(encoding="utf-8")
    if "TODO" in rubric:
        raise SystemExit(f"finish {RUBRIC} (worked examples, no TODO left) before dealing: the rubric is fixed first")
    rows, y = load()
    arms, rng, mask = read("arms.json"), np.random.default_rng(BLIND_SEED), masker(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    key = {}
    for fmt, items in (("pairs", pairs_from(arms, rows, y, rng)), ("triads", triads_from(arms, rows, y, rng))):
        items = [items[i] for i in rng.permutation(len(items))]
        lines = [f"# Blind reading: {fmt}", "", rubric, ""]
        for n, it in enumerate(items):
            iid = f"{fmt[0]}{n:03d}"
            if fmt == "pairs":
                shown = [it["k"], it["l"]] if rng.random() < 0.5 else [it["l"], it["k"]]
                lines += [f"## {iid}", "", f"**Passage 1.** {mask(rows[shown[0]]['text'])}", "",
                          f"**Passage 2.** {mask(rows[shown[1]]['text'])}", ""]
            else:
                cands = [("bridge", it["bridge"]), ("rival", it["rival"])]
                if rng.random() < 0.5:
                    cands.reverse()
                it["order"] = [c[0] for c in cands]
                lines += [f"## {iid}", "", f"**Anchor.** {mask(rows[it['anchor']]['text'])}", "",
                          f"**Candidate 1.** {mask(rows[cands[0][1]]['text'])}", "",
                          f"**Candidate 2.** {mask(rows[cands[1][1]]['text'])}", ""]
            it["uids"] = {k: rows[v]["uid"] for k, v in it.items() if k in ("k", "l", "anchor", "bridge", "rival")}
            key[iid] = it
        (OUT / f"sheet_{fmt}.md").write_text("\n".join(lines), encoding="utf-8")
    template = dict(reader="", pairs={i: None for i in key if i[0] == "p"}, triads={i: None for i in key if i[0] == "t"})
    write("scores_TEMPLATE.json", template, OUT, indent=1)
    prereg = dict(dealt=datetime.now(timezone.utc).isoformat(), seed=BLIND_SEED, arms=list(BLIND_ARMS),
                  per_arm=BLIND_PER_ARM, triads=sum(i[0] == "t" for i in key),
                  rubric_sha1=hashlib.sha1(rubric.encode()).hexdigest(),
                  key_sha1=hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest(),
                  analysis="pairs: consensus = readers' mean score, same thought = mean >= 1.5, any kinship = mean >= 0.5; "
                           "per arm: exact CI of same-thought and any-kinship shares; "
                           "bridges vs each arm: common-language effect and one-sided Mann-Whitney; "
                           "triads: share choosing the bridge, exact CI, one-sided binomial test against 0.5; "
                           "agreement: quadratic-weighted Cohen's kappa per reader pair (pairs), raw agreement (triads)")
    write("preregistration.json", prereg, OUT, indent=1)
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(key)} items dealt to {OUT}; key at {key_path} (sha1 {prereg['key_sha1'][:12]} in preregistration.json)")


def score(key_path):
    key = json.loads(key_path.read_text(encoding="utf-8"))
    readers = [read(p.name, OUT) for p in sorted(OUT.glob("scores_*.json")) if p.name != "scores_TEMPLATE.json"]
    if not readers:
        raise SystemExit(f"no scores_<reader>.json in {OUT}")
    out = dict(readers=[r["reader"] for r in readers], pairs={}, triads={}, agreement={})

    pids = sorted(i for i in key if i[0] == "p" and all(r["pairs"].get(i) is not None for r in readers))
    mean = {i: float(np.mean([r["pairs"][i] for r in readers])) for i in pids}
    by = {arm: np.array([mean[i] for i in pids if key[i]["arm"] == arm]) for arm in BLIND_ARMS}
    for arm, v in by.items():
        same, any_ = int((v >= 1.5).sum()), int((v >= 0.5).sum())
        out["pairs"][arm] = dict(n=len(v), mean=float(v.mean()), same=same / len(v), same_ci=exact_ci(same, len(v)),
                                 any=any_ / len(v), any_ci=exact_ci(any_, len(v)))
    for arm in BLIND_ARMS[1:]:
        a, b = by["final"], by[arm]
        cl = float(((a[:, None] > b[None]) + 0.5 * (a[:, None] == b[None])).mean())
        out["pairs"][f"final>{arm}"] = dict(auc=cl, p=float(mannwhitneyu(a, b, alternative="greater").pvalue))
    for (i, r), (j, s) in combinations(enumerate(readers), 2):
        out["agreement"][f"{r['reader']}~{s['reader']}"] = dict(
            pairs_kappa=float(cohen_kappa_score([r["pairs"][p] for p in pids], [s["pairs"][p] for p in pids], weights="quadratic")),
            triads_agree=float(np.mean([r["triads"][t] == s["triads"][t] for t in key if t[0] == "t"
                                        and r["triads"].get(t) is not None and s["triads"].get(t) is not None])))

    # a triad answer is 1 or 2, the candidate chosen; majority over readers, ties dropped
    for arm in ("unerased", "words", "tfidf", "all"):
        wins = []
        for t, it in key.items():
            if t[0] != "t" or (arm != "all" and it["rival_arm"] != arm):
                continue
            votes = [it["order"][r["triads"][t] - 1] == "bridge" for r in readers if r["triads"].get(t) in (1, 2)]
            if votes and np.mean(votes) != 0.5:
                wins.append(np.mean(votes) > 0.5)
        k = int(sum(wins))
        out["triads"][arm] = dict(n=len(wins), bridge_preferred=k / max(1, len(wins)), ci=exact_ci(k, len(wins)) if wins else None,
                                  p=float(binomtest(k, len(wins), 0.5, alternative="greater").pvalue) if wins else None)
    write("result.json", out, OUT, indent=1)
    write_js(VIZ_DATA / "blind.js", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["deal", "score"])
    ap.add_argument("--key", type=Path, required=True)
    a = ap.parse_args()
    {"deal": deal, "score": score}[a.action](a.key.expanduser())
