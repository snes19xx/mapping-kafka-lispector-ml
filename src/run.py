"""Run the analysis in order, logging each stage to data/logs/<stage>.log; stops at the first failure.

    python -m src.run                  show the stages
    python -m src.run all              every stage
    python -m src.run --from atlas     atlas and everything after it
    python -m src.run erase atlas      only these, in stage order

Run from analysis/.
"""
import subprocess
import sys
import time

from src.core.config import HERE, LOGS

# The stages are in order, and the first column is the name used on the command line. The second is the module to run, and the third is a short description of what it does.
STAGES = [
    ("clean", "src.prep.clean", "raw_txt -> clean_txt, bracket report"),
    ("chunk", "src.prep.chunk", "passages.json, twins.json (the Muirs' Metamorphosis, held out)"),
    ("embed", "src.embedding.embed", "all readings; cached passages are not encoded again"),
    ("erase", "src.erasure.erase", "LEACE + CORAL per reading; erasure_metrics.json"),
    ("atlas", "src.mapping.atlas", "layouts, constellations, bridges, blind-reading arms"),
    ("refs", "src.mapping.refs", "curated names and close-ups mapped onto this run -> OUT/data/refs.js"),
    ("export", "src.mapping.export", "OUT/data/atlas.js"),
    ("tree", "src.mapping.tree", "OUT/data/tree.js"),
    ("benchmarks", "src.evaluation.benchmarks", "leakage, retrieval, mixing, book signal, nulls -> OUT/data/bench.js"),
    ("stability", "src.evaluation.stability", "UMAP seeds, method variants -> stability.json"),
    ("translation", "src.evaluation.translation", "Bernofsky vs Muir -> translation.json"),
]
MODULE = {s: m for s, m, _ in STAGES}


def run(stage):
    LOGS.mkdir(parents=True, exist_ok=True)
    t = time.time()
    print(f"\n=== {stage}", flush=True)
    with open(LOGS / f"{stage}.log", "w", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable, "-u", "-m", MODULE[stage]], cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
        p.wait()
    print(f"=== {stage}: {'ok' if p.returncode == 0 else 'FAILED'} in {time.time() - t:.0f} s", flush=True)
    return p.returncode == 0


def main(args):
    names = [s for s, _, _ in STAGES]
    if not args:
        for s, _, what in STAGES:
            print(f"{s:12s} {what}")
        return
    if args[0] == "all":
        todo = names
    elif args[0] == "--from":
        todo = names[names.index(args[1]):]
    else:
        unknown = set(args) - set(names)
        if unknown:
            sys.exit(f"unknown stage(s): {', '.join(sorted(unknown))}")
        todo = [s for s in names if s in args]
    for s in todo:
        if not run(s):
            sys.exit(f"stopped at {s}; see {LOGS / (s + '.log')}")


if __name__ == "__main__":
    main(sys.argv[1:])
