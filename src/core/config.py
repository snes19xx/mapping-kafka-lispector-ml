"""Every path and parameter of the analysis."""
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parent
DATA = HERE / "data"
# embeddings by passage uid, one .npz per reading
CACHE = DATA / "cache"
LOGS = DATA / "logs"
# modules the June pieces import: atlas.js, bench.js, tree.js, refs.js
VIZ_DATA = ROOT / "OUT" / "data"
RAW_EPUB, RAW_TXT, CLEAN_TXT = ROOT / "raw_epubs", ROOT / "raw_txt", ROOT / "clean_txt"
CURATION = HERE / "curation.json"
# passage uids left out one per line, "uid  reason"
EXCLUDE = HERE / "exclude.txt"

# ---- corpus
MIN_WORDS, MAX_WORDS, DROP_BELOW = 70, 280, 35

# cleaner: the start of each work in the pandoc text (front matter is cut before it)
CLEAN_START = {
    "1": "IT’S WITH SUCH PROFOUND HAPPINESS", "2": "As he entered New York Harbor", "3": "THE ZÜRAU APHORISMS",
    "4": "BEFORE THE LAW", "5": "The Triumph", "6": "WHEN GREGOR SAMSA WOKE", "7": "LETTERS TO MILENA",
    "8": "I’M SEARCHING, I’M SEARCHING", "9": "All the world began with a yes", "10": "THE MIRACULOUS FISH",
}

# chunker: start = first line of the work (nth occurrence); end = first line after it.
# twin = a stretch of the book held out of the corpus: the Muirs' Metamorphosis inside Glatzer's Complete Stories,
# which duplicates Bernofsky's and is kept apart for the translation experiment.
BOOKS = {
    "1": dict(author="Lispector", title="Água Viva", year=1973, key="agua", end="THIS IS JUST THE BEGINNING"),
    "2": dict(author="Kafka", title="Amerika", year=1927, key="amerika", end="ACKNOWLEDGMENTS"),
    "3": dict(author="Kafka", title="Aphorisms", year=1918, key="aphorisms", end="THE SCHOCKEN KAFKA LIBRARY"),
    "4": dict(author="Kafka", title="The Complete Stories", year=1924, key="kstories",
              twin=dict(start="The Metamorphosis", end="In the Penal Colony", key="metamorphosis_muir",
                        title="The Metamorphosis (tr. Willa and Edwin Muir)")),
    "5": dict(author="Lispector", title="The Complete Stories", year=1977, key="lstories",
              start=("The Triumph", 2), end="Appendix: The Useless Explanation"),
    "6": dict(author="Kafka", title="The Metamorphosis", year=1915, key="metamorphosis", end="AFTERWORD"),
    "7": dict(author="Kafka", title="Letters to Milena", year=1923, key="milena", end="NOTES"),
    "8": dict(author="Lispector", title="The Passion According to G.H.", year=1964, key="gh", end="Translator’s Note"),
    "9": dict(author="Lispector", title="The Hour of the Star", year=1977, key="hour", end="Yes.", end_inclusive=True),
    "10": dict(author="Lispector", title="Too Much of Life (crônicas)", year=1973, key="cronicas", end="The making of"),
}
# continuous texts:
NO_HEADINGS = ("agua", "gh", "hour")

# ---- readings: a model and an instruction. Revisions pin the exact weights used.
MODELS = {
    "mpnet": dict(name="sentence-transformers/all-mpnet-base-v2", revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
                  max_len=384),
    "qwen": dict(name="Qwen/Qwen3-Embedding-0.6B", revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", max_len=512),
}
THEME = ("Instruct: Identify the existential and philosophical theme of this literary passage, "
         "setting aside its plot, characters, setting and prose style\nQuery: ")
NEUTRAL = "Instruct: Identify the theme of this literary passage\nQuery: "
# "qwen" isolates the instruction from the change of model; "qwen-neutral" isolates the existential framing
READINGS = {
    "mpnet": dict(model="mpnet", prompt=None),
    "qwen": dict(model="qwen", prompt=None),
    "qwen-theme": dict(model="qwen", prompt=THEME),
    "qwen-neutral": dict(model="qwen", prompt=NEUTRAL),
}
FINAL = "qwen-theme"
RAW = "mpnet"
SHARD = 256

# ---- erasure, neighbours, bridges
CORAL_DIMS = 256
CSLS_K = 10
NEIGHBOURS = 15
N_SHOWN = 300
# same-author passages in different books above this mpnet cosine are reported as one text
DUPLICATE_COS = 0.85

# ---- map
SEED = 42
UMAP_2D = dict(n_components=2, n_neighbors=50, min_dist=0.1, metric="cosine", init="spectral")
UMAP_10D = dict(n_components=10, n_neighbors=30, min_dist=0.0, metric="cosine")
K_CONSTELLATIONS = 18

# ---- statistics
BOOT, BOOT_SEED = 2000, 7
PERMUTATIONS = 200
SHUFFLES = 20
KS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500]

# ---- stability
UMAP_SEEDS = [42, 1, 2, 3, 4]
CORAL_DIMS_VARIANTS = [128, 512]
CSLS_K_VARIANTS = [5, 20]

# ---- blind reading, version 2 (human readers)
BLIND_SEED = 1924
BLIND_PER_ARM = 35
BLIND_TRIADS = 70
BLIND_ARMS = ("final", "unerased", "cosine", "words", "tfidf", "random")
