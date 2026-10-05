"""Paragraph-bounded passages from clean_txt/: short paragraphs merge forward, long ones split at sentence ends.

Writes data/passages.json (the corpus), data/twins.json (held-out duplicate text) and data/chunk_report.json.
"""
import re
import string
from collections import Counter

from src.core.common import uid, write
from src.core.config import (BOOKS, CLEAN_TXT, DATA, DROP_BELOW, EXCLUDE,
                             MAX_WORDS, MIN_WORDS, NO_HEADINGS)

DROP = [
    re.compile(r"^Translated by "),
    re.compile(r"^[*†‡§¶]\s"),               # footnotes
    re.compile(r"^[-_·—\s*]+$"),             # rules and ornaments
    re.compile(r"^\d{1,3}$"),                # aphorism numbers
]
SUPERSCRIPT = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+|fn\d+")
SENTENCE = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][”’\"')]))\s+(?=[“\"'A-ZÀ-Ý—])")
ROMAN = re.compile(r"^[IVXLC]+$")


def is_heading(line):
    words = line.split()
    if len(words) > 9:
        return False
    return not re.search(r"[.!?…:;,”\"’)]$", line)


def section_name(line):
    if ROMAN.match(line):
        return line
    return string.capwords(line) if line.isupper() else line


def book_lines(num, meta):
    text = (CLEAN_TXT / f"{num}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lo = 0
    if "start" in meta:
        marker, nth = meta["start"]
        lo = [i for i, l in enumerate(lines) if l == marker][nth - 1]
    hi = len(lines)
    if "end" in meta:
        hits = [i for i, l in enumerate(lines) if l == meta["end"] and i > lo]
        hi = hits[0] + (1 if meta.get("end_inclusive") else 0)
    return lines[lo:hi]


def cut_twin(lines, twin):
    """The book without the twin's stretch, and the stretch itself (from its title line to the next work's)."""
    s = lines.index(twin["start"])
    e = next(i for i in range(s + 1, len(lines)) if lines[i] == twin["end"])
    return lines[:s] + lines[e:], lines[s:e]


def split_long(par):
    words = len(par.split())
    if words <= MAX_WORDS:
        return [par]
    sents = SENTENCE.split(par)
    parts = max(2, round(words / 200))
    target = words / parts
    out, cur, n = [], [], 0
    for s in sents:
        cur.append(s)
        n += len(s.split())
        if n >= target and len(out) < parts - 1:
            out.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        out.append(" ".join(cur))
    return out


def passages(lines, key):
    """Units of text with their section; fragments under DROP_BELOW that cannot join the previous unit are counted."""
    section, buf, units, dropped = None, [], [], []

    def flush():
        nonlocal buf
        if not buf:
            return
        text = " ".join(buf)
        n = len(text.split())
        if n >= DROP_BELOW:
            units.append(dict(section=section, text=text))
        elif units and units[-1]["section"] == section and len(units[-1]["text"].split()) + n <= MAX_WORDS + 60:
            units[-1]["text"] += " " + text
        else:
            dropped.append(text)
        buf = []

    for raw in lines:
        line = SUPERSCRIPT.sub("", raw).strip()
        if not line or any(p.search(line) for p in DROP):
            continue
        if is_heading(line) and key not in NO_HEADINGS:
            flush()
            section = section_name(line)
            continue
        for part in split_long(line):
            if buf and len(" ".join(buf).split()) + len(part.split()) > MAX_WORDS:
                flush()
            buf.append(part)
            if len(" ".join(buf).split()) >= MIN_WORDS:
                flush()
    flush()
    return units, dropped


def rows_of(units, author, key, title, year):
    return [dict(id=f"{key}_{i:04d}", uid=uid(u["text"]), author=author, book=key, title=title, year=year,
                 section=u["section"], index=i, position=round(i / max(1, len(units) - 1), 4),
                 words=len(u["text"].split()), text=u["text"]) for i, u in enumerate(units)]


def excluded():
    if not EXCLUDE.exists():
        return {}
    pairs = [l.split(None, 1) for l in EXCLUDE.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return {p[0]: (p[1] if len(p) > 1 else "") for p in pairs}


def main():
    DATA.mkdir(exist_ok=True)
    rows, twins, report = [], [], {}
    for num, meta in BOOKS.items():
        lines = book_lines(num, meta)
        if "twin" in meta:
            lines, twin_lines = cut_twin(lines, meta["twin"])
            t = meta["twin"]
            units, dropped = passages(twin_lines, t["key"])
            twins += rows_of(units, meta["author"], t["key"], t["title"], meta["year"])
            report[t["key"]] = dict(passages=len(units), dropped=len(dropped), dropped_words=sum(len(d.split()) for d in dropped))
        units, dropped = passages(lines, meta["key"])
        rows += rows_of(units, meta["author"], meta["key"], meta["title"], meta["year"])
        report[meta["key"]] = dict(passages=len(units), dropped=len(dropped), dropped_words=sum(len(d.split()) for d in dropped),
                                   dropped_examples=dropped[:5])
    # leave out passages by decision; the next passage is marked so no false "next paragraph" is formed across the gap
    ex = excluded()
    kept = []
    for r in rows:
        if r["uid"] in ex:
            if kept and kept[-1]["book"] == r["book"]:
                kept[-1]["gap_after"] = True
            continue
        kept.append(r)
    report["excluded"] = [dict(uid=u, reason=why, found=any(r["uid"] == u for r in rows)) for u, why in ex.items()]
    dup = [u for u, c in Counter(r["uid"] for r in kept).items() if c > 1]
    report["identical_passages"] = dup
    for key, r in report.items():
        if isinstance(r, dict):
            print(f"{key:20s} {r['passages']:5d} passages  {r['dropped']:3d} fragments dropped ({r['dropped_words']} words)")
    by = Counter(r["author"] for r in kept)
    print(f"total {len(kept)} {dict(by)}; twins {len(twins)}; excluded {len(rows) - len(kept)}; identical {len(dup)}")
    write("passages.json", kept)
    write("twins.json", twins)
    write("chunk_report.json", report, indent=1)


if __name__ == "__main__":
    main()
