"""Stage 0: EPUB -> raw_txt (pandoc) -> clean_txt, with front matter cut and editorial marks removed.

    python -m src.prep.clean            raw_txt -> clean_txt, and data/clean_report.json
    python -m src.prep.clean convert    raw_epubs -> raw_txt first (only when the EPUBs change: pandoc versions differ)

Formerly cleaner.py and raw_epubs/converter.sh; the output is the same text with LF line ends.
"""
import re
import subprocess
import sys
from collections import Counter

from src.core.common import write
from src.core.config import CLEAN_START, CLEAN_TXT, DATA, RAW_EPUB, RAW_TXT

BRACKETS = re.compile(r"\[.*?\]")
# TOC entries like "1. vii"; bracketed images, datelines, elisions and editorial notes; runs of blank lines
PATTERNS = [
    (re.compile(r"^\d+\.\s+[ivxlc\d]+\s*$", re.M), ""),
    (BRACKETS, ""),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def convert():
    for i in range(1, 11):
        subprocess.run(["pandoc", str(RAW_EPUB / f"{i}.epub"), "-t", "plain", "--wrap=none",
                        "-o", str(RAW_TXT / f"{i}.txt")], check=True)
        print(f"converted {i}.epub")


def clean(num):
    text = (RAW_TXT / f"{num}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
    at = text.find(CLEAN_START[num])
    if at != -1:
        text = text[at:]
    removed = Counter(BRACKETS.findall(text))
    for pattern, repl in PATTERNS:
        text = pattern.sub(repl, text)
    (CLEAN_TXT / f"{num}.txt").write_text(text.strip(), encoding="utf-8")
    return dict(start_found=at != -1, brackets_removed=sum(removed.values()), brackets=removed.most_common())


def main():
    if sys.argv[1:] == ["convert"]:
        convert()
    CLEAN_TXT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    report = {str(n): clean(str(n)) for n in range(1, 11)}
    for n, r in report.items():
        print(f"{n:>2}: start {'ok' if r['start_found'] else 'MISSING'}, {r['brackets_removed']} bracketed spans removed")
    write("clean_report.json", report, indent=1)


if __name__ == "__main__":
    main()
