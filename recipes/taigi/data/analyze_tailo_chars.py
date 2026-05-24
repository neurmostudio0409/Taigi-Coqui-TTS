"""Scan prepared metadata for Tâi-lô character distribution.

Reports:
  - All unique characters (NFC and NFD views)
  - Frequency counts
  - Combining marks (tone diacritics) used
  - Suspicious / non-Tâi-lô characters that may leak from source

Usage:
    python recipes/taigi/data/analyze_tailo_chars.py
"""

import argparse
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Windows consoles default to cp950 / cp1252 and can't print Tâi-lô diacritics.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DIR = Path("data/common_voice_nan_tw/prepared")
SPLITS = ("train", "dev", "test", "cloning_eval")


def char_repr(c: str) -> str:
    name = unicodedata.name(c, "?")
    cat = unicodedata.category(c)
    return f"U+{ord(c):04X}  {cat}  {name}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()

    nfc_counter: Counter = Counter()
    nfd_counter: Counter = Counter()
    combining_counter: Counter = Counter()
    total_chars = 0

    for split in SPLITS:
        path = args.prepared_dir / f"metadata_{split}.csv"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            tailo = parts[1]
            total_chars += len(tailo)
            for ch in tailo:
                nfc_counter[ch] += 1
            for ch in unicodedata.normalize("NFD", tailo):
                nfd_counter[ch] += 1
                if unicodedata.combining(ch):
                    combining_counter[ch] += 1

    print(f"=== NFC view ({len(nfc_counter)} unique chars, {total_chars} total chars) ===")
    for ch, n in nfc_counter.most_common():
        print(f"  {n:>7}  {ch!r:<8}  {char_repr(ch)}")

    print(f"\n=== NFD view ({len(nfd_counter)} unique chars) ===")
    for ch, n in nfd_counter.most_common():
        flag = "  ← combining" if unicodedata.combining(ch) else ""
        print(f"  {n:>7}  {ch!r:<8}  {char_repr(ch)}{flag}")

    print(f"\n=== Combining marks ({len(combining_counter)} unique) ===")
    for ch, n in combining_counter.most_common():
        print(f"  {n:>7}  {char_repr(ch)}")

    # Suspicious characters: anything outside [a-zA-Z0-9 .,?!:;'\"()-] and known Tâi-lô combining marks
    safe_base = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789 .,?!:;'\"()-—–")
    known_tailo_combining = {
        "̀",  # grave  — tone 3
        "́",  # acute  — tone 2
        "̂",  # circumflex — tone 5
        "̄",  # macron — tone 7
        "̌",  # caron  — tone 6 (rare)
        "̍",  # vertical line above — tone 8 (入聲)
        "͘",  # combining dot above right — POJ "o͘"
        "̃",  # tilde — POJ nasalization (rare in Tâi-lô)
    }
    known_other = {"ⁿ"}  # ⁿ superscript n (Tâi-lô nasalization)
    suspicious: dict[str, int] = {}
    for ch, n in nfd_counter.items():
        if ch in safe_base:
            continue
        if ch in known_tailo_combining:
            continue
        if ch in known_other:
            continue
        suspicious[ch] = n

    if suspicious:
        print(f"\n=== Suspicious chars ({len(suspicious)}) — investigate ===")
        for ch, n in sorted(suspicious.items(), key=lambda x: -x[1]):
            print(f"  {n:>7}  {ch!r:<8}  {char_repr(ch)}")
    else:
        print("\n=== No suspicious characters — clean Tâi-lô set ===")


if __name__ == "__main__":
    main()
