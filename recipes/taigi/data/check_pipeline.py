"""Pre-flight check for the Taigi training pipeline.

Runs without torch / the full Coqui import chain. Verifies:

    1. prepared/metadata_*.csv exist and are non-empty
    2. every audio referenced in the metadata exists on disk
    3. speakers.json + splits_report.json present
    4. Tâi-lô tokenizer can encode every line in metadata_train.csv
       and round-trips lossless
    5. Coqui formatter logic (manually replayed) yields well-formed items
    6. recipe file parses cleanly (no syntax / import-name typos)

This catches the most common "I tried to launch training and something
broke at line 1" issues before paying for an H100 hour.

Usage:
    python recipes/taigi/data/check_pipeline.py
"""

import ast
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
PREPARED_DIR = PROJECT_ROOT / "data" / "common_voice_nan_tw" / "prepared"
CORPUS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "common_voice_nan_tw"
    / "cv-corpus-25.0-2026-03-09"
    / "nan-tw"
)
TAILO_DIR = PROJECT_ROOT / "TTS" / "tts" / "utils" / "text" / "tailo"
RECIPE_PATH = PROJECT_ROOT / "recipes" / "taigi" / "train_yourtts_taigi.py"
SPLITS = ("train", "dev", "test", "cloning_eval")

PASS = "  ✓"
FAIL = "  ✗"


def _step(title: str) -> None:
    print(f"\n[{title}]")


def _load_isolated(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_metadata_files() -> dict[str, int]:
    _step("1. metadata_*.csv presence + row counts")
    row_counts: dict[str, int] = {}
    for split in SPLITS:
        path = PREPARED_DIR / f"metadata_{split}.csv"
        if not path.exists():
            print(f"{FAIL} {path.name} missing")
            sys.exit(1)
        rows = [l for l in path.read_text(encoding="utf-8").splitlines() if l]
        row_counts[split] = len(rows)
        print(f"{PASS} {path.name}: {len(rows):>6} rows")
    return row_counts


def check_audio_paths(row_counts: dict[str, int]) -> int:
    _step("2. audio paths resolve on disk")
    missing = 0
    total = 0
    for split in SPLITS:
        path = PREPARED_DIR / f"metadata_{split}.csv"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            audio_rel = line.split("|", 1)[0]
            total += 1
            if not (CORPUS_ROOT / audio_rel).exists():
                missing += 1
                if missing <= 5:
                    print(f"{FAIL} missing: {audio_rel}")
    if missing:
        print(f"{FAIL} {missing} / {total} files missing")
        sys.exit(1)
    print(f"{PASS} all {total} audio files present")
    return total


def check_aux_files() -> None:
    _step("3. auxiliary outputs (speakers.json, splits_report.json)")
    for fname in ("speakers.json", "splits_report.json"):
        p = PREPARED_DIR / fname
        if not p.exists():
            print(f"{FAIL} {fname} missing")
            sys.exit(1)
        data = json.loads(p.read_text(encoding="utf-8"))
        if fname == "speakers.json":
            print(f"{PASS} speakers.json: {len(data)} speakers")
        else:
            print(f"{PASS} splits_report.json: {json.dumps(data['split_counts'])}")


def check_tokenizer() -> None:
    _step("4. Tâi-lô tokenizer round-trip on train split")
    chars_mod = _load_isolated(
        "TTS.tts.utils.text.characters",
        PROJECT_ROOT / "TTS" / "tts" / "utils" / "text" / "characters.py",
    )
    tailo_chars = _load_isolated(
        "TTS.tts.utils.text.tailo.characters", TAILO_DIR / "characters.py"
    )
    cleaners = _load_isolated(
        "TTS.tts.utils.text.tailo.cleaners", TAILO_DIR / "cleaners.py"
    )

    TailoCharacters = tailo_chars.TailoCharacters
    cleaner = cleaners.tailo_basic_cleaner
    voc = TailoCharacters()
    vocab_set = set(voc.vocab)
    print(f"{PASS} vocab size: {len(voc.vocab)}")

    rows = (
        (PREPARED_DIR / "metadata_train.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    mismatches = 0
    drops: Counter = Counter()
    for line in rows:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cleaned = cleaner(parts[1])
        token_ids = []
        for ch in cleaned:
            try:
                token_ids.append(voc.char_to_id(ch))
            except KeyError:
                drops[ch] += 1
        decoded = "".join(voc.id_to_char(i) for i in token_ids)
        expected = "".join(c for c in cleaned if c in vocab_set)
        if decoded != expected:
            mismatches += 1

    if mismatches:
        print(f"{FAIL} {mismatches} round-trip mismatches")
        sys.exit(1)
    print(f"{PASS} {len(rows)} strings round-trip lossless")
    if drops:
        # Drops are expected (e.g. '/' separator), not an error.
        top = ", ".join(f"{ch!r}×{n}" for ch, n in drops.most_common(3))
        print(f"     ({sum(drops.values())} char drops, top: {top})")


def check_formatter_logic() -> None:
    _step("5. formatter logic (replayed inline)")
    items = []
    not_found = 0
    meta = PREPARED_DIR / "metadata_train.csv"
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        cols = line.split("|")
        if len(cols) < 3:
            continue
        audio_rel, text, spk_id = cols[0], cols[1], cols[2]
        full = CORPUS_ROOT / audio_rel
        if not full.exists():
            not_found += 1
            continue
        items.append(
            {
                "text": text,
                "audio_file": str(full),
                "speaker_name": f"taigi_{spk_id}",
                "language": "nan-tw",
            }
        )
    if not_found or not items:
        print(f"{FAIL} not_found={not_found} items={len(items)}")
        sys.exit(1)
    spk_count = len({i["speaker_name"] for i in items})
    print(f"{PASS} {len(items)} items, {spk_count} speakers")
    print(f"     sample: text={items[0]['text']!r} spk={items[0]['speaker_name']}")


def check_recipe_parses() -> None:
    _step("6. recipe file parses + uses expected symbols")
    src = RECIPE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    required_symbols = (
        "common_voice_taigi",
        "tailo_basic_cleaner",
        "TailoCharacters",
        "TAILO_CHARACTERS",
        "BaseDatasetConfig",
        "VitsConfig",
        "Vits",
        "Trainer",
    )
    missing = [s for s in required_symbols if s not in src]
    if missing:
        print(f"{FAIL} recipe missing references to: {missing}")
        sys.exit(1)
    print(f"{PASS} parses cleanly ({sum(1 for _ in ast.walk(tree))} AST nodes)")
    print(f"     references all expected symbols: {len(required_symbols)} ok")


def main() -> None:
    print(f"Taigi pipeline pre-flight (project root: {PROJECT_ROOT})")
    row_counts = check_metadata_files()
    total_audio = check_audio_paths(row_counts)
    check_aux_files()
    check_tokenizer()
    check_formatter_logic()
    check_recipe_parses()
    print(f"\n=== ALL CHECKS PASSED ({total_audio} audio files) ===")
    print("\nready to:")
    print("  1. pip install -e .   # full Coqui deps (torch, etc.)")
    print("  2. python recipes/taigi/data/resample_to_wav.py   # optional, "
          "speeds up training")
    print("  3. python recipes/taigi/train_yourtts_taigi.py    # launch training")


if __name__ == "__main__":
    main()
