"""Round-trip test for the Tâi-lô tokenizer over the prepared corpus.

Pipeline tested:
    raw_sentence
       → tailo_basic_cleaner   (normalize: NFD, lowercase, fullwidth, drop CJK)
       → char_to_id            (cleaned → token ids)
       → id_to_char            (token ids → string)
       == cleaned   ← invariant: encode/decode must be lossless

We bypass the full ``TTS.tts.utils.text`` package __init__ (which eagerly
imports every language phonemizer and pulls in bangla/japanese/etc.) by
loading the two files we actually exercise via importlib.

Usage:
    python recipes/taigi/data/test_tailo_tokenizer.py
"""

import importlib.util
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TAILO_DIR = PROJECT_ROOT / "TTS" / "tts" / "utils" / "text" / "tailo"
CHARS_FILE = PROJECT_ROOT / "TTS" / "tts" / "utils" / "text" / "characters.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Order matters: characters.py is imported by tailo/characters.py via the
# package path "TTS.tts.utils.text.characters". We register it under that
# name first so the subsequent ``from TTS.tts.utils.text.characters import
# BaseCharacters`` succeeds without triggering the package __init__.
sys.path.insert(0, str(PROJECT_ROOT))
_chars_mod = _load("TTS.tts.utils.text.characters", CHARS_FILE)
tailo_chars_mod = _load(
    "TTS.tts.utils.text.tailo.characters", TAILO_DIR / "characters.py"
)
cleaners_mod = _load(
    "TTS.tts.utils.text.tailo.cleaners", TAILO_DIR / "cleaners.py"
)

TailoCharacters = tailo_chars_mod.TailoCharacters
tailo_basic_cleaner = cleaners_mod.tailo_basic_cleaner

PREPARED_DIR = Path("data/common_voice_nan_tw/prepared")
SPLITS = ("train", "dev", "test", "cloning_eval")


def main() -> None:
    chars = TailoCharacters()
    vocab = chars.vocab
    print(f"vocab size: {len(vocab)}")
    print(f"vocab: {vocab!r}\n")

    total = 0
    mismatches = 0
    mismatch_examples: list[tuple[str, str, str]] = []
    char_drops: Counter = Counter()
    drop_examples: dict[str, str] = {}

    for split in SPLITS:
        path = PREPARED_DIR / f"metadata_{split}.csv"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            raw = parts[1]
            cleaned = tailo_basic_cleaner(raw)

            token_ids = []
            for ch in cleaned:
                try:
                    token_ids.append(chars.char_to_id(ch))
                except KeyError:
                    char_drops[ch] += 1
                    drop_examples.setdefault(ch, raw)

            decoded = "".join(chars.id_to_char(i) for i in token_ids)
            expected = "".join(c for c in cleaned if c in vocab)

            total += 1
            if decoded != expected:
                mismatches += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append((raw, cleaned, decoded))

    print("=== round-trip stats ===")
    print(f"strings tested:        {total}")
    print(f"perfect round-trips:   {total - mismatches}")
    print(f"mismatches:            {mismatches}")
    print(f"unique chars dropped:  {len(char_drops)}")

    if char_drops:
        print("\n=== dropped chars (not in vocab) ===")
        for ch, n in char_drops.most_common():
            cat = unicodedata.category(ch)
            name = unicodedata.name(ch, "?")
            sample = drop_examples[ch][:50]
            print(f"  {n:>5}  U+{ord(ch):04X} {cat} {name!s:<35}  e.g. {sample!r}")

    if mismatch_examples:
        print("\n=== mismatch examples ===")
        for raw, cleaned, decoded in mismatch_examples:
            print(f"  raw:     {raw!r}")
            print(f"  cleaned: {cleaned!r}")
            print(f"  decoded: {decoded!r}\n")
    else:
        print("\nALL ROUND-TRIPS LOSSLESS ✓")


if __name__ == "__main__":
    main()
