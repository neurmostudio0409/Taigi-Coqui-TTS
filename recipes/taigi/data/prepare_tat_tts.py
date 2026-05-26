"""Prepare TAT-TTS corpus for Coqui-TTS training.

Reads each clip's JSON metadata, extracts the Tâi-lô (台羅) field, resamples
the 48 kHz / 24-bit WAV down to 16 kHz mono, applies the same tailo_basic_cleaner
as CV nan-tw, and writes LJSpeech-style metadata.

Speaker IDs assigned: tat_F1 / tat_F2 / tat_M1 / tat_M2 (→ formatter outputs
speaker_name 'taigi_tat_F1' etc.).

The TAT-TTS per-clip JSON layout (per upstream docs) commonly has fields like:
  - "音檔" or "audio_file" — wav filename
  - "TL" / "台羅" / "Tai-lo" — Tâi-lô romanization
  - "POJ" / "白話字" — POJ romanization (alternative)
  - "漢字" — Han characters
  - "speaker" / "language" — metadata

This script auto-detects which keys are present and picks the best Tâi-lô
field. If your TAT-TTS variant uses different keys, edit TAILO_KEY_CANDIDATES.

Usage:
    python recipes/taigi/data/prepare_tat_tts.py
"""

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_TAT_DIR = Path("data/tat_tts")
DEFAULT_OUT_DIR = Path("data/tat_tts")
TARGET_SR = 16000

TAILO_KEY_CANDIDATES = (
    "TL", "Tai-lo", "tai-lo", "tailo", "台羅", "Tâi-lô", "tâi-lô",
    "Lo-má-jī", "Lo-mâ-jī", "lomaji",
)
AUDIO_KEY_CANDIDATES = (
    "音檔", "音檔名", "audio_file", "audio", "wav", "filename", "file",
)


def _pick_key(d: dict, candidates: tuple[str, ...]) -> str | None:
    """Case- and NFC-insensitive key lookup."""
    norm = {unicodedata.normalize("NFC", k).lower(): k for k in d}
    for c in candidates:
        target = unicodedata.normalize("NFC", c).lower()
        if target in norm:
            return norm[target]
    return None


def _find_json_audio_pairs(speaker_dir: Path) -> list[tuple[Path, Path]]:
    """Find all (json, wav) pairs under a speaker directory."""
    pairs = []
    for json_path in speaker_dir.rglob("*.json"):
        wav = json_path.with_suffix(".wav")
        if wav.exists():
            pairs.append((json_path, wav))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tat-dir", type=Path, default=DEFAULT_TAT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dev-frac", type=float, default=0.02,
                   help="Per-speaker held-out fraction (default 2%)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.tat_dir.exists():
        sys.exit(
            f"error: {args.tat_dir} not found. Run:\n"
            f"  python recipes/taigi/data/download_tat_tts.py"
        )

    # Locate the Tâi-lô cleaner (same as for CV / 媠聲).
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    import importlib.util
    cleaner_spec = importlib.util.spec_from_file_location(
        "tailo_cleaners",
        project_root / "TTS" / "tts" / "utils" / "text" / "tailo" / "cleaners.py",
    )
    cleaner_mod = importlib.util.module_from_spec(cleaner_spec)
    cleaner_spec.loader.exec_module(cleaner_mod)
    tailo_basic_cleaner = cleaner_mod.tailo_basic_cleaner

    try:
        import librosa
        import soundfile as sf
        from tqdm import tqdm
    except ImportError as exc:
        sys.exit(f"missing dependency: {exc.name}. Run `pip install -e .` first.")

    out_clips = args.out_dir / "clips_wav"
    out_clips.mkdir(parents=True, exist_ok=True)

    train_rows: list[str] = []
    dev_rows: list[str] = []
    stats = {"total": 0, "ok": 0, "no_tailo": 0, "no_wav": 0, "decode_fail": 0}

    speakers = sorted(
        d for d in args.tat_dir.iterdir()
        if d.is_dir() and d.name in {"F1", "F2", "M1", "M2"}
    )
    if not speakers:
        sys.exit(
            f"error: no F1/F2/M1/M2 subdirs found under {args.tat_dir}.\n"
            f"  Expected layout: {args.tat_dir}/<speaker>/<clip>.wav + <clip>.json"
        )

    rng = random.Random(args.seed)

    for spk_dir in speakers:
        spk_id = f"tat_{spk_dir.name}"
        pairs = _find_json_audio_pairs(spk_dir)
        if not pairs:
            print(f"  [{spk_id}] no (json, wav) pairs found, skipping")
            continue
        print(f"\n[{spk_id}] {len(pairs)} clips")

        spk_rows: list[str] = []
        for json_path, wav_path in tqdm(pairs, desc=spk_id):
            stats["total"] += 1
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                stats["decode_fail"] += 1
                continue

            tailo_key = _pick_key(meta, TAILO_KEY_CANDIDATES)
            if tailo_key is None or not meta.get(tailo_key):
                stats["no_tailo"] += 1
                continue
            tailo_raw = str(meta[tailo_key])
            cleaned = tailo_basic_cleaner(tailo_raw)
            if not cleaned or len(cleaned) < 2:
                stats["no_tailo"] += 1
                continue

            dst_name = f"{spk_id}_{wav_path.stem}.wav"
            dst = out_clips / dst_name
            if not dst.exists() or dst.stat().st_size == 0:
                try:
                    audio, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
                    sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
                except Exception:
                    stats["decode_fail"] += 1
                    continue
            spk_rows.append(f"clips_wav/{dst_name}|{cleaned}|{spk_id}")
            stats["ok"] += 1

        # Per-speaker shuffle + split so dev contains samples from all 4 speakers.
        rng.shuffle(spk_rows)
        n_dev = max(1, int(len(spk_rows) * args.dev_frac))
        dev_rows.extend(spk_rows[:n_dev])
        train_rows.extend(spk_rows[n_dev:])

    train_path = args.out_dir / "metadata_tat_train.csv"
    dev_path = args.out_dir / "metadata_tat_dev.csv"
    train_path.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    dev_path.write_text("\n".join(dev_rows) + "\n", encoding="utf-8")

    print()
    print("=== summary ===")
    print(f"total scanned : {stats['total']}")
    print(f"kept          : {stats['ok']}")
    print(f"  train       : {len(train_rows)}  → {train_path}")
    print(f"  dev         : {len(dev_rows)}  → {dev_path}")
    print(f"skipped       : {{ {', '.join(f'{k}={v}' for k,v in stats.items() if k != 'ok' and k != 'total')} }}")
    print(f"speakers      : taigi_tat_F1 / F2 / M1 / M2")


if __name__ == "__main__":
    main()
