"""Prepare MOE 教育部臺灣台語常用詞辭典 audio for Coqui-TTS training.

Focuses on leku (例句 — full example sentences). sutiau (詞條 — single
words / 1-3 sec clips) is supported via --include-sutiau but skipped by
default because single-word clips don't help VITS learn sentence prosody.

⚠️  LICENSE: CC-BY-NC-ND 2.5 Taiwan. Internal/research use only.

Pipeline:
    1. Parse kautian.ods (master dictionary) to build a {audio_id: Tâi-lô}
       lookup. Falls back to scanning filename↔.txt sidecars if no .ods.
    2. Match each leku/*.wav (and optionally sutiau/*.wav) to its Tâi-lô.
    3. Resample 22-44 kHz → 16 kHz mono.
    4. Apply tailo_basic_cleaner.
    5. Output metadata_moe_{train,dev}.csv with speaker_id "moe_leku"
       (and "moe_sutiau" if enabled).

Usage:
    python recipes/taigi/data/prepare_moe.py [--include-sutiau]
"""

import argparse
import random
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MOE_DIR = Path("data/moe_sutian")
TARGET_SR = 16000

# Likely column headers in kautian.ods for audio ID and Tâi-lô text.
AUDIO_ID_COL_CANDIDATES = (
    "音檔", "音檔ID", "audio_id", "audio", "id", "編號", "音檔編號",
)
TAILO_COL_CANDIDATES = (
    "羅馬字", "台羅", "Tâi-lô", "Tai-lo", "TL", "Lo-má-jī", "lomaji",
)
LEKU_TEXT_COL_CANDIDATES = (
    "例句", "例句羅馬字", "例句台羅", "例句Tâi-lô", "leku",
)


def _load_kautian_index(ods_path: Path) -> dict[str, str] | None:
    """Parse kautian.ods → {audio_id: tai_lo_text}.

    Returns None if the .ods isn't there or can't be parsed; caller should
    fall back to filename-only mode.
    """
    if not ods_path.exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not available, skipping .ods parse")
        return None
    try:
        sheets = pd.read_excel(ods_path, sheet_name=None, engine="odf")
    except Exception as exc:
        print(f"  could not parse {ods_path}: {exc}")
        print("  (try `pip install odfpy`)")
        return None

    index: dict[str, str] = {}
    for sheet_name, df in sheets.items():
        norm_cols = {
            unicodedata.normalize("NFC", str(c)).lower(): c for c in df.columns
        }
        audio_col = next(
            (norm_cols[unicodedata.normalize("NFC", c).lower()]
             for c in AUDIO_ID_COL_CANDIDATES
             if unicodedata.normalize("NFC", c).lower() in norm_cols),
            None,
        )
        # Prefer leku-specific column if present (example-sentence Tâi-lô),
        # else fall back to single-word Tâi-lô.
        tailo_col = next(
            (norm_cols[unicodedata.normalize("NFC", c).lower()]
             for c in LEKU_TEXT_COL_CANDIDATES + TAILO_COL_CANDIDATES
             if unicodedata.normalize("NFC", c).lower() in norm_cols),
            None,
        )
        if audio_col is None or tailo_col is None:
            continue
        for _, row in df.iterrows():
            aid = row.get(audio_col)
            tl = row.get(tailo_col)
            if isinstance(aid, str) and isinstance(tl, str):
                index[aid.strip()] = tl.strip()
    print(f"  parsed {ods_path.name}: {len(index)} entries indexed")
    return index or None


def _resolve_tailo(wav_path: Path, index: dict[str, str] | None) -> str | None:
    """Look up Tâi-lô for one wav file."""
    stem = wav_path.stem
    if index and stem in index:
        return index[stem]
    # Some MOE releases ship sidecar .txt with Tâi-lô next to each wav.
    sidecar = wav_path.with_suffix(".txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--moe-dir", type=Path, default=DEFAULT_MOE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_MOE_DIR)
    p.add_argument(
        "--include-sutiau", action="store_true",
        help="Also process single-word clips (default off — low ROI for prosody)",
    )
    p.add_argument("--dev-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.moe_dir.exists():
        sys.exit(
            f"error: {args.moe_dir} not found. Run:\n"
            f"  python recipes/taigi/data/download_moe.py"
        )

    # Deferred imports.
    try:
        import librosa
        import soundfile as sf
        from tqdm import tqdm
    except ImportError as exc:
        sys.exit(f"missing dependency: {exc.name}. Run `pip install -e .`")

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

    index = _load_kautian_index(args.moe_dir / "kautian.ods")

    out_clips = args.out_dir / "clips_wav"
    out_clips.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Path]] = []
    # Leku (例句) is the primary source.
    leku_root = next(args.moe_dir.rglob("leku*"), None)
    if leku_root and leku_root.is_dir():
        sources.append(("moe_leku", leku_root))
    if args.include_sutiau:
        sutiau_root = next(args.moe_dir.rglob("sutiau*"), None)
        if sutiau_root and sutiau_root.is_dir():
            sources.append(("moe_sutiau", sutiau_root))

    if not sources:
        sys.exit(
            f"error: no leku/ or sutiau/ dir under {args.moe_dir}.\n"
            f"  Did download_moe.py finish extracting?"
        )

    rng = random.Random(args.seed)
    train_rows: list[str] = []
    dev_rows: list[str] = []
    stats = {"total": 0, "ok": 0, "no_tailo": 0, "decode_fail": 0}

    for spk_id, root in sources:
        wavs = sorted(root.rglob("*.wav"))
        print(f"\n[{spk_id}] {len(wavs)} clips under {root}")
        rows: list[str] = []
        for wav in tqdm(wavs, desc=spk_id):
            stats["total"] += 1
            tl_raw = _resolve_tailo(wav, index)
            if not tl_raw:
                stats["no_tailo"] += 1
                continue
            cleaned = tailo_basic_cleaner(tl_raw)
            if not cleaned or len(cleaned) < 2:
                stats["no_tailo"] += 1
                continue
            dst_name = f"{spk_id}_{wav.stem}.wav"
            dst = out_clips / dst_name
            if not dst.exists() or dst.stat().st_size == 0:
                try:
                    audio, _ = librosa.load(str(wav), sr=TARGET_SR, mono=True)
                    sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
                except Exception:
                    stats["decode_fail"] += 1
                    continue
            rows.append(f"clips_wav/{dst_name}|{cleaned}|{spk_id}")
            stats["ok"] += 1

        rng.shuffle(rows)
        n_dev = max(1, int(len(rows) * args.dev_frac))
        dev_rows.extend(rows[:n_dev])
        train_rows.extend(rows[n_dev:])

    train_path = args.out_dir / "metadata_moe_train.csv"
    dev_path = args.out_dir / "metadata_moe_dev.csv"
    train_path.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    dev_path.write_text("\n".join(dev_rows) + "\n", encoding="utf-8")

    print()
    print("=== summary ===")
    print(f"total wavs    : {stats['total']}")
    print(f"kept          : {stats['ok']}")
    print(f"  train       : {len(train_rows)}  → {train_path}")
    print(f"  dev         : {len(dev_rows)}  → {dev_path}")
    print(f"skipped       : no_tailo={stats['no_tailo']}, decode_fail={stats['decode_fail']}")
    print(f"⚠️  License : CC-BY-NC-ND 2.5 — internal/research use only.")


if __name__ == "__main__":
    main()
