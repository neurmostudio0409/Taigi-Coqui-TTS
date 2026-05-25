"""Prepare the Suí-siann (媠聲) corpus for Coqui-TTS training.

Pipeline:
    1. Read the dataset CSV (auto-locates `SuiSiann.csv` inside the extracted dir).
    2. Resample every clip 44.1 kHz → 16 kHz mono (matches our recipe SAMPLE_RATE).
    3. Apply the same Tâi-lô cleaner used for CV nan-tw — outputs LJSpeech-style
       metadata with format ``clips_wav/xxx.wav|tailo|speaker_id``.
    4. 95/5 train/dev per-speaker split (single speaker, so just shuffled).

Output (under ``data/suisiann/``):
    clips_wav/xxx.wav                 (16 kHz mono)
    metadata_suisiann_train.csv
    metadata_suisiann_dev.csv

Suí-siann is a single speaker (Ng Siù-iông 王秀容) — we assign speaker_id "100"
to keep it numerically distinct from CV nan-tw's 0-93 speaker range.

Usage:
    python recipes/taigi/data/prepare_suisiann.py
"""

import argparse
import csv
import random
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_SUISIANN_DIR = Path("data/suisiann")
SUISIANN_SPEAKER_ID = "100"  # outside CV's 0-93 range → speaker_name "taigi_100"
TARGET_SR = 16000

# Candidate column names for the Tâi-lô romanization column (Suí-siann CSVs
# have varied across releases / community forks).
TAILO_COL_CANDIDATES = (
    "羅馬字", "Tâi-lô", "tai-lo", "tailo", "lomaji", "Lo-má-jī",
    "Lo-má-tsū", "romanization", "Romaji",
)
AUDIO_COL_CANDIDATES = (
    "音檔", "音檔名", "audio", "wav", "filename", "file", "path",
)


def _find_csv(suisiann_dir: Path) -> Path:
    """Locate the metadata CSV inside the extracted Suí-siann directory."""
    candidates = list(suisiann_dir.rglob("SuiSiann*.csv"))
    candidates += list(suisiann_dir.rglob("*.csv"))
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    if not unique:
        sys.exit(f"error: no CSV found under {suisiann_dir}; did you --extract?")
    if len(unique) > 1:
        print(f"  multiple CSVs found, using first: {unique[0]}")
    return unique[0]


def _find_col(header: list[str], candidates: tuple[str, ...]) -> int | None:
    """Case- and NFC-insensitive column name match."""
    norm_header = [unicodedata.normalize("NFC", h.strip()).lower() for h in header]
    for cand in candidates:
        target = unicodedata.normalize("NFC", cand).lower()
        if target in norm_header:
            return norm_header.index(target)
    return None


def _resolve_audio_path(suisiann_dir: Path, filename: str) -> Path | None:
    """Find the audio file given its name, searching nested dirs."""
    direct = suisiann_dir / filename
    if direct.is_file():
        return direct
    # The wav may live in a subfolder (e.g. wav/, audio/).
    matches = list(suisiann_dir.rglob(filename))
    return matches[0] if matches else None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suisiann-dir", type=Path, default=DEFAULT_SUISIANN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_SUISIANN_DIR)
    p.add_argument("--dev-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4, help="Parallel resample jobs")
    args = p.parse_args()

    if not args.suisiann_dir.exists():
        sys.exit(
            f"error: {args.suisiann_dir} not found. Run:\n"
            f"  python recipes/taigi/data/download_suisiann.py --extract"
        )

    csv_path = _find_csv(args.suisiann_dir)
    print(f"using CSV: {csv_path}")

    # Read header to figure out which columns are audio vs Tâi-lô.
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    audio_col = _find_col(header, AUDIO_COL_CANDIDATES)
    tailo_col = _find_col(header, TAILO_COL_CANDIDATES)
    if audio_col is None or tailo_col is None:
        sys.exit(
            f"error: could not identify audio / Tâi-lô columns.\n"
            f"  header     = {header}\n"
            f"  audio_col  = {audio_col}\n"
            f"  tailo_col  = {tailo_col}\n"
            f"  candidates audio = {AUDIO_COL_CANDIDATES}\n"
            f"  candidates Tâi-lô = {TAILO_COL_CANDIDATES}"
        )
    print(f"  audio column #{audio_col}: {header[audio_col]!r}")
    print(f"  Tâi-lô column #{tailo_col}: {header[tailo_col]!r}")
    print(f"  total rows: {len(rows)}")

    # Deferred imports — keeps --help fast.
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        sys.exit(f"missing dependency: {exc.name}. Run `pip install -e .` first.")

    # Reuse the Tâi-lô cleaner so Suí-siann uses the same normalization rules
    # (NFD, lowercase, fullwidth → ASCII, drop CJK) as CV nan-tw.
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

    out_clips = args.out_dir / "clips_wav"
    out_clips.mkdir(parents=True, exist_ok=True)

    # Process every row: resample + extract Tâi-lô.
    items: list[tuple[str, str]] = []  # (relative wav path, cleaned tailo)
    skipped = {"missing_audio": 0, "empty_tailo": 0, "decode_fail": 0}

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    for row in tqdm(rows, desc="prepare"):
        if len(row) <= max(audio_col, tailo_col):
            continue
        audio_name = row[audio_col].strip()
        tailo_raw = row[tailo_col].strip()
        cleaned = tailo_basic_cleaner(tailo_raw)
        if not cleaned or len(cleaned) < 2:
            skipped["empty_tailo"] += 1
            continue

        src = _resolve_audio_path(args.suisiann_dir, audio_name)
        if src is None:
            skipped["missing_audio"] += 1
            continue

        # Resample 44.1k → 16k mono.
        dst_name = Path(audio_name).stem + ".wav"
        dst = out_clips / dst_name
        if not dst.exists() or dst.stat().st_size == 0:
            try:
                audio, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
                sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
            except Exception:
                skipped["decode_fail"] += 1
                continue
        items.append((f"clips_wav/{dst_name}", cleaned))

    # Per-speaker shuffle + split. Single speaker so just shuffle the whole list.
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_dev = max(1, int(len(items) * args.dev_frac))
    dev = items[:n_dev]
    train = items[n_dev:]

    def write_meta(rows: list[tuple[str, str]], path: Path) -> None:
        lines = [f"{p}|{t}|{SUISIANN_SPEAKER_ID}" for p, t in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    train_path = args.out_dir / "metadata_suisiann_train.csv"
    dev_path = args.out_dir / "metadata_suisiann_dev.csv"
    write_meta(train, train_path)
    write_meta(dev, dev_path)

    print()
    print(f"=== summary ===")
    print(f"rows processed   : {len(rows)}")
    print(f"items kept       : {len(items)}")
    print(f"  train          : {len(train)}  → {train_path}")
    print(f"  dev            : {len(dev)}  → {dev_path}")
    print(f"skipped reasons  : {skipped}")
    print(f"speaker_id       : {SUISIANN_SPEAKER_ID}  (→ speaker_name 'taigi_{SUISIANN_SPEAKER_ID}')")


if __name__ == "__main__":
    main()
