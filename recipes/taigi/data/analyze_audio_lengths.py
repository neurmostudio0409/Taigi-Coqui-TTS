"""Audit audio-clip length distribution across all prepared corpora.

Helps you decide TAIGI_MIN_AUDIO_LEN / TAIGI_MAX_AUDIO_LEN. Reports:
  - Per-corpus: total clips, min/median/max, fraction in each bucket
  - Combined: how many clips a given (min, max) filter would keep

Reads metadata_*.csv produced by prepare_*.py scripts. Audio file paths in the
metadata are relative to that dataset's root.

Usage:
    python recipes/taigi/data/analyze_audio_lengths.py
    python recipes/taigi/data/analyze_audio_lengths.py --bins 0.5,1,2,3,5,10,15
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# (corpus_label, metadata_path, audio_root) tuples.
# audio paths in metadata are relative to audio_root.
CORPORA = [
    ("cv_nan_tw_25",
     PROJECT_ROOT / "data" / "common_voice_nan_tw" / "prepared" / "metadata_train_wav.csv",
     PROJECT_ROOT / "data" / "common_voice_nan_tw" / "cv-corpus-25.0-2026-03-09" / "nan-tw"),
    ("suisiann",
     PROJECT_ROOT / "data" / "suisiann" / "metadata_suisiann_train.csv",
     PROJECT_ROOT / "data" / "suisiann"),
    ("tat_tts",
     PROJECT_ROOT / "data" / "tat_tts" / "metadata_tat_train.csv",
     PROJECT_ROOT / "data" / "tat_tts"),
    ("moe_leku",
     PROJECT_ROOT / "data" / "moe_sutian" / "metadata_moe_train.csv",
     PROJECT_ROOT / "data" / "moe_sutian"),
]


def _audio_seconds(path: Path) -> float | None:
    """Probe audio duration without decoding full file (uses soundfile)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bins", default="0.5,1,2,3,5,10,15",
        help="Comma-separated bin edges in seconds",
    )
    args = p.parse_args()

    bin_edges = sorted(float(x) for x in args.bins.split(","))
    # buckets: <bin0, [bin0,bin1), [bin1,bin2), ..., >= last
    bucket_labels = (
        [f"<{bin_edges[0]:.1f}"]
        + [f"[{a:.1f},{b:.1f})" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
        + [f">={bin_edges[-1]:.1f}"]
    )

    def _bucket(sec: float) -> int:
        for i, edge in enumerate(bin_edges):
            if sec < edge:
                return i
        return len(bin_edges)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    per_corpus: dict[str, dict] = {}
    all_secs: list[float] = []

    for label, meta_path, audio_root in CORPORA:
        if not meta_path.exists():
            print(f"skip {label}: {meta_path} not found")
            continue
        rows = [
            l for l in meta_path.read_text(encoding="utf-8").splitlines() if l
        ]
        secs = []
        for line in tqdm(rows, desc=label):
            parts = line.split("|")
            if len(parts) < 1:
                continue
            audio_path = audio_root / parts[0]
            dur = _audio_seconds(audio_path)
            if dur is not None:
                secs.append(dur)

        if not secs:
            continue
        secs_sorted = sorted(secs)
        n = len(secs_sorted)
        counts = [0] * (len(bin_edges) + 1)
        for s in secs:
            counts[_bucket(s)] += 1
        per_corpus[label] = {
            "n": n,
            "min": secs_sorted[0],
            "median": secs_sorted[n // 2],
            "max": secs_sorted[-1],
            "total_hr": sum(secs) / 3600,
            "counts": counts,
        }
        all_secs.extend(secs)

    # Print per-corpus.
    print()
    print("=" * 90)
    print(f"{'corpus':<15} {'n':>7} {'min':>6} {'med':>6} {'max':>7} {'hr':>6}  buckets")
    print(f"{'':<15} {'':>7} {'':>6} {'':>6} {'':>7} {'':>6}  " + " ".join(f"{b:>9}" for b in bucket_labels))
    print("=" * 90)
    for label, stats in per_corpus.items():
        bucket_str = " ".join(
            f"{c:>9}" if c > 0 else f"{'.':>9}" for c in stats["counts"]
        )
        print(
            f"{label:<15} {stats['n']:>7} {stats['min']:>6.2f} {stats['median']:>6.2f} "
            f"{stats['max']:>7.2f} {stats['total_hr']:>6.2f}  {bucket_str}"
        )

    # Combined.
    if all_secs:
        n = len(all_secs)
        total_hr = sum(all_secs) / 3600
        print()
        print(f"combined: {n} clips / {total_hr:.2f} hr")

        # Filter simulator
        print()
        print("If you set TAIGI_MIN_AUDIO_LEN to:")
        for cutoff in (0.5, 1.0, 1.5, 2.0, 3.0):
            kept = sum(1 for s in all_secs if s >= cutoff)
            kept_hr = sum(s for s in all_secs if s >= cutoff) / 3600
            dropped_pct = 100 * (1 - kept / n)
            print(
                f"  {cutoff:.1f} s  →  keep {kept:>6} clips / {kept_hr:5.2f} hr  "
                f"(drop {dropped_pct:.1f}%)"
            )


if __name__ == "__main__":
    main()
