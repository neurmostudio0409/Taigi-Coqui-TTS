"""Resample CV nan-tw MP3 clips to 16 kHz mono WAV.

Why pre-convert
---------------
- avoid MP3 decode on every training batch (librosa is slow on MP3)
- catch broken / unreadable files upfront, not 3 epochs in
- before pushing to Vast.ai, a 16 kHz mono WAV corpus is ~4x smaller
  than the original CV MP3 archive (faster upload)

Behavior
--------
- Reads every metadata_{train,dev,test,cloning_eval}.csv to gather the
  set of source MP3 paths (skips audio not referenced by any split).
- Decodes via librosa → writes 16-bit PCM WAV via soundfile.
- Outputs to ``clips_wav/`` (sibling of ``clips/``) — originals untouched.
- Skips files whose WAV already exists (idempotent / resumable).
- Writes ``metadata_*_wav.csv`` mirrors with ``clips_wav/xxx.wav`` paths.

Usage:
    python recipes/taigi/data/resample_to_wav.py [--workers N] [--sr 16000]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CORPUS = Path(
    "data/common_voice_nan_tw/cv-corpus-25.0-2026-03-09/nan-tw"
)
DEFAULT_PREPARED = Path("data/common_voice_nan_tw/prepared")
SPLITS = ("train", "dev", "test", "cloning_eval")


def _convert_one(args: tuple[Path, Path, int]) -> tuple[Path, str | None]:
    """Worker: decode one MP3 → write one WAV. Returns (path, error_or_None)."""
    src, dst, sr = args
    if dst.exists() and dst.stat().st_size > 0:
        return src, None  # already done
    try:
        import librosa
        import soundfile as sf

        audio, _ = librosa.load(str(src), sr=sr, mono=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 16-bit PCM is standard for VITS training
        sf.write(str(dst), audio, sr, subtype="PCM_16")
        return src, None
    except Exception as exc:  # pragma: no cover — surfaces in summary
        return src, f"{type(exc).__name__}: {exc}"


def _gather_jobs(
    prepared_dir: Path, corpus_root: Path, sr: int
) -> tuple[list[tuple[Path, Path, int]], dict[str, list[str]]]:
    """Collect (src_mp3, dst_wav, sr) tuples + per-split rewritten metadata."""
    jobs_by_src: dict[Path, Path] = {}
    rewritten: dict[str, list[str]] = {}

    for split in SPLITS:
        meta_path = prepared_dir / f"metadata_{split}.csv"
        if not meta_path.exists():
            continue
        out_lines: list[str] = []
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            audio_rel, text, speaker_id = parts[0], parts[1], parts[2]
            # clips/foo.mp3 → clips_wav/foo.wav
            src = corpus_root / audio_rel
            new_rel = audio_rel.replace("clips/", "clips_wav/").replace(
                ".mp3", ".wav"
            )
            dst = corpus_root / new_rel
            jobs_by_src.setdefault(src, dst)
            out_lines.append(f"{new_rel}|{text}|{speaker_id}")
        rewritten[split] = out_lines

    jobs = [(s, d, sr) for s, d in jobs_by_src.items()]
    return jobs, rewritten


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel processes (default: CPU-1)",
    )
    args = p.parse_args()

    if not args.corpus_root.exists():
        sys.exit(f"error: corpus root not found: {args.corpus_root}")
    if not args.prepared_dir.exists():
        sys.exit(f"error: prepared dir not found: {args.prepared_dir}")

    print(f"gathering jobs from {args.prepared_dir}...")
    jobs, rewritten = _gather_jobs(args.prepared_dir, args.corpus_root, args.sr)
    pending = [j for j in jobs if not (j[1].exists() and j[1].stat().st_size > 0)]
    already_done = len(jobs) - len(pending)
    print(
        f"  unique audio files: {len(jobs)}  "
        f"already converted: {already_done}  pending: {len(pending)}"
    )

    failed: list[tuple[Path, str]] = []
    if pending:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **k: x  # noqa: E731

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_convert_one, j) for j in pending]
            for fut in tqdm(
                as_completed(futures), total=len(pending), desc="resample"
            ):
                src, err = fut.result()
                if err:
                    failed.append((src, err))

    # Write parallel metadata files (always — fast and ensures latest).
    for split, lines in rewritten.items():
        out_path = args.prepared_dir / f"metadata_{split}_wav.csv"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote metadata_*_wav.csv mirrors to {args.prepared_dir}")
    print(f"converted ok: {len(pending) - len(failed)}")
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for src, err in failed[:20]:
            print(f"  {src.name}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
