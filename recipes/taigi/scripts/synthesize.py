"""Quick inference for a trained YourTTS-Taigi checkpoint.

Two modes:
    1. Speaker by ID    — use a speaker that was in the training set
    2. Speaker by wav    — zero-shot voice cloning from a 3–10 sec reference

Defaults
--------
- If --checkpoint omitted: auto-pick the latest ``best_model*.pth`` under
  ``recipes/taigi/runs/`` (handy right after Ctrl+C).
- If --text omitted: synthesize the 5 built-in Taigi test sentences.

Usage
-----
    # Single sentence, by speaker ID
    python recipes/taigi/scripts/synthesize.py \\
        --text "Lí hó, sè-kài." --speaker taigi_0 --out hello.wav

    # All 5 default test sentences (output to out/ dir)
    python recipes/taigi/scripts/synthesize.py \\
        --speaker taigi_0 --out-dir out/

    # Zero-shot voice cloning from a reference wav
    python recipes/taigi/scripts/synthesize.py \\
        --text "Lí hó." --speaker-wav reference.wav --out cloned.wav

    # Explicit checkpoint + batch from file (one sentence per line)
    python recipes/taigi/scripts/synthesize.py \\
        --checkpoint recipes/taigi/runs/YourTTS-.../best_model_2940.pth \\
        --text-file my_sentences.txt --speaker taigi_5 --out-dir out/
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / "recipes" / "taigi" / "runs"

DEFAULT_SENTENCES = [
    "Lí hó, sè-kài.",
    "Tâi-uân ê thinn-khì tsiok hó.",
    "A-bú beh tsia̍h pn̄g.",
    "Guá ài lim ka-pi.",
    "Kin-á-jit lí beh khì tó-uī?",
]


def _latest_checkpoint() -> Path:
    """Pick the newest best_model*.pth under recipes/taigi/runs/."""
    if not RUNS_DIR.exists():
        sys.exit(f"error: no runs dir at {RUNS_DIR}")
    candidates = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        for ckpt in run_dir.glob("best_model*.pth"):
            candidates.append(ckpt)
        # fall back to numbered checkpoints if no best_model present
        for ckpt in run_dir.glob("checkpoint_*.pth"):
            candidates.append(ckpt)
    if not candidates:
        sys.exit(f"error: no checkpoints found under {RUNS_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _sentences_from_args(args) -> list[tuple[str, str]]:
    """Return [(sentence, filename_stem), ...] based on --text / --text-file."""
    if args.text:
        return [(args.text, "synth")]
    if args.text_file:
        lines = [
            l.strip()
            for l in Path(args.text_file).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        return [(s, f"line{i:03d}") for i, s in enumerate(lines)]
    return [(s, f"test{i}") for i, s in enumerate(DEFAULT_SENTENCES)]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="best_model*.pth path (default: latest under recipes/taigi/runs/)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.json (default: alongside checkpoint)",
    )
    p.add_argument(
        "--speakers-file",
        type=Path,
        default=None,
        help="speakers.pth (default: alongside checkpoint)",
    )

    text_group = p.add_mutually_exclusive_group()
    text_group.add_argument("--text", help="Single sentence to synthesize")
    text_group.add_argument(
        "--text-file", type=Path, help="File with one sentence per line"
    )

    spk_group = p.add_mutually_exclusive_group()
    spk_group.add_argument(
        "--speaker", default="taigi_0",
        help="Training-set speaker name (default: taigi_0)",
    )
    spk_group.add_argument(
        "--speaker-wav", type=Path,
        help="Reference wav for zero-shot cloning (3-10 sec)",
    )

    p.add_argument(
        "--out", type=Path, default=None,
        help="Output wav path (single-sentence mode)",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("out"),
        help="Output dir for batch mode (default: out/)",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (default: use CUDA if available)",
    )
    args = p.parse_args()

    # Resolve paths.
    checkpoint = args.checkpoint or _latest_checkpoint()
    if not checkpoint.exists():
        sys.exit(f"error: checkpoint not found: {checkpoint}")
    run_dir = checkpoint.parent
    config = args.config or (run_dir / "config.json")
    speakers_file = args.speakers_file or (run_dir / "speakers.pth")
    if not config.exists():
        sys.exit(f"error: config.json not found: {config}")

    print(f"checkpoint: {checkpoint}")
    print(f"config:     {config}")
    print(f"speakers:   {speakers_file if speakers_file.exists() else '(none)'}")

    # Deferred imports — keeps --help cheap.
    try:
        import torch
        from TTS.utils.synthesizer import Synthesizer
    except ImportError as exc:
        sys.exit(f"missing dependency: {exc.name}. Run `pip install -e .` first.")

    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"device:     {'cuda' if use_cuda else 'cpu'}\n")

    syn = Synthesizer(
        tts_checkpoint=str(checkpoint),
        tts_config_path=str(config),
        tts_speakers_file=str(speakers_file) if speakers_file.exists() else "",
        use_cuda=use_cuda,
    )

    sentences = _sentences_from_args(args)
    batch_mode = args.out is None or len(sentences) > 1
    if batch_mode:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        spk_tag = args.speaker_wav.stem if args.speaker_wav else args.speaker

    for sent, stem in sentences:
        kwargs = {"text": sent, "language_name": "nan-tw", "split_sentences": False}
        if args.speaker_wav:
            kwargs["speaker_wav"] = str(args.speaker_wav)
        else:
            kwargs["speaker_name"] = args.speaker

        print(f"→ {stem}: {sent!r}")
        wav = syn.tts(**kwargs)

        out_path = args.out if (not batch_mode) else (
            args.out_dir / f"{stem}_{spk_tag}.wav"
        )
        syn.save_wav(wav, str(out_path))
        print(f"  wrote {out_path}\n")

    print("done.")


if __name__ == "__main__":
    main()
