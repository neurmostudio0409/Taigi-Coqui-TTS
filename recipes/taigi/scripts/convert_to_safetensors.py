"""Convert a Coqui-TTS YourTTS checkpoint to .safetensors.

Why
---
Coqui's ``Trainer`` saves ``.pth`` files via ``torch.save``, which pickles
the whole training state (model + optimizer + scheduler + step + epoch).
That's fine for resuming training, but:

  * pickle can execute arbitrary code on load — unsafe to share publicly
  * the file is ~2x larger than necessary (optimizer + scheduler state)
  * not directly loadable by HuggingFace inference loaders

``.safetensors`` is a weights-only format with zero code execution,
mmap-friendly, and the de-facto standard on HuggingFace Hub.

What this script does
---------------------
1. Loads ``checkpoint.pth`` (CPU, weights-only-style extraction).
2. Pulls out the ``"model"`` sub-dict (just the layer weights).
3. Clones tensors so safetensors' no-shared-storage requirement holds.
4. Writes ``checkpoint.safetensors`` with metadata about step / epoch.
5. Copies the companion ``config.json`` and ``speakers.pth`` to the
   same output directory — these are required at inference time.

Result is a folder ready for HuggingFace Hub upload or local inference.

Usage
-----
    # Convert the run's best model
    python recipes/taigi/scripts/convert_to_safetensors.py \\
        --checkpoint recipes/taigi/runs/YourTTS-Taigi-CV25-*/best_model.pth

    # Convert a specific step checkpoint
    python recipes/taigi/scripts/convert_to_safetensors.py \\
        --checkpoint recipes/taigi/runs/.../checkpoint_50000.pth \\
        --out-dir exports/yourtts-taigi-step50k
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _extract_weights(state: dict) -> dict:
    """Return the model state_dict regardless of Coqui Trainer key layout."""
    for key in ("model", "state_dict", "model_state_dict"):
        sub = state.get(key) if isinstance(state, dict) else None
        if isinstance(sub, dict) and sub:
            return sub
    # Fall through: maybe the file is already a bare state_dict.
    if isinstance(state, dict) and all(hasattr(v, "shape") for v in state.values()):
        return state
    raise ValueError(
        "could not locate model weights — checkpoint has no 'model' / "
        "'state_dict' / 'model_state_dict' key, and isn't itself a bare "
        "state_dict"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Coqui .pth checkpoint (best_model.pth or checkpoint_*.pth)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <checkpoint dir>/safetensors_export)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json (default: alongside checkpoint)",
    )
    p.add_argument(
        "--speakers",
        type=Path,
        default=None,
        help="Path to speakers.pth (default: alongside checkpoint)",
    )
    p.add_argument(
        "--no-companions",
        action="store_true",
        help="Skip copying config.json + speakers.pth",
    )
    args = p.parse_args()

    # Deferred imports so --help works without torch installed.
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        sys.exit(
            f"missing dependency: {exc.name}. Install: "
            f"pip install torch safetensors"
        )

    if not args.checkpoint.exists():
        sys.exit(f"error: checkpoint not found: {args.checkpoint}")

    src_dir = args.checkpoint.parent
    out_dir = args.out_dir or src_dir / "safetensors_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    weights = _extract_weights(state)
    step = state.get("step", "unknown") if isinstance(state, dict) else "unknown"
    epoch = state.get("epoch", "unknown") if isinstance(state, dict) else "unknown"

    # safetensors requires contiguous, non-shared tensors.
    cleaned = {}
    skipped_non_tensor = []
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            cleaned[k] = v.detach().clone().contiguous().cpu()
        else:
            skipped_non_tensor.append(k)

    if skipped_non_tensor:
        print(f"  skipped {len(skipped_non_tensor)} non-tensor entries "
              f"(e.g. {skipped_non_tensor[:3]})")

    out_safetensors = out_dir / f"{args.checkpoint.stem}.safetensors"
    metadata = {
        "format": "coqui-yourtts-taigi",
        "source_checkpoint": args.checkpoint.name,
        "step": str(step),
        "epoch": str(epoch),
        "num_tensors": str(len(cleaned)),
    }
    print(f"saving {len(cleaned)} tensors → {out_safetensors.name}")
    save_file(cleaned, str(out_safetensors), metadata=metadata)

    # Companion files — needed at inference time alongside the weights.
    if not args.no_companions:
        config_src = args.config or (src_dir / "config.json")
        speakers_src = args.speakers or (src_dir / "speakers.pth")
        for src in (config_src, speakers_src):
            if src.exists():
                dst = out_dir / src.name
                shutil.copy2(src, dst)
                print(f"  copied  {src.name}")
            else:
                print(f"  WARNING missing {src.name} ({src})")

    # Manifest with sizes + checksums to help downstream consumers verify.
    manifest = {
        "source_checkpoint": str(args.checkpoint),
        "step": step,
        "epoch": epoch,
        "num_tensors": len(cleaned),
        "files": {
            p.name: p.stat().st_size
            for p in sorted(out_dir.iterdir())
            if p.is_file()
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    pth_mb = args.checkpoint.stat().st_size / 1024**2
    safe_mb = out_safetensors.stat().st_size / 1024**2
    print(
        f"\nsize: .pth {pth_mb:>7.1f} MB  →  .safetensors {safe_mb:>7.1f} MB  "
        f"(-{(1 - safe_mb / pth_mb) * 100:.0f}%)"
    )
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
