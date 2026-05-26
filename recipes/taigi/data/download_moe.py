"""Download MOE 教育部臺灣台語常用詞辭典 audio corpus.

⚠️  LICENSE WARNING: Per the dictionary site, audio is CC-BY-NC-ND 2.5 Taiwan.
- Personal / research use: OK
- Train a TTS model and **publicly release the checkpoint**: NOT OK (NoDerivatives)
- Keep any model trained on this data INTERNAL — don't push to HF Hub etc.

Downloads:
  - kautian.ods       (master dictionary spreadsheet — entry IDs ↔ Tâi-lô ↔ 漢字)
  - sutiau-wav.zip    (詞條 single-word audio)  ← short, low ROI for prosody
  - leku-wav.zip      (例句 example sentences) ← higher ROI for TTS

By default this script grabs leku + kautian only. Pass --include-sutiau to
also grab single-word clips.

Usage:
    python recipes/taigi/data/download_moe.py [--include-sutiau]
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_OUT_DIR = Path("data/moe_sutian")
CHUNK_SIZE = 1 << 20
MAX_RETRIES = 10
RETRY_BACKOFF_SEC = 5

SOURCES = {
    "kautian.ods": "https://sutian.moe.edu.tw/media/senn/ods/kautian.ods",
    "leku-wav.zip": "https://sutian.moe.edu.tw/media/senn/leku-wav.zip",
    "sutiau-wav.zip": "https://sutian.moe.edu.tw/media/senn/sutiau-wav.zip",
}


def download(url: str, dest: Path) -> None:
    """Resume-capable download (MOE server can drop large transfers)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if have and r.status_code == 200:
                    dest.unlink()
                    have = 0
                elif have and r.status_code == 416:
                    print(f"  already complete: {dest.name}")
                    return
                else:
                    r.raise_for_status()
                content_length = int(r.headers.get("Content-Length", 0))
                total = content_length + have if content_length else None
                mode = "ab" if have else "wb"
                with open(dest, mode) as f, tqdm(
                    total=total, initial=have, unit="B", unit_scale=True,
                    unit_divisor=1024,
                    desc=f"{dest.name} (try {attempt}/{MAX_RETRIES})",
                ) as bar:
                    for block in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not block:
                            continue
                        f.write(block)
                        bar.update(len(block))
            return  # cleanly finished
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as exc:
            print(f"  {type(exc).__name__}: {exc}; retry in {RETRY_BACKOFF_SEC}s")
            time.sleep(RETRY_BACKOFF_SEC)
    sys.exit(f"error: download still failing after {MAX_RETRIES} tries.")


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {zip_path.name} → {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        for m in tqdm(members, desc="unzip"):
            z.extract(m, path=out_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--include-sutiau", action="store_true",
        help="Also download single-word (詞條) clips. Default off — low ROI for TTS prosody.",
    )
    p.add_argument("--no-extract", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = ["kautian.ods", "leku-wav.zip"]
    if args.include_sutiau:
        targets.append("sutiau-wav.zip")

    for fname in targets:
        dest = args.out_dir / fname
        print(f"\n→ {fname}")
        download(SOURCES[fname], dest)

    if not args.no_extract:
        for fname in targets:
            if fname.endswith(".zip"):
                extract_zip(args.out_dir / fname, args.out_dir)

    print(f"\ndone. data under {args.out_dir}")
    print("⚠️  Reminder: MOE data is CC-BY-NC-ND. Don't publish trained models.")


if __name__ == "__main__":
    main()
