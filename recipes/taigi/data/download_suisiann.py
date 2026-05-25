"""Download the Suí-siann (媠聲) Taiwanese Hokkien speech corpus.

Source : 意傳科技 ÌTHUÂN KHOKI (https://suisiann-dataset.ithuan.tw/)
License: CC BY-SA 4.0 (audio); 漢字 transcription read-only
Content: 3,467 clips × ~4.9 sec, ~4.75 hr, single speaker (王秀容 Ng Siù-iông)
Audio  : 44.1 kHz, 16-bit WAV (also has 48 kHz studio variant in archive)
Text   : CSV mapping each clip to source text, Hàn-jī, **Tâi-lô**, duration

Why this dataset matters for us:
- Single highly-clean studio-recorded speaker → boosts mel quality
- Tâi-lô labels are PRIMARY (no parenthetical extraction needed)
- Complements CV nan-tw (which is crowdsourced / variable quality)

Usage:
    python recipes/taigi/data/download_suisiann.py [--extract]
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = "https://tongan-puntiunn.ithuan.tw/SuiSiann/SuiSiann-0.2.1.tar"
DEFAULT_OUT_DIR = Path("data/suisiann")
DEFAULT_FILENAME = "SuiSiann-0.2.1.tar"
CHUNK_SIZE = 1 << 20  # 1 MiB
EXPECTED_MIN_BYTES = 100 * 1024 * 1024  # corpus is ~500MB-1GB; under 100MB = bad


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resume_pos = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if resume_pos and r.status_code == 200:
            dest.unlink()
            resume_pos = 0
        elif resume_pos and r.status_code == 416:
            print(f"already complete: {dest}")
            return
        else:
            r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + resume_pos
        mode = "ab" if resume_pos else "wb"
        with open(dest, mode) as f, tqdm(
            total=total,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for block in r.iter_content(chunk_size=CHUNK_SIZE):
                if not block:
                    continue
                f.write(block)
                bar.update(len(block))


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name} → {out_dir}")
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for m in tqdm(members, unit="file", desc="extract"):
            tar.extract(m, path=out_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--filename", default=DEFAULT_FILENAME)
    p.add_argument("--extract", action="store_true", help="Extract tar after download")
    args = p.parse_args()

    dest = args.out_dir / args.filename
    if not dest.exists() or dest.stat().st_size < EXPECTED_MIN_BYTES:
        print(f"[1/2] downloading → {dest}")
        download(args.url, dest)
        size = dest.stat().st_size
        if size < EXPECTED_MIN_BYTES:
            sys.exit(
                f"error: download too small ({size / 1024**2:.1f} MB); "
                f"expected >= {EXPECTED_MIN_BYTES / 1024**2:.0f} MB. "
                f"Delete and re-run."
            )
    else:
        print(f"[1/2] already downloaded: {dest}")

    if args.extract:
        print("[2/2] extracting")
        extract(dest, args.out_dir)
        print(f"done. data in {args.out_dir}")
    else:
        print(f"done. archive at {dest}  (run with --extract to unpack)")


if __name__ == "__main__":
    main()
