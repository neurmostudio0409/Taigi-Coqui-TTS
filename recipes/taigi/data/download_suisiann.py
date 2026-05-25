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
import time
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = "https://tongan-puntiunn.ithuan.tw/SuiSiann/SuiSiann-0.2.1.tar"
DEFAULT_OUT_DIR = Path("data/suisiann")
DEFAULT_FILENAME = "SuiSiann-0.2.1.tar"
CHUNK_SIZE = 1 << 20  # 1 MiB
MAX_RETRIES = 20
RETRY_BACKOFF_SEC = 5


def _expected_total(url: str) -> int | None:
    """HEAD the URL to learn the true file size, so we know when we're done."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=15)
        if r.ok and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except requests.RequestException:
        pass
    return None


def download(url: str, dest: Path) -> None:
    """Download with automatic resume on connection drop.

    The 媠聲 tar is ~2.3 GB and the ithuan.tw mirror drops connections at
    ~1 GB occasionally. We loop with exponential-ish backoff, each iteration
    using a Range header to resume from the current file size.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _expected_total(url)
    if expected:
        print(f"  expected total: {expected / 1024**3:.2f} GB")

    for attempt in range(1, MAX_RETRIES + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            print(f"already complete: {dest} ({have / 1024**3:.2f} GB)")
            return

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if have and r.status_code == 200:
                    # Server ignored Range — restart from zero
                    dest.unlink()
                    have = 0
                elif have and r.status_code == 416:
                    # Already complete per server
                    print(f"already complete: {dest}")
                    return
                else:
                    r.raise_for_status()

                content_length = int(r.headers.get("Content-Length", 0))
                total = content_length + have if content_length else expected
                mode = "ab" if have else "wb"

                with open(dest, mode) as f, tqdm(
                    total=total,
                    initial=have,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"{dest.name} (try {attempt}/{MAX_RETRIES})",
                ) as bar:
                    for block in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not block:
                            continue
                        f.write(block)
                        bar.update(len(block))

            # Loop exited cleanly. Check whether we got everything.
            now = dest.stat().st_size
            if expected is None or now >= expected:
                return
            print(
                f"  short read: have {now / 1024**3:.2f} GB / expected "
                f"{expected / 1024**3:.2f} GB — retrying"
            )
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.Timeout,
        ) as exc:
            now = dest.stat().st_size if dest.exists() else 0
            print(
                f"  connection error at {now / 1024**3:.2f} GB "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(RETRY_BACKOFF_SEC)

    sys.exit(
        f"error: download still incomplete after {MAX_RETRIES} retries. "
        f"Have {dest.stat().st_size / 1024**3:.2f} GB; "
        f"expected {expected / 1024**3 if expected else '?':.2f} GB. "
        f"Re-run the script to keep trying."
    )


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
    print(f"[1/2] downloading → {dest}  (auto-resume on drop)")
    download(args.url, dest)

    if args.extract:
        print("[2/2] extracting")
        extract(dest, args.out_dir)
        print(f"done. data in {args.out_dir}")
    else:
        print(f"done. archive at {dest}  (run with --extract to unpack)")


if __name__ == "__main__":
    main()
