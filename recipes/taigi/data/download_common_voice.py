"""Download Common Voice nan-tw (Taiwanese Minnan) from Mozilla Data Collective.

Dataset: Common Voice Scripted Speech 25.0 - Taiwanese (Minnan)
  21.78 hr validated / 299 speakers / 32,426 clips / CC0-1.0

The API key is loaded, in order of precedence:
    1. --api-key flag
    2. MDC_API_KEY environment variable
    3. MDC_API_KEY in a ``.env`` file at the project root (gitignored)

To set up:
    cp .env.example .env       # then edit .env to fill MDC_API_KEY=...
    python recipes/taigi/data/download_common_voice.py --extract
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


def _load_dotenv(env_path: Path) -> None:
    """Tiny .env loader (no python-dotenv dependency).

    Only sets keys not already present in os.environ, so shell env wins.
    """
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

API_BASE = "https://mozilladatacollective.com/api/datasets"
DEFAULT_DATASET_ID = "cmn2cyd8901jemm0738nubysq"
DEFAULT_FILENAME = "common-voice-scripted-speech-25-0-taiwan.tar.gz"
DEFAULT_OUT_DIR = Path("data/common_voice_nan_tw")
CHUNK_SIZE = 1 << 20  # 1 MiB


def get_download_url(dataset_id: str, api_key: str) -> str:
    url = f"{API_BASE}/{dataset_id}/download"
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    if r.status_code == 401:
        sys.exit("error: 401 unauthorized — check MDC_API_KEY")
    if r.status_code == 404:
        sys.exit(f"error: dataset {dataset_id} not found")
    r.raise_for_status()
    payload = r.json()
    if "downloadUrl" not in payload:
        sys.exit(f"error: no downloadUrl in response: {json.dumps(payload)[:200]}")
    return payload["downloadUrl"]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resume_pos = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if resume_pos and r.status_code == 200:
            # Server ignored Range — start over
            dest.unlink()
            resume_pos = 0
        elif resume_pos and r.status_code == 416:
            print(f"already complete: {dest}")
            return
        else:
            r.raise_for_status()

        content_length = int(r.headers.get("Content-Length", 0))
        total = content_length + resume_pos
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
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in tqdm(members, unit="file", desc="extract"):
            tar.extract(m, path=out_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("MDC_API_KEY"),
        help="Mozilla Data Collective API key (env: MDC_API_KEY)",
    )
    p.add_argument(
        "--dataset-id",
        default=os.environ.get("MDC_DATASET_ID", DEFAULT_DATASET_ID),
        help=f"Dataset ID (default: CV 25.0 nan-tw, {DEFAULT_DATASET_ID})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="Output filename for the archive",
    )
    p.add_argument(
        "--extract",
        action="store_true",
        help="Extract the tar.gz after download",
    )
    p.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the tar.gz file after extracting (default: keep)",
    )
    args = p.parse_args()

    # Late-load .env so it can fill MDC_API_KEY if neither flag nor shell set one.
    if not args.api_key:
        project_root = Path(__file__).resolve().parents[3]
        _load_dotenv(project_root / ".env")
        args.api_key = os.environ.get("MDC_API_KEY")

    if not args.api_key:
        sys.exit(
            "error: missing API key — set MDC_API_KEY in .env, "
            "export MDC_API_KEY=..., or pass --api-key"
        )

    dest = args.out_dir / args.filename
    print(f"[1/3] requesting presigned URL for dataset {args.dataset_id}")
    download_url = get_download_url(args.dataset_id, args.api_key)

    print(f"[2/3] downloading → {dest}")
    download(download_url, dest)

    if args.extract:
        print(f"[3/3] extracting")
        extract(dest, args.out_dir)
        print(f"done. data in {args.out_dir}")
    else:
        print(f"done. archive at {dest}  (run with --extract to unpack)")


if __name__ == "__main__":
    main()
