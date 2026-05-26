"""Download the TAT-TTS corpus (NTUT × 李江卻基金會).

NOTE: TAT-TTS is gated behind ACLCLP authorization
(https://www.aclclp.org.tw/use_mat.php). You must have applied for + been
granted access before running this script. The 4 Google Drive folders below
are useless without that authorization — Google will eventually rate-limit
or block downloads if you're not on the access list.

Corpus stats (per upstream page):
  - 40.5 hr total / 4 speakers (2 male + 2 female)
  - ~46,496 clips
  - 48 kHz, 24-bit mono WAV
  - Per-clip JSON metadata with Tâi-lô / POJ / 漢字 / 漢羅
  - 2 dialects (Zhangzhou + Quanzhou)

Folder mapping (provided by user):
  F1 = 1cCIeT4Q5o1cKgxI-jzyJuPW6ks9SVuJM
  F2 = 16JWa03V76nAxNDTCFS06tg6RWy0eixhH
  M1 = 1IUZbxPs-v7v1PIO4doGMZtVPxT0mn_m2
  M2 = 1PWKHhV3Kj4_5xli7Ues-den3oF4EWEOf

Usage:
    pip install gdown          # one-time
    python recipes/taigi/data/download_tat_tts.py
"""

import argparse
import sys
from pathlib import Path

DEFAULT_OUT_DIR = Path("data/tat_tts")
FOLDERS = {
    "F1": "1cCIeT4Q5o1cKgxI-jzyJuPW6ks9SVuJM",
    "F2": "16JWa03V76nAxNDTCFS06tg6RWy0eixhH",
    "M1": "1IUZbxPs-v7v1PIO4doGMZtVPxT0mn_m2",
    "M2": "1PWKHhV3Kj4_5xli7Ues-den3oF4EWEOf",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--speakers", nargs="+", default=list(FOLDERS),
        help=f"Subset of speakers (any of {list(FOLDERS)})",
    )
    args = p.parse_args()

    try:
        import gdown
    except ImportError:
        sys.exit(
            "error: gdown not installed. Run `pip install gdown` first."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spk in args.speakers:
        if spk not in FOLDERS:
            print(f"skip unknown speaker: {spk}")
            continue
        folder_id = FOLDERS[spk]
        target = args.out_dir / spk
        target.mkdir(exist_ok=True)
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        print(f"\n[{spk}] downloading from {url}")
        print(f"        → {target}")
        try:
            gdown.download_folder(
                url=url,
                output=str(target),
                quiet=False,
                use_cookies=False,
                remaining_ok=True,  # tolerate partial / continue across runs
            )
        except Exception as exc:
            print(f"  WARNING: {spk} failed: {exc}")
            print(f"  Re-run script to retry; gdown will skip already-downloaded files.")

    print(f"\ndone. all speakers under {args.out_dir}")


if __name__ == "__main__":
    main()
