"""Prepare Common Voice nan-tw corpus for Coqui-TTS training.

Reads validated.tsv (the full validated pool), extracts Tâi-lô romanization
from the parenthetical annotation, then re-splits speaker-stratified for
zero-shot voice-cloning training (the official train/dev/test split is
ASR-oriented — only 5 train speakers — and unusable for YourTTS).

Strategy
--------
1. Drop rows with no Tâi-lô annotation.
2. Drop speakers with < --min-utts-per-speaker utterances (too few to learn).
3. Hold out --num-holdout-speakers from the medium-utterance tier as
   "unseen speakers" for cloning generalization eval.
4. For remaining speakers, shuffle utterances and split 80/10/10 into
   train/dev/test (each speaker appears in all three splits).
5. Optional --max-utts-per-speaker cap on the train slice only, to keep
   the heavy-tail top speakers from dominating.

Outputs (in --out-dir):
    metadata_train.csv          path|tailo|speaker_id
    metadata_dev.csv
    metadata_test.csv
    metadata_cloning_eval.csv   unseen speakers, for zero-shot eval
    speakers.json               {client_id_hash: int_id}
    splits_report.json          stats summary
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PAREN_RE = re.compile(r"（([^（）]+)）\s*$")


def extract_tailo(sentence: str) -> str | None:
    m = PAREN_RE.search(sentence.strip())
    if not m:
        return None
    first = m.group(1).split("|", 1)[0].strip()
    return first or None


def load_durations(path: Path) -> dict[str, float]:
    """clip_durations.tsv: clip <tab> duration[ms]"""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2:
                out[row[0]] = float(row[1]) / 1000.0  # ms → s
    return out


def split_speaker_utts(
    utts: list[tuple[str, str]],
    rng: random.Random,
    train_frac: float = 0.8,
    dev_frac: float = 0.1,
) -> tuple[list, list, list]:
    """Per-speaker 80/10/10 split. Guarantees ≥1 in each split when possible."""
    shuffled = utts[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_frac))
    n_dev = max(1, int(n * dev_frac)) if n >= 3 else 0
    train = shuffled[:n_train]
    dev = shuffled[n_train : n_train + n_dev]
    test = shuffled[n_train + n_dev :]
    return train, dev, test


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(
            "data/common_voice_nan_tw/cv-corpus-25.0-2026-03-09/nan-tw"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/common_voice_nan_tw/prepared"),
    )
    p.add_argument("--min-utts-per-speaker", type=int, default=20)
    p.add_argument(
        "--max-utts-per-speaker",
        type=int,
        default=1500,
        help="Cap on train-slice utterances per speaker (0 = no cap)",
    )
    p.add_argument("--num-holdout-speakers", type=int, default=15)
    p.add_argument(
        "--holdout-min-utts",
        type=int,
        default=30,
        help="Holdout speakers picked from those with ≥ this many utts",
    )
    p.add_argument(
        "--holdout-max-utts",
        type=int,
        default=200,
        help="...and ≤ this many (avoid removing top-heavy speakers)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.corpus_dir.exists():
        sys.exit(f"error: corpus dir not found: {args.corpus_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    validated_path = args.corpus_dir / "validated.tsv"
    durations = load_durations(args.corpus_dir / "clip_durations.tsv")

    # Pass 1 — read, extract Tâi-lô, bucket by speaker.
    per_speaker: dict[str, list[tuple[str, str]]] = defaultdict(list)
    rejected = Counter()
    total = 0
    with validated_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            total += 1
            tailo = extract_tailo(row.get("sentence", ""))
            if tailo is None:
                rejected["no_tailo"] += 1
                continue
            if len(tailo) < 2:
                rejected["too_short"] += 1
                continue
            per_speaker[row["client_id"]].append((row["path"], tailo))

    # Pass 2 — filter speakers, pick cloning holdout.
    eligible = {sp: u for sp, u in per_speaker.items() if len(u) >= args.min_utts_per_speaker}
    dropped_speakers = len(per_speaker) - len(eligible)

    holdout_pool = [
        sp for sp, u in eligible.items()
        if args.holdout_min_utts <= len(u) <= args.holdout_max_utts
    ]
    rng.shuffle(holdout_pool)
    holdout_speakers = set(holdout_pool[: args.num_holdout_speakers])
    training_speakers = {sp: u for sp, u in eligible.items() if sp not in holdout_speakers}

    # Pass 3 — assign integer speaker IDs (deterministic by hash sort).
    speaker_map: dict[str, int] = {}
    for sp in sorted(training_speakers.keys()):
        speaker_map[sp] = len(speaker_map)
    for sp in sorted(holdout_speakers):
        speaker_map[sp] = len(speaker_map)

    # Pass 4 — per-speaker 80/10/10 split for training speakers.
    splits = {"train": [], "dev": [], "test": []}
    split_durations = {"train": 0.0, "dev": 0.0, "test": 0.0, "cloning_eval": 0.0}
    cap = args.max_utts_per_speaker

    for sp, utts in training_speakers.items():
        train, dev, test = split_speaker_utts(utts, rng)
        if cap and len(train) > cap:
            train = train[:cap]
        for bucket_name, bucket in (("train", train), ("dev", dev), ("test", test)):
            for path, tailo in bucket:
                splits[bucket_name].append(
                    f"clips/{path}|{tailo}|{speaker_map[sp]}"
                )
                split_durations[bucket_name] += durations.get(path, 0.0)

    # Pass 5 — cloning eval (all utts from holdout speakers).
    cloning_rows = []
    for sp in holdout_speakers:
        for path, tailo in per_speaker[sp]:
            cloning_rows.append(f"clips/{path}|{tailo}|{speaker_map[sp]}")
            split_durations["cloning_eval"] += durations.get(path, 0.0)

    # Write outputs.
    for name, rows in splits.items():
        (args.out_dir / f"metadata_{name}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
    (args.out_dir / "metadata_cloning_eval.csv").write_text(
        "\n".join(cloning_rows) + "\n", encoding="utf-8"
    )
    (args.out_dir / "speakers.json").write_text(
        json.dumps(speaker_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "source_total_rows": total,
        "rejected": dict(rejected),
        "speakers_total": len(per_speaker),
        "speakers_dropped_low_utts": dropped_speakers,
        "speakers_eligible": len(eligible),
        "speakers_training": len(training_speakers),
        "speakers_holdout": len(holdout_speakers),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "cloning_eval_count": len(cloning_rows),
        "split_hours": {k: round(v / 3600, 2) for k, v in split_durations.items()},
        "config": {
            "min_utts_per_speaker": args.min_utts_per_speaker,
            "max_utts_per_speaker": args.max_utts_per_speaker,
            "num_holdout_speakers": args.num_holdout_speakers,
            "holdout_min_utts": args.holdout_min_utts,
            "holdout_max_utts": args.holdout_max_utts,
            "seed": args.seed,
        },
    }
    (args.out_dir / "splits_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"source rows:      {total:>6}")
    print(f"rejected:         {dict(rejected)}")
    print(
        f"speakers:         total={len(per_speaker)} "
        f"eligible={len(eligible)} "
        f"training={len(training_speakers)} "
        f"holdout={len(holdout_speakers)}"
    )
    print("split           rows     hours")
    print("---------------------------------")
    for name in ("train", "dev", "test", "cloning_eval"):
        rows = splits[name] if name in splits else cloning_rows
        hrs = report["split_hours"][name]
        print(f"{name:<14} {len(rows):>6}    {hrs:>6.2f}")
    print(f"\nwrote → {args.out_dir}")


if __name__ == "__main__":
    main()
