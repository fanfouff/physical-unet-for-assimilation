#!/usr/bin/env python3
"""
Build a reproducible train/val/test split JSON for the nested NPZ layout.

Default layout:
  /data/lrx_true/era_obs/npz/
    train/*.npz
    val/*.npz
    test/*.npz
    test_2/*.npz
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np


def _collect_npz_files(path: Path) -> List[str]:
    if not path.exists():
        return []

    files = []
    skipped_invalid = 0
    for p in sorted(path.glob("*.npz")):
        if p.name in {"stats.npz", "increment_stats.npz", "dataset_split.json"}:
            continue
        try:
            with np.load(p) as data:
                target = data["target"]
                if target.size == 0 or not np.isfinite(target).all() or np.all(target == 0):
                    skipped_invalid += 1
                    continue
        except Exception:
            skipped_invalid += 1
            continue
        files.append(str(p.resolve()))

    if skipped_invalid:
        print(f"[INFO] {path}: skipped invalid files = {skipped_invalid}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed split JSON for train_ddp.py")
    parser.add_argument("--data_root", type=str, required=True, help="Parent npz directory")
    parser.add_argument("--train_dir", type=str, default="train", help="Training pool directory name")
    parser.add_argument("--val_dir", type=str, default="val", help="Validation directory name")
    parser.add_argument(
        "--test_dirs",
        nargs="+",
        default=["test", "test_2"],
        help="One or more test directory names",
    )
    parser.add_argument(
        "--val_ratio_from_train",
        type=float,
        default=0.1,
        help="When val_dir is empty, reserve this fraction from the train pool as validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train->val split")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()

    train_files = _collect_npz_files(root / args.train_dir)
    val_files = _collect_npz_files(root / args.val_dir)
    test_files: List[str] = []
    for name in args.test_dirs:
        test_files.extend(_collect_npz_files(root / name))
    test_files = sorted(test_files)

    if not train_files:
        raise ValueError(f"No training NPZ files found under {root / args.train_dir}")
    if not test_files:
        raise ValueError(f"No test NPZ files found under any of {args.test_dirs}")

    if not val_files:
        rng = random.Random(args.seed)
        shuffled = train_files.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_ratio_from_train))
        val_files = sorted(shuffled[:n_val])
        train_files = sorted(shuffled[n_val:])

    split: Dict[str, List[str]] = {
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)

    print(f"Saved fixed split to: {output}")
    print(f"train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")


if __name__ == "__main__":
    main()
