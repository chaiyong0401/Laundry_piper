#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR
EPISODES_DIR = TRAIN_DIR
INDEX_DIR = TRAIN_DIR / "index"

SEED = 42
CLASSES = ["NC", "G", "C"]

WINDOW_LENGTH = 300
STRIDE = 1

BUCKET_KEYS = [
    "NC|stable",
    "G|stable",
    "C|stable",
    "NC->G",
    "G->C",
    "C->NC",
    "G->NC",
]

DEFAULT_WEIGHTS = [2, 2, 2, 1, 1, 1, 1]


def _bucket_of_window(left: str, right: str) -> str:
    if left in CLASSES and right in CLASSES and left == right:
        return f"{left}|stable"

    key = f"{left}->{right}"
    if key in {"NC->G", "G->C", "C->NC", "G->NC"}:
        return key

    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default=",".join(map(str, DEFAULT_WEIGHTS)))
    parser.add_argument("--ignore_ratio_max", type=float, default=0.5,
                        help="Drop window if IGNORE ratio > this value (default=0.5)")
    args = parser.parse_args()

    ignore_ratio_max = float(args.ignore_ratio_max)

    weights = [int(x) for x in args.weights.split(",")]
    weight_map = {k: weights[i] for i, k in enumerate(BUCKET_KEYS)}

    rng = np.random.default_rng(SEED)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(EPISODES_DIR.glob("*.npz"))
    if not files:
        print("[ERROR] no episode files found")
        return

    pool: Dict[str, List[Tuple[str, int]]] = {k: [] for k in BUCKET_KEYS}
    availability = {k: 0 for k in BUCKET_KEYS}

    dropped_ignore_ratio = 0

    for ep_fp in files:
        with np.load(ep_fp, mmap_mode="r", allow_pickle=True) as d:
            labels = d["train_labels"]

        labels = np.asarray(labels)
        N = len(labels)

        for start in range(0, N - WINDOW_LENGTH + 1, STRIDE):
            window = labels[start:start + WINDOW_LENGTH]

            # 🔥 IGNORE ratio 계산
            ignore_ratio = np.mean(window == "IGNORE")
            if ignore_ratio > ignore_ratio_max:
                dropped_ignore_ratio += 1
                continue

            left = str(window[0])
            right = str(window[-1])

            b = _bucket_of_window(left, right)
            if b == "":
                continue

            pool[b].append((ep_fp.name, start))
            availability[b] += 1

    print("availability:", availability)
    print("dropped_ignore_ratio:", dropped_ignore_ratio)

    # ---------------- quota 계산 ----------------
    sum_w = sum(weights)
    bottleneck = []

    for b in BUCKET_KEYS:
        w = weight_map[b]
        if w <= 0:
            continue
        if availability[b] == 0:
            raise ValueError(f"{b} has no samples")
        bottleneck.append(int(np.floor(availability[b] * sum_w / w)))

    T_max = min(bottleneck)

    quota = {}
    for b in BUCKET_KEYS:
        w = weight_map[b]
        quota[b] = int(np.floor(T_max * w / sum_w)) if w > 0 else 0

    print("quota:", quota)

    # ---------------- 샘플링 ----------------
    sel_episode, sel_start, sel_bucket = [], [], []

    for b in BUCKET_KEYS:
        k = quota[b]
        if k <= 0:
            continue

        idx = rng.choice(len(pool[b]), size=k, replace=False)
        for i in idx:
            ep, s = pool[b][int(i)]
            sel_episode.append(ep)
            sel_start.append(s)
            sel_bucket.append(b)

    perm = rng.permutation(len(sel_episode))

    np.savez_compressed(
        INDEX_DIR / "selected.npz",
        episode_name=np.asarray(sel_episode)[perm],
        start_idx=np.asarray(sel_start)[perm],
        bucket=np.asarray(sel_bucket)[perm],
    )

    print("[DONE] episode-based index saved.")


if __name__ == "__main__":
    main()