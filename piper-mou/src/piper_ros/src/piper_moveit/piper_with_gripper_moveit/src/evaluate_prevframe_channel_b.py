#!/usr/bin/env python3
"""Evaluate a phase-independent previous-frame tactile change guard.

The detector does not need a stable initial contact reference.  In both grasp
and transport it:

1. computes the same C8D+MRL 9-D descriptor;
2. compares the current descriptor with a short mean of preceding frames;
3. robustly scales each feature using train-success deltas;
4. thresholds the RMS scaled change for consecutive frames.

Thresholds and robust scales are phase-specific, but the detector structure
and feature semantics are identical.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from transport_features import compute_contact_mrl_9d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["grasp", "transport"],
                        required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--signal", choices=["all9", "contact_peak", "mrl"],
        default="all9",
        help="Shared per-frame quantity whose change is monitored")
    parser.add_argument("--reference_frames", type=int, default=3)
    parser.add_argument("--consec", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threshold_pct", type=float, default=95.0)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--event_log_dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if min(args.reference_frames, args.consec, args.warmup) <= 0:
        parser.error("frame counts must be positive")
    if not 0 < args.threshold_pct < 100:
        parser.error("--threshold_pct must be between 0 and 100")
    return args


def select_signal(feature: np.ndarray, signal: str) -> np.ndarray:
    if signal == "contact_peak":
        return feature[:, [7, 8]]
    if signal == "mrl":
        return feature[:, [0]]
    return feature


def load_raw_episodes(data_dir: Path, prefix: str,
                      label: str) -> list[tuple[str, np.ndarray]]:
    paths = sorted(glob.glob(
        str(data_dir / f"{prefix}_*_{label}_raw.npy")))
    result = []
    for path in paths:
        raw = np.load(path, allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] != 48:
            continue
        feature = compute_contact_mrl_9d(raw)
        result.append((Path(path).name, feature))
    return result


def split_success(episodes: list[tuple[str, np.ndarray]],
                  val_ratio: float, test_ratio: float):
    count = len(episodes)
    val_count = max(1, int(round(count * val_ratio)))
    test_count = max(1, int(round(count * test_ratio)))
    train_count = count - val_count - test_count
    return (
        episodes[:train_count],
        episodes[train_count:train_count + val_count],
        episodes[train_count + val_count:],
    )


def feature_deltas(feature: np.ndarray, reference_frames: int) -> np.ndarray:
    deltas = np.zeros_like(feature, dtype=np.float64)
    for frame in range(1, len(feature)):
        start = max(0, frame - reference_frames)
        reference = np.mean(feature[start:frame], axis=0)
        deltas[frame] = feature[frame] - reference
    return deltas


def robust_delta_scale(train: list[tuple[str, np.ndarray]],
                       reference_frames: int, warmup: int,
                       signal: str) -> np.ndarray:
    rows = []
    for _, full_feature in train:
        feature = select_signal(full_feature, signal)
        delta = feature_deltas(feature, reference_frames)
        if len(delta) > warmup:
            rows.append(delta[warmup:])
    merged = np.vstack(rows)
    median = np.median(merged, axis=0)
    mad_scale = 1.4826 * np.median(
        np.abs(merged - median[None, :]), axis=0)
    std = np.std(merged, axis=0)
    # MAD may collapse for sparse contact-count dimensions.
    return np.maximum(mad_scale, np.maximum(std * 0.1, 1e-6))


def change_score(feature: np.ndarray, scale: np.ndarray,
                 reference_frames: int, signal: str) -> np.ndarray:
    feature = select_signal(feature, signal)
    delta = feature_deltas(feature, reference_frames)
    return np.sqrt(np.mean((delta / scale[None, :]) ** 2, axis=1))


def episode_stat(scores: np.ndarray, warmup: int, consec: int) -> float:
    usable = scores[warmup:]
    if len(usable) < consec:
        return 0.0
    return max(
        float(np.min(usable[start:start + consec]))
        for start in range(len(usable) - consec + 1)
    )


def first_trigger(scores: np.ndarray, threshold: float,
                  warmup: int, consec: int) -> int | None:
    count = 0
    for frame in range(warmup, len(scores)):
        if scores[frame] > threshold:
            count += 1
        else:
            count = 0
        if count >= consec:
            return frame
    return None


def evaluate_group(episodes, scale, args, threshold):
    rows = []
    for name, feature in episodes:
        scores = change_score(
            feature, scale, args.reference_frames, args.signal)
        trigger = first_trigger(
            scores, threshold, args.warmup, args.consec)
        rows.append({
            "file": name,
            "frames": len(feature),
            "detected": trigger is not None,
            "trigger_frame": trigger,
            "max_score": float(np.max(scores[args.warmup:])),
        })
    return rows


def first_event(meta: dict[str, Any], name: str) -> int | None:
    for event in meta.get("events", []):
        if event.get("name") == name:
            return int(event["timestamp_ns"])
    return None


def evaluate_transport_events(log_dir: Path, scale: np.ndarray,
                              args, threshold: float) -> dict[str, Any]:
    controlled = []
    stable = []
    for meta_path in sorted(log_dir.glob("transport_event_*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            arrays = np.load(
                meta_path.with_name(meta["npz_file"]),
                allow_pickle=False)
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        trial_type = meta.get("trial_type")
        if trial_type not in {"stable", "abrupt", "gradual"}:
            continue
        if trial_type == "gradual" and meta.get("status") not in {
                "gradual_loss", "gradual_no_loss"}:
            continue
        raw = arrays["model_raw_48d"]
        timestamps = arrays["sensor_timestamp_ns"]
        feature = compute_contact_mrl_9d(raw)
        scores = change_score(
            feature, scale, args.reference_frames, args.signal)
        trigger_frame = first_trigger(
            scores, threshold, args.warmup, args.consec)
        trigger_time = (
            int(timestamps[trigger_frame])
            if trigger_frame is not None else None)
        onset = first_event(meta, "perturb_start")
        loss = first_event(meta, "contact_loss")
        row = {
            "file": meta_path.name,
            "trial_type": trial_type,
            "status": meta.get("status"),
            "trigger_frame": trigger_frame,
            "trigger_time": trigger_time,
            "onset": onset,
            "loss": loss,
        }
        if trial_type == "stable":
            stable.append(row)
        elif onset is not None and loss is not None:
            controlled.append(row)

    true_positive = 0
    pre_onset = 0
    false_negative = 0
    for row in controlled:
        trigger = row["trigger_time"]
        deadline = (
            row["loss"] if row["trial_type"] == "gradual"
            else row["onset"] + 2_000_000_000)
        if trigger is not None and trigger < row["onset"]:
            pre_onset += 1
            false_negative += 1
        elif trigger is not None and trigger <= deadline:
            true_positive += 1
        else:
            false_negative += 1
    stable_fp = sum(row["trigger_time"] is not None for row in stable)
    return {
        "controlled_failure_trials": len(controlled),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "pre_onset_trigger_trials": pre_onset,
        "stable_trials": len(stable),
        "stable_false_positive_trials": stable_fp,
        "rows": controlled + stable,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    success = load_raw_episodes(data_dir, args.phase, "success")
    failure = load_raw_episodes(data_dir, args.phase, "failure")
    train, validation, test = split_success(
        success, args.val_ratio, args.test_ratio)
    scale = robust_delta_scale(
        train, args.reference_frames, args.warmup, args.signal)
    val_stats = np.asarray([
        episode_stat(
            change_score(
                feature, scale, args.reference_frames, args.signal),
            args.warmup, args.consec)
        for _, feature in validation
    ])
    threshold = float(np.percentile(
        val_stats, args.threshold_pct))
    test_rows = evaluate_group(test, scale, args, threshold)
    failure_rows = evaluate_group(failure, scale, args, threshold)
    test_fp = sum(row["detected"] for row in test_rows)
    failure_tp = sum(row["detected"] for row in failure_rows)

    report: dict[str, Any] = {
        "phase": args.phase,
        "config": vars(args),
        "threshold": threshold,
        "scale": scale.tolist(),
        "split": {
            "train_success": len(train),
            "validation_success": len(validation),
            "test_success": len(test),
            "failure": len(failure),
        },
        "historical_episode_eval": {
            "test_success_false_positive": test_fp,
            "test_success_total": len(test_rows),
            "failure_detected": failure_tp,
            "failure_total": len(failure_rows),
            "test_success_rows": test_rows,
            "failure_rows": failure_rows,
        },
    }
    if args.event_log_dir and args.phase == "transport":
        report["event_eval"] = evaluate_transport_events(
            Path(args.event_log_dir).expanduser().resolve(),
            scale, args, threshold)

    print(f"[Previous-frame B] phase={args.phase}, "
          f"threshold={threshold:.6f}")
    print(f"[Historical] test success FP={test_fp}/{len(test_rows)}, "
          f"failure detected={failure_tp}/{len(failure_rows)}")
    if "event_eval" in report:
        event = report["event_eval"]
        print(
            "[Event] "
            f"TP={event['true_positive']}/"
            f"{event['controlled_failure_trials']}, "
            f"pre-onset={event['pre_onset_trigger_trials']}, "
            f"stable FP={event['stable_false_positive_trials']}/"
            f"{event['stable_trials']}")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output else
        Path(f"prevframe_b_{args.phase}_report.json").resolve())
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"[Done] {output}")


if __name__ == "__main__":
    main()
