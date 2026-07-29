#!/usr/bin/env python3
"""Evaluate Channel A, Channel B, and OR fusion from transport event logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


CONFIGURATIONS = ("A", "B", "A_OR_B")


def first_true_time(
    timestamps: np.ndarray,
    flags: np.ndarray,
) -> Optional[int]:
    indices = np.flatnonzero(flags)
    return int(timestamps[indices[0]]) if len(indices) else None


def first_event_time(meta: Dict[str, Any], name: str) -> Optional[int]:
    for event in meta.get("events", []):
        if event.get("name") == name:
            return int(event["timestamp_ns"])
    return None


def merge_local_markers(meta_path: Path, meta: Dict[str, Any]) -> None:
    """Use the keyboard marker stream when ROS delivery did not reach the node."""
    marker_path = meta_path.parent / "marker_stream.jsonl"
    if not marker_path.exists():
        return
    start_ns = int(meta["start_monotonic_ns"])
    end_ns = int(meta["end_monotonic_ns"])
    existing_names = {
        event.get("name") for event in meta.get("events", [])}
    with marker_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                marker = json.loads(line)
                timestamp_ns = int(marker["timestamp_ns"])
                name = str(marker["name"])
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
            if (start_ns <= timestamp_ns <= end_ns
                    and name not in existing_names):
                meta.setdefault("events", []).append({
                    "name": name,
                    "timestamp_ns": timestamp_ns,
                    "elapsed_s": (timestamp_ns - start_ns) / 1e9,
                    "note": "keyboard marker (evaluation fallback)",
                })
                existing_names.add(name)
    meta.setdefault("events", []).sort(
        key=lambda event: int(event["timestamp_ns"]))


def load_trial(meta_path: Path) -> Dict[str, Any]:
    with meta_path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    merge_local_markers(meta_path, meta)
    npz_path = meta_path.with_name(meta["npz_file"])
    arrays = np.load(npz_path, allow_pickle=False)

    a_time = first_true_time(arrays["a_timestamp_ns"], arrays["a_confirmed"])
    b_time = first_true_time(
        arrays["sensor_timestamp_ns"], arrays["b_confirmed"])
    fusion_candidates = [value for value in (a_time, b_time) if value is not None]
    return {
        "meta": meta,
        "trigger": {
            "A": a_time,
            "B": b_time,
            "A_OR_B": min(fusion_candidates) if fusion_candidates else None,
        },
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = (
        z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denom
    )
    return center - margin, center + margin


def evaluate(
    trials: list[Dict[str, Any]],
    deadline_s: float,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"configurations": {}, "trial_count": len(trials)}
    for configuration in CONFIGURATIONS:
        failure_total = 0
        true_positive = 0
        false_negative = 0
        stable_total = 0
        stable_false_positive = 0
        pre_onset_false_positive = 0
        latencies_s: list[float] = []
        gradual_lead_s: list[float] = []
        gradual_no_loss_total = 0
        gradual_no_loss_trigger = 0

        for trial in trials:
            meta = trial["meta"]
            trial_type = str(meta.get("trial_type", "")).lower()
            trigger = trial["trigger"][configuration]
            onset = first_event_time(meta, "perturb_start")
            loss = first_event_time(meta, "contact_loss")

            if trial_type == "stable":
                stable_total += 1
                if trigger is not None:
                    stable_false_positive += 1
                continue

            if trial_type not in {"abrupt", "gradual"}:
                # Keep uncontrolled failures and resisted perturbations in the
                # archive without mixing them into the controlled denominator.
                continue

            if trial_type == "gradual" and onset is not None and loss is None:
                gradual_no_loss_total += 1
                if trigger is not None and trigger >= onset:
                    gradual_no_loss_trigger += 1
                continue

            # A controlled failure requires both the imposed-perturbation marker
            # and confirmation that the garment physically left the grasp.
            if onset is None or loss is None:
                continue
            failure_total += 1
            # Gradual release asks whether a warning occurred before physical
            # loss; an arbitrary short post-onset deadline would discard valid
            # early warnings during a deliberately slow release.
            deadline = (
                loss if trial_type == "gradual"
                else onset + int(deadline_s * 1e9))
            if trigger is not None and trigger < onset:
                pre_onset_false_positive += 1
                false_negative += 1
                continue
            if trigger is not None and trigger <= deadline:
                true_positive += 1
                latencies_s.append((trigger - onset) / 1e9)
                if trial_type == "gradual" and loss is not None:
                    gradual_lead_s.append((loss - trigger) / 1e9)
            else:
                false_negative += 1

        recall = true_positive / failure_total if failure_total else float("nan")
        false_positive = stable_false_positive + pre_onset_false_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else float("nan")
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall)
            and precision + recall > 0.0 else float("nan")
        )
        false_trial_rate = (
            stable_false_positive / stable_total
            if stable_total else float("nan")
        )
        ci_low, ci_high = wilson_interval(true_positive, failure_total)
        report["configurations"][configuration] = {
            "failure_trials": failure_total,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "event_precision": precision,
            "event_recall": recall,
            "event_f1": f1,
            "event_recall_95ci": [ci_low, ci_high],
            "stable_trials": stable_total,
            "stable_false_positive_trials": stable_false_positive,
            "stable_false_trial_rate": false_trial_rate,
            "pre_onset_false_positive_trials": pre_onset_false_positive,
            "median_latency_s": (
                float(np.median(latencies_s)) if latencies_s else None
            ),
            "latency_iqr_s": (
                [
                    float(np.percentile(latencies_s, 25)),
                    float(np.percentile(latencies_s, 75)),
                ]
                if latencies_s else None
            ),
            "gradual_detected_before_loss": sum(
                lead > 0.0 for lead in gradual_lead_s
            ),
            "gradual_lead_count": len(gradual_lead_s),
            "median_gradual_lead_s": (
                float(np.median(gradual_lead_s))
                if gradual_lead_s else None
            ),
            "gradual_no_loss_trials": gradual_no_loss_total,
            "gradual_no_loss_trigger_trials": gradual_no_loss_trigger,
            "gradual_no_loss_trigger_rate": (
                gradual_no_loss_trigger / gradual_no_loss_total
                if gradual_no_loss_total else float("nan")
            ),
        }
    return report


def format_pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{100.0 * value:.1f}%"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        default="/home/dyros/mcy_ws/transport_event_eval",
    )
    parser.add_argument(
        "--deadline_s",
        type=float,
        default=2.0,
        help="Maximum delay after perturb_start counted as a detection",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.deadline_s <= 0:
        parser.error("--deadline_s must be positive")

    log_dir = Path(args.log_dir).expanduser()
    meta_paths = sorted(log_dir.glob("transport_event_*.json"))
    trials = []
    for path in meta_paths:
        try:
            trials.append(load_trial(path))
        except KeyError:
            # Ignore aggregate reports or unrelated JSON files sharing the
            # prefix; trial metadata always contains ``npz_file``.
            continue
    report = evaluate(trials, args.deadline_s)

    print(f"[Event Eval] trials={len(trials)}, deadline={args.deadline_s:.3f}s")
    print("Config   Precision  Recall     F1         Stable FP   Median latency   Gradual lead")
    for name in CONFIGURATIONS:
        result = report["configurations"][name]
        latency = result["median_latency_s"]
        lead = result["median_gradual_lead_s"]
        print(
            f"{name:<8} "
            f"{format_pct(result['event_precision']):>9}  "
            f"{result['true_positive']:>3}/{result['failure_trials']:<3} "
            f"({format_pct(result['event_recall']):>6})  "
            f"{format_pct(result['event_f1']):>9}  "
            f"{result['stable_false_positive_trials']:>3}/"
            f"{result['stable_trials']:<3} "
            f"({format_pct(result['stable_false_trial_rate']):>6})  "
            f"{'N/A' if latency is None else f'{latency:.3f}s':>14}  "
            f"{'N/A' if lead is None else f'{lead:.3f}s':>12}"
        )
    if any(
            report["configurations"][name]["gradual_no_loss_trials"]
            for name in CONFIGURATIONS):
        print("[Gradual no-loss] detector responses after automatic release:")
        for name in CONFIGURATIONS:
            result = report["configurations"][name]
            print(
                f"  {name:<8} "
                f"{result['gradual_no_loss_trigger_trials']}/"
                f"{result['gradual_no_loss_trials']} "
                f"({format_pct(result['gradual_no_loss_trigger_rate'])})")

    output_path = (
        Path(args.output).expanduser()
        if args.output else log_dir / "transport_event_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(report), stream, ensure_ascii=False, indent=2)
    print(f"[Done] {output_path}")


if __name__ == "__main__":
    main()
