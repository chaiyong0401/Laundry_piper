#!/usr/bin/env python3
"""Evaluate the final phase-specific Channel-A and shared Channel-B design.

Channel A
---------
* grasp: absolute/cumulative grasp 6D, no relative transform
* transport: stable-grasp relative 9D

The GRU-AE backbone is identical; weights, scalers, and thresholds are
phase-specific.

Channel B
---------
Both phases use the same short-term detector:

* active-contact and peak-force from the C8D+MRL descriptor;
* current frame minus the immediately preceding frame;
* phase-specific robust scale and validation p95 threshold;
* three consecutive threshold exceedances after 30 warm-up frames.

For Transport event logs, the script additionally compares the recomputed
previous-frame B with the stable-baseline B recorded by ``basket_ori.py``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


LEGACY_ROOT = Path("/home/dyros/mcy_ws")
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from train_gru_ae import GRUAE, apply_relative_transform  # noqa: E402

from evaluate_prevframe_channel_b import (  # noqa: E402
    change_score,
    episode_stat,
    first_trigger,
    robust_delta_scale,
)
from evaluate_transport_events import (  # noqa: E402
    evaluate as evaluate_transport_event_triggers,
    first_event_time,
    first_true_time,
    load_trial,
)
from transport_features import (  # noqa: E402
    compute_contact_mrl_9d,
    compute_transport_9d_relative,
)


B_REFERENCE_FRAMES = 1
B_CONSECUTIVE = 3
B_WARMUP = 30
B_THRESHOLD_PERCENTILE = 95.0
B_SIGNAL = "contact_peak"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grasp_data_dir", default="/home/dyros/mcy_ws/grasp_new")
    parser.add_argument(
        "--grasp_model_dir",
        default="grasp_new_comparison/grasp_6d_absolute")
    parser.add_argument(
        "--transport_data_dir", default="/home/dyros/mcy_ws/transport_new")
    parser.add_argument(
        "--transport_model_dir",
        default=(
            "/home/dyros/mcy_ws/transport_new/"
            "codex_model_9d_relative_firstframe"))
    parser.add_argument(
        "--event_log_dir", default="/home/dyros/mcy_ws/transport_event_eval")
    parser.add_argument(
        "--output",
        default="grasp_new_comparison/final_phase_design_report.json")
    parser.add_argument("--deadline_s", type=float, default=2.0)
    return parser.parse_args()


def artifact_paths(phase: str, model_dir: Path) -> dict[str, Path]:
    suffix = "" if phase == "grasp" else "_9d_relative"
    return {
        "report": model_dir / f"{phase}_codex_report{suffix}.json",
        "checkpoint": model_dir / f"{phase}_gru_ae{suffix}.pt",
        "scaler": model_dir / f"{phase}_scaler{suffix}.pkl",
        "threshold": model_dir / f"{phase}_threshold{suffix}.npy",
        "infer_cfg": model_dir / f"{phase}_infer_cfg{suffix}.pkl",
    }


def load_artifacts(phase: str, model_dir: Path):
    paths = artifact_paths(phase, model_dir)
    with paths["report"].open(encoding="utf-8") as stream:
        report = json.load(stream)
    with paths["infer_cfg"].open("rb") as stream:
        infer_cfg = pickle.load(stream)
    scaler = joblib.load(paths["scaler"])
    config = report["config"]
    model = GRUAE(
        input_dim=int(scaler.n_features_in_),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        seq_len=int(config["seq_len"]),
    )
    model.load_state_dict(torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    threshold = float(np.load(paths["threshold"])[0])
    return model, scaler, threshold, infer_cfg, report


def raw_name(feature_name: str) -> str:
    return (
        feature_name if feature_name.endswith("_raw.npy")
        else feature_name[:-4] + "_raw.npy")


def load_raw(data_dir: Path, name: str) -> np.ndarray:
    raw = np.load(data_dir / raw_name(name), allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 48:
        raise ValueError(f"invalid raw episode {name}: {raw.shape}")
    return raw.astype(np.float32)


def load_a_feature(
    phase: str,
    data_dir: Path,
    name: str,
    train_config: dict[str, Any],
) -> np.ndarray:
    if phase == "grasp":
        feature = np.load(data_dir / name, allow_pickle=False)
        if feature.ndim != 2 or feature.shape[1] != 6:
            raise ValueError(f"invalid grasp feature {name}: {feature.shape}")
        return feature.astype(np.float32)
    raw = load_raw(data_dir, name)
    return compute_transport_9d_relative(
        raw,
        baseline_frames=int(train_config["baseline_frames"]),
        change_threshold=float(train_config["change_threshold"]),
        change_sharpness=float(train_config["change_sharpness"]),
    )


def load_b_feature(data_dir: Path, name: str) -> np.ndarray:
    return compute_contact_mrl_9d(load_raw(data_dir, name))


def select_b_signal(feature: np.ndarray) -> np.ndarray:
    # active_contact, peak_force
    return feature[:, [7, 8]]


def channel_a(
    feature: np.ndarray,
    model: GRUAE,
    scaler,
    threshold: float,
    infer_cfg: dict[str, Any],
) -> tuple[bool, int | None, float]:
    seq_len = int(infer_cfg["seq_len"])
    stride = int(infer_cfg["inference_stride"])
    consecutive_needed = int(infer_cfg.get("consec", 2))
    cutoff = float(infer_cfg.get("cutoff_multiplier", 2.2))
    relative = str(infer_cfg.get("relative", "none"))
    n_rel_dims = infer_cfg.get("n_rel_dims")
    weight_alpha = float(infer_cfg.get("weight_alpha", 0.0))
    consecutive = 0
    max_score = float("-inf")
    for start in range(0, len(feature) - seq_len + 1, stride):
        window = scaler.transform(
            feature[start:start + seq_len]).astype(np.float32)
        window = apply_relative_transform(
            window, relative, n_rel_dims=n_rel_dims)
        x = torch.as_tensor(window).unsqueeze(0)
        with torch.no_grad():
            score = float(model.reconstruction_error(
                x, weight_alpha=weight_alpha).item())
        max_score = max(max_score, score)
        if score > threshold * cutoff:
            return True, start + seq_len - 1, max_score
        if score > threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= consecutive_needed:
            return True, start + seq_len - 1, max_score
    return False, None, max_score


def b_scale_and_threshold(
    data_dir: Path,
    train_names: list[str],
    validation_names: list[str],
) -> tuple[np.ndarray, float, list[float]]:
    train = [
        (name, select_b_signal(load_b_feature(data_dir, name)))
        for name in train_names
    ]
    scale = robust_delta_scale(
        train,
        B_REFERENCE_FRAMES,
        B_WARMUP,
        signal="all9",
    )
    validation_stats = []
    for name in validation_names:
        feature = select_b_signal(load_b_feature(data_dir, name))
        scores = change_score(
            feature, scale, B_REFERENCE_FRAMES, signal="all9")
        validation_stats.append(episode_stat(
            scores, B_WARMUP, B_CONSECUTIVE))
    threshold = float(np.percentile(
        np.asarray(validation_stats), B_THRESHOLD_PERCENTILE))
    return scale, threshold, validation_stats


def channel_b(
    full_feature: np.ndarray,
    scale: np.ndarray,
    threshold: float,
) -> tuple[bool, int | None, float]:
    feature = select_b_signal(full_feature)
    scores = change_score(
        feature, scale, B_REFERENCE_FRAMES, signal="all9")
    trigger = first_trigger(
        scores, threshold, B_WARMUP, B_CONSECUTIVE)
    max_score = (
        float(np.max(scores[B_WARMUP:]))
        if len(scores) > B_WARMUP else 0.0)
    return trigger is not None, trigger, max_score


def confusion(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    tp = sum(row["label"] == 1 and row[key] for row in rows)
    fp = sum(row["label"] == 0 and row[key] for row in rows)
    fn = sum(row["label"] == 1 and not row[key] for row in rows)
    tn = sum(row["label"] == 0 and not row[key] for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None
        and precision + recall else None)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def historical_phase_eval(
    phase: str,
    data_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    model, scaler, threshold_a, infer_cfg, train_report = load_artifacts(
        phase, model_dir)
    manifest = train_report["manifest"]
    scale_b, threshold_b, validation_stats = b_scale_and_threshold(
        data_dir,
        manifest["train_success"],
        manifest["validation_success"],
    )
    rows = []
    groups = (
        (0, "test_success", manifest["test_success"]),
        (1, "failure", manifest["test_failure"]),
    )
    for label, group, names in groups:
        for name in names:
            a_feature = load_a_feature(
                phase, data_dir, name, train_report["config"])
            b_feature = load_b_feature(data_dir, name)
            a_hit, a_frame, max_a = channel_a(
                a_feature, model, scaler, threshold_a, infer_cfg)
            b_hit, b_frame, max_b = channel_b(
                b_feature, scale_b, threshold_b)
            rows.append({
                "file": name,
                "group": group,
                "label": label,
                "a": a_hit,
                "b": b_hit,
                "ab": a_hit or b_hit,
                "a_frame": a_frame,
                "b_frame": b_frame,
                "max_a": max_a,
                "max_b": max_b,
            })
    failures = [row for row in rows if row["label"] == 1]
    return {
        "phase": phase,
        "channel_a": (
            "absolute grasp 6D"
            if phase == "grasp" else "stable-baseline relative transport 9D"),
        "channel_b": {
            "signal": B_SIGNAL,
            "reference_frames": B_REFERENCE_FRAMES,
            "consecutive": B_CONSECUTIVE,
            "warmup": B_WARMUP,
            "threshold_percentile": B_THRESHOLD_PERCENTILE,
            "scale": scale_b.tolist(),
            "threshold": threshold_b,
            "validation_episode_stats": validation_stats,
        },
        "threshold_a": threshold_a,
        "metrics": {
            key: confusion(rows, key) for key in ("a", "b", "ab")
        },
        "failure_overlap": {
            "both": sum(row["a"] and row["b"] for row in failures),
            "a_only": sum(row["a"] and not row["b"] for row in failures),
            "b_only": sum(row["b"] and not row["a"] for row in failures),
            "neither": sum(not row["a"] and not row["b"] for row in failures),
        },
        "rows": rows,
    }


def event_eval(
    data_dir: Path,
    train_report: dict[str, Any],
    event_log_dir: Path,
    deadline_s: float,
) -> dict[str, Any]:
    manifest = train_report["manifest"]
    scale_b, threshold_b, _ = b_scale_and_threshold(
        data_dir,
        manifest["train_success"],
        manifest["validation_success"],
    )
    previous_trials = []
    baseline_trials = []
    for meta_path in sorted(event_log_dir.glob("transport_event_*.json")):
        try:
            baseline = load_trial(meta_path)
            meta = baseline["meta"]
            arrays = np.load(
                meta_path.with_name(meta["npz_file"]),
                allow_pickle=False)
            raw = arrays["model_raw_48d"]
            timestamps = arrays["sensor_timestamp_ns"]
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        b_feature = compute_contact_mrl_9d(raw)
        _, previous_frame, _ = channel_b(
            b_feature, scale_b, threshold_b)
        previous_time = (
            int(timestamps[previous_frame])
            if previous_frame is not None else None)
        a_time = first_true_time(
            arrays["a_timestamp_ns"], arrays["a_confirmed"])
        previous_candidates = [
            value for value in (a_time, previous_time)
            if value is not None]
        previous_trials.append({
            "meta": meta,
            "trigger": {
                "A": a_time,
                "B": previous_time,
                "A_OR_B": (
                    min(previous_candidates)
                    if previous_candidates else None),
            },
        })
        baseline_trials.append(baseline)
    return {
        "previous_frame_b": evaluate_transport_event_triggers(
            previous_trials, deadline_s),
        "stable_baseline_b": evaluate_transport_event_triggers(
            baseline_trials, deadline_s),
        "previous_frame_by_type": event_metrics_by_type(
            previous_trials, deadline_s),
        "stable_baseline_by_type": event_metrics_by_type(
            baseline_trials, deadline_s),
        "previous_frame_overlap": event_detection_overlap(
            previous_trials, deadline_s),
        "stable_baseline_overlap": event_detection_overlap(
            baseline_trials, deadline_s),
        "previous_frame_b_threshold": threshold_b,
    }


def event_metrics_by_type(
    trials: list[dict[str, Any]],
    deadline_s: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for trial_type in ("abrupt", "gradual"):
        type_result: dict[str, Any] = {}
        for configuration in ("A", "B", "A_OR_B"):
            total = 0
            true_positive = 0
            false_negative = 0
            pre_onset = 0
            latencies = []
            lead_times = []
            for trial in trials:
                meta = trial["meta"]
                if str(meta.get("trial_type", "")).lower() != trial_type:
                    continue
                onset = first_event_time(meta, "perturb_start")
                loss = first_event_time(meta, "contact_loss")
                if onset is None or loss is None:
                    continue
                total += 1
                trigger = trial["trigger"][configuration]
                deadline = (
                    loss if trial_type == "gradual"
                    else onset + int(deadline_s * 1e9))
                if trigger is not None and trigger < onset:
                    pre_onset += 1
                    false_negative += 1
                elif trigger is not None and trigger <= deadline:
                    true_positive += 1
                    latencies.append((trigger - onset) / 1e9)
                    if trial_type == "gradual":
                        lead_times.append((loss - trigger) / 1e9)
                else:
                    false_negative += 1
            type_result[configuration] = {
                "failure_trials": total,
                "true_positive": true_positive,
                "false_negative": false_negative,
                "pre_onset_trigger_trials": pre_onset,
                "recall": true_positive / total if total else None,
                "median_latency_s": (
                    float(np.median(latencies)) if latencies else None),
                "median_lead_before_loss_s": (
                    float(np.median(lead_times))
                    if lead_times else None),
            }
        result[trial_type] = type_result
    return result


def event_detection_overlap(
    trials: list[dict[str, Any]],
    deadline_s: float,
) -> dict[str, int]:
    overlap = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for trial in trials:
        meta = trial["meta"]
        trial_type = str(meta.get("trial_type", "")).lower()
        onset = first_event_time(meta, "perturb_start")
        loss = first_event_time(meta, "contact_loss")
        if trial_type not in {"abrupt", "gradual"}:
            continue
        if onset is None or loss is None:
            continue
        deadline = (
            loss if trial_type == "gradual"
            else onset + int(deadline_s * 1e9))

        def valid(trigger: int | None) -> bool:
            return (
                trigger is not None
                and onset <= trigger <= deadline)

        a_hit = valid(trial["trigger"]["A"])
        b_hit = valid(trial["trigger"]["B"])
        if a_hit and b_hit:
            overlap["both"] += 1
        elif a_hit:
            overlap["a_only"] += 1
        elif b_hit:
            overlap["b_only"] += 1
        else:
            overlap["neither"] += 1
    return overlap


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def print_historical(result: dict[str, Any]) -> None:
    print(f"\n[{result['phase'].upper()} historical episodes]")
    print("Config  Precision  Recall     F1       TP FP FN TN")
    for key, label in (("a", "A"), ("b", "B-prev"), ("ab", "A OR B")):
        metric = result["metrics"][key]
        print(
            f"{label:<7} {format_metric(metric['precision']):>9} "
            f"{format_metric(metric['recall']):>8} "
            f"{format_metric(metric['f1']):>8}  "
            f"{metric['tp']:>2} {metric['fp']:>2} "
            f"{metric['fn']:>2} {metric['tn']:>2}")
    print("[Failure overlap] " + ", ".join(
        f"{key}={value}"
        for key, value in result["failure_overlap"].items()))


def print_event(result: dict[str, Any]) -> None:
    print("\n[TRANSPORT event-labelled]")
    for b_name, label in (
        ("previous_frame_b", "Previous-frame B"),
        ("stable_baseline_b", "Stable-baseline B"),
    ):
        print(f"  {label}")
        configs = result[b_name]["configurations"]
        for key, display in (
            ("A", "A"),
            ("B", "B"),
            ("A_OR_B", "A OR B"),
        ):
            metric = configs[key]
            print(
                f"    {display:<6} "
                f"TP={metric['true_positive']}/"
                f"{metric['failure_trials']} "
                f"stableFP={metric['stable_false_positive_trials']}/"
                f"{metric['stable_trials']} "
                f"F1={format_metric(metric['event_f1'])}")
        overlap_key = (
            "previous_frame_overlap"
            if b_name == "previous_frame_b"
            else "stable_baseline_overlap")
        print("    overlap: " + ", ".join(
            f"{key}={value}"
            for key, value in result[overlap_key].items()))
        by_type_key = (
            "previous_frame_by_type"
            if b_name == "previous_frame_b"
            else "stable_baseline_by_type")
        for trial_type in ("abrupt", "gradual"):
            values = result[by_type_key][trial_type]
            summary = []
            for key, display in (
                ("A", "A"),
                ("B", "B"),
                ("A_OR_B", "OR"),
            ):
                metric = values[key]
                summary.append(
                    f"{display}={metric['true_positive']}/"
                    f"{metric['failure_trials']}")
            print(f"    {trial_type}: " + ", ".join(summary))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    grasp_data_dir = Path(args.grasp_data_dir).expanduser().resolve()
    grasp_model_dir = Path(args.grasp_model_dir).expanduser().resolve()
    transport_data_dir = Path(
        args.transport_data_dir).expanduser().resolve()
    transport_model_dir = Path(
        args.transport_model_dir).expanduser().resolve()

    grasp = historical_phase_eval(
        "grasp", grasp_data_dir, grasp_model_dir)
    transport = historical_phase_eval(
        "transport", transport_data_dir, transport_model_dir)
    _, _, _, _, transport_train_report = load_artifacts(
        "transport", transport_model_dir)
    events = event_eval(
        transport_data_dir,
        transport_train_report,
        Path(args.event_log_dir).expanduser().resolve(),
        args.deadline_s,
    )

    report = {
        "design": {
            "channel_a_backbone": {
                "seq_len": 50,
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "grasp_a": "absolute 6D",
            "transport_a": "stable-baseline relative 9D",
            "shared_b": {
                "signal": B_SIGNAL,
                "reference_frames": B_REFERENCE_FRAMES,
                "consecutive": B_CONSECUTIVE,
                "warmup": B_WARMUP,
                "threshold_percentile": B_THRESHOLD_PERCENTILE,
                "phase_specific_scale_and_threshold": True,
            },
            "fusion": "A OR B",
        },
        "historical_episode_eval": {
            "grasp": grasp,
            "transport": transport,
        },
        "transport_event_eval": events,
        "limitations": [
            "Historical failure files have episode-level labels only.",
            "Historical raw files are cumulative range-normalised displacement.",
            "Event-labelled Transport results are the primary comparison for "
            "previous-frame versus stable-baseline Channel B.",
        ],
    }
    print_historical(grasp)
    print_historical(transport)
    print_event(events)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(report), stream, ensure_ascii=False, indent=2)
    print(f"\n[Done] {output}")


if __name__ == "__main__":
    main()
