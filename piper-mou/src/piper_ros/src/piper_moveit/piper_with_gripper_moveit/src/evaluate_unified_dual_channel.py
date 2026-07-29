#!/usr/bin/env python3
"""Offline evaluation of the shared C9D dual-channel detector.

Channel A uses the same C8D+MRL GRU-AE architecture for grasp and transport,
with phase-specific weights. Channel B uses the same framewise coherent-contact
change rule in both phases. Historical raw files are cumulative displacement,
not calibrated absolute force, so this evaluator cannot reproduce the newer
absolute-contact guard used by the online transport detector.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


LEGACY_ROOT = Path("/home/dyros/mcy_ws")
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from train_gru_ae import GRUAE, apply_relative_transform  # noqa: E402
from transport_features import compute_contact_mrl_9d  # noqa: E402


SUFFIX = "_c9d_unified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["grasp", "transport"],
                        required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--b_threshold_pct", type=float, default=95.0)
    parser.add_argument("--b_warmup", type=int, default=30)
    parser.add_argument("--b_consec", type=int, default=5)
    parser.add_argument("--b_smooth", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not 0 < args.b_threshold_pct < 100:
        parser.error("--b_threshold_pct must be between 0 and 100")
    if min(args.b_warmup, args.b_consec, args.b_smooth) <= 0:
        parser.error("B parameters must be positive")
    return args


def load_artifacts(args: argparse.Namespace):
    model_dir = Path(args.model_dir).expanduser().resolve()
    prefix = args.phase
    report_path = model_dir / f"{prefix}_codex_report{SUFFIX}.json"
    checkpoint_path = model_dir / f"{prefix}_gru_ae{SUFFIX}.pt"
    scaler_path = model_dir / f"{prefix}_scaler{SUFFIX}.pkl"
    threshold_path = model_dir / f"{prefix}_threshold{SUFFIX}.npy"
    config_path = model_dir / f"{prefix}_infer_cfg{SUFFIX}.pkl"
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    with config_path.open("rb") as stream:
        infer_cfg = pickle.load(stream)
    config = report["config"]
    scaler = joblib.load(scaler_path)
    model = GRUAE(
        input_dim=int(scaler.n_features_in_),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        seq_len=int(config["seq_len"]),
    )
    model.load_state_dict(torch.load(
        checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    threshold_a = float(np.load(threshold_path)[0])
    return model, scaler, threshold_a, infer_cfg, report


def load_feature(path: Path, active_threshold: float,
                 active_sharpness: float) -> np.ndarray:
    raw = np.load(path, allow_pickle=False)
    return compute_contact_mrl_9d(
        raw,
        active_threshold=active_threshold,
        active_sharpness=active_sharpness,
    )


def channel_a(
    feature: np.ndarray,
    model: GRUAE,
    scaler,
    threshold: float,
    infer_cfg: dict[str, Any],
) -> tuple[bool, int | None, float]:
    seq_len = int(infer_cfg["seq_len"])
    stride = int(infer_cfg["inference_stride"])
    consecutive_needed = int(
        infer_cfg.get("consec", 2))
    cutoff = float(infer_cfg.get("cutoff_multiplier", 2.2))
    relative = str(infer_cfg.get("relative", "none"))
    weight_alpha = float(infer_cfg.get("weight_alpha", 0.0))
    consecutive = 0
    max_score = float("-inf")
    for start in range(0, len(feature) - seq_len + 1, stride):
        window = scaler.transform(
            feature[start:start + seq_len]).astype(np.float32)
        window = apply_relative_transform(window, relative)
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


def channel_b_deltas(feature: np.ndarray, smooth: int) -> np.ndarray:
    # Shared fast signal: directional coherence weighted by active contact.
    signal = feature[:, 0] * feature[:, 7] / 16.0
    history: deque[float] = deque(maxlen=smooth)
    deltas = np.zeros(len(signal), dtype=np.float64)
    previous = None
    for index, value_raw in enumerate(signal):
        value = float(value_raw)
        if previous is None:
            previous = value
            history.append(value)
            continue
        reference = (
            float(np.mean(history))
            if len(history) == smooth else previous)
        deltas[index] = abs(value - reference)
        history.append(value)
        previous = value
    return deltas


def b_episode_stat(deltas: np.ndarray, warmup: int, consecutive: int) -> float:
    usable = deltas[warmup:]
    if len(usable) < consecutive:
        return 0.0
    # Largest threshold for which a consecutive-length run would trigger.
    return max(
        float(np.min(usable[start:start + consecutive]))
        for start in range(len(usable) - consecutive + 1)
    )


def channel_b(deltas: np.ndarray, threshold: float, warmup: int,
              consecutive_needed: int) -> tuple[bool, int | None, float]:
    consecutive = 0
    max_delta = float(np.max(deltas[warmup:])) if len(deltas) > warmup else 0.0
    for frame in range(warmup, len(deltas)):
        if deltas[frame] > threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= consecutive_needed:
            return True, frame, max_delta
    return False, None, max_delta


def confusion(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    tp = sum(row["label"] == 1 and row[key] for row in rows)
    fn = sum(row["label"] == 1 and not row[key] for row in rows)
    fp = sum(row["label"] == 0 and row[key] for row in rows)
    tn = sum(row["label"] == 0 and not row[key] for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None
        and precision + recall else None)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    model, scaler, threshold_a, infer_cfg, train_report = load_artifacts(args)
    train_cfg = train_report["config"]
    active_threshold = float(train_cfg["acontact_thresh"])
    active_sharpness = float(train_cfg["delta_active_sharpness"])

    validation_features = []
    for name in train_report["manifest"]["validation_success"]:
        validation_features.append(load_feature(
            data_dir / name, active_threshold, active_sharpness))
    validation_b_stats = np.asarray([
        b_episode_stat(
            channel_b_deltas(feature, args.b_smooth),
            args.b_warmup, args.b_consec)
        for feature in validation_features
    ])
    threshold_b = float(np.percentile(
        validation_b_stats, args.b_threshold_pct))

    rows: list[dict[str, Any]] = []
    groups = (
        (0, "test_success", train_report["manifest"]["test_success"]),
        (1, "failure", train_report["manifest"]["test_failure"]),
    )
    for label, group, names in groups:
        for name in names:
            feature = load_feature(
                data_dir / name, active_threshold, active_sharpness)
            a_hit, a_frame, max_a = channel_a(
                feature, model, scaler, threshold_a, infer_cfg)
            deltas = channel_b_deltas(feature, args.b_smooth)
            b_hit, b_frame, max_b = channel_b(
                deltas, threshold_b, args.b_warmup, args.b_consec)
            rows.append({
                "file": name,
                "group": group,
                "label": label,
                "frames": len(feature),
                "a": a_hit,
                "b": b_hit,
                "ab": a_hit or b_hit,
                "a_frame": a_frame,
                "b_frame": b_frame,
                "max_a": max_a,
                "max_b": max_b,
            })

    metrics = {
        key: confusion(rows, key) for key in ("a", "b", "ab")
    }
    failures = [row for row in rows if row["label"] == 1]
    unique = {
        "both": sum(row["a"] and row["b"] for row in failures),
        "a_only": sum(row["a"] and not row["b"] for row in failures),
        "b_only": sum(row["b"] and not row["a"] for row in failures),
        "neither": sum(not row["a"] and not row["b"] for row in failures),
    }
    report = {
        "phase": args.phase,
        "feature": "shared C8D+MRL 9D",
        "threshold_a": threshold_a,
        "threshold_b": threshold_b,
        "b_calibration_episode_stats": validation_b_stats.tolist(),
        "b_config": {
            "signal": "MRL * active_contact / 16",
            "threshold_pct": args.b_threshold_pct,
            "warmup": args.b_warmup,
            "consecutive": args.b_consec,
            "smooth": args.b_smooth,
        },
        "metrics": metrics,
        "failure_overlap": unique,
        "rows": rows,
        "limitations": [
            "Failure labels cover whole episodes and contain no event onset.",
            "Historical raw files are cumulative displacement, not calibrated absolute force.",
            "The absolute initial-contact and force-drop guard cannot be evaluated from these files.",
        ],
    }

    print(f"[Unified C9D] phase={args.phase}")
    print(f"[Thresholds] A={threshold_a:.6f}, B={threshold_b:.6f} "
          f"(validation episodes={len(validation_b_stats)})")
    print("Config  Precision  Recall     F1        TP FP FN TN")
    for key, label in (("a", "A"), ("b", "B"), ("ab", "A OR B")):
        value = metrics[key]
        print(
            f"{label:<7} {fmt(value['precision']):>9} "
            f"{fmt(value['recall']):>8} {fmt(value['f1']):>9}  "
            f"{value['tp']:>2} {value['fp']:>2} "
            f"{value['fn']:>2} {value['tn']:>2}")
    print("[Failure overlap] " + ", ".join(
        f"{key}={value}" for key, value in unique.items()))

    output = (
        Path(args.output).expanduser().resolve()
        if args.output else
        Path(args.model_dir).expanduser().resolve()
        / f"{args.phase}_unified_dual_channel_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f"[Done] {output}")


if __name__ == "__main__":
    main()
