#!/usr/bin/env python3
"""Diagnose transport GRU-AE detections episode by episode.

The script loads a ``train_gru_ae_codex.py`` artifact directory, reproduces
runtime decisions, and writes:

* one time-series plot per episode;
* a compact score overview;
* CSV and JSON summaries for TP/FN/FP/TN review.

Without an external event annotation, ``sensor_candidate_frame`` is only the
largest frame-to-frame change in scaler-normalised 11D features.  It must not
be interpreted as ground-truth laundry-loss timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_gru_ae_codex import (
    GRUAE,
    Episode,
    load_episodes,
    runtime_detect,
    score_episode,
)


DEFAULT_FEATURE_NAMES = (
    "mrl",
    "mean_delta_norm",
    "top3_delta_norm",
    "delta_fx",
    "delta_fy",
    "delta_fz",
    "delta_mx",
    "delta_my",
    "delta_mz",
    "active_delta_count",
    "std_delta_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create TP/FN/FP/TN time-series diagnostics for Codex 11D")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument(
        "--output_dir", default=None,
        help="Default: ARTIFACT_DIR/diagnostics")
    parser.add_argument(
        "--labels", nargs="+",
        choices=["failure", "test_success", "validation_success"],
        default=["failure", "test_success"],
        help="Episode groups to diagnose")
    parser.add_argument(
        "--event_csv", default=None,
        help="Optional CSV containing file,event_frame ground-truth columns")
    parser.add_argument(
        "--dpi", type=int, default=130,
        help="Individual plot resolution")
    return parser.parse_args()


def load_event_frames(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    result: dict[str, int] = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"file", "event_frame"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("event CSV must contain file,event_frame columns")
        for row in reader:
            result[os.path.basename(row["file"])] = int(row["event_frame"])
    return result


def find_artifact(artifact_dir: Path, stem: str) -> Path:
    matches = sorted(artifact_dir.glob(stem))
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise RuntimeError(
            f"Expected exactly one artifact matching {stem!r}; found {names}")
    return matches[0]


def load_bundle(artifact_dir: Path):
    report_path = find_artifact(
        artifact_dir, "transport_codex_report_11d_codex.json")
    model_path = find_artifact(
        artifact_dir, "transport_gru_ae_11d_codex.pt")
    scaler_path = find_artifact(
        artifact_dir, "transport_scaler_11d_codex.pkl")
    threshold_path = find_artifact(
        artifact_dir, "transport_threshold_11d_codex.npy")
    config_path = find_artifact(
        artifact_dir, "transport_infer_cfg_11d_codex.pkl")

    with open(report_path) as stream:
        report = json.load(stream)
    with open(config_path, "rb") as stream:
        infer_cfg = pickle.load(stream)

    # Training arguments contain architecture fields; inference settings from
    # the separately saved config remain the runtime source of truth.
    config = dict(report["config"])
    config.update(infer_cfg)
    args = argparse.Namespace(**config)
    scaler = joblib.load(scaler_path)
    threshold = float(np.asarray(np.load(threshold_path)).reshape(-1)[0])

    model = GRUAE(
        input_dim=int(scaler.n_features_in_),
        hidden_dim=int(report["config"]["hidden_dim"]),
        num_layers=int(report["config"]["num_layers"]),
        dropout=float(report["config"]["dropout"]),
        seq_len=int(args.seq_len),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return report, args, scaler, threshold, model


def select_episodes(data_dir: Path, report: dict, args: argparse.Namespace,
                    labels: list[str]) -> list[tuple[str, Episode]]:
    success = {
        os.path.basename(ep.path): ep
        for ep in load_episodes(data_dir, "transport", "success", args)
    }
    failure = {
        os.path.basename(ep.path): ep
        for ep in load_episodes(data_dir, "transport", "failure", args)
    }
    manifest = report["manifest"]
    selected: list[tuple[str, Episode]] = []

    group_specs = {
        "failure": ("test_failure", failure),
        "test_success": ("test_success", success),
        "validation_success": ("validation_success", success),
    }
    for label in labels:
        manifest_key, episode_map = group_specs[label]
        for filename in manifest[manifest_key]:
            episode = episode_map.get(filename)
            if episode is None:
                print(f"[SKIP] Missing episode from manifest: {filename}")
                continue
            selected.append((label, episode))
    return selected


def diagnostic_category(detected: bool, score_count: int, max_score: float,
                        threshold: float) -> str:
    if detected:
        return "detected"
    if score_count:
        return "isolated_above_threshold"
    if max_score >= 0.5 * threshold:
        return "subthreshold_near"
    return "low_score"


def analyse_episode(label: str, episode: Episode, model: GRUAE, scaler,
                    threshold: float, args: argparse.Namespace,
                    feature_names: tuple[str, ...],
                    event_frames: dict[str, int]) -> tuple[dict, dict]:
    scores, starts = score_episode(model, episode, scaler, args, None)
    score_frames = np.asarray(starts, dtype=int) + args.seq_len - 1
    detected, detection_frame, reason = runtime_detect(
        scores, starts, episode, threshold, args, has_mrl=True)

    scaled = scaler.transform(episode.features).astype(np.float32)
    feature_step = np.zeros(len(scaled), dtype=np.float32)
    if len(scaled) > 1:
        feature_step[1:] = np.linalg.norm(
            np.diff(scaled, axis=0), axis=1) / np.sqrt(scaled.shape[1])
    # Frames before seq_len - 1 cannot produce an online GRU decision and are
    # commonly dominated by sensor/baseline settling.  Diagnose the largest
    # observable transition only after the first complete window.
    candidate_search_start = min(args.seq_len - 1, len(feature_step) - 1)
    sensor_candidate = int(
        candidate_search_start +
        np.argmax(feature_step[candidate_search_start:])
    )
    dominant_index = int(np.argmax(np.abs(
        scaled[sensor_candidate] - scaled[max(0, sensor_candidate - 1)])))
    max_score_index = int(np.argmax(scores))
    max_score_frame = int(score_frames[max_score_index])
    filename = os.path.basename(episode.path)
    event_frame = event_frames.get(filename)

    is_failure = label == "failure"
    outcome = (
        "TP" if is_failure and detected else
        "FN" if is_failure else
        "FP" if detected else
        "TN"
    )
    score_count = int(np.count_nonzero(scores > threshold))
    row = {
        "file": filename,
        "label": label,
        "outcome": outcome,
        "category": diagnostic_category(
            detected, score_count, float(scores[max_score_index]), threshold),
        "frames": int(len(episode.features)),
        "windows": int(len(scores)),
        "detected": bool(detected),
        "detection_frame": detection_frame,
        "detection_reason": reason,
        "event_frame": event_frame,
        "detection_delay": (
            int(detection_frame - event_frame)
            if detection_frame is not None and event_frame is not None else None
        ),
        "max_score": float(scores[max_score_index]),
        "mean_score": float(scores.mean()),
        "max_score_frame": max_score_frame,
        "windows_above_threshold": score_count,
        "windows_above_cutoff": int(np.count_nonzero(
            scores > threshold * args.cutoff_multiplier)),
        "sensor_candidate_frame": sensor_candidate,
        "sensor_candidate_search_start": candidate_search_start,
        "sensor_candidate_step": float(feature_step[sensor_candidate]),
        "sensor_candidate_dominant_feature": feature_names[dominant_index],
    }
    series = {
        "features": episode.features,
        "scaled_features": scaled,
        "feature_step": feature_step,
        "scores": scores,
        "score_frames": score_frames,
    }
    return row, series


def add_frame_markers(ax, row: dict) -> None:
    ax.axvline(
        row["sensor_candidate_frame"], color="darkorange", linestyle=":",
        linewidth=1.1, label="sensor candidate")
    if row["detection_frame"] is not None:
        ax.axvline(
            row["detection_frame"], color="crimson", linestyle="-.",
            linewidth=1.1, label="detection")
    if row["event_frame"] is not None:
        ax.axvline(
            row["event_frame"], color="black", linestyle="--",
            linewidth=1.2, label="ground-truth event")


def plot_episode(output_path: Path, row: dict, series: dict,
                 threshold: float, cutoff_multiplier: float,
                 feature_names: tuple[str, ...], dpi: int) -> None:
    t = np.arange(row["frames"])
    features = series["features"]
    scaled = series["scaled_features"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        f'{row["outcome"]} | {row["file"]}\n'
        f'max={row["max_score"]:.4f}, category={row["category"]}',
        fontsize=11)

    ax = axes[0]
    ax.plot(series["score_frames"], series["scores"], marker="o",
            markersize=2.5, linewidth=1.1, color="navy")
    ax.axhline(threshold, color="crimson", linestyle="--",
               label=f"threshold={threshold:.4f}")
    ax.axhline(threshold * cutoff_multiplier, color="purple", linestyle=":",
               label=f"single cutoff={threshold * cutoff_multiplier:.4f}")
    add_frame_markers(ax, row)
    ax.set_ylabel("GRU-AE score")
    ax.legend(fontsize=7, ncol=5)
    ax.grid(alpha=0.2)

    ax = axes[1]
    for index in (0, 1, 2, 9, 10):
        ax.plot(t, scaled[:, index], linewidth=0.9,
                label=f"{feature_names[index]} (z)")
    add_frame_markers(ax, row)
    ax.set_ylabel("contact features\ntrain z-score")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.2)

    ax = axes[2]
    for index in (3, 4, 5):
        ax.plot(t, scaled[:, index], linewidth=0.9,
                label=f"{feature_names[index]} (z)")
    add_frame_markers(ax, row)
    ax.set_ylabel("resultant force\ntrain z-score")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.2)

    ax = axes[3]
    ax.plot(t, series["feature_step"], color="darkorange", linewidth=1.0,
            label="11D frame step")
    for index in (6, 7, 8):
        ax.plot(t, scaled[:, index], linewidth=0.75, alpha=0.7,
                label=f"{feature_names[index]} (z)")
    add_frame_markers(ax, row)
    ax.set_ylabel("moment z-score /\nfeature step")
    ax.set_xlabel("frame")
    ax.legend(fontsize=7, ncol=5)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_overview(path: Path, rows: list[dict], threshold: float) -> None:
    ordered = sorted(
        rows, key=lambda row: (row["label"], row["max_score"]), reverse=True)
    colors = {
        "TP": "crimson",
        "FN": "salmon",
        "FP": "darkorange",
        "TN": "steelblue",
    }
    labels = [
        row["file"].replace("transport_", "").replace("_raw.npy", "")
        for row in ordered
    ]
    values = [row["max_score"] for row in ordered]
    fig_height = max(6.0, len(rows) * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(rows))
    ax.barh(y, values, color=[colors[row["outcome"]] for row in ordered])
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"threshold={threshold:.4f}")
    ax.set_yticks(y, labels=labels, fontsize=7)
    ax.set_xlabel("episode maximum reconstruction score")
    ax.set_title("Transport Codex 11D episode diagnostics")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args_cli = parse_args()
    data_dir = Path(args_cli.data_dir).expanduser().resolve()
    artifact_dir = Path(args_cli.artifact_dir).expanduser().resolve()
    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir else artifact_dir / "diagnostics"
    )
    plots_dir = output_dir / "episodes"

    report, args, scaler, threshold, model = load_bundle(artifact_dir)
    feature_names = tuple(report.get("feature_names") or DEFAULT_FEATURE_NAMES)
    if len(feature_names) != scaler.n_features_in_:
        raise RuntimeError("Feature names and scaler dimensions do not match")
    event_frames = load_event_frames(args_cli.event_csv)
    episodes = select_episodes(data_dir, report, args, args_cli.labels)
    if not episodes:
        raise RuntimeError("No episodes selected")

    plots_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for label, episode in episodes:
        row, series = analyse_episode(
            label, episode, model, scaler, threshold, args,
            feature_names, event_frames)
        rows.append(row)
        plot_episode(
            plots_dir / f'{row["outcome"]}_{Path(row["file"]).stem}.png',
            row, series, threshold, args.cutoff_multiplier,
            feature_names, args_cli.dpi)

    write_csv(output_dir / "transport_diagnostics.csv", rows)
    payload = {
        "artifact_dir": str(artifact_dir),
        "threshold": threshold,
        "cutoff_multiplier": args.cutoff_multiplier,
        "consecutive": args.consec,
        "inference_stride": args.inference_stride,
        "event_annotation_supplied": bool(event_frames),
        "sensor_candidate_note": (
            "Largest frame-to-frame change in scaler-normalised 11D after "
            "the first complete GRU window; not a ground-truth loss event."),
        "episodes": rows,
    }
    with open(output_dir / "transport_diagnostics.json", "w") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    plot_overview(output_dir / "transport_diagnostics_overview.png",
                  rows, threshold)

    counts = {
        outcome: sum(row["outcome"] == outcome for row in rows)
        for outcome in ("TP", "FN", "FP", "TN")
    }
    categories: dict[str, int] = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    print(f"[Model] threshold={threshold:.6f}, "
          f"cutoff={threshold * args.cutoff_multiplier:.6f}, "
          f"consec={args.consec}, stride={args.inference_stride}")
    print(f"[Outcome] {counts}")
    print(f"[Categories] {categories}")
    print(f"[Done] {output_dir}")


if __name__ == "__main__":
    main()
