#!/usr/bin/env python3
"""Leakage-resistant GRU-AE training for tactile one-class detection.

Key differences from train_gru_ae.py:
  * split success data by episode before making sliding windows;
  * fit StandardScaler on train-success windows only;
  * select threshold on validation-success windows only;
  * report final metrics on held-out test-success and failure episodes;
  * ignore inference-generated ``*_unknown*.npy`` files.

The feature and GRU implementations are imported from train_gru_ae.py so that
online feature generation in basket_ori.py remains numerically compatible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


LEGACY_TRAIN_DIR = Path("/home/dyros/mcy_ws")
if str(LEGACY_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_TRAIN_DIR))

from train_gru_ae import (  # noqa: E402
    GRUAE,
    apply_relative_transform,
    build_hybrid_window,
    compute_8d_from_raw,
    compute_9d_from_raw,
    compute_10d_from_raw,
    compute_11d_from_raw,
)
from transport_features import (  # noqa: E402
    CONTACT_MRL_9D_NAMES,
    TRANSPORT_9D_RELATIVE_NAMES,
    TRANSPORT_11D_CODEX_NAMES,
    compute_contact_mrl_9d,
    compute_transport_9d_relative,
    compute_transport_11d_delta,
)


@dataclass
class Episode:
    path: str
    features: np.ndarray


class WindowDataset(Dataset):
    def __init__(self, windows: list[np.ndarray]):
        self.windows = [torch.as_tensor(w, dtype=torch.float32) for w in windows]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Episode-split GRU-AE one-class trainer")
    parser.add_argument("--model", choices=["grasp", "insert", "transport"],
                        default="transport")
    parser.add_argument("--data_dir", required=True,
                        help="Ground-truth data directory; unknown files are ignored")
    parser.add_argument(
        "--output_dir", default=None,
        help="Artifact directory (default: DATA_DIR/codex_model). "
             "Use the data_dir itself only when intentionally deploying.")

    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument("--use_raw", action="store_true")
    feature_group.add_argument("--use_9d", action="store_true")
    feature_group.add_argument(
        "--use_c9d_unified", action="store_true",
        help="Shared C8D+MRL 9D representation for grasp and transport")
    feature_group.add_argument("--use_8d", action="store_true")
    feature_group.add_argument("--use_10d", action="store_true")
    feature_group.add_argument("--use_11d", action="store_true")
    feature_group.add_argument(
        "--use_9d_relative", action="store_true",
        help=("Stable-grasp relative C9D: MRL, delta F/M, "
              "changed-taxel count, peak delta norm"))
    feature_group.add_argument(
        "--use_11d_codex", action="store_true",
        help="Revised delta 11D: MRL, mean/top3/std delta norm, F, M, active count")

    # Accepted for command compatibility. This trainer always uses sliding
    # windows because that is the online inference representation.
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--win_stride", type=int, default=10,
                        help="Training/validation sliding-window stride")
    parser.add_argument("--infer_stride", type=int, default=None,
                        help="Episode evaluation stride (default: win_stride)")
    parser.add_argument("--split", choices=["chronological", "random"],
                        default="chronological",
                        help="Success episode split method")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--threshold_pct", type=float, default=99.0)
    parser.add_argument("--threshold_scope", choices=["window", "episode_max"],
                        default="window",
                        help="Validation-success statistic used for threshold")
    parser.add_argument("--relative", choices=["none", "window_mean", "first_frame"],
                        default="none")
    parser.add_argument("--weight_alpha", type=float, default=0.0)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--acontact_thresh", type=float, default=0.1)
    parser.add_argument("--delta_active_thresh", type=float, default=0.1,
                        help="Revised 11D active-taxel delta-norm threshold")
    parser.add_argument("--delta_active_sharpness", type=float, default=10.0,
                        help="Revised 11D soft active-count sigmoid sharpness")
    parser.add_argument("--baseline_frames", type=int, default=30,
                        help="Stable-grasp median frames for relative C9D")
    parser.add_argument("--change_threshold", type=float, default=0.1,
                        help="Relative C9D changed-taxel delta-norm threshold")
    parser.add_argument("--change_sharpness", type=float, default=10.0,
                        help="Relative C9D changed-taxel sigmoid sharpness")
    parser.add_argument("--mrl_thresh", type=float, default=0.3)
    parser.add_argument(
        "--disable_mrl_guard", action="store_true",
        help="Evaluate Channel A as a pure reconstruction detector")
    parser.add_argument("--cutoff_multiplier", type=float, default=2.2)
    parser.add_argument("--consec", type=int, default=2,
                        help="Regular-threshold consecutive inference count")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load/split/window data without training or writing artifacts")
    args = parser.parse_args()

    if args.infer_stride is None:
        args.infer_stride = args.win_stride
    if args.seq_len <= 0 or args.win_stride <= 0 or args.infer_stride <= 0:
        parser.error("seq_len and strides must be positive")
    if args.baseline_frames <= 0 or args.baseline_frames > args.seq_len:
        parser.error("baseline_frames must be in [1, seq_len]")
    if args.change_threshold <= 0 or args.change_sharpness <= 0:
        parser.error("change_threshold and change_sharpness must be positive")
    if not 0.0 < args.threshold_pct < 100.0:
        parser.error("threshold_pct must be between 0 and 100")
    if args.val_ratio <= 0 or args.test_ratio <= 0:
        parser.error("val_ratio and test_ratio must be positive")
    if args.val_ratio + args.test_ratio >= 1.0:
        parser.error("val_ratio + test_ratio must be less than 1")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_suffix(args: argparse.Namespace) -> str:
    if args.use_c9d_unified:
        return "_c9d_unified"
    if args.use_9d_relative:
        return "_9d_relative"
    if args.use_11d_codex:
        return "_11d_codex"
    if args.use_11d:
        return "_11d"
    if args.use_10d:
        return "_10d"
    if args.use_8d:
        return "_8d"
    if args.use_9d:
        return "_9d"
    if args.use_raw:
        return "_raw"
    return ""


def raw_feature_builder(args: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
    if args.use_c9d_unified:
        return lambda raw: compute_contact_mrl_9d(
            raw,
            active_threshold=args.acontact_thresh,
            active_sharpness=args.delta_active_sharpness,
        )
    if args.use_9d_relative:
        return lambda raw: compute_transport_9d_relative(
            raw,
            baseline_frames=args.baseline_frames,
            change_threshold=args.change_threshold,
            change_sharpness=args.change_sharpness,
        )
    if args.use_11d_codex:
        return lambda raw: compute_transport_11d_delta(
            raw,
            active_threshold=args.delta_active_thresh,
            active_sharpness=args.delta_active_sharpness,
        )
    if args.use_11d:
        return lambda raw: compute_11d_from_raw(raw, args.acontact_thresh)
    if args.use_10d:
        return lambda raw: compute_10d_from_raw(raw, args.acontact_thresh)
    if args.use_8d:
        return lambda raw: compute_8d_from_raw(raw, args.acontact_thresh)
    if args.use_9d:
        return compute_9d_from_raw
    if args.use_raw:
        return lambda raw: raw.astype(np.float32)
    raise ValueError("Raw feature builder requested for precomputed feature mode")


def load_episodes(data_dir: Path, prefix: str, label: str,
                  args: argparse.Namespace) -> list[Episode]:
    uses_raw_file = any((args.use_raw, args.use_9d, args.use_c9d_unified,
                         args.use_8d,
                         args.use_10d, args.use_11d, args.use_11d_codex,
                         args.use_9d_relative))
    if uses_raw_file:
        files = sorted(glob.glob(str(data_dir / f"{prefix}_*_{label}_raw.npy")))
        builder = raw_feature_builder(args)
    else:
        candidates = sorted(glob.glob(str(data_dir / f"{prefix}_*_{label}.npy")))
        files = [p for p in candidates
                 if not p.endswith(("_raw.npy", "_abs.npy"))]
        builder = lambda arr: arr.astype(np.float32)

    episodes: list[Episode] = []
    for path in files:
        arr = np.load(path)
        if arr.ndim != 2:
            print(f"[SKIP] {os.path.basename(path)}: shape={arr.shape}")
            continue
        feat = builder(arr)
        if feat.ndim != 2 or len(feat) < args.seq_len:
            print(f"[SKIP] {os.path.basename(path)}: feature shape={feat.shape}")
            continue
        if not np.isfinite(feat).all():
            print(f"[SKIP] {os.path.basename(path)}: non-finite feature")
            continue
        episodes.append(Episode(path=path, features=feat.astype(np.float32)))
    return episodes


def split_success_episodes(episodes: list[Episode], args: argparse.Namespace
                           ) -> tuple[list[Episode], list[Episode], list[Episode]]:
    if len(episodes) < 3:
        raise RuntimeError("At least three valid success episodes are required")
    ordered = list(episodes)
    if args.split == "random":
        rng = np.random.default_rng(args.seed)
        rng.shuffle(ordered)

    count = len(ordered)
    test_count = max(1, int(round(count * args.test_ratio)))
    val_count = max(1, int(round(count * args.val_ratio)))
    if test_count + val_count >= count:
        test_count = 1
        val_count = 1
    train_count = count - val_count - test_count
    train = ordered[:train_count]
    val = ordered[train_count:train_count + val_count]
    test = ordered[train_count + val_count:]
    return train, val, test


def episode_windows(episode: Episode, args: argparse.Namespace,
                    stride: int | None = None) -> list[np.ndarray]:
    actual_stride = args.win_stride if stride is None else stride
    windows = []
    for start in range(0, len(episode.features) - args.seq_len + 1, actual_stride):
        window = episode.features[start:start + args.seq_len]
        if args.hybrid:
            window = build_hybrid_window(window)
        windows.append(window.astype(np.float32))
    return windows


def collect_windows(episodes: list[Episode], args: argparse.Namespace,
                    stride: int | None = None) -> list[np.ndarray]:
    return [window for episode in episodes
            for window in episode_windows(episode, args, stride)]


def fit_train_scaler(train_windows: list[np.ndarray]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.vstack(train_windows))
    return scaler


def preprocess_windows(windows: list[np.ndarray], scaler: StandardScaler,
                       args: argparse.Namespace, n_rel_dims: int | None
                       ) -> list[np.ndarray]:
    result = []
    for window in windows:
        scaled = scaler.transform(window).astype(np.float32)
        transformed = apply_relative_transform(
            scaled, args.relative, n_rel_dims=n_rel_dims)
        result.append(transformed.astype(np.float32))
    return result


def temporal_mse(recon: torch.Tensor, target: torch.Tensor,
                 alpha: float) -> torch.Tensor:
    frame_mse = ((recon - target) ** 2).mean(dim=2)
    if alpha <= 0:
        return frame_mse.mean()
    weights = torch.exp(torch.linspace(-alpha, 0.0, target.shape[1],
                                       device=target.device))
    weights = weights / weights.sum()
    return (frame_mse * weights.unsqueeze(0)).sum(dim=1).mean()


def train_model(model: GRUAE, train_windows: list[np.ndarray],
                val_windows: list[np.ndarray], args: argparse.Namespace,
                checkpoint_path: Path) -> tuple[list[float], list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(
        WindowDataset(train_windows), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(
        WindowDataset(val_windows), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    best_loss = float("inf")
    patience_count = 0
    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = temporal_mse(model(batch), batch, args.weight_alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * len(batch)
        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = temporal_mse(model(batch), batch, args.weight_alpha)
                val_sum += loss.item() * len(batch)
        val_loss = val_sum / len(val_loader.dataset)
        train_history.append(train_loss)
        val_history.append(val_loss)
        scheduler.step(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[EarlyStopping] epoch={epoch}, best_val={best_loss:.6f}")
                break
    return train_history, val_history


def score_preprocessed(model: GRUAE, windows: list[np.ndarray],
                       args: argparse.Namespace) -> np.ndarray:
    if not windows:
        return np.empty(0, dtype=np.float64)
    device = next(model.parameters()).device
    loader = DataLoader(WindowDataset(windows), batch_size=256, shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            frame_mse = ((batch - recon) ** 2).mean(dim=2)
            if args.weight_alpha > 0:
                weights = torch.exp(torch.linspace(
                    -args.weight_alpha, 0.0, batch.shape[1], device=device))
                weights = weights / weights.sum()
                error = (frame_mse * weights.unsqueeze(0)).sum(dim=1)
            else:
                error = frame_mse.mean(dim=1)
            scores.extend(error.cpu().numpy())
    return np.asarray(scores, dtype=np.float64)


def score_episode(model: GRUAE, episode: Episode, scaler: StandardScaler,
                  args: argparse.Namespace, n_rel_dims: int | None
                  ) -> tuple[np.ndarray, list[int]]:
    starts = list(range(0, len(episode.features) - args.seq_len + 1,
                        args.infer_stride))
    windows = []
    for start in starts:
        window = episode.features[start:start + args.seq_len]
        if args.hybrid:
            window = build_hybrid_window(window)
        windows.append(window.astype(np.float32))
    prepared = preprocess_windows(windows, scaler, args, n_rel_dims)
    return score_preprocessed(model, prepared, args), starts


def runtime_detect(scores: np.ndarray, starts: list[int], episode: Episode,
                   threshold: float, args: argparse.Namespace,
                   has_mrl: bool) -> tuple[bool, int | None, str | None]:
    consecutive = 0
    for index, (score, start) in enumerate(zip(scores, starts)):
        rule1 = False
        if has_mrl:
            observed = episode.features[:start + args.seq_len, 0]
            rule1 = float(np.max(observed)) < args.mrl_thresh
        rule2 = score > threshold * args.cutoff_multiplier
        rule3 = score > threshold
        if rule1 or rule2:
            consecutive = args.consec
        elif rule3:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= args.consec:
            reason = "rule1" if rule1 else ("rule2" if rule2 else "rule3")
            return True, start + args.seq_len - 1, reason
    return False, None, None


def threshold_from_validation(model: GRUAE, val_episodes: list[Episode],
                              scaler: StandardScaler, args: argparse.Namespace,
                              n_rel_dims: int | None) -> tuple[float, np.ndarray]:
    per_episode = []
    for episode in val_episodes:
        scores, _ = score_episode(model, episode, scaler, args, n_rel_dims)
        if len(scores):
            per_episode.append(scores)
    if not per_episode:
        raise RuntimeError("No validation windows available for threshold")
    all_scores = np.concatenate(per_episode)
    calibration = (np.asarray([scores.max() for scores in per_episode])
                   if args.threshold_scope == "episode_max" else all_scores)
    threshold = float(np.percentile(calibration, args.threshold_pct))
    print(f"[Threshold] scope={args.threshold_scope}, "
          f"p{args.threshold_pct:g}={threshold:.6f}, n={len(calibration)}")
    return threshold, all_scores


def evaluate_episodes(model: GRUAE, groups: dict[str, list[Episode]],
                      scaler: StandardScaler, threshold: float,
                      args: argparse.Namespace, n_rel_dims: int | None,
                      has_mrl: bool) -> tuple[dict, dict[str, np.ndarray]]:
    metrics = {}
    score_groups: dict[str, np.ndarray] = {}
    for label, episodes in groups.items():
        detected = 0
        max_scores = []
        window_scores = []
        reasons: dict[str, int] = {}
        details = []
        for episode in episodes:
            scores, starts = score_episode(model, episode, scaler, args, n_rel_dims)
            if not len(scores):
                continue
            hit, frame, reason = runtime_detect(
                scores, starts, episode, threshold, args, has_mrl)
            detected += int(hit)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            max_scores.append(float(scores.max()))
            window_scores.extend(scores.tolist())
            details.append({
                "file": os.path.basename(episode.path),
                "frames": len(episode.features),
                "detected": hit,
                "detection_frame": frame,
                "reason": reason,
                "max_score": float(scores.max()),
                "mean_score": float(scores.mean()),
            })
        total = len(details)
        metrics[label] = {
            "episodes": total,
            "detected": detected,
            "detection_rate": detected / total if total else None,
            "first_reason": reasons,
            "details": details,
        }
        score_groups[label] = np.asarray(window_scores)
        print(f"[Episode Eval] {label:<13} {detected}/{total} "
              f"({detected / total * 100 if total else 0:.1f}%)")
    return metrics, score_groups


def save_plots(output_dir: Path, prefix: str, suffix: str,
               train_history: list[float], val_history: list[float],
               score_groups: dict[str, np.ndarray], threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(train_history, label="train")
    axes[0].plot(val_history, label="validation")
    axes[0].set_yscale("log")
    axes[0].set_title("Episode-split learning curve")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    colors = {"test_success": "steelblue", "failure": "crimson"}
    for label, scores in score_groups.items():
        if len(scores):
            axes[1].hist(scores, bins=35, alpha=0.55,
                         label=label, color=colors.get(label))
    axes[1].axvline(threshold, color="black", linestyle="--",
                    label=f"threshold={threshold:.4f}")
    axes[1].set_title("Held-out window scores")
    axes[1].set_xlabel("reconstruction error")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_gru_ae{suffix}_codex_eval.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = (Path(args.output_dir).expanduser().resolve()
                  if args.output_dir else data_dir / "codex_model")
    prefix = args.model
    suffix = feature_suffix(args)

    success = load_episodes(data_dir, prefix, "success", args)
    failure = load_episodes(data_dir, prefix, "failure", args)
    train_eps, val_eps, test_eps = split_success_episodes(success, args)
    print(f"[Episode Split] method={args.split}, success total={len(success)} | "
          f"train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")
    print(f"[Ground Truth] failure={len(failure)}, unknown files are not loaded")

    train_windows_raw = collect_windows(train_eps, args)
    val_windows_raw = collect_windows(val_eps, args)
    test_windows_raw = collect_windows(test_eps, args)
    if not train_windows_raw or not val_windows_raw or not test_windows_raw:
        raise RuntimeError("One or more success splits contain no valid windows")
    print(f"[Sliding Windows] seq_len={args.seq_len}, train/val stride={args.win_stride}, "
          f"infer stride={args.infer_stride}")
    print(f"[Window Counts] train={len(train_windows_raw)}, "
          f"val={len(val_windows_raw)}, test={len(test_windows_raw)}")

    if args.dry_run:
        print("[Dry Run] Data pipeline is valid; no artifacts were written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = fit_train_scaler(train_windows_raw)
    raw_feature_dim = train_windows_raw[0].shape[1] - (1 if args.hybrid else 0)
    n_rel_dims = raw_feature_dim if args.hybrid else None
    train_windows = preprocess_windows(train_windows_raw, scaler, args, n_rel_dims)
    val_windows = preprocess_windows(val_windows_raw, scaler, args, n_rel_dims)

    input_dim = train_windows[0].shape[1]
    model = GRUAE(input_dim=input_dim, hidden_dim=args.hidden_dim,
                  num_layers=args.num_layers, dropout=args.dropout,
                  seq_len=args.seq_len)
    checkpoint_path = output_dir / f"{prefix}_gru_ae{suffix}.pt"
    scaler_path = output_dir / f"{prefix}_scaler{suffix}.pkl"
    threshold_path = output_dir / f"{prefix}_threshold{suffix}.npy"
    config_path = output_dir / f"{prefix}_infer_cfg{suffix}.pkl"
    joblib.dump(scaler, scaler_path)

    print(f"[Model] input_dim={input_dim}, params={sum(p.numel() for p in model.parameters()):,}")
    train_history, val_history = train_model(
        model, train_windows, val_windows, args, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    threshold, val_scores = threshold_from_validation(
        model, val_eps, scaler, args, n_rel_dims)
    np.save(threshold_path, np.asarray([threshold], dtype=np.float64))

    infer_cfg = {
        "relative": args.relative,
        "weight_alpha": args.weight_alpha,
        "use_raw": args.use_raw,
        "use_9d": args.use_9d,
        "use_c9d_unified": args.use_c9d_unified,
        "use_8d": args.use_8d,
        "use_10d": args.use_10d,
        "use_11d": args.use_11d,
        "use_11d_codex": args.use_11d_codex,
        "use_9d_relative": args.use_9d_relative,
        "acontact_thresh": args.acontact_thresh,
        "delta_active_thresh": args.delta_active_thresh,
        "delta_active_sharpness": args.delta_active_sharpness,
        "baseline_frames": args.baseline_frames,
        "change_threshold": args.change_threshold,
        "change_sharpness": args.change_sharpness,
        "seq_len": args.seq_len,
        "hybrid": args.hybrid,
        "n_rel_dims": n_rel_dims,
        "mrl_thresh": args.mrl_thresh,
        "mrl_guard_enabled": (
            not args.disable_mrl_guard and not args.use_9d_relative),
        "cutoff_multiplier": args.cutoff_multiplier,
        "consec": args.consec,
        "train_window_stride": args.win_stride,
        "inference_stride": args.infer_stride,
        "threshold_pct": args.threshold_pct,
        "threshold_scope": args.threshold_scope,
        "episode_split": args.split,
        "seed": args.seed,
    }
    with open(config_path, "wb") as stream:
        pickle.dump(infer_cfg, stream)

    # Relative C9D contains change-vector MRL, not absolute contact MRL, so it
    # must not drive the "never grasped" miss guard.
    has_mrl = (
        not args.disable_mrl_guard
        and not (args.use_raw or args.use_8d or args.use_9d_relative))
    metrics, score_groups = evaluate_episodes(
        model,
        {"test_success": test_eps, "failure": failure},
        scaler, threshold, args, n_rel_dims, has_mrl)

    if len(score_groups.get("test_success", [])) and len(score_groups.get("failure", [])):
        y_true = np.concatenate((
            np.zeros(len(score_groups["test_success"]), dtype=int),
            np.ones(len(score_groups["failure"]), dtype=int)))
        y_score = np.concatenate((
            score_groups["test_success"], score_groups["failure"]))
        metrics["window_roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["window_average_precision"] = float(
            average_precision_score(y_true, y_score))
        print(f"[Window Eval] ROC-AUC={metrics['window_roc_auc']:.4f}, "
              f"AP={metrics['window_average_precision']:.4f}")

    manifest = {
        "train_success": [os.path.basename(ep.path) for ep in train_eps],
        "validation_success": [os.path.basename(ep.path) for ep in val_eps],
        "test_success": [os.path.basename(ep.path) for ep in test_eps],
        "test_failure": [os.path.basename(ep.path) for ep in failure],
    }
    if args.use_c9d_unified:
        feature_names = list(CONTACT_MRL_9D_NAMES)
    elif args.use_9d_relative:
        feature_names = list(TRANSPORT_9D_RELATIVE_NAMES)
    elif args.use_11d_codex:
        feature_names = list(TRANSPORT_11D_CODEX_NAMES)
    else:
        feature_names = None

    report = {
        "config": vars(args),
        "feature_names": feature_names,
        "threshold": threshold,
        "validation_score_mean": float(val_scores.mean()),
        "validation_score_max": float(val_scores.max()),
        "metrics": metrics,
        "manifest": manifest,
    }
    with open(output_dir / f"{prefix}_codex_report{suffix}.json", "w") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)

    if args.eval:
        save_plots(output_dir, prefix, suffix, train_history, val_history,
                   score_groups, threshold)
    print(f"[Done] artifacts: {output_dir}")
    print(f"[Deploy] basket_ori.py loads artifacts from {data_dir}; "
          "copy/deploy only after reviewing held-out metrics.")


if __name__ == "__main__":
    main()
