"""
GRU Autoencoder 학습 스크립트
==============================
One-class anomaly detection: success 데이터만으로 학습하여
재구성 오차(reconstruction error)를 anomaly score로 사용.

Usage:
  # grasp 모델 학습 (6D feature)
  python -m tactile_anomaly.train_gru_ae --model grasp

  # insert 모델 학습
  python -m tactile_anomaly.train_gru_ae --model insert

  # raw 48D taxel 데이터로 학습
  python -m tactile_anomaly.train_gru_ae --model grasp --use_raw

  # 학습 후 평가 곡선/분포 저장
  python -m tactile_anomaly.train_gru_ae --model grasp --eval
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  하이퍼파라미터 기본값
# ─────────────────────────────────────────────
DEFAULTS = dict(
    seq_len      = 50,       # 고정 시퀀스 길이 (리샘플링 목표)
    hidden_dim   = 64,       # GRU hidden size
    num_layers   = 2,        # GRU 레이어 수
    dropout      = 0.2,
    batch_size   = 32,
    epochs       = 1000,
    lr           = 1e-3,
    weight_decay = 1e-4,
    patience     = 20,       # early stopping
    threshold_pct= 95,       # 학습 재구성 오차의 N 백분위수를 임계값으로 사용
    val_ratio    = 0.15,
)


class TactileDataset(Dataset):

    def __init__(self, sequences: list[np.ndarray]):
        self.data = [torch.tensor(s, dtype=torch.float32) for s in sequences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        out, h_n = self.gru(x)
        return out, h_n



class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_n):
        B = h_n.shape[1]
        last_h = h_n[-1]                          # (B, hidden_dim)
        inp = last_h.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, T, hidden_dim)
        out, _ = self.gru(inp, h_n)               # (B, T, hidden_dim)
        recon = self.fc(out)                       # (B, T, output_dim)
        return recon

class GRUAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = GRUDecoder(hidden_dim, input_dim, num_layers, dropout, seq_len)

    def forward(self, x):
        _, h_n = self.encoder(x)
        recon = self.decoder(h_n)
        return recon

    def reconstruction_error(self, x, weight_alpha=0.0):
        recon = self.forward(x)
        seq_len = x.shape[1]
        if weight_alpha > 0:
            weights = torch.exp(
                torch.linspace(-weight_alpha, 0.0, seq_len, device=x.device)
            )
            weights = weights / weights.sum()         
            frame_err = ((x - recon) ** 2).mean(dim=2) # (B, T)
            err = (frame_err * weights.unsqueeze(0)).sum(dim=1)  # (B,)
        else:
            err = ((x - recon) ** 2).mean(dim=(1, 2))  # (B,)
        return err


def build_hybrid_window(window: np.ndarray) -> np.ndarray:
    per_frame_norm = np.linalg.norm(window, axis=1, keepdims=True)  # (T, 1)
    return np.concatenate([window, per_frame_norm], axis=1).astype(np.float32)


def apply_relative_transform(window: np.ndarray, mode: str,
                              n_rel_dims: int = None) -> np.ndarray:
    if mode == 'none':
        return window.copy()
    d = window.shape[1] if n_rel_dims is None else n_rel_dims
    feat = window[:, :d].copy()
    rest = window[:, d:]           
    if mode == 'window_mean':
        feat = feat - feat.mean(axis=0, keepdims=True)
    elif mode == 'first_frame':
        feat = feat - feat[0:1]
    if rest.shape[1] > 0:
        return np.concatenate([feat, rest], axis=1)
    return feat


def apply_relative_transform_tensor(x: torch.Tensor, mode: str,
                                     n_rel_dims: int = None) -> torch.Tensor:

    if mode == 'none':
        return x
    d = x.shape[2] if n_rel_dims is None else n_rel_dims
    feat = x[:, :, :d]
    rest = x[:, :, d:]
    if mode == 'window_mean':
        feat = feat - feat.mean(dim=1, keepdim=True)
    elif mode == 'first_frame':
        feat = feat - feat[:, 0:1, :]
    if rest.shape[2] > 0:
        return torch.cat([feat, rest], dim=2)
    return feat


def load_sequences_raw(data_dir: str, prefix: str, label: str,
                       use_raw: bool = False) -> list[np.ndarray]:
    """리샘플링 없이 원본 길이 그대로 로드 (threshold_B 캘리브레이션용)."""
    if use_raw:
        pattern = os.path.join(data_dir, f"{prefix}_*_{label}_raw.npy")
        files = sorted(glob.glob(pattern))
    else:
        pattern = os.path.join(data_dir, f"{prefix}_*_{label}.npy")
        files = sorted([f for f in glob.glob(pattern) if not f.endswith("_raw.npy")])
    sequences = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[0] < 5:
            continue
        sequences.append(arr.astype(np.float32))
    return sequences


def load_sequences(data_dir: str, prefix: str, label: str,
                   use_raw: bool, seq_len: int) -> list[np.ndarray]:

    suffix = "_raw.npy" if use_raw else ".npy"
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}{suffix}")
    files = sorted(glob.glob(pattern))

    if not use_raw:
        files = [f for f in files if not f.endswith("_raw.npy")]

    sequences = []
    for f in files:
        arr = np.load(f)            # [T, D]
        if arr.ndim != 2 or arr.shape[0] < 5:
            print(f"[SKIP] {os.path.basename(f)}: shape={arr.shape}")
            continue
        arr_resampled = resample(arr, seq_len, axis=0).astype(np.float32)
        sequences.append(arr_resampled)
        print(f"[LOAD] {os.path.basename(f)}: {arr.shape} → {arr_resampled.shape}")

    return sequences


def load_sequences_windowed(data_dir: str, prefix: str, label: str,
                             use_raw: bool, seq_len: int,
                             stride: int) -> list[np.ndarray]:

    suffix = "_raw.npy" if use_raw else ".npy"
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}{suffix}")
    files = sorted(glob.glob(pattern))
    if not use_raw:
        files = [f for f in files if not f.endswith("_raw.npy")]

    windows = []
    for f in files:
        arr = np.load(f)           
        if arr.ndim != 2 or arr.shape[0] < seq_len:
            print(f"[SKIP] {os.path.basename(f)}: shape={arr.shape} (seq_len={seq_len} 미만)")
            continue
        n_win = 0
        for start in range(0, arr.shape[0] - seq_len + 1, stride):
            windows.append(arr[start:start + seq_len].astype(np.float32))
            n_win += 1
        print(f"[LOAD] {os.path.basename(f)}: {arr.shape} → {n_win} windows (stride={stride})")

    return windows


def fit_scaler(sequences: list[np.ndarray]) -> StandardScaler:
    """success 시퀀스 전체를 합쳐서 StandardScaler fit."""
    all_frames = np.vstack(sequences)   
    scaler = StandardScaler()
    scaler.fit(all_frames)
    return scaler


def apply_scaler(sequences: list[np.ndarray],
                 scaler: StandardScaler) -> list[np.ndarray]:
    return [scaler.transform(s) for s in sequences]



def weighted_mse_loss(recon: torch.Tensor, x: torch.Tensor,
                      weight_alpha: float) -> torch.Tensor:

    if weight_alpha <= 0:
        return nn.functional.mse_loss(recon, x)
    seq_len = x.shape[1]
    weights = torch.exp(
        torch.linspace(-weight_alpha, 0.0, seq_len, device=x.device)
    )
    weights = weights / weights.sum()               # (T,)
    frame_mse = ((recon - x) ** 2).mean(dim=2)
    return (frame_mse * weights.unsqueeze(0)).mean()


def train(model, train_loader, val_loader, cfg, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    relative_mode = cfg.get("relative", "none")
    weight_alpha  = cfg.get("weight_alpha", 0.0)
    n_rel_dims    = cfg.get("n_rel_dims", None)

    best_val_loss = float("inf")
    patience_cnt = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device)
            x = apply_relative_transform_tensor(x, relative_mode, n_rel_dims)
            recon = model(x)
            loss = weighted_mse_loss(recon, x, weight_alpha)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x = apply_relative_transform_tensor(x, relative_mode, n_rel_dims)
                recon = model(x)
                val_loss += weighted_mse_loss(recon, x, weight_alpha).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d}/{cfg['epochs']} | "
                  f"train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"[EarlyStopping] epoch {epoch}, best_val={best_val_loss:.6f}")
                break

    print(f"[Train] 완료. best val loss = {best_val_loss:.6f}")
    return train_losses, val_losses


def compute_threshold(model, sequences, scaler, cfg, device):
    relative_mode = cfg.get("relative", "none")
    weight_alpha  = cfg.get("weight_alpha", 0.0)
    n_rel_dims    = cfg.get("n_rel_dims", None)
    model.eval()
    errors = []
    with torch.no_grad():
        for seq in sequences:
            x = torch.tensor(scaler.transform(seq), dtype=torch.float32)
            x = x.unsqueeze(0).to(device)
            x = apply_relative_transform_tensor(x, relative_mode, n_rel_dims)
            err = model.reconstruction_error(x, weight_alpha=weight_alpha)
            errors.append(err.item())
    errors = np.array(errors)
    threshold = float(np.percentile(errors, cfg["threshold_pct"]))
    print(f"[Threshold] {cfg['threshold_pct']}th percentile = {threshold:.6f}")
    print(f"[Threshold] success 오차: mean={errors.mean():.6f}, "
          f"std={errors.std():.6f}, max={errors.max():.6f}")
    return threshold, errors



def evaluate_and_plot(model, all_seqs_by_label: dict,
                      scaler, threshold, out_dir, prefix, use_raw,
                      relative_mode: str = "none", weight_alpha: float = 0.0,
                      n_rel_dims: int = None):
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"success": "steelblue", "perturbation": "orange", "failure": "crimson"}
    all_errors = {}

    for label, seqs in all_seqs_by_label.items():
        if not seqs:
            continue
        errors = []
        with torch.no_grad():
            for seq in seqs:
                x = torch.tensor(scaler.transform(seq), dtype=torch.float32)
                x = x.unsqueeze(0).to(device)
                x = apply_relative_transform_tensor(x, relative_mode, n_rel_dims)
                err = model.reconstruction_error(x, weight_alpha=weight_alpha)
                errors.append(err.item())
        all_errors[label] = np.array(errors)
        axes[0].hist(errors, bins=20, alpha=0.6,
                     color=colors.get(label, "gray"), label=label)

    axes[0].axvline(threshold, color="black", linestyle="--", label=f"threshold ({threshold:.4f})")
    axes[0].set_xlabel("Reconstruction Error (MSE)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{prefix} GRU-AE: Error Distribution")
    axes[0].legend()

    if all_errors:
        labels_sorted = [l for l in ["success", "perturbation", "failure"] if l in all_errors]
        data_sorted = [all_errors[l] for l in labels_sorted]
        bp = axes[1].boxplot(data_sorted, labels=labels_sorted, patch_artist=True)
        for patch, label in zip(bp["boxes"], labels_sorted):
            patch.set_facecolor(colors.get(label, "gray"))
        axes[1].axhline(threshold, color="black", linestyle="--", label="threshold")
        axes[1].set_ylabel("Reconstruction Error (MSE)")
        axes[1].set_title(f"{prefix} GRU-AE: Error by Label")
        axes[1].legend()

    suffix = "_raw" if use_raw else ""
    fname = os.path.join(out_dir, f"{prefix}_gru_ae{suffix}_eval.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[Eval] 플롯 저장: {fname}")

    for label, errors in all_errors.items():
        above = np.sum(errors > threshold)
        print(f"  [{label}] n={len(errors)}, "
              f"mean={errors.mean():.4f}, "
              f"anomaly rate = {above}/{len(errors)} = {above/len(errors)*100:.1f}%")


def evaluate_channel_b_sequences(raw_seqs_by_label: dict, threshold_b: float,
                                  smooth_window: int = 3) -> dict:
    """
    채널 B (MRL delta) 시퀀스 단위 감지율 평가.
    ANY 프레임에서 delta > threshold_b 이면 해당 시퀀스를 이상으로 판정.
    반환: {label: (detected, total)}
    """
    from .detector import FramewiseDeltaDetector
    results = {}
    for label, seqs in raw_seqs_by_label.items():
        detected = 0
        for seq in seqs:
            det = FramewiseDeltaDetector(threshold_b, smooth_window)
            triggered = any(det.update(frame)[1] for frame in seq)
            if triggered:
                detected += 1
        results[label] = (detected, len(seqs))
    return results


def evaluate_channel_a_sequences(model, raw_seqs_by_label: dict, scaler, threshold_a: float,
                                  seq_len: int, relative_mode: str, weight_alpha: float,
                                  n_rel_dims, device) -> dict:
    results = {}
    model.eval()
    with torch.no_grad():
        for label, seqs in raw_seqs_by_label.items():
            detected = 0
            for seq in seqs:
                T = len(seq)
                if T < seq_len:
                    continue
                triggered = False
                for start in range(0, T - seq_len + 1):
                    window = seq[start:start + seq_len]
                    x = apply_relative_transform(
                        scaler.transform(window).astype(np.float32),
                        relative_mode, n_rel_dims)
                    x_t = torch.tensor(x).unsqueeze(0).to(device)
                    score = model.reconstruction_error(x_t, weight_alpha).item()
                    if score > threshold_a:
                        triggered = True
                        break
                if triggered:
                    detected += 1
            results[label] = (detected, len(seqs))
    return results


def print_dual_channel_eval(results_a: dict, results_b: dict):
    labels = [l for l in ["success", "perturbation", "failure"]
              if l in results_a or l in results_b]
    print("\n[Dual-Channel 시퀀스 단위 평가]")
    print(f"{'Label':<14} {'Ch-A':>10} {'Ch-B':>10} {'A OR B':>10}")
    print("-" * 46)
    for label in labels:
        det_a, tot_a = results_a.get(label, (0, 0))
        det_b, tot_b = results_b.get(label, (0, 0))
        tot = max(tot_a, tot_b)
        if tot == 0:
            continue
        pct_a  = det_a  / tot * 100 if tot else 0
        pct_b  = det_b  / tot * 100 if tot else 0
        print(f"{label:<14} {det_a}/{tot} ({pct_a:5.1f}%)  "
              f"{det_b}/{tot} ({pct_b:5.1f}%)")


def evaluate_dual_channel_combined(model, raw_seqs_by_label: dict, scaler,
                                    threshold_a: float, threshold_b: float,
                                    seq_len: int, relative_mode: str, weight_alpha: float,
                                    n_rel_dims, device, smooth_window: int = 3,
                                    consec_a: int = 1, consec_b: int = 1,
                                    infer_stride: int = 5) -> dict:

    from .detector import FramewiseDeltaDetector
    results = {}
    model.eval()

    for label, seqs in raw_seqs_by_label.items():
        det_a = det_b = det_ab = 0
        tot = len(seqs)
        with torch.no_grad():
            for seq in seqs:
                T = len(seq)

                det_b_obj = FramewiseDeltaDetector(threshold_b, smooth_window)
                b_consec = 0
                flag_b = False
                for frame in seq:
                     _, trig = det_b_obj.update(frame)
                     if trig:
                         b_consec += 1
                         if b_consec >= consec_b:
                             flag_b = True
                             break
                     else:
                         b_consec = 0

                flag_a = False
                if T >= seq_len:
                    a_consec = 0
                    for start in range(0, T - seq_len + 1, infer_stride):
                        window = seq[start:start + seq_len]
                        x = apply_relative_transform(
                            scaler.transform(window).astype(np.float32),
                            relative_mode, n_rel_dims)
                        x_t = torch.tensor(x).unsqueeze(0).to(device)
                        score = model.reconstruction_error(x_t, weight_alpha).item()
                        if score > threshold_a:
                            a_consec += 1
                            if a_consec >= consec_a:
                                flag_a = True
                                break
                        else:
                            a_consec = 0

                if flag_a:
                    det_a += 1
                if flag_b:
                    det_b += 1
                if flag_a or flag_b:
                    det_ab += 1

        results[label] = {'a': (det_a, tot), 'b': (det_b, tot), 'ab': (det_ab, tot)}
    return results



def plot_learning_curve(train_losses, val_losses, out_dir, prefix, use_raw):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{prefix} GRU-AE: Learning Curve")
    plt.legend()
    plt.tight_layout()
    suffix = "_raw" if use_raw else ""
    fname = os.path.join(out_dir, f"{prefix}_gru_ae{suffix}_curve.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[Plot] 학습 곡선 저장: {fname}")


def main():
    parser = argparse.ArgumentParser(description="GRU-AE 학습")
    parser.add_argument("--model", type=str, default="grasp",
                        choices=["grasp", "insert", "transport"],
                        help="grasp: close_gripper→ba_init, transport: ba_init→wm_end")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="데이터 디렉토리 (기본: grasp→grasp_data, insert→insert_data)")
    parser.add_argument("--use_raw", action="store_true",
                        help="raw 48D taxel 데이터 사용 (기본: 6D feature)")
    parser.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--hidden_dim", type=int, default=DEFAULTS["hidden_dim"])
    parser.add_argument("--num_layers", type=int, default=DEFAULTS["num_layers"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--threshold_pct", type=float, default=DEFAULTS["threshold_pct"],
                        help="임계값 백분위수 (기본: 95, 소수점 가능: 99.5 등)")
    parser.add_argument("--eval", action="store_true",
                        help="학습 후 perturbation/failure 데이터로 평가 플롯 생성")
    parser.add_argument("--windowed", action="store_true",
                        help="슬라이딩 윈도우 방식으로 학습 (실시간 inference용, resample 없이 원본 프레임 사용)")
    parser.add_argument("--win_stride", type=int, default=10,
                        help="윈도우 추출 stride (기본: 10프레임)")
    parser.add_argument("--relative", type=str, default="none",
                        choices=["none", "window_mean", "first_frame"],
                        help="상대 특징 변환 방식 (none/window_mean/first_frame)")
    parser.add_argument("--weight_alpha", type=float, default=0.0,
                        help="지수 가중치 MSE alpha (0=균일, 2.0 권장)")
    parser.add_argument("--hybrid", action="store_true",
                        help="접촉 강도(per-frame norm)를 마지막 차원으로 추가 (D→D+1). "
                             "relative와 함께 사용 시 norm 차원은 변환 제외 → "
                             "TYPE-A(no-contact)와 TYPE-B(slip) 동시 감지")
    args = parser.parse_args()

    ws = os.getcwd()
    if args.data_dir:
        data_dir = args.data_dir
    else:
        default_dirs = {"grasp": "grasp_data", "insert": "insert_data", "transport": "transport_data"}
        data_dir = os.path.join(ws, default_dirs.get(args.model, "grasp_data"))
    os.makedirs(data_dir, exist_ok=True)

    prefix = args.model   # "grasp" or "insert"
    suffix = "_raw" if args.use_raw else ""

    model_save_path   = os.path.join(data_dir, f"{prefix}_gru_ae{suffix}.pt")
    scaler_save_path  = os.path.join(data_dir, f"{prefix}_scaler{suffix}.pkl")
    thresh_save_path  = os.path.join(data_dir, f"{prefix}_threshold{suffix}.npy")
    thresh_b_save_path= os.path.join(data_dir, f"{prefix}_threshold_b{suffix}.npy")
    infer_cfg_path    = os.path.join(data_dir, f"{prefix}_infer_cfg{suffix}.pkl")

    cfg = {k: getattr(args, k, v) for k, v in DEFAULTS.items()}
    cfg.update(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] model={prefix}, use_raw={args.use_raw}, windowed={args.windowed}, "
          f"seq_len={cfg['seq_len']}, hidden={cfg['hidden_dim']}, "
          f"layers={cfg['num_layers']}, device={device}")
    print(f"[Config] data_dir={data_dir}")
    print(f"[Config] relative={args.relative}, weight_alpha={args.weight_alpha}, hybrid={args.hybrid}")
    if args.windowed:
        print(f"[Config] win_stride={args.win_stride} (슬라이딩 윈도우 모드)")

    if args.windowed:
        success_seqs = load_sequences_windowed(data_dir, prefix, "success",
                                               args.use_raw, cfg["seq_len"],
                                               args.win_stride)
    else:
        success_seqs = load_sequences(data_dir, prefix, "success",
                                      args.use_raw, cfg["seq_len"])
    if len(success_seqs) == 0:
        print(f"[ERROR] success 데이터가 없습니다: {data_dir}/{prefix}_*_success*.npy")
        print("         데이터를 수집 및 확보한 후 학습을 다시 시도해주세요.")
        return

    print(f"\n[Data] success 샘플 수: {len(success_seqs)}")
    raw_feat_dim = success_seqs[0].shape[1]

    if args.hybrid:
        success_seqs = [build_hybrid_window(w) for w in success_seqs]
        print(f"[Hybrid] per-frame norm 추가: {raw_feat_dim}D → {raw_feat_dim+1}D")

    input_dim = success_seqs[0].shape[1]
    n_rel_dims = raw_feat_dim if args.hybrid else None 
    cfg["n_rel_dims"] = n_rel_dims
    print(f"[Data] feature dim: {input_dim}, seq_len: {cfg['seq_len']}")

    scaler = fit_scaler(success_seqs)
    success_seqs_norm = apply_scaler(success_seqs, scaler)
    joblib.dump(scaler, scaler_save_path)
    print(f"[Scaler] 저장: {scaler_save_path}")

    dataset = TactileDataset(success_seqs_norm)
    val_size = max(1, int(len(dataset) * DEFAULTS["val_ratio"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)
    print(f"[Data] train={train_size}, val={val_size}")

    model = GRUAE(
        input_dim  = input_dim,
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        dropout    = DEFAULTS["dropout"],
        seq_len    = cfg["seq_len"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    train_losses, val_losses = train(model, train_loader, val_loader,
                                     cfg, model_save_path)
    plot_learning_curve(train_losses, val_losses, data_dir, prefix, args.use_raw)

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)

    threshold, train_errors = compute_threshold(
        model, success_seqs, scaler, cfg, device
    )
    np.save(thresh_save_path, np.array([threshold]))
    print(f"[Threshold A] 저장: {thresh_save_path} (value={threshold:.6f})")

    from .detector import calibrate_threshold_b
    raw_success = load_sequences_raw(data_dir, prefix, "success", use_raw=args.use_raw)
    if not raw_success:
        raw_success = success_seqs   
    threshold_b = calibrate_threshold_b(raw_success, smooth_window=3, percentile=99)
    np.save(thresh_b_save_path, np.array([threshold_b]))
    print(f"[Threshold B] 저장: {thresh_b_save_path} (value={threshold_b:.6f})")

    infer_cfg = {
        "relative":     args.relative,
        "weight_alpha": args.weight_alpha,
        "use_raw":      args.use_raw,
        "seq_len":      cfg["seq_len"],
        "hybrid":       args.hybrid,
        "n_rel_dims":   n_rel_dims,
    }
    with open(infer_cfg_path, "wb") as f:
        pickle.dump(infer_cfg, f)
    print(f"[InferCfg] 저장: {infer_cfg_path} → {infer_cfg}")

    if args.eval:
        all_seqs = {"success": success_seqs}
        for label in ["perturbation", "failure"]:
            if args.windowed:
                seqs = load_sequences_windowed(data_dir, prefix, label,
                                               args.use_raw, cfg["seq_len"],
                                               args.win_stride)
            else:
                seqs = load_sequences(data_dir, prefix, label,
                                      args.use_raw, cfg["seq_len"])
            if seqs:
                if args.hybrid:
                    seqs = [build_hybrid_window(w) for w in seqs]
                all_seqs[label] = seqs
                print(f"[Eval] {label} 샘플 수: {len(seqs)}")
        evaluate_and_plot(model, all_seqs, scaler, threshold,
                          data_dir, prefix, args.use_raw,
                          relative_mode=args.relative,
                          weight_alpha=args.weight_alpha,
                          n_rel_dims=n_rel_dims)

        # ── 채널 B + 통합 평가 (시퀀스 단위)
        infer_stride = args.win_stride
        print(f"\n[Dual-Channel 시퀀스 단위 평가] (A OR B, stride={infer_stride} 슬라이딩)")
        print(f"  Threshold A = {threshold:.6f}, Threshold B = {threshold_b:.6f}")
        raw_seqs_by_label = {}
        for label in ["success", "perturbation", "failure"]:
            seqs = load_sequences_raw(data_dir, prefix, label, use_raw=args.use_raw)
            if seqs:
                raw_seqs_by_label[label] = seqs
        if raw_seqs_by_label:
            for consec_a, consec_b in [(1, 1), (2, 3), (2, 4), (2, 5), (3, 3)]:
                combined = evaluate_dual_channel_combined(
                    model, raw_seqs_by_label, scaler,
                    threshold_a=threshold, threshold_b=threshold_b,
                    seq_len=cfg["seq_len"], relative_mode=args.relative,
                    weight_alpha=args.weight_alpha, n_rel_dims=n_rel_dims,
                    device=device, smooth_window=3,
                    consec_a=consec_a, consec_b=consec_b,
                    infer_stride=infer_stride,
                )
                print(f"\n  [consec_A={consec_a}, consec_B={consec_b}]")
                print(f"  {'Label':<14} {'Ch-A':>14} {'Ch-B':>14} {'A OR B':>14}")
                print("  " + "-" * 56)
                for label, res in combined.items():
                    det_a, tot = res['a']
                    det_b, _   = res['b']
                    det_ab, _  = res['ab']
                    print(f"  {label:<14} "
                          f"{det_a}/{tot} ({det_a/tot*100:5.1f}%)  "
                          f"{det_b}/{tot} ({det_b/tot*100:5.1f}%)  "
                          f"{det_ab}/{tot} ({det_ab/tot*100:5.1f}%)")

    print("\n[완료] 저장된 파일:")
    print(f"  모델        : {model_save_path}")
    print(f"  스케일러    : {scaler_save_path}")
    print(f"  Threshold A : {thresh_save_path}")
    print(f"  Threshold B : {thresh_b_save_path}")
    print(f"  추론 설정   : {infer_cfg_path}")


if __name__ == "__main__":
    main()
