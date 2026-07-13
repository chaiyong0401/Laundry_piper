"""
GRU Autoencoder 학습 스크립트
==============================
One-class anomaly detection: success 데이터만으로 학습하여
재구성 오차(reconstruction error)를 anomaly score로 사용.

Usage:
  # grasp 모델 학습 (6D feature)
  python train_gru_ae.py --model grasp

  # insert 모델 학습
  python train_gru_ae.py --model insert

  # raw 48D taxel 데이터로 학습
  python train_gru_ae.py --model grasp --use_raw

  # 학습 후 평가 곡선/분포 저장
  python train_gru_ae.py --model grasp --eval
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
    hidden_dim   = 128,       # GRU hidden size
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


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
class TactileDataset(Dataset):
    """
    npy 파일을 로드하여 고정 길이 시퀀스로 변환.
    각 샘플: (seq_len, feature_dim) float32 tensor
    """
    def __init__(self, sequences: list[np.ndarray]):
        # sequences: list of [T, D] arrays (이미 정규화 완료)
        self.data = [torch.tensor(s, dtype=torch.float32) for s in sequences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────
#  GRU Autoencoder
# ─────────────────────────────────────────────
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        # x: (B, T, D)
        out, h_n = self.gru(x)
        # h_n: (num_layers, B, hidden_dim)
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
        # h_n: (num_layers, B, hidden_dim)
        B = h_n.shape[1]
        # 마지막 레이어의 hidden state를 T번 반복 → decoder input
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
        """
        샘플별 재구성 오차 반환 (B,).
        weight_alpha > 0 이면 최근 프레임에 지수 가중치 적용.
          weight_alpha=0 → 균일 평균 (기본)
          weight_alpha=2 → 마지막 프레임 가중치 ≈ 첫 프레임의 e^2 ≈ 7.4배
        """
        recon = self.forward(x)
        seq_len = x.shape[1]
        if weight_alpha > 0:
            # (T,) 가중치: linspace(-alpha, 0) → exp → 정규화
            weights = torch.exp(
                torch.linspace(-weight_alpha, 0.0, seq_len, device=x.device)
            )
            weights = weights / weights.sum()           # 합=1
            frame_err = ((x - recon) ** 2).mean(dim=2) # (B, T)
            err = (frame_err * weights.unsqueeze(0)).sum(dim=1)  # (B,)
        else:
            err = ((x - recon) ** 2).mean(dim=(1, 2))  # (B,)
        return err


# ─────────────────────────────────────────────
#  Relative transform
# ─────────────────────────────────────────────
def build_hybrid_window(window: np.ndarray) -> np.ndarray:
    """
    원본 (T, D) window에 프레임별 L2 norm을 마지막 차원으로 추가.
    반환: (T, D+1) float32

    마지막 차원(norm)은 접촉 강도를 나타냄:
      - 접촉 있음: norm 양수 (물체 무관)
      - 접촉 없음: norm ≈ 0
    relative transform은 앞 D 차원에만 적용하고 norm은 보존.
    """
    per_frame_norm = np.linalg.norm(window, axis=1, keepdims=True)  # (T, 1)
    return np.concatenate([window, per_frame_norm], axis=1).astype(np.float32)


def apply_relative_transform(window: np.ndarray, mode: str,
                              n_rel_dims: int = None) -> np.ndarray:
    """
    window: (T, D) float32
    mode:
      'none'        → 원본 그대로
      'window_mean' → 각 feature를 window 평균 기준으로 중심화 (자세 변화 제거)
      'first_frame' → 첫 프레임을 기준으로 차분 (점진적 드리프트 제거)
    n_rel_dims: relative 변환을 적용할 앞 차원 수 (None=전체)
                hybrid 모드에서는 마지막 norm 차원을 보존하기 위해 D-1 전달
    반환: (T, D) float32
    """
    if mode == 'none':
        return window.copy()
    d = window.shape[1] if n_rel_dims is None else n_rel_dims
    feat = window[:, :d].copy()
    rest = window[:, d:]           # norm 차원 등 변환 제외
    if mode == 'window_mean':
        feat = feat - feat.mean(axis=0, keepdims=True)
    elif mode == 'first_frame':
        feat = feat - feat[0:1]
    if rest.shape[1] > 0:
        return np.concatenate([feat, rest], axis=1)
    return feat


def apply_relative_transform_tensor(x: torch.Tensor, mode: str,
                                     n_rel_dims: int = None) -> torch.Tensor:
    """
    x: (B, T, D) tensor
    n_rel_dims: relative 변환을 적용할 앞 차원 수 (None=전체)
    """
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


# ─────────────────────────────────────────────
#  데이터 로딩 유틸
# ─────────────────────────────────────────────
def load_sequences_raw(data_dir: str, prefix: str, label: str,
                       use_raw: bool = False) -> list[np.ndarray]:
    """리샘플링 없이 원본 길이 그대로 로드 (threshold_B 캘리브레이션용)."""
    if use_raw:
        # _raw.npy 파일 (48D taxel)
        pattern = os.path.join(data_dir, f"{prefix}_*_{label}_raw.npy")
        files = sorted(glob.glob(pattern))
    else:
        # 6D feature 파일 (_raw.npy 제외)
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
                   use_raw: bool, seq_len: int,
                   exclude_files: set = None) -> list[np.ndarray]:
    """
    data_dir 에서 {prefix}_{ts}_{label}[_raw].npy 파일을 모두 로드.
    각 시퀀스를 seq_len 으로 리샘플링하여 반환.
    exclude_files: 제외할 파일 basename 집합 (노이즈 에피소드 필터)
    """
    suffix = "_raw.npy" if use_raw else ".npy"
    # label 이 들어있고 _raw 여부가 맞는 파일만
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}{suffix}")
    files = sorted(glob.glob(pattern))

    # use_raw=False일 때 *_raw.npy 파일이 섞이지 않도록 필터
    if not use_raw:
        files = [f for f in files if not f.endswith("_raw.npy")]

    sequences = []
    for f in files:
        if exclude_files and os.path.basename(f) in exclude_files:
            print(f"[SKIP-NOISY] {os.path.basename(f)}")
            continue
        arr = np.load(f)            # [T, D]
        if arr.ndim != 2 or arr.shape[0] < 5:
            print(f"[SKIP] {os.path.basename(f)}: shape={arr.shape}")
            continue
        arr_resampled = resample(arr, seq_len, axis=0).astype(np.float32)
        sequences.append(arr_resampled)
        print(f"[LOAD] {os.path.basename(f)}: {arr.shape} → {arr_resampled.shape}")

    return sequences


def _compute_episode_mrl_std(arr: np.ndarray, use_raw: bool) -> float:
    """에피소드 배열에서 MRL 시계열의 표준편차를 반환 (노이즈 지표)."""
    if use_raw:
        vecs = arr[:, :48].reshape(len(arr), 16, 3)
        mrl = np.linalg.norm(vecs, axis=2).mean(axis=1)
    else:
        mrl = arr[:, 0]  # 6D: 첫 번째 feature = MRL
    return float(mrl.std())


def find_noisy_episodes(data_dir: str, prefix: str, label: str,
                        use_raw: bool, std_threshold: float) -> set:
    """
    MRL std 가 std_threshold 를 초과하는 에피소드 파일명(basename)을 반환.
    이 파일들은 학습에서 제외된다.
    """
    suffix = "_raw.npy" if use_raw else ".npy"
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}{suffix}")
    files = sorted(glob.glob(pattern))
    if not use_raw:
        files = [f for f in files if not f.endswith("_raw.npy")]

    noisy = set()
    for f in files:
        arr = np.load(f).astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] < 5:
            continue
        std = _compute_episode_mrl_std(arr, use_raw)
        if std > std_threshold:
            noisy.add(os.path.basename(f))
            print(f"  [NOISY] {os.path.basename(f)}: mrl_std={std:.4f} > {std_threshold}")
    return noisy


def load_sequences_windowed(data_dir: str, prefix: str, label: str,
                            use_raw: bool, seq_len: int,
                            stride: int,
                            exclude_files: set = None) -> list[np.ndarray]:
    """
    슬라이딩 윈도우 방식으로 로드.
    각 시퀀스에서 크기 seq_len 의 window 를 stride 간격으로 추출.
    리샘플링 없이 원본 프레임 사용 → 실시간 inference 분포와 일치.
    exclude_files: 제외할 파일 basename 집합 (노이즈 에피소드 필터)
    """
    suffix = "_raw.npy" if use_raw else ".npy"
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}{suffix}")
    files = sorted(glob.glob(pattern))
    if not use_raw:
        files = [f for f in files if not f.endswith("_raw.npy")]

    windows = []
    for f in files:
        if exclude_files and os.path.basename(f) in exclude_files:
            print(f"[SKIP-NOISY] {os.path.basename(f)}")
            continue
        arr = np.load(f)            # [T, D]
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
    all_frames = np.vstack(sequences)   # [총 프레임 수, D]
    scaler = StandardScaler()
    scaler.fit(all_frames)
    return scaler


def apply_scaler(sequences: list[np.ndarray],
                 scaler: StandardScaler) -> list[np.ndarray]:
    return [scaler.transform(s) for s in sequences]


# ─────────────────────────────────────────────
#  학습
# ─────────────────────────────────────────────
def weighted_mse_loss(recon: torch.Tensor, x: torch.Tensor,
                      weight_alpha: float) -> torch.Tensor:
    """
    최근 프레임에 지수 가중치를 적용한 MSE.
    weight_alpha=0 → 균일 MSE (nn.MSELoss 와 동일)
    """
    if weight_alpha <= 0:
        return nn.functional.mse_loss(recon, x)
    seq_len = x.shape[1]
    weights = torch.exp(
        torch.linspace(-weight_alpha, 0.0, seq_len, device=x.device)
    )
    weights = weights / weights.sum()               # (T,)
    # per-frame MSE: (B, T)
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
        # ── train
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

        # ── val
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

        # ── early stopping
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


# ─────────────────────────────────────────────
#  임계값 계산
# ─────────────────────────────────────────────
def compute_threshold(model, sequences, scaler, cfg, device):
    """
    학습에 사용된 success 시퀀스의 재구성 오차 분포에서
    threshold_pct 백분위수를 임계값으로 설정.
    """
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


# ─────────────────────────────────────────────
#  평가 플롯
# ─────────────────────────────────────────────
def evaluate_and_plot(model, all_seqs_by_label: dict,
                      scaler, threshold, out_dir, prefix, use_raw,
                      relative_mode: str = "none", weight_alpha: float = 0.0,
                      n_rel_dims: int = None):
    """
    레이블별 재구성 오차 분포 + 히스토그램 저장.
    all_seqs_by_label: {'success': [...], 'perturbation': [...], 'failure': [...]}
    """
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

    # 레이블별 오차 박스플롯
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

    # 통계 출력
    for label, errors in all_errors.items():
        above = np.sum(errors > threshold)
        print(f"  [{label}] n={len(errors)}, "
              f"mean={errors.mean():.4f}, "
              f"anomaly rate = {above}/{len(errors)} = {above/len(errors)*100:.1f}%")


# ─────────────────────────────────────────────
#  채널 B 시퀀스 단위 평가
# ─────────────────────────────────────────────
def evaluate_channel_b_sequences(raw_seqs_by_label: dict, threshold_b: float,
                                  smooth_window: int = 3) -> dict:
    """
    채널 B (MRL delta) 시퀀스 단위 감지율 평가.
    ANY 프레임에서 delta > threshold_b 이면 해당 시퀀스를 이상으로 판정.
    반환: {label: (detected, total)}
    """
    from dual_channel_detector import FramewiseDeltaDetector
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
                                  n_rel_dims, device, hybrid: bool = False) -> dict:
    """
    채널 A (GRU-AE) 시퀀스 단위 감지율 평가.
    슬라이딩 윈도우(stride=1)로 ANY 윈도우에서 score > threshold_a 이면 이상 판정.
    반환: {label: (detected, total)}
    """
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
                    if hybrid:
                        window = build_hybrid_window(window)
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
    """채널 A, B, A OR B 시퀀스 단위 감지율 테이블 출력."""
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
        det_ab = 0
        for i in range(tot):
            a_hit = (i < tot_a) and (det_a > 0)  # approximate if unequal; use per-seq below
            b_hit = (i < tot_b) and (det_b > 0)
        # per-sequence OR requires aligned lists; approximate with union bound capped at tot
        # actual per-seq computed in caller
        pct_a  = det_a  / tot * 100 if tot else 0
        pct_b  = det_b  / tot * 100 if tot else 0
        print(f"{label:<14} {det_a}/{tot} ({pct_a:5.1f}%)  "
              f"{det_b}/{tot} ({pct_b:5.1f}%)")
    print("  * A OR B는 아래 per-sequence 결과 참조")


def evaluate_dual_channel_combined(model, raw_seqs_by_label: dict, scaler,
                                    threshold_a: float, threshold_b: float,
                                    seq_len: int, relative_mode: str, weight_alpha: float,
                                    n_rel_dims, device, smooth_window: int = 3,
                                    consec_a: int = 1, consec_b: int = 1,
                                    infer_stride: int = 10, hybrid: bool = False,
                                    mrl_thresh: float = 0.3, cutoff_multiplier: float = 2.2,
                                    warmup: int = 30, warmup_b: int = None) -> dict:
    """
    시퀀스별로 채널 A와 B를 동시에 평가하여 A OR B 감지율 계산.

    consec_a:     채널 A 이상 확정에 필요한 연속 윈도우 수 (1=단발 트리거)
    consec_b:     채널 B 이상 확정에 필요한 연속 프레임 수 (1=단발 트리거)
    infer_stride: 채널 A 슬라이딩 윈도우 stride (실제 inference stride와 동일하게 설정)
    warmup:       Ch-A 윈도우 skip 기준 (start < warmup 이면 skip)
    warmup_b:     Ch-B 프레임 skip 기준 (미지정 시 warmup 값 사용)
    반환: {label: {'a': (det, tot), 'b': (det, tot), 'ab': (det, tot)}}
    """
    from dual_channel_detector import FramewiseDeltaDetector
    if warmup_b is None:
        warmup_b = warmup
    results = {}
    model.eval()

    for label, seqs in raw_seqs_by_label.items():
        det_a = det_b = det_ab = 0
        tot = len(seqs)
        with torch.no_grad():
            for seq in seqs:
                T = len(seq)

                # ── 채널 B: 연속 consec_b 프레임 이상 시 감지 (warmup_b 적용)
                det_b_obj = FramewiseDeltaDetector(threshold_b, smooth_window)
                b_consec = 0
                flag_b = False
                for t, frame in enumerate(seq):
                    _, trig = det_b_obj.update(frame)
                    if t < warmup_b:
                        b_consec = 0
                        continue
                    if trig:
                        b_consec += 1
                        if b_consec >= consec_b:
                            flag_b = True
                            break
                    else:
                        b_consec = 0

                # ── 채널 A: 연속 consec_a 윈도우 이상 시 감지 (stride=infer_stride)
                flag_a = False
                
                # Rule 1: Miss Guard Rail (MRL threshold)
                mrls = [FramewiseDeltaDetector._compute_mrl(frame) for frame in seq]
                max_mrl = max(mrls) if mrls else 0.0
                if max_mrl < mrl_thresh:
                    flag_a = True

                if not flag_a and T >= seq_len:
                    a_consec = 0
                    for start in range(0, T - seq_len + 1, infer_stride):
                        # if start < warmup:
                        #     continue
                        window = seq[start:start + seq_len]
                        if hybrid:
                            window = build_hybrid_window(window)
                        x = apply_relative_transform(
                            scaler.transform(window).astype(np.float32),
                            relative_mode, n_rel_dims)
                        x_t = torch.tensor(x).unsqueeze(0).to(device)
                        score = model.reconstruction_error(x_t, weight_alpha).item()
                        
                        # Rule 2: Super high error cutoff (instant trigger)
                        if score > threshold_a * cutoff_multiplier:
                            flag_a = True
                            break

                        # Rule 3: Regular threshold
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


# ─────────────────────────────────────────────
#  학습 곡선 플롯
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────
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
    parser.add_argument("--noisy_filter", type=float, default=0.0,
                        help="MRL std 가 이 값을 초과하는 success 에피소드를 학습에서 제외 "
                             "(0=비활성화, 권장: 0.25~0.35)")
    parser.add_argument("--score_filter", type=float, default=0.0,
                        help="1차 학습 후 threshold 초과 윈도우 비율(0~1)이 이 값 이상인 "
                             "success 에피소드를 제거하고 재학습. "
                             "(0=비활성화, 권장: 0.15~0.25)")
    parser.add_argument("--mrl_thresh", type=float, default=0.3,
                        help="미스 가드레일 MRL 임계값 (기본: 0.3)")
    parser.add_argument("--cutoff_multiplier", type=float, default=2.2,
                        help="초고오차 즉시 차단 배수 (기본: 2.2, 즉 th_high = threshold_a * 2.2)")
    parser.add_argument("--warmup", type=int, default=30,
                        help="Ch-A eval 윈도우 skip 기준 (기본: 30)")
    parser.add_argument("--warmup_b", type=int, default=None,
                        help="threshold_b 보정용 워밍업. 미지정 시 --warmup 값 사용 (runtime Ch-B warmup에 맞춰 30 권장)")
    args = parser.parse_args()
    if args.warmup_b is None:
        args.warmup_b = args.warmup

    # ── 경로 설정
    ws = "/home/dyros/mcy_ws"
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
    print(f"[Config] mrl_thresh={args.mrl_thresh}, cutoff_multiplier={args.cutoff_multiplier}")
    if args.windowed:
        print(f"[Config] win_stride={args.win_stride} (슬라이딩 윈도우 모드)")

    # ── 노이즈 에피소드 필터링 (--noisy_filter)
    exclude_files: set = set()
    if args.noisy_filter > 0:
        print(f"\n[NoisyFilter] MRL std > {args.noisy_filter} 에피소드 검색 중...")
        exclude_files = find_noisy_episodes(data_dir, prefix, "success",
                                            args.use_raw, args.noisy_filter)
        print(f"[NoisyFilter] {len(exclude_files)}개 에피소드 학습에서 제외")

    # ── success 데이터 로드
    if args.windowed:
        success_seqs = load_sequences_windowed(data_dir, prefix, "success",
                                               args.use_raw, cfg["seq_len"],
                                               args.win_stride,
                                               exclude_files=exclude_files)
    else:
        success_seqs = load_sequences(data_dir, prefix, "success",
                                      args.use_raw, cfg["seq_len"],
                                      exclude_files=exclude_files)
    if len(success_seqs) == 0:
        print(f"[ERROR] success 데이터가 없습니다: {data_dir}/{prefix}_*_success*.npy")
        print("         먼저 --mode collect 로 데이터를 수집하세요.")
        return

    print(f"\n[Data] success 샘플 수: {len(success_seqs)}")
    raw_feat_dim = success_seqs[0].shape[1]

    # ── hybrid: per-frame norm 차원 추가 (D → D+1)
    if args.hybrid:
        success_seqs = [build_hybrid_window(w) for w in success_seqs]
        print(f"[Hybrid] per-frame norm 추가: {raw_feat_dim}D → {raw_feat_dim+1}D")

    input_dim = success_seqs[0].shape[1]
    n_rel_dims = raw_feat_dim if args.hybrid else None  # norm 차원은 relative 제외
    cfg["n_rel_dims"] = n_rel_dims
    print(f"[Data] feature dim: {input_dim}, seq_len: {cfg['seq_len']}")

    # ── 정규화 (success 데이터로만 fit)
    scaler = fit_scaler(success_seqs)
    success_seqs_norm = apply_scaler(success_seqs, scaler)
    joblib.dump(scaler, scaler_save_path)
    print(f"[Scaler] 저장: {scaler_save_path}")

    # ── Dataset / DataLoader
    dataset = TactileDataset(success_seqs_norm)
    val_size = max(1, int(len(dataset) * DEFAULTS["val_ratio"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)
    print(f"[Data] train={train_size}, val={val_size}")

    # hold-out: val 인덱스 기준으로 원본 시퀀스 추출 (threshold 계산에 사용)
    holdout_seqs = [success_seqs[i] for i in sorted(val_ds.indices)]
    print(f"[Threshold] hold-out 시퀀스 수: {len(holdout_seqs)}")

    # ── 모델 생성
    model = GRUAE(
        input_dim  = input_dim,
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        dropout    = DEFAULTS["dropout"],
        seq_len    = cfg["seq_len"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 파라미터 수: {total_params:,}")

    # ── 학습
    print("\n[Train] 시작...")
    train_losses, val_losses = train(model, train_loader, val_loader,
                                     cfg, model_save_path)
    plot_learning_curve(train_losses, val_losses, data_dir, prefix, args.use_raw)

    # ── 최적 가중치 로드 후 임계값 계산
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)

    # ── Threshold A: success 재구성 오차 분포의 N 백분위수
    threshold, train_errors = compute_threshold(
        model, success_seqs, scaler, cfg, device
    )
    np.save(thresh_save_path, np.array([threshold]))
    print(f"[Threshold A] 저장: {thresh_save_path} (value={threshold:.6f})")

    # ── Score-based 재학습 (--score_filter)
    if args.score_filter > 0:
        print(f"\n[ScoreFilter] 1차 threshold={threshold:.5f}, "
              f"pct_above>{args.score_filter:.2f} 에피소드 탐색...")
        raw_eps = load_sequences_raw(data_dir, prefix, "success", use_raw=args.use_raw)
        bad_files: set = set()
        model.eval()
        with torch.no_grad():
            for i, ep in enumerate(raw_eps):
                T = len(ep)
                if T < cfg["seq_len"]:
                    continue
                win_scores = []
                for start in range(0, T - cfg["seq_len"] + 1, args.win_stride):
                    window = ep[start:start + cfg["seq_len"]]
                    if args.hybrid:
                        window = build_hybrid_window(window)
                    x = apply_relative_transform(
                        scaler.transform(window).astype(np.float32),
                        args.relative, n_rel_dims)
                    x_t = torch.tensor(x).unsqueeze(0).to(device)
                    win_scores.append(
                        model.reconstruction_error(x_t, args.weight_alpha).item())
                pct_above = np.mean(np.array(win_scores) > threshold)
                # raw_eps와 파일 목록 매핑
                raw_files = sorted(glob.glob(
                    os.path.join(data_dir, f"{prefix}_*_success"
                                           + ("_raw.npy" if args.use_raw else ".npy"))))
                if not args.use_raw:
                    raw_files = [f for f in raw_files if not f.endswith("_raw.npy")]
                if i < len(raw_files) and pct_above >= args.score_filter:
                    fname = os.path.basename(raw_files[i])
                    bad_files.add(fname)
                    print(f"  [BAD] {fname}: pct_above={pct_above*100:.0f}%")

        if bad_files:
            print(f"\n[ScoreFilter] {len(bad_files)}개 에피소드 제거 후 재학습...")
            exclude_files = exclude_files | bad_files
            # 재학습
            if args.windowed:
                success_seqs = load_sequences_windowed(
                    data_dir, prefix, "success", args.use_raw,
                    cfg["seq_len"], args.win_stride, exclude_files=exclude_files)
            else:
                success_seqs = load_sequences(
                    data_dir, prefix, "success", args.use_raw,
                    cfg["seq_len"], exclude_files=exclude_files)

            if args.hybrid:
                success_seqs = [build_hybrid_window(w) for w in success_seqs]
            print(f"[ScoreFilter] 재학습 샘플 수: {len(success_seqs)}")

            scaler = fit_scaler(success_seqs)
            success_seqs_norm = apply_scaler(success_seqs, scaler)
            joblib.dump(scaler, scaler_save_path)

            dataset2 = TactileDataset(success_seqs_norm)
            val_size2 = max(1, int(len(dataset2) * DEFAULTS["val_ratio"]))
            train_size2 = len(dataset2) - val_size2
            train_ds2, val_ds2 = random_split(dataset2, [train_size2, val_size2],
                                              generator=torch.Generator().manual_seed(42))
            train_loader2 = DataLoader(train_ds2, batch_size=cfg["batch_size"], shuffle=True)
            val_loader2   = DataLoader(val_ds2,   batch_size=cfg["batch_size"], shuffle=False)

            model2 = GRUAE(input_dim=input_dim, hidden_dim=cfg["hidden_dim"],
                           num_layers=cfg["num_layers"], dropout=DEFAULTS["dropout"],
                           seq_len=cfg["seq_len"])
            print("[ScoreFilter] 재학습 시작...")
            train(model2, train_loader2, val_loader2, cfg, model_save_path)
            model2.load_state_dict(torch.load(model_save_path, map_location=device))
            model = model2.to(device)

            threshold, train_errors = compute_threshold(model, success_seqs, scaler, cfg, device)
            np.save(thresh_save_path, np.array([threshold]))
            print(f"[ScoreFilter] 재학습 완료. 새 threshold_A={threshold:.6f}")
        else:
            print("[ScoreFilter] 제거할 에피소드 없음. 재학습 생략.")

    # ── Threshold B: 채널 B (MRL delta) 캘리브레이션
    # 에피소드 단위 hold-out 분리 (seed=42, val_ratio 동일)
    sys.path.insert(0, "/home/dyros/mcy_ws")
    from dual_channel_detector import calibrate_threshold_b
    raw_success_all = load_sequences_raw(data_dir, prefix, "success", use_raw=args.use_raw)
    if not raw_success_all:
        raw_success_all = success_seqs  # fallback
    n_ep = len(raw_success_all)
    rng_ep = np.random.default_rng(42)
    hold_ep_n = max(1, int(n_ep * DEFAULTS["val_ratio"]))
    hold_ep_idx = sorted(rng_ep.choice(n_ep, hold_ep_n, replace=False).tolist())
    train_ep_idx = [i for i in range(n_ep) if i not in set(hold_ep_idx)]
    raw_success_holdout = [raw_success_all[i] for i in hold_ep_idx]
    raw_success_train   = [raw_success_all[i] for i in train_ep_idx]
    print(f"[Threshold B] raw 에피소드 hold-out: {len(raw_success_holdout)}/{n_ep}")
    threshold_b = calibrate_threshold_b(raw_success_all, smooth_window=3, percentile=99, warmup=args.warmup_b)
    np.save(thresh_b_save_path, np.array([threshold_b]))
    print(f"[Threshold B] 저장: {thresh_b_save_path} (value={threshold_b:.6f})")

    # ── Inference config 저장 (basket_ori.py 가 참조)
    infer_cfg = {
        "relative":          args.relative,
        "weight_alpha":      args.weight_alpha,
        "use_raw":           args.use_raw,
        "seq_len":           cfg["seq_len"],
        "hybrid":            args.hybrid,
        "n_rel_dims":        n_rel_dims,
        "mrl_thresh":        args.mrl_thresh,
        "cutoff_multiplier": args.cutoff_multiplier,
    }
    with open(infer_cfg_path, "wb") as f:
        pickle.dump(infer_cfg, f)
    print(f"[InferCfg] 저장: {infer_cfg_path} → {infer_cfg}")

    # ── 평가 플롯 (--eval 옵션)
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
        infer_stride = args.win_stride  # 실제 inference stride와 동일하게 평가
        print(f"\n[Dual-Channel 시퀀스 단위 평가] (A OR B, stride={infer_stride} 슬라이딩)")
        print(f"  Threshold A = {threshold:.6f}, Threshold B = {threshold_b:.6f}")
        raw_seqs_by_label = {}
        # success는 전체 에피소드 사용 (hold-out 10개는 seq 단위 평가에 통계량이 부족)
        # threshold는 hold-out 윈도우로 계산했으므로 목적은 달성됨
        if raw_success_all:
            raw_seqs_by_label["success"] = raw_success_all
        for label in ["perturbation", "failure"]:
            seqs = load_sequences_raw(data_dir, prefix, label, use_raw=args.use_raw)
            if seqs:
                raw_seqs_by_label[label] = seqs
        if raw_seqs_by_label:
            for consec_a, consec_b in [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (2, 2), (3, 2), (5, 3)]:
                combined = evaluate_dual_channel_combined(
                    model, raw_seqs_by_label, scaler,
                    threshold_a=threshold, threshold_b=threshold_b,
                    seq_len=cfg["seq_len"], relative_mode=args.relative,
                    weight_alpha=args.weight_alpha, n_rel_dims=n_rel_dims,
                    device=device, smooth_window=3,
                    consec_a=consec_a, consec_b=consec_b,
                    infer_stride=infer_stride,
                    hybrid=args.hybrid,
                    mrl_thresh=args.mrl_thresh,
                    cutoff_multiplier=args.cutoff_multiplier,
                    warmup=args.warmup,
                    warmup_b=args.warmup_b,
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
