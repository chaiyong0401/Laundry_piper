"""
실제 sliding window 기반 score 분포 확인.
training data에서 stride=5 sliding window로 score 계산 → runtime 분포와 직접 비교.
"""
import numpy as np
import glob
import pickle
import joblib
import sys
sys.path.insert(0, '/home/dyros/mcy_ws')
import torch
from train_gru_ae import GRUAE, apply_relative_transform

DATA_DIR = 'grasp_05'
SEQ_LEN  = 50
STRIDE   = 5

success_files = sorted(glob.glob(f'{DATA_DIR}/grasp_*_success_raw.npy'))
failure_files = sorted(glob.glob(f'{DATA_DIR}/grasp_*_failure_raw.npy'))
debug_files   = sorted(glob.glob(f'{DATA_DIR}/infer_debug_*_raw.npy'))

scaler = joblib.load(f'{DATA_DIR}/grasp_scaler_raw.pkl')
with open(f'{DATA_DIR}/grasp_infer_cfg_raw.pkl', 'rb') as f:
    cfg = pickle.load(f)

print(f"infer_cfg: {cfg}")

checkpoint = torch.load(f'{DATA_DIR}/grasp_gru_ae_raw.pt', map_location='cpu')
model = GRUAE(input_dim=48, hidden_dim=64, num_layers=2, dropout=0.2, seq_len=50)
model.load_state_dict(checkpoint)
model.eval()

relative_mode = cfg.get('relative', 'none')
weight_alpha  = cfg.get('weight_alpha', 0.0)

thr = np.load(f'{DATA_DIR}/grasp_threshold_raw.npy')[0]
print(f"Current threshold: {thr:.6f}")
print(f"n_success={len(success_files)}, n_failure={len(failure_files)}, n_debug={len(debug_files)}\n")


def score_windows(fpath, stride=STRIDE):
    seq = np.load(fpath).astype(np.float32)
    T = len(seq)
    if T < SEQ_LEN:
        return []
    scores = []
    for start in range(0, T - SEQ_LEN + 1, stride):
        w = seq[start:start+SEQ_LEN]
        w_s = scaler.transform(w)
        w_r = apply_relative_transform(w_s, relative_mode)
        x = torch.tensor(w_r, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            scores.append(float(model.reconstruction_error(x, weight_alpha=weight_alpha).item()))
    return scores


def score_max_window(fpath):
    """시퀀스 단위 이상 판단: any window > threshold"""
    scores = score_windows(fpath)
    return scores if scores else []


# ── success
s_all = []
s_max = []
s_seq_lengths = []
for f in success_files:
    seq = np.load(f)
    s_seq_lengths.append(len(seq))
    wins = score_windows(f)
    s_all.extend(wins)
    if wins:
        s_max.append(max(wins))

# ── failure
f_all = []
f_max = []
f_seq_lengths = []
for f in failure_files:
    seq = np.load(f)
    f_seq_lengths.append(len(seq))
    wins = score_windows(f)
    f_all.extend(wins)
    if wins:
        f_max.append(max(wins))

# ── debug (runtime)
d_all = []
for f in debug_files:
    wins = score_windows(f)
    d_all.extend(wins)

print("=== 시퀀스 길이 ===")
print(f"Success T: min={min(s_seq_lengths)}, max={max(s_seq_lengths)}, mean={np.mean(s_seq_lengths):.1f}")
print(f"Failure T: min={min(f_seq_lengths)}, max={max(f_seq_lengths)}, mean={np.mean(f_seq_lengths):.1f}")
print()

print("=== 슬라이딩 윈도우 score 분포 (stride=5) ===")
s_all = np.array(s_all)
f_all = np.array(f_all)
print(f"Success  n_windows={len(s_all)}: mean={np.mean(s_all):.6f}  std={np.std(s_all):.6f}  "
      f"p50={np.percentile(s_all,50):.6f}  p95={np.percentile(s_all,95):.6f}  "
      f"p99={np.percentile(s_all,99):.6f}  max={np.max(s_all):.6f}")
print(f"Failure  n_windows={len(f_all)}: mean={np.mean(f_all):.6f}  std={np.std(f_all):.6f}  "
      f"p5={np.percentile(f_all,5):.6f}  p50={np.percentile(f_all,50):.6f}  "
      f"min={np.min(f_all):.6f}  max={np.max(f_all):.6f}")
if d_all:
    d_all = np.array(d_all)
    print(f"Runtime  n_windows={len(d_all)}: mean={np.mean(d_all):.6f}  std={np.std(d_all):.6f}  "
          f"min={np.min(d_all):.6f}  max={np.max(d_all):.6f}")
print(f"Threshold A: {thr:.6f}")
print()

# 시퀀스 단위 이상 판단 (any window > thr)
print("=== 시퀀스 단위 감지율 (any window > threshold) ===")
s_fp = sum(1 for m in s_max if m > thr)
f_tp = sum(1 for m in f_max if m > thr)
print(f"Success: {s_fp}/{len(s_max)} above threshold  (False Positive rate)")
print(f"Failure: {f_tp}/{len(f_max)} above threshold  (True Positive rate)")
print()

# threshold sweep
print("=== Threshold sweep (window-level) ===")
print(f"{'Thr':>8}  {'S_above':>14}  {'F_above':>14}  {'F1':>8}")
best_f1, best_thr = 0.0, 0.0
for t in np.arange(0.005, 0.20, 0.005):
    fp = np.sum(s_all > t)
    tp = np.sum(f_all > t)
    fn = len(f_all) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    flag = " <-- current" if abs(t - thr) < 0.003 else ""
    if f1 > best_f1:
        best_f1, best_thr = f1, t
    print(f"{t:8.3f}  {fp:>5}/{len(s_all)}={fp/len(s_all)*100:4.1f}%  "
          f"{tp:>5}/{len(f_all)}={tp/len(f_all)*100:4.1f}%  {f1:8.4f}{flag}")

print(f"\nBest window-level threshold: {best_thr:.3f}  F1={best_f1:.4f}")

# 시퀀스 단위 threshold sweep (max-score per seq)
print("\n=== Threshold sweep (sequence-level, max window score per seq) ===")
print(f"{'Thr':>8}  {'FP_seq':>14}  {'TP_seq':>14}  {'F1_seq':>8}")
best_f1s, best_thrs = 0.0, 0.0
s_max_arr = np.array(s_max)
f_max_arr = np.array(f_max)
for t in np.arange(0.005, 0.20, 0.005):
    fp = np.sum(s_max_arr > t)
    tp = np.sum(f_max_arr > t)
    fn = len(f_max_arr) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    flag = " <-- current" if abs(t - thr) < 0.003 else ""
    if f1 > best_f1s:
        best_f1s, best_thrs = f1, t
    print(f"{t:8.3f}  {fp:>3}/{len(s_max)}={fp/len(s_max)*100:4.1f}%  "
          f"{tp:>3}/{len(f_max)}={tp/len(f_max)*100:4.1f}%  {f1:8.4f}{flag}")

print(f"\nBest sequence-level threshold: {best_thrs:.3f}  F1={best_f1s:.4f}")
