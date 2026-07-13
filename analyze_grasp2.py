import numpy as np
import glob
import pickle
import joblib
import sys
sys.path.insert(0, '/home/dyros/mcy_ws')
import torch
from train_gru_ae import GRUAE, apply_relative_transform
from scipy.signal import resample

success_files = sorted(glob.glob('grasp_05/grasp_*_success_raw.npy'))
failure_files = sorted(glob.glob('grasp_05/grasp_*_failure_raw.npy'))
debug_files   = sorted(glob.glob('grasp_05/infer_debug_*_raw.npy'))

scaler = joblib.load('grasp_05/grasp_scaler_raw.pkl')
with open('grasp_05/grasp_infer_cfg_raw.pkl', 'rb') as f:
    cfg = pickle.load(f)

checkpoint = torch.load('grasp_05/grasp_gru_ae_raw.pt', map_location='cpu')
model = GRUAE(input_dim=48, hidden_dim=64, num_layers=2, dropout=0.2, seq_len=50)
model.load_state_dict(checkpoint)
model.eval()

relative_mode = cfg.get('relative', 'none')
weight_alpha  = cfg.get('weight_alpha', 0.0)

def score_full(fpath):
    seq = np.load(fpath).astype(np.float32)
    seq_fixed = resample(seq, 50, axis=0).astype(np.float32)
    seq_s = scaler.transform(seq_fixed)
    seq_r = apply_relative_transform(seq_s, relative_mode)
    x = torch.tensor(seq_r, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return float(model.reconstruction_error(x, weight_alpha=weight_alpha).item())

def score_sliding(fpath):
    seq = np.load(fpath).astype(np.float32)
    if len(seq) < 50:
        return None
    window = seq[-50:]
    seq_s = scaler.transform(window)
    seq_r = apply_relative_transform(seq_s, relative_mode)
    x = torch.tensor(seq_r, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return float(model.reconstruction_error(x, weight_alpha=weight_alpha).item())

s_scores = [score_full(f) for f in success_files]
f_scores = [score_full(f) for f in failure_files]
d_scores = [score_sliding(f) for f in debug_files]
d_scores = [s for s in d_scores if s is not None]

thr = np.load('grasp_05/grasp_threshold_raw.npy')[0]

print("=== Score 분포 ===")
print(f"Success  n={len(s_scores)}: mean={np.mean(s_scores):.4f}  std={np.std(s_scores):.4f}  p50={np.percentile(s_scores,50):.4f}  p90={np.percentile(s_scores,90):.4f}  p95={np.percentile(s_scores,95):.4f}  max={np.max(s_scores):.4f}")
print(f"Failure  n={len(f_scores)}: mean={np.mean(f_scores):.4f}  std={np.std(f_scores):.4f}  p5={np.percentile(f_scores,5):.4f}  p50={np.percentile(f_scores,50):.4f}  min={np.min(f_scores):.4f}")
print(f"Runtime  n={len(d_scores)}: mean={np.mean(d_scores):.4f}  std={np.std(d_scores):.4f}  min={np.min(d_scores):.4f}  max={np.max(d_scores):.4f}")
print(f"Current threshold A: {thr:.6f}")

# threshold가 어떻게 계산됐는지 확인 (95th percentile of success)
pct95_s = np.percentile(s_scores, 95)
print(f"95th pct of success: {pct95_s:.6f}  (threshold 설정 기준)")
print()

# 분포 overlap 시각화 (텍스트)
print("=== Score 히스토그램 (텍스트) ===")
bins = np.arange(0, 0.6, 0.05)
s_hist, _ = np.histogram(s_scores, bins=bins)
f_hist, _ = np.histogram(f_scores, bins=bins)
print(f"{'Bin':>12}  {'Success':>10}  {'Failure':>10}")
for i in range(len(bins)-1):
    marker = " <<< overlap" if s_hist[i] > 0 and f_hist[i] > 0 else ""
    print(f"{bins[i]:.2f}~{bins[i+1]:.2f}   {s_hist[i]:>8}    {f_hist[i]:>8}{marker}")
print()

# threshold sweep
print("=== Threshold별 성능 ===")
print(f"{'Thr':>8}  {'FP':>12}  {'TP':>12}  {'F1':>8}")
best_f1, best_thr = 0, 0
for t in np.arange(0.05, 0.55, 0.02):
    fp = sum(s > t for s in s_scores)
    tp = sum(s > t for s in f_scores)
    fn = len(f_scores) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    flag = " <-- current" if abs(t - thr) < 0.015 else ""
    if f1 > best_f1:
        best_f1, best_thr = f1, t
    print(f"{t:8.3f}  {fp:>3}/{len(s_scores)}={fp/len(s_scores)*100:4.1f}%  {tp:>3}/{len(f_scores)}={tp/len(f_scores)*100:4.1f}%  {f1:8.4f}{flag}")

print(f"\nBest threshold: {best_thr:.3f}  F1={best_f1:.4f}")

# 현재 threshold vs best threshold 비교
print()
print("=== Runtime 24개 샘플에서 각 threshold 판정 비교 ===")
fp_cur = sum(s > thr for s in d_scores)
fp_best = sum(s > best_thr for s in d_scores)
print(f"Current thr={thr:.4f}: {fp_cur}/{len(d_scores)} anomaly 판정")
print(f"Best    thr={best_thr:.3f}: {fp_best}/{len(d_scores)} anomaly 판정")
print(f"Runtime score 분포: {sorted(d_scores)}")
