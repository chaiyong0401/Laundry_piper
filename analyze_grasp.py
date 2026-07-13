import numpy as np
import glob
import pickle
import joblib
import sys

sys.path.insert(0, "/home/dyros/mcy_ws")

success_files = sorted(glob.glob('grasp_05/grasp_*_success_raw.npy'))
failure_files = sorted(glob.glob('grasp_05/grasp_*_failure_raw.npy'))
success_6d    = sorted(glob.glob('grasp_05/grasp_*_success.npy'))
failure_6d    = sorted(glob.glob('grasp_05/grasp_*_failure.npy'))

print(f"=== 데이터 현황 ===")
print(f"Success (48D raw): {len(success_files)}")
print(f"Failure (48D raw): {len(failure_files)}")
print(f"Success (6D):      {len(success_6d)}")
print(f"Failure (6D):      {len(failure_6d)}")

with open('grasp_05/grasp_infer_cfg_raw.pkl', 'rb') as f:
    cfg = pickle.load(f)
print(f"\ninfer_cfg (raw): {cfg}")

try:
    with open('grasp_05/grasp_infer_cfg.pkl', 'rb') as f:
        cfg6 = pickle.load(f)
    print(f"infer_cfg (6D):  {cfg6}")
except:
    pass

shapes_s = [np.load(f).shape for f in success_files]
shapes_f = [np.load(f).shape for f in failure_files]
print(f"\n=== 시퀀스 길이 (48D raw) ===")
print(f"Success T: min={min(s[0] for s in shapes_s)}, max={max(s[0] for s in shapes_s)}, mean={np.mean([s[0] for s in shapes_s]):.1f}")
print(f"Failure T: min={min(s[0] for s in shapes_f)}, max={max(s[0] for s in shapes_f)}, mean={np.mean([s[0] for s in shapes_f]):.1f}")

thr   = np.load('grasp_05/grasp_threshold_raw.npy')
thr_b = np.load('grasp_05/grasp_threshold_b_raw.npy')
thr6  = np.load('grasp_05/grasp_threshold.npy')
thr6b = np.load('grasp_05/grasp_threshold_b.npy')
print(f"\n=== Threshold ===")
print(f"48D raw  - A: {thr[0]:.6f}, B: {thr_b[0]:.6f}")
print(f"6D feat  - A: {thr6[0]:.6f}, B: {thr6b[0]:.6f}")

# 실제 runtime inference score 확인
print(f"\n=== Runtime inference scores (infer_debug) ===")
debug_files = sorted(glob.glob('grasp_05/infer_debug_*_raw.npy'))
print(f"Debug files: {len(debug_files)}")
for f in debug_files:
    d = np.load(f)
    fname = f.split("/")[-1]
    print(f"  {fname}: shape={d.shape}")

# GRU-AE로 training set 재구성 오차 계산
print(f"\n=== GRU-AE reconstruction error 분포 (train set 평가) ===")
try:
    import torch
    from train_gru_ae import GRUAE, apply_relative_transform, build_hybrid_window
    from scipy.signal import resample

    scaler = joblib.load('grasp_05/grasp_scaler_raw.pkl')
    checkpoint = torch.load('grasp_05/grasp_gru_ae_raw.pt', map_location='cpu')
    input_dim = scaler.n_features_in_
    model = GRUAE(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2, seq_len=50)
    model.load_state_dict(checkpoint)
    model.eval()

    relative_mode = cfg.get("relative", "none")
    weight_alpha  = cfg.get("weight_alpha", 0.0)
    hybrid        = cfg.get("hybrid", False)
    n_rel_dims    = cfg.get("n_rel_dims", None)

    def get_score(fpath):
        seq = np.load(fpath).astype(np.float32)
        seq_fixed = resample(seq, 50, axis=0).astype(np.float32)
        if hybrid:
            seq_fixed = build_hybrid_window(seq_fixed)
        seq_s = scaler.transform(seq_fixed)
        seq_r = apply_relative_transform(seq_s, relative_mode, n_rel_dims=n_rel_dims)
        x = torch.tensor(seq_r, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = float(model.reconstruction_error(x, weight_alpha=weight_alpha).item())
        return score

    s_scores = [get_score(f) for f in success_files]
    f_scores = [get_score(f) for f in failure_files]

    print(f"Success scores: mean={np.mean(s_scores):.6f}, std={np.std(s_scores):.6f}, min={np.min(s_scores):.6f}, max={np.max(s_scores):.6f}")
    print(f"Failure scores: mean={np.mean(f_scores):.6f}, std={np.std(f_scores):.6f}, min={np.min(f_scores):.6f}, max={np.max(f_scores):.6f}")
    print(f"Threshold A: {thr[0]:.6f}")
    print(f"  Success above threshold: {sum(s > thr[0] for s in s_scores)}/{len(s_scores)} (false positive)")
    print(f"  Failure above threshold: {sum(s > thr[0] for s in f_scores)}/{len(f_scores)} (true positive)")

    # 분리도 확인
    overlap = sum(1 for s in s_scores if s > thr[0]) + sum(1 for s in f_scores if s <= thr[0])
    print(f"  Misclassified: {overlap} / {len(s_scores)+len(f_scores)}")

except Exception as e:
    print(f"  모델 평가 실패: {e}")
