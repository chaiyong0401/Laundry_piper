"""학습 없이 eval 결과만 재계산 (hybrid 버그 수정 후 검증용)."""
import sys, glob, pickle, os
import numpy as np
import torch
import joblib
sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import (
    GRUAE, apply_relative_transform, build_hybrid_window,
    load_sequences_raw, evaluate_dual_channel_combined
)

DATA_DIR  = 'grasp_05'
PREFIX    = 'grasp'
USE_RAW   = True
SEQ_LEN   = 50
WIN_STRIDE = 10

scaler = joblib.load(f'{DATA_DIR}/{PREFIX}_scaler_raw.pkl')
with open(f'{DATA_DIR}/{PREFIX}_infer_cfg_raw.pkl', 'rb') as f:
    cfg = pickle.load(f)

relative_mode = cfg.get('relative', 'none')
weight_alpha  = cfg.get('weight_alpha', 0.0)
n_rel_dims    = cfg.get('n_rel_dims', None)
hybrid        = cfg.get('hybrid', False)

threshold   = np.load(f'{DATA_DIR}/{PREFIX}_threshold_raw.npy')[0]
threshold_b = np.load(f'{DATA_DIR}/{PREFIX}_threshold_b_raw.npy')[0]

print(f"hybrid={hybrid}, relative={relative_mode}, n_rel_dims={n_rel_dims}")
print(f"Threshold A={threshold:.6f}, Threshold B={threshold_b:.6f}")
print(f"Scaler n_features={scaler.n_features_in_}")

input_dim = scaler.n_features_in_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUAE(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2, seq_len=SEQ_LEN)
model.load_state_dict(torch.load(f'{DATA_DIR}/{PREFIX}_gru_ae_raw.pt', map_location=device))
model = model.to(device)
model.eval()

raw_seqs = {}
for label in ['success', 'failure']:
    seqs = load_sequences_raw(DATA_DIR, PREFIX, label, use_raw=USE_RAW)
    if seqs:
        raw_seqs[label] = seqs
        print(f"  {label}: {len(seqs)} seqs")

print(f"\n[Dual-Channel 시퀀스 단위 평가] (stride={WIN_STRIDE})")
for consec_a, consec_b in [(1, 1), (2, 3), (2, 4), (3, 3), (3, 5)]:
    combined = evaluate_dual_channel_combined(
        model, raw_seqs, scaler,
        threshold_a=threshold, threshold_b=threshold_b,
        seq_len=SEQ_LEN, relative_mode=relative_mode,
        weight_alpha=weight_alpha, n_rel_dims=n_rel_dims,
        device=device, smooth_window=3,
        consec_a=consec_a, consec_b=consec_b,
        infer_stride=WIN_STRIDE,
        hybrid=hybrid,
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
