"""
visualize_fn_vs_tp.py
FN(놓친 실패) vs TP(감지된 실패) 촉각 신호 비교 시각화
6D feature: [mrl, norm, max_norm, unit_x, unit_y, unit_z]
"""

import os, sys, pickle
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import GRUAE, apply_relative_transform

GRASP_DIR = '/home/dyros/mcy_ws/grasp_new'
SEQ_LEN   = 50
WIN_STRIDE = 10
SMOOTH_WIN = 3
CONSEC_A  = 2
CONSEC_B  = 5

# FN 파일명 (eval 결과)
FN_FILES = {
    'grasp_20260522_150424_467429_failure.npy',
    'grasp_20260527_193308_761677_failure.npy',
    'grasp_20260527_193500_463084_failure.npy',
    'grasp_20260527_200123_761854_failure.npy',
    'grasp_20260527_204011_004913_failure.npy',
    'grasp_20260527_204450_061379_failure.npy',
    'grasp_20260527_204525_294943_failure.npy',
    'grasp_20260527_210134_409778_failure.npy',
}

def load_model():
    prefix = 'grasp'
    pt  = os.path.join(GRASP_DIR, f'{prefix}_gru_ae.pt')
    pkl = os.path.join(GRASP_DIR, f'{prefix}_scaler.pkl')
    cfg = os.path.join(GRASP_DIR, f'{prefix}_infer_cfg.pkl')
    thr = os.path.join(GRASP_DIR, f'{prefix}_threshold.npy')
    thrb = os.path.join(GRASP_DIR, f'{prefix}_threshold_b.npy')

    state  = torch.load(pt, map_location='cpu', weights_only=False)
    hidden = state['encoder.gru.weight_hh_l0'].shape[1]
    scaler = joblib.load(pkl)
    input_dim = scaler.n_features_in_
    model = GRUAE(input_dim=input_dim, hidden_dim=hidden, num_layers=2,
                  dropout=0.0, seq_len=SEQ_LEN)
    model.load_state_dict(state)
    model.eval()
    with open(cfg, 'rb') as f:
        infer_cfg = pickle.load(f)
    threshold_a = float(np.load(thr)[0])
    threshold_b = float(np.load(thrb)[0])
    return model, scaler, infer_cfg, threshold_a, threshold_b

def get_recon_errors(seq, model, scaler, infer_cfg):
    """각 윈도우의 reconstruction error + 윈도우 중간 시간 인덱스 반환."""
    errors, times = [], []
    for start in range(0, len(seq) - SEQ_LEN + 1, WIN_STRIDE):
        window = seq[start:start+SEQ_LEN]
        x = scaler.transform(window).astype(np.float32)
        x = apply_relative_transform(x, infer_cfg.get('relative','none'),
                                     infer_cfg.get('n_rel_dims'))
        x_t = torch.tensor(x).unsqueeze(0)
        with torch.no_grad():
            e = float(model.reconstruction_error(
                x_t, infer_cfg.get('weight_alpha', 0.0)).item())
        errors.append(e)
        times.append(start + SEQ_LEN // 2)
    return np.array(times), np.array(errors)

def get_mrl_series(seq):
    """frame별 MRL (6D[0]) 시계열."""
    return seq[:, 0]

def get_norm_series(seq):
    """frame별 norm (6D[1]) 시계열."""
    return seq[:, 1]

def get_mrl_delta_series(seq):
    """frame별 |MRL - smoothed_MRL| 시계열."""
    hist = deque(maxlen=SMOOTH_WIN)
    prev = None
    deltas = []
    for frame in seq:
        mrl = float(frame[0])
        if prev is None:
            prev = mrl; hist.append(mrl); deltas.append(0.0); continue
        ref = float(np.mean(hist)) if len(hist) == SMOOTH_WIN else prev
        deltas.append(abs(mrl - ref))
        hist.append(mrl); prev = mrl
    return np.array(deltas)

def load_all_failures():
    fn_seqs, tp_seqs = {}, {}
    for f in sorted(os.listdir(GRASP_DIR)):
        if not f.endswith('.npy') or 'raw' in f: continue
        if 'grasp_2' not in f or 'failure' not in f: continue
        arr = np.load(os.path.join(GRASP_DIR, f)).astype(np.float32)
        if f in FN_FILES:
            fn_seqs[f] = arr
        else:
            tp_seqs[f] = arr
    return fn_seqs, tp_seqs

def plot_comparison(fn_seqs, tp_seqs, model, scaler, infer_cfg, thr_a, thr_b):
    fn_list = sorted(fn_seqs.items())
    tp_list = sorted(tp_seqs.items())

    n_fn = len(fn_list)
    n_tp = len(tp_list)

    fig = plt.figure(figsize=(22, n_fn * 3.5 + 2))
    fig.suptitle('FN (Missed) vs TP (Detected) — Grasp Failure Comparison\n'
                 'Red=FN(missed), Blue=TP(detected)', fontsize=13, y=0.99)

    # 3열: MRL | Norm | Recon Error
    cols = 3
    rows = max(n_fn, n_tp)

    gs = gridspec.GridSpec(rows * 2 + 1, cols, figure=fig,
                           hspace=0.55, wspace=0.35,
                           top=0.96, bottom=0.03)

    def short_name(fname):
        return fname.replace('grasp_', '').replace('_failure.npy', '')

    def plot_seq(seq, fname, row, color, label_prefix):
        mrl   = get_mrl_series(seq)
        norm  = get_norm_series(seq)
        delta = get_mrl_delta_series(seq)
        t_win, errs = get_recon_errors(seq, model, scaler, infer_cfg)
        t = np.arange(len(seq))
        title = f"{label_prefix}: {short_name(fname)}"

        # MRL
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.plot(t, mrl, color=color, lw=1.0)
        ax0.axhline(thr_b, color='gray', ls='--', lw=0.8, label=f'thr_B={thr_b:.3f}')
        ax0.set_title(title, fontsize=7.5, pad=2)
        ax0.set_ylabel('MRL', fontsize=7); ax0.set_ylim(0, 1.05)
        ax0.tick_params(labelsize=6); ax0.legend(fontsize=6)

        # Norm
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.plot(t, norm, color=color, lw=1.0)
        ax1.set_title(title, fontsize=7.5, pad=2)
        ax1.set_ylabel('Norm', fontsize=7)
        ax1.tick_params(labelsize=6)

        # Recon Error
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.plot(t_win, errs, color=color, lw=1.2, marker='o', ms=3)
        ax2.axhline(thr_a, color='gray', ls='--', lw=0.8, label=f'thr_A={thr_a:.3f}')
        ax2.set_title(title, fontsize=7.5, pad=2)
        ax2.set_ylabel('Recon Error', fontsize=7)
        ax2.tick_params(labelsize=6); ax2.legend(fontsize=6)

    # FN rows
    for i, (fname, seq) in enumerate(fn_list):
        plot_seq(seq, fname, i, color='crimson', label_prefix='FN')

    # separator row (empty)
    sep_row = n_fn
    ax_sep = fig.add_subplot(gs[sep_row, :])
    ax_sep.axis('off')
    ax_sep.text(0.5, 0.5, '─── TP (Detected) below ───',
                ha='center', va='center', fontsize=10, color='steelblue',
                transform=ax_sep.transAxes)

    # TP rows
    for i, (fname, seq) in enumerate(tp_list):
        plot_seq(seq, fname, sep_row + 1 + i, color='steelblue', label_prefix='TP')

    # column headers
    for col, title in enumerate(['MRL over time', 'Norm over time', 'Recon Error (Ch-A)']):
        fig.text(0.18 + col * 0.28, 0.975, title,
                 ha='center', fontsize=10, fontweight='bold')

    out = '/home/dyros/mcy_ws/fn_vs_tp_comparison.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"[SAVE] {out}")
    return out

def main():
    print("Loading model...")
    model, scaler, infer_cfg, thr_a, thr_b = load_model()

    # calibrate thr_a (μ+3σ on train success)
    import random
    rng = random.Random(42)
    succ_files = sorted([f for f in os.listdir(GRASP_DIR)
                         if f.endswith('.npy') and 'raw' not in f
                         and 'grasp_2' in f and 'success' in f])
    succ_seqs = [np.load(os.path.join(GRASP_DIR, f)).astype(np.float32)
                 for f in succ_files]
    rng.shuffle(succ_seqs)
    n_test = max(1, round(len(succ_seqs) * 0.20))
    train_succ = succ_seqs[n_test:]

    from eval_dual_channel_combined import score_channel_a
    print(f"Calibrating thr_A on {len(train_succ)} train success seqs...")
    scores = [score_channel_a(s, model, scaler, infer_cfg) for s in train_succ]
    thr_a_cal = float(np.mean(scores) + 3.0 * np.std(scores))
    print(f"  thr_A_calibrated={thr_a_cal:.4f}  thr_B={thr_b:.4f}")

    fn_seqs, tp_seqs = load_all_failures()
    print(f"FN={len(fn_seqs)}, TP={len(tp_seqs)}")

    out = plot_comparison(fn_seqs, tp_seqs, model, scaler, infer_cfg, thr_a_cal, thr_b)
    print(f"Done: {out}")

if __name__ == '__main__':
    main()
