"""
FN vs TP 수치 통계 + 대표 샘플 클린 시각화
"""
import os, sys, pickle, random
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import GRUAE, apply_relative_transform

GRASP_DIR  = '/home/dyros/mcy_ws/grasp_new'
SEQ_LEN    = 50
WIN_STRIDE = 10
SMOOTH_WIN = 3

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
    pt   = os.path.join(GRASP_DIR, 'grasp_gru_ae.pt')
    pkl  = os.path.join(GRASP_DIR, 'grasp_scaler.pkl')
    cfg  = os.path.join(GRASP_DIR, 'grasp_infer_cfg.pkl')
    thrb = os.path.join(GRASP_DIR, 'grasp_threshold_b.npy')
    state  = torch.load(pt, map_location='cpu', weights_only=False)
    hidden = state['encoder.gru.weight_hh_l0'].shape[1]
    scaler = joblib.load(pkl)
    model  = GRUAE(input_dim=scaler.n_features_in_, hidden_dim=hidden,
                   num_layers=2, dropout=0.0, seq_len=SEQ_LEN)
    model.load_state_dict(state); model.eval()
    with open(cfg, 'rb') as f: infer_cfg = pickle.load(f)
    thr_b = float(np.load(thrb)[0])
    return model, scaler, infer_cfg, thr_b

def recon_errors(seq, model, scaler, cfg):
    times, errs = [], []
    for s in range(0, len(seq) - SEQ_LEN + 1, WIN_STRIDE):
        w = seq[s:s+SEQ_LEN]
        x = scaler.transform(w).astype(np.float32)
        x = apply_relative_transform(x, cfg.get('relative','none'), cfg.get('n_rel_dims'))
        with torch.no_grad():
            e = float(model.reconstruction_error(
                torch.tensor(x).unsqueeze(0), cfg.get('weight_alpha',0.0)).item())
        times.append(s + SEQ_LEN//2); errs.append(e)
    return np.array(times), np.array(errs)

def mrl_delta(seq):
    hist = deque(maxlen=SMOOTH_WIN); prev = None; d = []
    for f in seq:
        m = float(f[0])
        if prev is None:
            prev = m; hist.append(m); d.append(0.0); continue
        ref = float(np.mean(hist)) if len(hist)==SMOOTH_WIN else prev
        d.append(abs(m - ref))
        hist.append(m); prev = m
    return np.array(d)

def calibrate_thr_a(model, scaler, cfg):
    rng = random.Random(42)
    files = sorted([f for f in os.listdir(GRASP_DIR)
                    if f.endswith('.npy') and 'raw' not in f
                    and 'grasp_2' in f and 'success' in f])
    seqs = [np.load(os.path.join(GRASP_DIR, f)).astype(np.float32) for f in files]
    rng.shuffle(seqs)
    train = seqs[max(1, round(len(seqs)*0.20)):]
    sc = []
    for s in train:
        _, e = recon_errors(s, model, scaler, cfg)
        sc.append(max(e) if len(e) else 0.0)
    return float(np.mean(sc) + 3.0*np.std(sc))

def load_failures():
    fn_d, tp_d = {}, {}
    for f in sorted(os.listdir(GRASP_DIR)):
        if not f.endswith('.npy') or 'raw' in f: continue
        if 'grasp_2' not in f or 'failure' not in f: continue
        arr = np.load(os.path.join(GRASP_DIR, f)).astype(np.float32)
        (fn_d if f in FN_FILES else tp_d)[f] = arr
    return fn_d, tp_d

def stats(seqs, model, scaler, cfg, thr_a, thr_b):
    rows = []
    for fname, seq in seqs.items():
        _, e = recon_errors(seq, model, scaler, cfg)
        d = mrl_delta(seq)
        rows.append({
            'file': fname,
            'len': len(seq),
            'mrl_mean': seq[:,0].mean(),
            'mrl_min':  seq[:,0].min(),
            'norm_mean': seq[:,1].mean(),
            'max_err':  e.max() if len(e) else 0.0,
            'max_delta': d.max(),
            'err_above_thr': (e > thr_a).sum(),
            'delta_above_thr': (d > thr_b).sum(),
        })
    return rows

# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_panels(fn_d, tp_d, model, scaler, cfg, thr_a, thr_b, n_show=5):
    """FN/TP 각 n_show개 대표 샘플, 3열 (MRL, Norm, Recon Error)."""
    fn_items = sorted(fn_d.items())[:n_show]
    tp_items = sorted(tp_d.items())[:n_show]
    n = max(len(fn_items), len(tp_items))

    fig, axes = plt.subplots(n, 6, figsize=(24, n * 2.8))
    if n == 1: axes = axes[np.newaxis]
    fig.suptitle('Grasp Failure — FN(red, missed) vs TP(blue, detected)\n'
                 'Left 3 cols: FN | Right 3 cols: TP', fontsize=12, y=1.01)

    cols_label = ['MRL', 'Norm', 'Recon Error(Ch-A)']

    for col, lbl in enumerate(cols_label):
        fig.text(0.08 + col * 0.155, 1.0, f'FN: {lbl}', ha='center',
                 fontsize=9, fontweight='bold', color='crimson')
        fig.text(0.565 + col * 0.155, 1.0, f'TP: {lbl}', ha='center',
                 fontsize=9, fontweight='bold', color='steelblue')

    def short(f): return f.replace('grasp_','').replace('_failure.npy','')[8:24]

    for row in range(n):
        for side, items, color, off in [('FN', fn_items, 'crimson', 0),
                                         ('TP', tp_items, 'steelblue', 3)]:
            if row >= len(items):
                for c in range(3): axes[row, off+c].axis('off')
                continue
            fname, seq = items[row]
            t = np.arange(len(seq))
            t_w, e_w = recon_errors(seq, model, scaler, cfg)
            d = mrl_delta(seq)

            # MRL
            ax = axes[row, off]
            ax.plot(t, seq[:,0], color=color, lw=1.2)
            ax.axhline(thr_b, color='gray', ls='--', lw=0.9)
            ax.set_ylim(-0.05, 1.1); ax.set_ylabel('MRL', fontsize=8)
            ax.set_title(short(fname), fontsize=7.5)
            ax.tick_params(labelsize=7)
            if row == 0: ax.set_xlabel('frame', fontsize=7)

            # Norm
            ax = axes[row, off+1]
            ax.plot(t, seq[:,1], color=color, lw=1.2)
            ax.set_ylabel('Norm', fontsize=8)
            ax.set_title(short(fname), fontsize=7.5)
            ax.tick_params(labelsize=7)

            # Recon Error
            ax = axes[row, off+2]
            ax.plot(t_w, e_w, color=color, lw=1.3, marker='o', ms=3)
            ax.axhline(thr_a, color='gray', ls='--', lw=0.9,
                       label=f'thr={thr_a:.3f}')
            ax.set_ylabel('Recon Err', fontsize=8)
            ax.set_title(short(fname), fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)

    plt.tight_layout()
    out = '/home/dyros/mcy_ws/fn_vs_tp_v2.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"[SAVE] {out}")
    plt.close()

# ── MRL 분포 비교 박스플롯 ────────────────────────────────────────────────────
def plot_distributions(fn_d, tp_d, model, scaler, cfg, thr_a, thr_b):
    fn_mrl_all, tp_mrl_all = [], []
    fn_err_all, tp_err_all = [], []
    fn_norm_all, tp_norm_all = [], []

    for seq in fn_d.values():
        fn_mrl_all.extend(seq[:,0].tolist())
        fn_norm_all.extend(seq[:,1].tolist())
        _, e = recon_errors(seq, model, scaler, cfg)
        fn_err_all.extend(e.tolist())

    for seq in tp_d.values():
        tp_mrl_all.extend(seq[:,0].tolist())
        tp_norm_all.extend(seq[:,1].tolist())
        _, e = recon_errors(seq, model, scaler, cfg)
        tp_err_all.extend(e.tolist())

    fn_max_err = [max(recon_errors(s, model, scaler, cfg)[1]) if len(s) >= SEQ_LEN else 0.0
                  for s in fn_d.values()]
    tp_max_err = [max(recon_errors(s, model, scaler, cfg)[1]) if len(s) >= SEQ_LEN else 0.0
                  for s in tp_d.values()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('FN vs TP 분포 비교 (grasp failure)', fontsize=12)

    # MRL 분포
    axes[0].boxplot([fn_mrl_all, tp_mrl_all], labels=['FN(missed)', 'TP(detected)'],
                    patch_artist=True,
                    boxprops=dict(facecolor='mistyrose'),
                    medianprops=dict(color='red'))
    axes[0].set_title('MRL (frame-level)', fontsize=10)
    axes[0].set_ylabel('MRL value'); axes[0].axhline(thr_b, color='gray', ls='--', label=f'thr_B={thr_b:.3f}')
    axes[0].legend(fontsize=8)

    # Norm 분포
    axes[1].boxplot([fn_norm_all, tp_norm_all], labels=['FN(missed)', 'TP(detected)'],
                    patch_artist=True,
                    boxprops=dict(facecolor='mistyrose'),
                    medianprops=dict(color='red'))
    axes[1].set_title('Norm (frame-level)', fontsize=10)
    axes[1].set_ylabel('Norm value')

    # Max Recon Error per sequence
    axes[2].boxplot([fn_max_err, tp_max_err], labels=['FN(missed)', 'TP(detected)'],
                    patch_artist=True,
                    boxprops=dict(facecolor='mistyrose'),
                    medianprops=dict(color='red'))
    axes[2].axhline(thr_a, color='blue', ls='--', label=f'thr_A={thr_a:.3f}')
    axes[2].set_title('Max Recon Error per sequence', fontsize=10)
    axes[2].set_ylabel('Recon Error'); axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = '/home/dyros/mcy_ws/fn_vs_tp_dist.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"[SAVE] {out}")
    plt.close()

def main():
    import warnings; warnings.filterwarnings('ignore')
    model, scaler, cfg, thr_b = load_model()
    thr_a = calibrate_thr_a(model, scaler, cfg)
    print(f"thr_A={thr_a:.4f}  thr_B={thr_b:.4f}\n")

    fn_d, tp_d = load_failures()

    # 수치 통계 출력
    print("=== FN (놓친 실패) 통계 ===")
    print(f"{'파일':<40} {'길이':>5} {'MRL평균':>8} {'MRL최소':>8} {'Norm평균':>9} {'MaxErr':>8} {'MaxDelta':>9}")
    for r in stats(fn_d, model, scaler, cfg, thr_a, thr_b):
        nm = r['file'].replace('grasp_','').replace('_failure.npy','')
        print(f"  {nm:<38} {r['len']:>5} {r['mrl_mean']:>8.3f} {r['mrl_min']:>8.3f} "
              f"{r['norm_mean']:>9.3f} {r['max_err']:>8.4f} {r['max_delta']:>9.4f}")

    print(f"\n=== TP (감지된 실패) 통계 ===")
    print(f"{'파일':<40} {'길이':>5} {'MRL평균':>8} {'MRL최소':>8} {'Norm평균':>9} {'MaxErr':>8} {'MaxDelta':>9}")
    for r in stats(tp_d, model, scaler, cfg, thr_a, thr_b):
        nm = r['file'].replace('grasp_','').replace('_failure.npy','')
        print(f"  {nm:<38} {r['len']:>5} {r['mrl_mean']:>8.3f} {r['mrl_min']:>8.3f} "
              f"{r['norm_mean']:>9.3f} {r['max_err']:>8.4f} {r['max_delta']:>9.4f}")

    fn_stats = stats(fn_d, model, scaler, cfg, thr_a, thr_b)
    tp_stats = stats(tp_d, model, scaler, cfg, thr_a, thr_b)
    print(f"\n=== 평균 비교 ===")
    for key, label in [('mrl_mean','MRL평균'),('mrl_min','MRL최소'),
                        ('norm_mean','Norm평균'),('max_err','MaxReconErr')]:
        fn_v = np.mean([r[key] for r in fn_stats])
        tp_v = np.mean([r[key] for r in tp_stats])
        print(f"  {label:<12}: FN={fn_v:.4f}  TP={tp_v:.4f}  diff={tp_v-fn_v:+.4f}")

    # 시각화
    plot_panels(fn_d, tp_d, model, scaler, cfg, thr_a, thr_b, n_show=8)
    plot_distributions(fn_d, tp_d, model, scaler, cfg, thr_a, thr_b)

if __name__ == '__main__':
    main()
