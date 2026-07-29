"""
Fx,Fy,Fz / Acontact 가 FN/TP 구별에 유의미한지 확인
"""
import os, sys, pickle, warnings
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import GRUAE, apply_relative_transform, compute_10d_from_raw

TRANSPORT_DIR = '/home/dyros/mcy_ws/transport_new'
SEQ_LEN    = 50
WIN_STRIDE = 10
CONSEC_A   = 2
EXCLUDE    = {'transport_20260527_210604_679439_failure_raw.npy'}

def load_model():
    pt   = os.path.join(TRANSPORT_DIR, 'transport_gru_ae_10d.pt')
    pkl  = os.path.join(TRANSPORT_DIR, 'transport_scaler_10d.pkl')
    thr  = os.path.join(TRANSPORT_DIR, 'transport_threshold_10d.npy')
    cfg  = os.path.join(TRANSPORT_DIR, 'transport_infer_cfg_10d.pkl')
    state  = torch.load(pt, map_location='cpu', weights_only=False)
    hidden = state['encoder.gru.weight_hh_l0'].shape[1]
    scaler = joblib.load(pkl)
    model  = GRUAE(input_dim=scaler.n_features_in_, hidden_dim=hidden,
                   num_layers=2, dropout=0.0, seq_len=SEQ_LEN)
    model.load_state_dict(state); model.eval()
    thr_a = float(np.load(thr)[0])
    with open(cfg, 'rb') as f:
        infer_cfg = pickle.load(f)
    return model, scaler, thr_a, infer_cfg

def is_tp(seq_10d, model, scaler, cfg, thr_a):
    consec = 0
    for start in range(0, len(seq_10d) - SEQ_LEN + 1, WIN_STRIDE):
        w = seq_10d[start:start+SEQ_LEN]
        x = scaler.transform(w).astype(np.float32)
        x = apply_relative_transform(x, cfg.get('relative','none'), cfg.get('n_rel_dims'))
        with torch.no_grad():
            s = float(model.reconstruction_error(
                torch.tensor(x).unsqueeze(0), cfg.get('weight_alpha',0.0)).item())
        if s >= thr_a:
            consec += 1
            if consec >= CONSEC_A: return True
        else:
            consec = 0
    return False

model, scaler, thr_a, cfg = load_model()

fail_files = sorted([f for f in os.listdir(TRANSPORT_DIR)
                     if f.endswith('_failure_raw.npy') and f not in EXCLUDE])
succ_files = sorted([f for f in os.listdir(TRANSPORT_DIR)
                     if f.endswith('_success_raw.npy')])

# ── 통계 수집 ─────────────────────────────────────────────────────────────
def get_stats(raw):
    """10D에서 Fx,Fy,Fz,Acontact 추출 후 F/A 계산."""
    seq = compute_10d_from_raw(raw)
    # 10D = [mrl, norm, Fx, Fy, Fz, Mx, My, Mz, Acontact, Fmax]
    Fx = seq[:, 2];  Fy = seq[:, 3];  Fz = seq[:, 4]
    Ac = seq[:, 8]
    Ac_safe = np.where(Ac > 0.5, Ac, np.nan)   # Acontact < 0.5이면 제외
    Fx_norm = Fx / Ac_safe
    Fy_norm = Fy / Ac_safe
    Fz_norm = Fz / Ac_safe
    F_norm  = np.sqrt(Fx**2 + Fy**2 + Fz**2) / Ac_safe

    return {
        'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'Ac': Ac,
        'Fx/A': Fx_norm, 'Fy/A': Fy_norm, 'Fz/A': Fz_norm, 'F/A': F_norm,
    }

fn_stats = []
tp_stats = []
suc_stats = []

for fname in fail_files:
    raw = np.load(os.path.join(TRANSPORT_DIR, fname)).astype(np.float32)
    seq = compute_10d_from_raw(raw)
    tp = is_tp(seq, model, scaler, cfg, thr_a)
    st = get_stats(raw)
    (tp_stats if tp else fn_stats).append(st)

for fname in succ_files:
    raw = np.load(os.path.join(TRANSPORT_DIR, fname)).astype(np.float32)
    suc_stats.append(get_stats(raw))

def nanmean_all(stats, key):
    return np.nanmean(np.concatenate([s[key] for s in stats]))
def nanstd_all(stats, key):
    return np.nanstd(np.concatenate([s[key] for s in stats]))

print(f"TP={len(tp_stats)}  FN={len(fn_stats)}  Succ={len(suc_stats)}\n")
print(f"{'특성':<10} {'성공':>10} {'TP':>10} {'FN':>10}   {'TP-FN분리도':>12}")
print("-" * 60)
for key in ['Fx', 'Fy', 'Fz', 'Ac', 'F/A', 'Fz/A']:
    sv = nanmean_all(suc_stats, key)
    tv = nanmean_all(tp_stats,  key)
    fv = nanmean_all(fn_stats,  key)
    ts = nanstd_all(tp_stats,   key)
    fs = nanstd_all(fn_stats,   key)
    sep = abs(tv - fv) / (0.5*(ts+fs)+1e-9)   # Cohen's d 근사
    print(f"  {key:<8}  {sv:>10.3f}  {tv:>10.3f}  {fv:>10.3f}   {sep:>12.2f}")

# ── 시각화 ────────────────────────────────────────────────────────────────
keys   = ['Fz', 'Fz/A', 'F/A', 'Ac']
labels = ['Fz (합력z)', 'Fz / Acontact', '|F| / Acontact', 'Acontact']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('FN vs TP vs Success: F/Acontact 비교\n(각 파일의 전체 시계열 평균값)', fontsize=12)

colors = {'Success': '#1f77b4', 'TP': '#ff7f0e', 'FN': '#2ca02c'}

for ax, key, label in zip(axes.flat, keys, labels):
    data  = {
        'Success': [nanmean_all([s], key) for s in suc_stats],
        'TP'     : [nanmean_all([s], key) for s in tp_stats],
        'FN'     : [nanmean_all([s], key) for s in fn_stats],
    }
    positions = [1, 2, 3]
    bp = ax.boxplot([data['Success'], data['TP'], data['FN']],
                    positions=positions, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    for patch, col in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(col); patch.set_alpha(0.6)

    # scatter
    for pos, (grp, vals) in zip(positions, data.items()):
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(pos + jitter, vals, color=colors[grp], s=25, alpha=0.8, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Success', 'TP', 'FN'], fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.grid(axis='y', lw=0.4, alpha=0.5)

plt.tight_layout()
out = '/home/dyros/mcy_ws/f_over_acontact.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"\n[SAVE] {out}")
plt.close()
