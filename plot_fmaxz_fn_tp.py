"""
Transport FN/TP 케이스별 Fmax_z 시계열 비교 (grasp_transport_combined 스타일)
- grasp 구간(보라) + transport 구간(FN=초록, TP=주황) 연결
- 각 행: 하나의 transport failure 케이스
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
GRASP_DIR     = '/home/dyros/mcy_ws/grasp_new'
SEQ_LEN       = 50
WIN_STRIDE    = 10
CONSEC_A      = 2
EXCLUDE       = {'transport_20260527_210604_679439_failure_raw.npy'}


# ── 모델 로드 ────────────────────────────────────────────────────────────────
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
    """consec_A 연속 윈도우 초과 시 TP."""
    consec = 0
    for start in range(0, len(seq_10d) - SEQ_LEN + 1, WIN_STRIDE):
        w = seq_10d[start:start+SEQ_LEN]
        x = scaler.transform(w).astype(np.float32)
        x = apply_relative_transform(x, cfg.get('relative', 'none'), cfg.get('n_rel_dims'))
        with torch.no_grad():
            s = float(model.reconstruction_error(
                torch.tensor(x).unsqueeze(0), cfg.get('weight_alpha', 0.0)).item())
        if s >= thr_a:
            consec += 1
            if consec >= CONSEC_A:
                return True
        else:
            consec = 0
    return False


def fmaxz_from_raw(raw: np.ndarray) -> np.ndarray:
    """48D raw → Fmax_z (z방향 최대 taxel 값) 시계열."""
    T = raw.shape[0]
    vecs = raw.reshape(T, 16, 3).astype(np.float64)
    return vecs[:, :, 2].max(axis=1).astype(np.float32)   # (T,)


def match_grasp(transport_fname: str):
    """transport 파일에 대응하는 grasp 파일 찾기 (timestamp 기준 15~60s 이전)."""
    # 타임스탬프 파싱: transport_YYYYMMDD_HHMMSS_XXXXXX_*
    parts = os.path.basename(transport_fname).split('_')
    date_str = parts[1]      # YYYYMMDD
    time_str = parts[2]      # HHMMSS
    t_sec = (int(time_str[:2]) * 3600
             + int(time_str[2:4]) * 60
             + int(time_str[4:6]))

    grasp_files = sorted([
        f for f in os.listdir(GRASP_DIR)
        if f.endswith('_raw.npy') and f.startswith(f'grasp_{date_str}_')
    ])

    best, best_diff = None, 1e9
    for gf in grasp_files:
        gparts = gf.split('_')
        gt_str = gparts[2]
        g_sec = (int(gt_str[:2]) * 3600
                 + int(gt_str[2:4]) * 60
                 + int(gt_str[4:6]))
        diff = t_sec - g_sec
        if 10 <= diff <= 90 and diff < best_diff:
            best, best_diff = gf, diff
    return best


# ── 메인 ─────────────────────────────────────────────────────────────────────
model, scaler, thr_a, cfg = load_model()
print(f"thr_A = {thr_a:.4f}")

fail_files = sorted([
    f for f in os.listdir(TRANSPORT_DIR)
    if f.endswith('_failure_raw.npy') and f not in EXCLUDE
])

cases = []
for fname in fail_files:
    raw = np.load(os.path.join(TRANSPORT_DIR, fname)).astype(np.float32)
    seq = compute_10d_from_raw(raw)
    tp = is_tp(seq, model, scaler, cfg, thr_a)
    fmz = fmaxz_from_raw(raw)

    grasp_fname = match_grasp(fname)
    if grasp_fname:
        g_raw = np.load(os.path.join(GRASP_DIR, grasp_fname)).astype(np.float32)
        g_fmz = fmaxz_from_raw(g_raw)
        g_label = 'success' if 'success' in grasp_fname else 'failure'
    else:
        g_raw = None; g_fmz = None; g_label = None

    cases.append({
        'fname'      : fname,
        'name'       : fname.replace('transport_', '').replace('_failure_raw.npy', ''),
        'tp'         : tp,
        'fmz'        : fmz,
        'g_fmz'      : g_fmz,
        'g_label'    : g_label,
    })

tp_cases = [c for c in cases if c['tp']]
fn_cases = [c for c in cases if not c['tp']]
print(f"TP={len(tp_cases)}  FN={len(fn_cases)}")

# ── 플롯 ─────────────────────────────────────────────────────────────────────
# 위쪽: FN 케이스들, 아래쪽: TP 케이스들
all_ordered = fn_cases + tp_cases
n = len(all_ordered)
n_fn = len(fn_cases)

fig, axes = plt.subplots(n, 1, figsize=(14, n * 1.8), sharex=False)
if n == 1:
    axes = [axes]

fig.suptitle('Transport Failure Cases: Fmax_z (z방향 최대 taxel)\n'
             '보라=grasp  |  초록=FN(미탐지)  |  주황=TP(탐지됨)',
             fontsize=12, y=1.01)

for i, c in enumerate(all_ordered):
    ax = axes[i]

    is_fn  = not c['tp']
    t_color = '#2ca02c' if is_fn else '#ff7f0e'   # FN=초록, TP=주황
    bg_color = '#e8f5e9' if is_fn else '#fff3e0'
    ax.set_facecolor(bg_color)

    offset = 0

    # grasp 구간
    if c['g_fmz'] is not None:
        g_t = np.arange(len(c['g_fmz']))
        ax.plot(g_t + offset, c['g_fmz'],
                color='#7b2d8b', lw=1.0, alpha=0.85, label='grasp')
        ax.axvspan(0, len(c['g_fmz']), color='#e8d5f0', alpha=0.3, zorder=0)
        offset += len(c['g_fmz'])
        ax.axvline(offset, color='gray', lw=0.8, ls='--')

    # transport 구간
    tr_t = np.arange(len(c['fmz']))
    ax.plot(tr_t + offset, c['fmz'],
            color=t_color, lw=1.2, label='transport')

    # y축 최대값 기준선
    all_vals = list(c['fmz'])
    if c['g_fmz'] is not None:
        all_vals += list(c['g_fmz'])
    ymax = max(max(all_vals) * 1.15, 0.5)
    ax.set_ylim(-0.1, ymax)

    tag = 'FN' if is_fn else 'TP'
    g_info = f"  (grasp={c['g_label']})" if c['g_label'] else ''
    ax.set_ylabel('Fmax_z', fontsize=7)
    ax.set_title(f"[{tag}] {c['name']}{g_info}", fontsize=8, loc='left', pad=2)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', lw=0.4, alpha=0.5)

    if i == n_fn - 1:
        ax.axhline(-0.08, color='dimgray', lw=1.5, ls='-', xmin=0, xmax=1, clip_on=False)

# 범례
axes[0].legend(fontsize=7, loc='upper right')

plt.tight_layout()
out = '/home/dyros/mcy_ws/fmaxz_fn_tp.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"[SAVE] {out}")
plt.close()
