"""
FN 케이스별 시계열 분석 및 패턴 분류
"""
import os, sys, pickle
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import GRUAE, apply_relative_transform, compute_10d_from_raw
from dual_channel_detector import FramewiseDeltaDetector

TRANSPORT_DIR = '/home/dyros/mcy_ws/transport_new'
SEQ_LEN    = 50
WIN_STRIDE = 10
SMOOTH_WIN = 3
CONSEC_A   = 2
CONSEC_B   = 5
WARMUP_B   = 30
EXCLUDE    = {'transport_20260527_210604_679439_failure_raw.npy'}

FN_FILES = [
    'transport_20260522_145733_815583_failure_raw.npy',
    'transport_20260522_152559_730544_failure_raw.npy',
    'transport_20260527_193101_293530_failure_raw.npy',
    'transport_20260527_194538_034669_failure_raw.npy',
    'transport_20260527_194749_330327_failure_raw.npy',
    'transport_20260527_200957_081843_failure_raw.npy',
    'transport_20260527_202136_344088_failure_raw.npy',
    'transport_20260527_202238_311722_failure_raw.npy',
    'transport_20260527_211214_914393_failure_raw.npy',
    'transport_20260720_190633_532068_failure_raw.npy',
    'transport_20260720_193024_441326_failure_raw.npy',
]


def load_model():
    pt   = os.path.join(TRANSPORT_DIR, 'transport_gru_ae_10d.pt')
    pkl  = os.path.join(TRANSPORT_DIR, 'transport_scaler_10d.pkl')
    thr  = os.path.join(TRANSPORT_DIR, 'transport_threshold_10d.npy')
    thrb = os.path.join(TRANSPORT_DIR, 'transport_threshold_b_10d.npy')
    cfg  = os.path.join(TRANSPORT_DIR, 'transport_infer_cfg_10d.pkl')
    state  = torch.load(pt, map_location='cpu', weights_only=False)
    hidden = state['encoder.gru.weight_hh_l0'].shape[1]
    scaler = joblib.load(pkl)
    model  = GRUAE(input_dim=scaler.n_features_in_, hidden_dim=hidden,
                   num_layers=2, dropout=0.0, seq_len=SEQ_LEN)
    model.load_state_dict(state); model.eval()
    thr_a = float(np.load(thr)[0])
    thr_b = float(np.load(thrb)[0])
    with open(cfg, 'rb') as f:
        infer_cfg = pickle.load(f)
    return model, scaler, thr_a, thr_b, infer_cfg


def score_series(seq_10d, model, scaler, cfg):
    """슬라이딩 윈도우별 score 시계열 반환 (center frame index, score)."""
    times, scores = [], []
    for start in range(0, len(seq_10d) - SEQ_LEN + 1, WIN_STRIDE):
        w = seq_10d[start:start+SEQ_LEN]
        x = scaler.transform(w).astype(np.float32)
        x = apply_relative_transform(x, cfg.get('relative','none'), cfg.get('n_rel_dims'))
        with torch.no_grad():
            s = float(model.reconstruction_error(
                torch.tensor(x).unsqueeze(0), cfg.get('weight_alpha', 0.0)).item())
        times.append(start + SEQ_LEN // 2)
        scores.append(s)
    return np.array(times), np.array(scores)


def delta_b_series(seq_10d, thr_b):
    """프레임별 MRL delta 시계열."""
    det = FramewiseDeltaDetector(thr_b, SMOOTH_WIN)
    deltas, flags = [], []
    for frame in seq_10d:
        d, f = det.update(np.array(frame, dtype=np.float32))
        deltas.append(d)
        flags.append(f)
    return np.array(deltas), np.array(flags)


def classify_case(info):
    """분석 지표 기반 케이스 분류."""
    max_score   = info['max_score']
    fmax_max    = info['fmax_max']
    seq_len     = info['len']
    mrl_drop_t  = info['mrl_drop_frame']   # MRL이 처음 0에 가까워지는 프레임
    score_peak  = info['score_peak_count'] # thr_A 초과 윈도우 수

    # 타입 D: thr_A 초과 윈도우가 존재하지만 consec 미달
    if score_peak >= 1 and max_score >= 0.1059:
        return 'D', '단발 임계초과 (1윈도우만 초과, consec 미달)'

    # 타입 A: 신호 자체가 극미약 (Fmax ≤ 1.1)
    if fmax_max <= 1.1:
        return 'A', '극미약 신호 (Fmax ≤ 1.1, 접촉 거의 없음)'

    # 타입 B: 짧은 시퀀스 + 약한 신호
    if seq_len <= 170:
        return 'B', '짧은 시퀀스 + 약신호 (탐지 윈도우 부족)'

    # 타입 E: 접촉 세기는 중간이지만 점수가 낮은 케이스
    if fmax_max >= 1.8 and max_score < 0.06:
        return 'E', '중간 접촉, 저이상점수 (패턴 변화 미미)'

    # 타입 C: 긴 시퀀스인데도 신호가 일관되게 약함
    return 'C', '약신호 지속 (긴 시퀀스에도 오차 낮음)'


def analyze_all():
    import warnings; warnings.filterwarnings('ignore')
    model, scaler, thr_a, thr_b, cfg = load_model()
    print(f"thr_A={thr_a:.4f}  thr_B={thr_b:.4f}\n")

    case_info = []
    for fname in FN_FILES:
        raw = np.load(os.path.join(TRANSPORT_DIR, fname)).astype(np.float32)
        seq = compute_10d_from_raw(raw)

        t_sc, sc = score_series(seq, model, scaler, cfg)
        deltas, flags = delta_b_series(seq, thr_b)

        mrl_series = seq[:, 0]
        fmax_series = seq[:, 9]
        norm_series = seq[:, 1]

        # MRL이 처음으로 0.05 이하로 떨어지는 프레임
        drop_frames = np.where(mrl_series < 0.05)[0]
        mrl_drop_t = int(drop_frames[0]) if len(drop_frames) > 0 else len(seq)

        score_above = int((sc >= thr_a).sum())

        info = {
            'file'           : fname,
            'name'           : fname.replace('transport_','').replace('_failure_raw.npy',''),
            'len'            : len(raw),
            'max_score'      : float(sc.max()) if len(sc) else 0.0,
            'score_peak_count': score_above,
            'fmax_max'       : float(fmax_series.max()),
            'norm_mean'      : float(norm_series.mean()),
            'mrl_mean'       : float(mrl_series.mean()),
            'mrl_drop_frame' : mrl_drop_t,
            'mrl_drop_pct'   : mrl_drop_t / len(seq) * 100,
            'sc_times'       : t_sc,
            'sc_vals'        : sc,
            'mrl_series'     : mrl_series,
            'fmax_series'    : fmax_series,
            'deltas'         : deltas,
        }
        case_type, case_desc = classify_case(info)
        info['case_type'] = case_type
        info['case_desc'] = case_desc
        case_info.append(info)

    # ── 텍스트 출력 ──────────────────────────────────────────────────────────
    print("=" * 85)
    print(f"{'케이스':<5} {'파일명':<30} {'길이':>5} {'Fmax':>6} {'MaxScore':>9} "
          f"{'MRL drop%':>9} {'Score>thr':>9}")
    print("=" * 85)

    by_type = {}
    for r in sorted(case_info, key=lambda x: x['case_type']):
        t = r['case_type']
        by_type.setdefault(t, []).append(r)
        print(f"  [{t}]  {r['name']:<30} {r['len']:>5} {r['fmax_max']:>6.2f} "
              f"{r['max_score']:>9.4f} {r['mrl_drop_pct']:>8.1f}% {r['score_peak_count']:>9}개")

    print("\n" + "=" * 85)
    print("케이스 분류 요약")
    print("=" * 85)
    for t in sorted(by_type.keys()):
        items = by_type[t]
        print(f"\n[타입 {t}] {items[0]['case_desc']}  ({len(items)}개)")
        for r in items:
            print(f"  • {r['name']}")
            print(f"    길이={r['len']}  Fmax={r['fmax_max']:.2f}  "
                  f"MaxScore={r['max_score']:.4f}  "
                  f"MRL drop 시점={r['mrl_drop_frame']}프레임({r['mrl_drop_pct']:.0f}%)")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    n = len(case_info)
    fig, axes = plt.subplots(n, 3, figsize=(18, n * 2.5))
    fig.suptitle('FN 케이스 시계열 분석 (10D 모델)', fontsize=13)

    type_colors = {'A': 'crimson', 'B': 'orange', 'C': 'purple', 'D': 'steelblue', 'E': 'darkgreen'}

    for i, r in enumerate(sorted(case_info, key=lambda x: x['case_type'])):
        color = type_colors.get(r['case_type'], 'gray')
        t = np.arange(r['len'])
        label = f"[{r['case_type']}] {r['name'][-20:]}"

        # MRL
        ax = axes[i, 0]
        ax.plot(t, r['mrl_series'], color=color, lw=1.2)
        ax.axvline(r['mrl_drop_frame'], color='gray', ls=':', lw=1.0)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel('MRL', fontsize=8)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)

        # Fmax
        ax = axes[i, 1]
        ax.plot(t, r['fmax_series'], color=color, lw=1.2)
        ax.set_ylabel('Fmax', fontsize=8)
        ax.tick_params(labelsize=7)

        # Recon Score
        ax = axes[i, 2]
        if len(r['sc_vals']) > 0:
            ax.plot(r['sc_times'], r['sc_vals'], color=color, lw=1.3, marker='o', ms=3)
        ax.axhline(thr_a, color='black', ls='--', lw=0.9, label=f'thr={thr_a:.3f}')
        ax.set_ylabel('Score', fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    axes[0, 0].set_title(f'[{case_info[0]["case_type"]}] MRL', fontsize=8)
    axes[0, 1].set_title('Fmax', fontsize=8)
    axes[0, 2].set_title('Recon Score (Ch-A)', fontsize=8)

    plt.tight_layout()
    out = '/home/dyros/mcy_ws/fn_cases_10d.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\n[SAVE] {out}")
    plt.close()


if __name__ == '__main__':
    analyze_all()
