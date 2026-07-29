"""
10D 모델 FN 분석: consec_A=2, consec_B=5 기준 미감지 실패 시퀀스 분석
"""
import os, sys, pickle
import joblib
import numpy as np
import torch
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


def fired_a(seq_10d, model, scaler, cfg, thr_a):
    """consec_A 연속 윈도우 초과 시 True. max score도 반환."""
    consec = 0; max_s = 0.0
    for start in range(0, len(seq_10d) - SEQ_LEN + 1, WIN_STRIDE):
        w = seq_10d[start:start+SEQ_LEN]
        x = scaler.transform(w).astype(np.float32)
        x = apply_relative_transform(x, cfg.get('relative','none'), cfg.get('n_rel_dims'))
        with torch.no_grad():
            s = float(model.reconstruction_error(
                torch.tensor(x).unsqueeze(0), cfg.get('weight_alpha', 0.0)).item())
        max_s = max(max_s, s)
        if s >= thr_a:
            consec += 1
            if consec >= CONSEC_A:
                return True, max_s
        else:
            consec = 0
    return False, max_s


def fired_b(seq_10d, thr_b):
    """consec_B 연속 프레임 초과 시 True."""
    det = FramewiseDeltaDetector(thr_b, SMOOTH_WIN)
    consec = 0
    for i, frame in enumerate(seq_10d):
        _, trig = det.update(np.array(frame, dtype=np.float32))
        if i < WARMUP_B:
            consec = 0; continue
        if trig:
            consec += 1
            if consec >= CONSEC_B:
                return True
        else:
            consec = 0
    return False


def main():
    import warnings; warnings.filterwarnings('ignore')
    model, scaler, thr_a, thr_b, cfg = load_model()
    print(f"thr_A={thr_a:.4f}  thr_B={thr_b:.4f}  consec_A={CONSEC_A}  consec_B={CONSEC_B}\n")

    fail_files = sorted([f for f in os.listdir(TRANSPORT_DIR)
                         if f.endswith('_raw.npy') and 'failure' in f
                         and f not in EXCLUDE])

    results = []
    for fname in fail_files:
        raw = np.load(os.path.join(TRANSPORT_DIR, fname)).astype(np.float32)
        seq = compute_10d_from_raw(raw)

        fa, max_score = fired_a(seq, model, scaler, cfg, thr_a)
        fb = fired_b(seq, thr_b)

        results.append({
            'file'      : fname,
            'len'       : len(raw),
            'fa'        : fa,
            'fb'        : fb,
            'or'        : fa or fb,
            'max_score' : max_score,
            'mrl_mean'  : float(seq[:, 0].mean()),
            'mrl_min'   : float(seq[:, 0].min()),
            'norm_mean' : float(seq[:, 1].mean()),
            'fmax_mean' : float(seq[:, 9].mean()),
            'fmax_max'  : float(seq[:, 9].max()),
        })

    tp = [r for r in results if r['or']]
    fn = [r for r in results if not r['or']]

    hdr = (f"{'파일명':<42} {'길이':>5} {'A':>3} {'B':>3} "
           f"{'MaxScore':>9} {'MRL평균':>8} {'MRL최소':>8} {'Norm평균':>9} {'Fmax최대':>9}")
    def row(r):
        nm = r['file'].replace('transport_','').replace('_failure_raw.npy','')
        return (f"  {nm:<40} {r['len']:>5} {'O' if r['fa'] else '-':>3} {'O' if r['fb'] else '-':>3} "
                f"{r['max_score']:>9.4f} {r['mrl_mean']:>8.3f} {r['mrl_min']:>8.3f} "
                f"{r['norm_mean']:>9.3f} {r['fmax_max']:>9.3f}")

    print(f"=== 전체 결과: TP={len(tp)}/{len(results)}  FN={len(fn)}/{len(results)} ===\n")

    print(f"── TP 감지된 실패 ({len(tp)}개) ──")
    print(hdr)
    for r in tp: print(row(r))

    print(f"\n── FN 놓친 실패 ({len(fn)}개) ──")
    print(hdr)
    for r in fn: print(row(r))

    print(f"\n── 평균 비교 (TP vs FN) ──")
    for key, label in [('mrl_mean','MRL평균'), ('mrl_min','MRL최소'),
                        ('norm_mean','Norm평균'), ('fmax_max','Fmax최대'),
                        ('max_score','MaxScore')]:
        tv = np.mean([r[key] for r in tp]) if tp else 0.0
        fv = np.mean([r[key] for r in fn]) if fn else 0.0
        print(f"  {label:<10}: TP={tv:.4f}  FN={fv:.4f}  diff={tv-fv:+.4f}")

    print(f"\n── A만 감지 / B만 감지 / 둘 다 감지 ──")
    a_only  = [r for r in tp if r['fa'] and not r['fb']]
    b_only  = [r for r in tp if r['fb'] and not r['fa']]
    ab_both = [r for r in tp if r['fa'] and r['fb']]
    print(f"  A only : {len(a_only)}개")
    print(f"  B only : {len(b_only)}개")
    print(f"  A & B  : {len(ab_both)}개")


if __name__ == '__main__':
    main()
