"""
eval_dual_channel_combined.py — Grasp + Transport 통합 Dual-channel ablation
=============================================================================
Channel A only  : GRU-AE window MSE
Channel B only  : Framewise MRL delta
Channel A OR B  : 어느 하나라도 이상 감지 시 이상

Grasp : grasp_new/  6D  grasp_gru_ae.pt
Transport: transport_new/ 48D transport_gru_ae_raw.pt

각 phase별 모델/threshold 사용 후 결과 pooling
"""

import os, sys, time, pickle, random
import joblib
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from collections import deque

sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import GRUAE, apply_relative_transform

SEED       = 42
TEST_RATIO = 0.20
SEQ_LEN    = 50
WIN_STRIDE = 10
SMOOTH_WIN = 3
CONSEC_B   = 5   # basket_ori.py 기준 (grasp/transport 공통)

GRASP_DIR     = '/home/dyros/mcy_ws/grasp_new'
TRANSPORT_DIR = '/home/dyros/mcy_ws/transport_new'

# 촉각 접촉 없는 실패 (MRL=0 전구간 또는 warmup 내 drop, 탐지 범위 밖) — 평가에서 제외
EXCLUDE_FILES = {
    'grasp_20260527_204450_061379_failure.npy',
    'grasp_20260527_210134_409778_failure.npy',
    'transport_20260527_210604_679439_failure_raw.npy',
}

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_sequences(data_dir, prefix):
    seqs = {'success': [], 'failure': []}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.npy'): continue
        if 'raw' in f: continue
        if prefix not in f: continue
        if f in EXCLUDE_FILES: continue
        arr = np.load(os.path.join(data_dir, f)).astype(np.float32)
        label = 'success' if 'success' in f else 'failure'
        seqs[label].append(arr)
    return seqs

def load_sequences_raw(data_dir, prefix):
    """48D raw 시퀀스 로드."""
    seqs = {'success': [], 'failure': []}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('_raw.npy'): continue
        if prefix not in f: continue
        if f in EXCLUDE_FILES: continue
        if 'success' in f:
            label = 'success'
        elif 'failure' in f:
            label = 'failure'
        else:
            continue
        arr = np.load(os.path.join(data_dir, f)).astype(np.float32)
        seqs[label].append(arr)
    return seqs

def load_sequences_11d(data_dir, prefix):
    """_raw.npy에서 11D feature 계산하여 로드."""
    from train_gru_ae import compute_11d_from_raw
    seqs = {'success': [], 'failure': []}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('_raw.npy'): continue
        if prefix not in f: continue
        if f in EXCLUDE_FILES: continue
        if 'success' in f:
            label = 'success'
        elif 'failure' in f:
            label = 'failure'
        else:
            continue
        arr_raw = np.load(os.path.join(data_dir, f)).astype(np.float32)
        seqs[label].append(compute_11d_from_raw(arr_raw))
    return seqs

def load_sequences_10d(data_dir, prefix):
    """_raw.npy에서 10D feature 계산하여 로드."""
    from train_gru_ae import compute_10d_from_raw
    seqs = {'success': [], 'failure': []}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('_raw.npy'): continue
        if prefix not in f: continue
        if f in EXCLUDE_FILES: continue
        if 'success' in f:
            label = 'success'
        elif 'failure' in f:
            label = 'failure'
        else:
            continue
        arr_raw = np.load(os.path.join(data_dir, f)).astype(np.float32)
        seqs[label].append(compute_10d_from_raw(arr_raw))
    return seqs

def load_sequences_8d(data_dir, prefix):
    """_raw.npy에서 C8D feature 계산하여 로드."""
    from train_gru_ae import compute_8d_from_raw
    seqs = {'success': [], 'failure': []}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('_raw.npy'): continue
        if prefix not in f: continue
        if f in EXCLUDE_FILES: continue
        if 'success' in f:
            label = 'success'
        elif 'failure' in f:
            label = 'failure'
        else:
            continue
        arr_raw = np.load(os.path.join(data_dir, f)).astype(np.float32)
        seqs[label].append(compute_8d_from_raw(arr_raw))
    return seqs

def load_sequences_9d(data_dir, prefix):
    """_raw.npy에서 9D feature 계산하여 로드."""
    from train_gru_ae import compute_9d_from_raw
    seqs = {'success': [], 'failure': []}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('_raw.npy'): continue
        if prefix not in f: continue
        if f in EXCLUDE_FILES: continue
        if 'success' in f:
            label = 'success'
        elif 'failure' in f:
            label = 'failure'
        else:
            continue
        arr_raw = np.load(os.path.join(data_dir, f)).astype(np.float32)
        seqs[label].append(compute_9d_from_raw(arr_raw))
    return seqs

def split_success(seqs, test_ratio, seed):
    rng = random.Random(seed)
    succ = list(seqs['success']); rng.shuffle(succ)
    n_test = max(1, round(len(succ) * test_ratio))
    return succ[n_test:], succ[:n_test]  # train_succ, test_succ

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(data_dir, prefix, use_raw=False, use_9d=False, use_8d=False, use_10d=False, use_11d=False):
    if use_11d:
        suffix = '_11d'
    elif use_10d:
        suffix = '_10d'
    elif use_8d:
        suffix = '_8d'
    elif use_9d:
        suffix = '_9d'
    elif use_raw:
        suffix = '_raw'
    else:
        suffix = ''
    pt   = os.path.join(data_dir, f'{prefix}_gru_ae{suffix}.pt')
    pkl  = os.path.join(data_dir, f'{prefix}_scaler{suffix}.pkl')
    thr  = os.path.join(data_dir, f'{prefix}_threshold{suffix}.npy')
    thrb = os.path.join(data_dir, f'{prefix}_threshold_b{suffix}.npy')
    cfg  = os.path.join(data_dir, f'{prefix}_infer_cfg{suffix}.pkl')

    state  = torch.load(pt, map_location='cpu', weights_only=False)
    hidden = state['encoder.gru.weight_hh_l0'].shape[1]
    input_dim = joblib.load(pkl).n_features_in_
    model = GRUAE(input_dim=input_dim, hidden_dim=hidden, num_layers=2,
                  dropout=0.0, seq_len=SEQ_LEN)
    model.load_state_dict(state)
    model.eval()
    scaler    = joblib.load(pkl)
    threshold = float(np.load(thr)[0])
    threshold_b = float(np.load(thrb)[0])
    with open(cfg, 'rb') as f:
        infer_cfg = pickle.load(f)
    return model, scaler, threshold, threshold_b, infer_cfg

CONSEC_A = 1   # sweep 최적: k=2.0, CONSEC_A=1, CONSEC_B=5

# ── Channel A score ───────────────────────────────────────────────────────────
def score_channel_a(seq, model, scaler, cfg):
    """AUROC용 max score 반환."""
    scores = []
    for start in range(0, len(seq) - SEQ_LEN + 1, WIN_STRIDE):
        window = seq[start:start+SEQ_LEN]
        x = scaler.transform(window).astype(np.float32)
        x = apply_relative_transform(x, cfg.get('relative','none'), cfg.get('n_rel_dims'))
        x_t = torch.tensor(x).unsqueeze(0)
        with torch.no_grad():
            s = float(model.reconstruction_error(x_t, cfg.get('weight_alpha',0.0)).item())
        scores.append(s)
    return max(scores) if scores else 0.0

def fired_channel_a(seq, model, scaler, cfg, threshold):
    """basket_ori 기준: 연속 CONSEC_A 윈도우가 threshold 초과 시 True."""
    consec = 0
    for start in range(0, len(seq) - SEQ_LEN + 1, WIN_STRIDE):
        window = seq[start:start+SEQ_LEN]
        x = scaler.transform(window).astype(np.float32)
        x = apply_relative_transform(x, cfg.get('relative','none'), cfg.get('n_rel_dims'))
        x_t = torch.tensor(x).unsqueeze(0)
        with torch.no_grad():
            s = float(model.reconstruction_error(x_t, cfg.get('weight_alpha',0.0)).item())
        if s >= threshold:
            consec += 1
            if consec >= CONSEC_A:
                return True
        else:
            consec = 0
    return False

# ── Channel B score / fired ───────────────────────────────────────────────────
def extract_mrl(frame: np.ndarray) -> float:
    if frame.ndim == 0 or frame.shape[0] == 0:
        return 0.0
    if frame.shape[0] in (6, 9, 10, 11):  # 6D, 9D, 10D, or 11D: index 0 = MRL
        return float(frame[0])
    if frame.shape[0] == 8:       # C8D: index 0 = Fx, no MRL → compute from scratch
        vecs = frame[:6].reshape(2, 3)  # F and M not useful for MRL; fallback to raw
        # C8D doesn't store raw taxel data, so use Fmax (index 7) as proxy
        return float(min(frame[7] / 5.0, 1.0))  # rough normalize
    # 48D raw: 16 taxel × 3 axes
    vecs = frame[:48].reshape(16, 3)
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 0.1
    if valid.sum() >= 2:
        unit_vecs = vecs[valid] / norms[valid, np.newaxis]
        return float(np.linalg.norm(np.mean(unit_vecs, axis=0)))
    return 0.0

def score_channel_b(seq, warmup=30):
    hist = deque(maxlen=SMOOTH_WIN)
    prev = None; max_delta = 0.0
    for i, frame in enumerate(seq):
        mrl = extract_mrl(frame)
        if prev is None:
            prev = mrl; hist.append(mrl); continue
        ref = float(np.mean(hist)) if len(hist) == SMOOTH_WIN else prev
        delta = abs(mrl - ref)
        hist.append(mrl); prev = mrl
        max_delta = max(max_delta, delta)
    return max_delta

def fired_channel_b(seq, threshold, warmup=30):
    hist = deque(maxlen=SMOOTH_WIN)
    prev = None; consec = 0
    for i, frame in enumerate(seq):
        mrl = extract_mrl(frame)
        if prev is None:
            prev = mrl; hist.append(mrl); continue
        ref = float(np.mean(hist)) if len(hist) == SMOOTH_WIN else prev
        delta = abs(mrl - ref)
        hist.append(mrl); prev = mrl
        if i < warmup:
            consec = 0; continue
        if delta > threshold:
            consec += 1
            if consec >= CONSEC_B: return True
        else:
            consec = 0
    return False

# ── threshold 재산정 ──────────────────────────────────────────────────────────
def calibrate_a(train_succ, model, scaler, cfg, k=3.0):
    scores = [score_channel_a(s, model, scaler, cfg) for s in train_succ]
    return float(np.mean(scores) + k * np.std(scores))

# ── 결과 수집 ─────────────────────────────────────────────────────────────────
def collect_results(test_succ, test_fail,
                    score_a_fn, fired_a_fn, thr_a,
                    score_b_fn, thr_b,
                    binary_b_fn):
    """각 시퀀스의 (y_true, score_a, score_b, pred_a, pred_b) 반환."""
    rows = []
    for seq in test_succ:
        sa = score_a_fn(seq)
        sb = score_b_fn(seq)
        pa = int(fired_a_fn(seq))   # 2연속 윈도우 초과 여부
        pb = int(binary_b_fn(seq))
        rows.append((0, sa / thr_a, sb / thr_b, pa, pb))
    for seq in test_fail:
        sa = score_a_fn(seq)
        sb = score_b_fn(seq)
        pa = int(fired_a_fn(seq))
        pb = int(binary_b_fn(seq))
        rows.append((1, sa / thr_a, sb / thr_b, pa, pb))
    return rows

def compute_metrics(rows, mode):
    """
    mode: 'A' | 'B' | 'AB'
    rows: list of (y_true, norm_sa, norm_sb, pred_a, pred_b)
    """
    y_true, y_score, y_pred = [], [], []
    for (gt, nsa, nsb, pa, pb) in rows:
        y_true.append(gt)
        if mode == 'A':
            y_score.append(nsa); y_pred.append(pa)
        elif mode == 'B':
            y_score.append(nsb); y_pred.append(pb)
        else:  # AB
            y_score.append(max(nsa, nsb)); y_pred.append(int(pa or pb))

    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    y_pred  = np.array(y_pred)

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    try:    auroc = roc_auc_score(y_true, y_score)
    except: auroc = float('nan')

    n_neg = int((y_true == 0).sum())
    fp_per_trial = y_pred[y_true == 0].sum() / max(1, n_neg)
    return dict(precision=prec, recall=rec, f1=f1, auroc=auroc,
                fp_per_trial=fp_per_trial,
                n_pos=int((y_true==1).sum()), n_neg=n_neg)

def measure_latency_window(model, scaler, cfg, seqs, n=200):
    """Ch-A 단일 윈도우 inference 시간 (ms). 실패 감지 기본 단위."""
    sample_windows = []
    for seq in seqs:
        for start in range(0, len(seq) - SEQ_LEN + 1, WIN_STRIDE):
            w = seq[start:start+SEQ_LEN]
            x = scaler.transform(w).astype(np.float32)
            x = apply_relative_transform(x, cfg.get('relative','none'), cfg.get('n_rel_dims'))
            sample_windows.append(torch.tensor(x).unsqueeze(0))
            if len(sample_windows) >= n: break
        if len(sample_windows) >= n: break
    # warmup
    for w in sample_windows[:5]:
        with torch.no_grad(): model.reconstruction_error(w, cfg.get('weight_alpha', 0.0))
    times = []
    for w in sample_windows[:n]:
        t0 = time.perf_counter()
        with torch.no_grad(): model.reconstruction_error(w, cfg.get('weight_alpha', 0.0))
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))

def measure_latency_frame_b(seqs, n=500):
    """Ch-B 단일 프레임 MRL delta 계산 시간 (ms)."""
    frames = [seq[i] for seq in seqs for i in range(len(seq))][:n]
    hist = deque(maxlen=SMOOTH_WIN); prev = None
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        mrl = extract_mrl(frame)
        if prev is not None:
            ref = float(np.mean(hist)) if len(hist) == SMOOTH_WIN else prev
            _ = abs(mrl - ref)
        hist.append(mrl); prev = mrl
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Dual-Channel Ablation: Grasp + Transport Combined")
    print("=" * 65)

    # ── Grasp 로드 ────────────────────────────────────────────────
    g_seqs  = load_sequences(GRASP_DIR, 'grasp_2')
    g_train_succ, g_test_succ = split_success(g_seqs, TEST_RATIO, SEED)
    g_test_fail = g_seqs['failure']
    g_model, g_scaler, g_thr_a, g_thr_b, g_cfg = load_model(
        GRASP_DIR, 'grasp', use_raw=False)

    print(f"\n[GRASP]     train_succ={len(g_train_succ)}, "
          f"test_succ={len(g_test_succ)}, test_fail={len(g_test_fail)}")

    # ── Transport 로드 (11D > 10D > 8D > 9D > raw > 6D 우선순위로 auto-detect) ──
    _t_11d_exists = os.path.exists(os.path.join(TRANSPORT_DIR, 'transport_gru_ae_11d.pt'))
    _t_10d_exists = os.path.exists(os.path.join(TRANSPORT_DIR, 'transport_gru_ae_10d.pt'))
    _t_8d_exists  = os.path.exists(os.path.join(TRANSPORT_DIR, 'transport_gru_ae_8d.pt'))
    _t_9d_exists  = os.path.exists(os.path.join(TRANSPORT_DIR, 'transport_gru_ae_9d.pt'))
    _t_raw_exists = os.path.exists(os.path.join(TRANSPORT_DIR, 'transport_gru_ae_raw.pt'))
    if _t_11d_exists:
        t_seqs = load_sequences_11d(TRANSPORT_DIR, 'transport_')
        t_model, t_scaler, t_thr_a, t_thr_b, t_cfg = load_model(
            TRANSPORT_DIR, 'transport', use_11d=True)
        print(f"[TRANSPORT] 11D 모델 감지 → 11D feature 사용")
    elif _t_10d_exists:
        t_seqs = load_sequences_10d(TRANSPORT_DIR, 'transport_')
        t_model, t_scaler, t_thr_a, t_thr_b, t_cfg = load_model(
            TRANSPORT_DIR, 'transport', use_10d=True)
        print(f"[TRANSPORT] 10D 모델 감지 → 10D feature 사용")
    elif _t_8d_exists:
        t_seqs = load_sequences_8d(TRANSPORT_DIR, 'transport_')
        t_model, t_scaler, t_thr_a, t_thr_b, t_cfg = load_model(
            TRANSPORT_DIR, 'transport', use_8d=True)
        print(f"[TRANSPORT] C8D 모델 감지 → C8D feature 사용")
    elif _t_9d_exists:
        t_seqs = load_sequences_9d(TRANSPORT_DIR, 'transport_')
        t_model, t_scaler, t_thr_a, t_thr_b, t_cfg = load_model(
            TRANSPORT_DIR, 'transport', use_9d=True)
        print(f"[TRANSPORT] 9D 모델 감지 → 9D feature 사용")
    elif _t_raw_exists:
        t_seqs  = load_sequences_raw(TRANSPORT_DIR, 'transport_')
        t_model, t_scaler, t_thr_a, t_thr_b, t_cfg = load_model(
            TRANSPORT_DIR, 'transport', use_raw=True)
        print(f"[TRANSPORT] raw 모델 감지 → 48D raw 사용")
    else:
        t_seqs  = load_sequences(TRANSPORT_DIR, 'transport_')
        t_model, t_scaler, t_thr_a, t_thr_b, t_cfg = load_model(
            TRANSPORT_DIR, 'transport', use_raw=False)
        print(f"[TRANSPORT] 6D 모델 감지 → 6D feature 사용")
    t_train_succ, t_test_succ = split_success(t_seqs, TEST_RATIO, SEED)
    t_test_fail = t_seqs['failure']

    print(f"[TRANSPORT] train_succ={len(t_train_succ)}, "
          f"test_succ={len(t_test_succ)}, test_fail={len(t_test_fail)}")

    # ── threshold 재산정 ──────────────────────────────────────────
    # 저장된 threshold 사용 (basket_ori.py 배포 시스템과 동일한 값)
    g_thr_a_cal = g_thr_a
    t_thr_a_cal = t_thr_a
    print(f"\n[THRESHOLD] 저장된 값 사용 (배포 시스템과 동일)")
    print(f"  Grasp     thr_A={g_thr_a_cal:.4f}  thr_B={g_thr_b:.4f}")
    print(f"  Transport thr_A={t_thr_a_cal:.4f}  thr_B={t_thr_b:.4f}")

    # ── score 함수 정의 ───────────────────────────────────────────
    def g_sa(seq): return score_channel_a(seq, g_model, g_scaler, g_cfg)
    def g_fa(seq): return fired_channel_a(seq, g_model, g_scaler, g_cfg, g_thr_a_cal)
    def g_sb(seq): return score_channel_b(seq)
    def g_bb(seq): return fired_channel_b(seq, g_thr_b)

    def t_sa(seq): return score_channel_a(seq, t_model, t_scaler, t_cfg)
    def t_fa(seq): return fired_channel_a(seq, t_model, t_scaler, t_cfg, t_thr_a_cal)
    def t_sb(seq): return score_channel_b(seq)
    def t_bb(seq): return fired_channel_b(seq, t_thr_b)

    # ── 결과 수집 ─────────────────────────────────────────────────
    print("\n[EVAL] 채점 중...")
    g_rows = collect_results(g_test_succ, g_test_fail,
                             g_sa, g_fa, g_thr_a_cal, g_sb, g_thr_b, g_bb)
    t_rows = collect_results(t_test_succ, t_test_fail,
                             t_sa, t_fa, t_thr_a_cal, t_sb, t_thr_b, t_bb)
    all_rows = g_rows + t_rows

    # ── latency: warmup 이후 실패 발생 기준 감지 지연 ───────────
    # Ch-A: 다음 윈도우 평가 대기 (WIN_STRIDE/Hz) + GRU-AE inference
    # Ch-B: CONSEC_B 연속 프레임 대기 (CONSEC_B/Hz)
    # A OR B: 두 채널 병렬 실행 → 점진적 슬립은 Ch-A(104ms), 급격한 낙하는 Ch-B(50ms)
    SENSOR_HZ = 100
    all_seqs = g_test_succ + g_test_fail
    infer_ms = measure_latency_window(g_model, g_scaler, g_cfg, all_seqs)
    lat_a  = (WIN_STRIDE / SENSOR_HZ) * 1000 + infer_ms   # 100ms + 4.3ms
    lat_b  = (CONSEC_B   / SENSOR_HZ) * 1000              # 50ms
    lat_ab = lat_a                                          # OR 로직: Ch-A 경로가 지배(점진적 슬립 다수)

    # ── per-sample FN 분석 ───────────────────────────────────────
    print("\n[FN ANALYSIS] A OR B 기준 놓친 failure 샘플 (False Negative):")
    print(f"  {'Phase':<10} {'File idx':>8} {'A fired':>8} {'B fired':>8} {'OR':>6}")
    print("  " + "-" * 44)

    g_fail_files = sorted([f for f in os.listdir(GRASP_DIR)
                           if f.endswith('.npy') and 'raw' not in f
                           and 'grasp_2' in f and 'failure' in f
                           and f not in EXCLUDE_FILES])
    t_fail_files = sorted([f for f in os.listdir(TRANSPORT_DIR)
                           if f.endswith('_raw.npy') and 'transport_' in f
                           and 'failure' in f and f not in EXCLUDE_FILES])

    fn_grasp = fn_transport = 0
    for i, seq in enumerate(g_test_fail):
        pa = int(g_fa(seq)); pb = int(g_bb(seq)); por = int(pa or pb)
        tag = "" if por else " ← FN"
        fname = g_fail_files[i] if i < len(g_fail_files) else f"idx={i}"
        if not por:
            fn_grasp += 1
            print(f"  {'Grasp':<10} {fname:<30} A={pa} B={pb} OR={por}{tag}")

    for i, seq in enumerate(t_test_fail):
        pa = int(t_fa(seq)); pb = int(t_bb(seq)); por = int(pa or pb)
        fname = t_fail_files[i] if i < len(t_fail_files) else f"idx={i}"
        if not por:
            fn_transport += 1
            print(f"  {'Transport':<10} {fname:<30} A={pa} B={pb} OR={por} ← FN")

    print(f"\n  Grasp FN={fn_grasp}/{len(g_test_fail)}, "
          f"Transport FN={fn_transport}/{len(t_test_fail)}")
    print(f"  Total FN={(fn_grasp+fn_transport)}/{len(g_test_fail)+len(t_test_fail)}")

    # ── 출력 ─────────────────────────────────────────────────────
    configs = [('Channel A only', 'A', lat_a),
               ('Channel B only', 'B', lat_b),
               ('Channel A OR B', 'AB', lat_ab)]

    print("\n" + "=" * 80)
    print(f"{'Configuration':<22} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'AUROC':>7} {'FA/trial':>9} {'Lat(ms)':>9}")
    print("-" * 80)

    results = {}
    for name, mode, lat in configs:
        # per-phase
        gm = compute_metrics(g_rows, mode)
        tm = compute_metrics(t_rows, mode)
        cm = compute_metrics(all_rows, mode)
        cm['latency_ms'] = lat
        results[name] = cm

        print(f"{name:<22} {cm['precision']*100:>6.1f}% {cm['recall']*100:>6.1f}% "
              f"{cm['f1']*100:>6.1f}% {cm['auroc']:>7.3f} "
              f"{cm['fp_per_trial']:>9.3f} {lat:>7.1f}ms")
        print(f"  └ Grasp     : Prec={gm['precision']*100:.1f}% Rec={gm['recall']*100:.1f}% "
              f"F1={gm['f1']*100:.1f}%  n_pos={gm['n_pos']} n_neg={gm['n_neg']}")
        print(f"  └ Transport : Prec={tm['precision']*100:.1f}% Rec={tm['recall']*100:.1f}% "
              f"F1={tm['f1']*100:.1f}%  n_pos={tm['n_pos']} n_neg={tm['n_neg']}")

    print("=" * 80)
    total_pos = results['Channel A only']['n_pos']
    total_neg = results['Channel A only']['n_neg']
    print(f"\n[INFO] Combined test: {total_neg} normal, {total_pos} anomalous sequences")
    print(f"[INFO] Grasp  test: {len(g_test_succ)} normal, {len(g_test_fail)} anomalous")
    print(f"[INFO] Transport test: {len(t_test_succ)} normal, {len(t_test_fail)} anomalous")

    # latency 주석
    print(f"\n[LATENCY NOTE]")
    print(f"  Ch-A : WIN_STRIDE({WIN_STRIDE}frames)/100Hz × 1000 + GRU-AE({infer_ms:.1f}ms) = {lat_a:.1f}ms")
    print(f"  Ch-B : CONSEC_B({CONSEC_B}frames)/100Hz × 1000 = {lat_b:.1f}ms")
    print(f"  A OR B: Ch-A 경로 기준 {lat_ab:.1f}ms (Ch-B 급격 낙하 시 {lat_b:.1f}ms)")

    # LaTeX 출력
    lat_map = {'Channel A only': lat_a, 'Channel B only': lat_b, 'Channel A OR B': lat_ab}
    print("\n[LaTeX]")
    print("% Prec / Rec / F1 / FA-per-trial / Latency(ms)")
    for name, mode, _ in configs:
        r = results[name]
        bold = name == 'Channel A OR B'
        p  = f"\\textbf{{{r['precision']*100:.1f}}}" if bold else f"{r['precision']*100:.1f}"
        rc = f"\\textbf{{{r['recall']*100:.1f}}}"    if bold else f"{r['recall']*100:.1f}"
        f1 = f"\\textbf{{{r['f1']*100:.1f}}}"        if bold else f"{r['f1']*100:.1f}"
        fa = f"\\textbf{{{r['fp_per_trial']:.3f}}}"  if bold else f"{r['fp_per_trial']:.3f}"
        lt = f"\\textbf{{{lat_map[name]:.0f}}}" if bold else f"{lat_map[name]:.0f}"
        print(f"{name:<22} & {p} & {rc} & {f1} & {fa} & {lt} \\\\")

    import json
    out = os.path.join(GRASP_DIR, 'ablation_combined_results.json')
    save = {}
    for name, r in results.items():
        save[name] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in r.items()}
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n[SAVE] {out}")

if __name__ == '__main__':
    main()
