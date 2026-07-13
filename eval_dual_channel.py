"""
eval_dual_channel.py  — Dual-channel tactile ablation evaluation
=================================================================
Channel A only  : GRU-AE window MSE
Channel B only  : Framewise MRL delta
Channel A OR B  : 어느 하나라도 이상 감지 시 이상

데이터: grasp_new (success/failure .npy)
평가: seed=42, 80/20 train-test stratified split
      threshold_a는 train 정상 시퀀스에서 재산정
      threshold_b는 기존 값 사용 (rule-based)

Metrics: Precision, Recall, F1, AUROC, Latency(ms), FP/trial
"""

import os, sys, time, pickle, random
import joblib
import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from collections import deque

sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import GRUAE, apply_relative_transform

# ── 설정 ─────────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/dyros/mcy_ws/grasp_new'
SEED       = 42
TEST_RATIO = 0.20
SEQ_LEN    = 50
WIN_STRIDE = 10   # 슬라이딩 윈도우 stride
SMOOTH_WIN = 3    # Channel B smoothing

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_sequences(data_dir):
    seqs = {'success': [], 'failure': []}
    files = sorted(os.listdir(data_dir))
    for f in files:
        if not f.endswith('.npy'): continue
        if 'raw' in f: continue
        if 'grasp_2' not in f: continue
        path = os.path.join(data_dir, f)
        arr = np.load(path).astype(np.float32)
        label = 'success' if 'success' in f else 'failure'
        seqs[label].append(arr)
    return seqs

def train_test_split(seqs, test_ratio, seed):
    rng = random.Random(seed)
    train, test = {'success': [], 'failure': []}, {'success': [], 'failure': []}
    for label in ['success', 'failure']:
        sl = list(seqs[label]); rng.shuffle(sl)
        n_test = max(1, round(len(sl) * test_ratio))
        test[label]  = sl[:n_test]
        train[label] = sl[n_test:]
    return train, test

# ── Channel A: GRU-AE ─────────────────────────────────────────────────────────
def load_gru_ae(data_dir):
    pt  = os.path.join(data_dir, 'grasp_gru_ae.pt')
    pkl = os.path.join(data_dir, 'grasp_scaler.pkl')
    state = torch.load(pt, map_location='cpu', weights_only=False)
    hidden = state['encoder.gru.weight_hh_l0'].shape[1]
    model = GRUAE(input_dim=6, hidden_dim=hidden, num_layers=2, dropout=0.0, seq_len=SEQ_LEN)
    model.load_state_dict(state)
    model.eval()
    scaler = joblib.load(pkl)
    return model, scaler

def score_channel_a(seq, model, scaler, cfg):
    """시퀀스 하나의 채널 A 최대 MSE score 반환."""
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

# ── Channel B: MRL delta ──────────────────────────────────────────────────────
CONSEC_B_NEEDED = 3   # 기존 5 → 3으로 변경

def score_channel_b(seq, warmup=30, consec_needed=CONSEC_B_NEEDED):
    """
    채널 B 시뮬레이션 — 배포 로직과 동일.
    score: 최대 MRL delta (AUROC용)
    fired: 연속 consec_needed 프레임 초과 여부 (binary용)
    """
    hist = deque(maxlen=SMOOTH_WIN)
    prev = None; consec = 0; max_delta = 0.0
    for i, frame in enumerate(seq):
        mrl = float(frame[0])
        if prev is None:
            prev = mrl; hist.append(mrl); continue
        ref = float(np.mean(hist)) if len(hist)==SMOOTH_WIN else prev
        delta = abs(mrl - ref)
        hist.append(mrl); prev = mrl
        max_delta = max(max_delta, delta)
    return max_delta

def fired_channel_b(seq, threshold, warmup=30, consec_needed=CONSEC_B_NEEDED):
    """배포 로직: warmup 후 연속 consec_needed 프레임 threshold 초과 시 True."""
    hist = deque(maxlen=SMOOTH_WIN)
    prev = None; consec = 0
    for i, frame in enumerate(seq):
        mrl = float(frame[0])
        if prev is None:
            prev = mrl; hist.append(mrl); continue
        ref = float(np.mean(hist)) if len(hist)==SMOOTH_WIN else prev
        delta = abs(mrl - ref)
        hist.append(mrl); prev = mrl
        if i < warmup:
            consec = 0; continue
        if delta > threshold:
            consec += 1
            if consec >= consec_needed: return True
        else:
            consec = 0
    return False

# ── threshold 재산정 (train 정상 시퀀스 기반) ──────────────────────────────────
def calibrate_threshold_a(train_succ, model, scaler, cfg, k_sigma=3.0):
    scores = [score_channel_a(s, model, scaler, cfg) for s in train_succ]
    mu, sigma = np.mean(scores), np.std(scores)
    return mu + k_sigma * sigma, scores

def calibrate_threshold_b(train_succ, k_sigma=3.0):
    scores = [score_channel_b(s) for s in train_succ]
    mu, sigma = np.mean(scores), np.std(scores)
    return mu + k_sigma * sigma, scores

# ── 평가 ─────────────────────────────────────────────────────────────────────
def evaluate_config(test, score_fn, threshold, label_name, binary_fn=None):
    """
    score_fn(seq) → float  연속 score (AUROC 용)
    threshold     → score >= threshold 이면 anomaly
    binary_fn     → 별도 binary 판정 함수 (Channel B 배포 로직 등)
                    None 이면 score >= threshold 사용
    """
    y_true, y_score, y_pred = [], [], []
    for seq in test['success']:
        s = score_fn(seq)
        p = int(binary_fn(seq)) if binary_fn else int(s >= threshold)
        y_true.append(0); y_score.append(s); y_pred.append(p)
    for seq in test['failure']:
        s = score_fn(seq)
        p = int(binary_fn(seq)) if binary_fn else int(s >= threshold)
        y_true.append(1); y_score.append(s); y_pred.append(p)

    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    y_pred  = np.array(y_pred)

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = float('nan')

    n_succ = len(test['success'])
    fp_per_trial = y_pred[:n_succ].sum() / max(1, n_succ)

    return dict(precision=prec, recall=rec, f1=f1, auroc=auroc,
                fp_per_trial=fp_per_trial,
                n_pos=len(test['failure']), n_neg=n_succ,
                threshold=threshold)

def measure_latency(score_fn, seqs, n=50):
    """score_fn 실행 시간 (ms) — warmup 후 평균."""
    all_seqs = seqs['success'][:10] + seqs['failure'][:5]
    # warmup
    for s in all_seqs[:3]: score_fn(s)
    times = []
    for _ in range(n):
        seq = all_seqs[_ % len(all_seqs)]
        t0 = time.perf_counter()
        score_fn(seq)
        times.append((time.perf_counter()-t0)*1000)
    return np.mean(times), np.std(times)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Dual-Channel Tactile Ablation Evaluation")
    print("="*60)

    # 데이터
    seqs = load_sequences(DATA_DIR)
    print(f"\n[DATA] success={len(seqs['success'])}, failure={len(seqs['failure'])}")

    # GRU-AE는 정상(success)만으로 학습하므로 failure는 모두 unseen anomaly.
    # → success만 train/test 분리, failure 전체를 test anomaly로 사용.
    rng = random.Random(SEED)
    succ = list(seqs['success']); rng.shuffle(succ)
    n_test_succ = max(1, round(len(succ) * TEST_RATIO))
    train_succ = succ[n_test_succ:]
    test_succ  = succ[:n_test_succ]
    test_fail  = seqs['failure']   # 전체 failure = unseen anomaly

    # evaluate_config 에서 쓸 형태로 묶기
    train = {'success': train_succ, 'failure': []}
    test  = {'success': test_succ,  'failure': test_fail}

    print(f"[SPLIT] train: success={len(train_succ)} (threshold 산정용)")
    print(f"[SPLIT] test:  success={len(test_succ)}, failure={len(test_fail)} (전체 anomaly)")

    # 모델
    model, scaler = load_gru_ae(DATA_DIR)
    with open(os.path.join(DATA_DIR, 'grasp_infer_cfg.pkl'), 'rb') as f:
        cfg = pickle.load(f)
    print(f"\n[CFG] {cfg}")

    # threshold 재산정
    print("\n[CALIBRATE] Train 정상 시퀀스로 threshold 재산정...")
    thr_a, train_scores_a = calibrate_threshold_a(train_succ, model, scaler, cfg)
    # Channel B는 rule-based: 원래 threshold 사용 (도메인 지식 기반)
    thr_b_orig = float(np.load(os.path.join(DATA_DIR,'grasp_threshold_b.npy'))[0])
    print(f"  threshold_a (3σ from train): {thr_a:.4f}  (orig: {np.load(os.path.join(DATA_DIR,'grasp_threshold.npy'))[0]:.4f})")
    print(f"  threshold_b (rule-based):    {thr_b_orig:.4f}  (no recalibration — domain-knowledge-based)")

    # score 함수 정의
    def score_a(seq): return score_channel_a(seq, model, scaler, cfg)
    def score_b(seq): return score_channel_b(seq)
    def score_ab(seq):
        sa = score_a(seq) / thr_a
        sb = score_b(seq) / thr_b_orig
        return max(sa, sb)

    # Channel B binary: 배포 로직 시뮬레이션
    def binary_b(seq): return float(fired_channel_b(seq, thr_b_orig))

    thr_ab = 1.0
    thr_b  = thr_b_orig

    # 평가
    print("\n[EVAL] 각 채널 평가 중...")
    # Channel B binary: 배포 로직 (연속 5프레임 체크)
    def binary_b_deploy(seq): return fired_channel_b(seq, thr_b_orig)
    # A OR B binary: 어느 하나가 fires
    def binary_ab(seq):
        fa = (score_a(seq) >= thr_a)
        fb = binary_b_deploy(seq)
        return fa or fb

    configs = [
        ("Channel A only (GRU-AE)",   score_a,  thr_a,    None),
        ("Channel B only (MRL delta)", score_b,  thr_b_orig, binary_b_deploy),
        ("Channel A OR Channel B",     score_ab, thr_ab,   binary_ab),
    ]

    results = {}
    for name, fn, thr, bfn in configs:
        r = evaluate_config(test, fn, thr, name, binary_fn=bfn)
        lat_mean, lat_std = measure_latency(fn, test)
        r['latency_ms'] = lat_mean
        r['latency_std'] = lat_std
        results[name] = r
        print(f"  {name}: F1={r['f1']:.3f}  AUROC={r['auroc']:.3f}  lat={lat_mean:.1f}ms")

    # 결과 출력
    print("\n" + "="*80)
    print(f"{'Configuration':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUROC':>7} {'Lat(ms)':>9} {'FP/trial':>9}")
    print("-"*80)
    for name, r in results.items():
        print(f"{name:<30} {r['precision']*100:>5.1f}% {r['recall']*100:>5.1f}% "
              f"{r['f1']*100:>5.1f}% {r['auroc']:>7.3f} "
              f"{r['latency_ms']:>7.1f}ms {r['fp_per_trial']:>9.3f}")
    print("="*80)
    print(f"\n[INFO] test split: {len(test['success'])} normal, {len(test['failure'])} anomalous sequences")

    # 저장
    import json
    out = os.path.join(DATA_DIR, 'ablation_dual_channel_results.json')
    save = {}
    for name, r in results.items():
        save[name] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in r.items()}
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"[SAVE] {out}")

if __name__ == '__main__':
    main()
