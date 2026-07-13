"""
Ablation Study: Dual-Channel Tactile Anomaly Detection
=======================================================
basket_ori.py의 실제 inference 파이프라인을 재현하여 아래 세 구성의 성능을 측정.

  Channel A only  : GRU-AE sliding-window MSE (stride=10, window=50)
  Channel B only  : FramewiseDeltaDetector (per-frame MRL delta)
  A OR B          : 둘 중 하나라도 이상 확정 시 anomaly

평가 기준
  Positive  = failure trial (이상 감지 대상)
  Negative  = success trial (정상 파지/이송)
  TP / FP / FN / TN → Precision, Recall, F1, AUROC, Latency(ms), FP/trial

두 Phase 합산:
  Grasp  (6D,  grasp_new/)
  Transport (48D raw, transport_new/)
"""

import os, sys, glob, time, argparse
import pickle, joblib
from collections import deque

import numpy as np
import torch
from scipy.signal import resample
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, "/home/dyros/mcy_ws")
from train_gru_ae import GRUAE, apply_relative_transform, build_hybrid_window
from dual_channel_detector import FramewiseDeltaDetector

# ──────────────────────────────────────────────────────────────
#  Config per phase
# ──────────────────────────────────────────────────────────────
PHASES = {
    "grasp": {
        "data_dir"  : "/home/dyros/mcy_ws/grasp_new",
        "prefix"    : "grasp",
        "use_raw"   : False,
        "suffix"    : "",
    },
    "transport": {
        "data_dir"  : "/home/dyros/mcy_ws/transport_new",
        "prefix"    : "transport",
        "use_raw"   : True,
        "suffix"    : "_raw",
    },
}

WARMUP          = 30   # 초기 모션 아티팩트 무시 프레임
MAX_NORM_GATE   = 0.3  # 최소 힘 게이트 (6D feature index 2 = max_norm)
SEQ_LEN         = 50
STRIDE          = 10
CONSEC_A_NEEDED = 2
CONSEC_B_NEEDED = 5
SMOOTH_WIN_B    = 3
LATENCY_REPEAT  = 200  # 지연시간 측정 반복 횟수


# ──────────────────────────────────────────────────────────────
#  데이터 로드
# ──────────────────────────────────────────────────────────────
def load_phase_data(cfg: dict):
    """
    Returns list of (seq: np.ndarray [T, D], label: int)
    label 1 = failure (positive), 0 = success (negative)
    """
    data_dir = cfg["data_dir"]
    prefix   = cfg["prefix"]
    use_raw  = cfg["use_raw"]
    suffix   = "_raw" if use_raw else ""

    pattern = os.path.join(data_dir, f"{prefix}_*.npy")
    all_files = sorted(glob.glob(pattern))
    # threshold / model 파일 등 제외 + 6D/48D 구분
    # use_raw=False → _raw.npy 제외, use_raw=True → _raw.npy만 포함
    data_files = []
    for f in all_files:
        bn = os.path.basename(f)
        if "success" not in bn and "failure" not in bn:
            continue
        is_raw = bn.endswith("_raw.npy")
        if use_raw and not is_raw:
            continue
        if not use_raw and is_raw:
            continue
        data_files.append(f)

    samples = []
    for fpath in data_files:
        seq = np.load(fpath).astype(np.float32)
        label = 1 if "failure" in os.path.basename(fpath) else 0
        samples.append((seq, label, fpath))
    return samples


# ──────────────────────────────────────────────────────────────
#  모델 로드
# ──────────────────────────────────────────────────────────────
def load_model(cfg: dict):
    data_dir = cfg["data_dir"]
    prefix   = cfg["prefix"]
    suffix   = cfg["suffix"]

    model_path  = os.path.join(data_dir, f"{prefix}_gru_ae{suffix}.pt")
    scaler_path = os.path.join(data_dir, f"{prefix}_scaler{suffix}.pkl")
    thresh_path = os.path.join(data_dir, f"{prefix}_threshold{suffix}.npy")
    thresh_b    = os.path.join(data_dir, f"{prefix}_threshold_b{suffix}.npy")
    cfg_path    = os.path.join(data_dir, f"{prefix}_infer_cfg{suffix}.pkl")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    scaler = joblib.load(scaler_path)
    input_dim  = scaler.n_features_in_
    hidden_dim = checkpoint["encoder.gru.weight_hh_l0"].shape[1]
    model = GRUAE(input_dim=input_dim, hidden_dim=hidden_dim,
                  num_layers=2, dropout=0.2, seq_len=SEQ_LEN)
    model.load_state_dict(checkpoint)
    model.eval()

    threshold_a = float(np.load(thresh_path)[0])
    threshold_b = float(np.load(thresh_b)[0])

    with open(cfg_path, "rb") as f:
        infer_cfg = pickle.load(f)

    return model, scaler, threshold_a, threshold_b, infer_cfg


# ──────────────────────────────────────────────────────────────
#  보조 함수
# ──────────────────────────────────────────────────────────────
def get_max_norm(frame: np.ndarray, use_raw: bool) -> float:
    """6D: index2=max_norm / 48D raw: max taxel norm 계산"""
    if not use_raw:
        return float(frame[2])
    vecs  = frame[:48].reshape(16, 3)
    norms = np.linalg.norm(vecs, axis=1)
    return float(norms.max())


def run_windowed_inference(buffer_frames, model, scaler, infer_cfg) -> float:
    """sliding window 추론 → reconstruction error (score)"""
    relative_mode = infer_cfg.get("relative", "none")
    weight_alpha  = infer_cfg.get("weight_alpha", 0.0)
    hybrid        = infer_cfg.get("hybrid", False)
    n_rel_dims    = infer_cfg.get("n_rel_dims", None)

    seq = np.array(buffer_frames, dtype=np.float32)   # [50, D]
    if hybrid:
        seq = build_hybrid_window(seq)
    seq_s = scaler.transform(seq).astype(np.float32)
    seq_s = np.clip(seq_s, -3.0, 3.0)
    seq_r = apply_relative_transform(seq_s, relative_mode, n_rel_dims=n_rel_dims)
    xt    = torch.tensor(seq_r).unsqueeze(0)           # [1, 50, D]
    with torch.no_grad():
        score = float(model.reconstruction_error(xt, weight_alpha=weight_alpha).item())
    return score


# ──────────────────────────────────────────────────────────────
#  채널별 trial 평가 (basket_ori.py 로직 재현)
# ──────────────────────────────────────────────────────────────
def evaluate_trial(seq, model, scaler, threshold_a, threshold_b,
                   infer_cfg, use_raw,
                   use_ch_a=True, use_ch_b=True):
    """
    Returns:
        detected   (bool)  — 이상 감지 여부
        score_a    (float) — 시퀀스 내 최대 채널A score (AUROC용)
        delta_b    (float) — 시퀀스 내 최대 채널B delta (AUROC용)
        detect_frame (int) — 최초 감지 프레임 번호 (latency 추정용, 미감지 시 -1)
    """
    buf   = deque(maxlen=SEQ_LEN)
    ch_b  = FramewiseDeltaDetector(delta_threshold=threshold_b,
                                   smooth_window=SMOOTH_WIN_B)

    consec_a = 0
    consec_b = 0
    frame_cnt = 0
    max_score_a = 0.0
    max_delta_b = 0.0
    detected    = False
    detect_frame = -1

    for frame in seq:
        buf.append(frame)
        frame_cnt += 1
        cur_max_norm = get_max_norm(frame, use_raw)

        # ── Channel A (sliding window, stride=10, consec>=2 필요)
        flag_a = False
        if use_ch_a and len(buf) >= SEQ_LEN and frame_cnt % STRIDE == 0:
            score_a = run_windowed_inference(list(buf), model, scaler, infer_cfg)
            max_score_a = max(max_score_a, score_a)
            if score_a > threshold_a:
                consec_a += 1
                if consec_a >= CONSEC_A_NEEDED:
                    flag_a = True
            else:
                consec_a = 0

        # ── Channel B (per-frame MRL delta, warmup + 최소힘 gate, consec>=5 필요)
        flag_b = False
        if (use_ch_b and
                frame_cnt > WARMUP and
                cur_max_norm > MAX_NORM_GATE):
            delta, is_anom = ch_b.update(frame)
            max_delta_b = max(max_delta_b, delta)
            if is_anom:
                consec_b += 1
                if consec_b >= CONSEC_B_NEEDED:
                    flag_b = True
            else:
                consec_b = 0

        if flag_a or flag_b:
            detected = True
            detect_frame = frame_cnt
            break

    # 미감지 시 시퀀스 전체 길이를 detection frame으로 표시
    if not detected:
        detect_frame = len(seq)

    return detected, max_score_a, max_delta_b, detect_frame


# ──────────────────────────────────────────────────────────────
#  지연시간 측정 (bench)
# ──────────────────────────────────────────────────────────────
def measure_latency(model, scaler, infer_cfg, input_dim,
                    threshold_b, repeat=LATENCY_REPEAT):
    """
    채널 A: (1, 50, D) 추론 시간
    채널 B: FramewiseDeltaDetector.update() 시간 (단일 프레임)
    """
    dummy_seq = np.random.randn(SEQ_LEN, input_dim).astype(np.float32)
    dummy_frames = list(dummy_seq)

    # warmup
    for _ in range(20):
        run_windowed_inference(dummy_frames, model, scaler, infer_cfg)

    times_a = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run_windowed_inference(dummy_frames, model, scaler, infer_cfg)
        times_a.append((time.perf_counter() - t0) * 1000)
    lat_a = float(np.mean(times_a))

    det_b = FramewiseDeltaDetector(delta_threshold=threshold_b,
                                   smooth_window=SMOOTH_WIN_B)
    dummy_frame = np.random.randn(input_dim).astype(np.float32)
    for _ in range(20):
        det_b.update(dummy_frame)
    times_b = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        det_b.update(dummy_frame)
        times_b.append((time.perf_counter() - t0) * 1000)
    lat_b = float(np.mean(times_b))

    return lat_a, lat_b


# ──────────────────────────────────────────────────────────────
#  메트릭 계산
# ──────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, scores):
    """
    y_true: list of int (1=failure, 0=success)
    y_pred: list of int (1=detected, 0=not detected)
    scores: list of float (continuous score for AUROC)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = np.array(scores)

    prec    = precision_score(y_true, y_pred, zero_division=0) * 100
    rec     = recall_score(y_true, y_pred, zero_division=0) * 100
    f1      = f1_score(y_true, y_pred, zero_division=0) * 100
    n_neg   = int((y_true == 0).sum())
    n_fp    = int(((y_pred == 1) & (y_true == 0)).sum())
    fp_per_trial = n_fp / n_neg if n_neg > 0 else 0.0

    try:
        auroc = roc_auc_score(y_true, scores)
    except Exception:
        auroc = float("nan")

    return prec, rec, f1, auroc, fp_per_trial


# ──────────────────────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["grasp", "transport", "all"],
                        default="all", help="평가할 phase (기본: 전체 합산)")
    parser.add_argument("--latex", action="store_true",
                        help="LaTeX 테이블 행 출력")
    args = parser.parse_args()

    phase_names = list(PHASES.keys()) if args.phase == "all" else [args.phase]

    # ── 전체 결과 누적용
    results_all = {cfg: {"y_true": [], "preds": [], "scores_a": [],
                          "scores_b": [], "scores_ab": []}
                   for cfg in ["A", "B", "AB"]}

    lat_a_all, lat_b_all = [], []

    for phase_name in phase_names:
        print(f"\n{'='*60}")
        print(f"  Phase: {phase_name}")
        print(f"{'='*60}")

        cfg     = PHASES[phase_name]
        samples = load_phase_data(cfg)
        model, scaler, thr_a, thr_b, infer_cfg = load_model(cfg)
        input_dim = scaler.n_features_in_
        use_raw   = cfg["use_raw"]

        n_pos = sum(1 for _, lbl, _ in samples if lbl == 1)
        n_neg = sum(1 for _, lbl, _ in samples if lbl == 0)
        print(f"  데이터: {len(samples)}개 (success={n_neg}, failure={n_pos})")
        print(f"  threshold_A={thr_a:.6f}, threshold_B={thr_b:.6f}")
        print(f"  input_dim={input_dim}, use_raw={use_raw}")

        # 지연시간 측정
        lat_a, lat_b = measure_latency(model, scaler, infer_cfg,
                                        input_dim, thr_b)
        lat_a_all.append(lat_a)
        lat_b_all.append(lat_b)
        print(f"  Latency  Ch-A={lat_a:.3f}ms  Ch-B={lat_b:.4f}ms/frame")

        # trial별 평가
        for seq, label, fpath in samples:
            fname = os.path.basename(fpath)

            # Ch-A only
            det_a, sa, _, _ = evaluate_trial(
                seq, model, scaler, thr_a, thr_b, infer_cfg, use_raw,
                use_ch_a=True, use_ch_b=False)

            # Ch-B only
            det_b, _, db, _ = evaluate_trial(
                seq, model, scaler, thr_a, thr_b, infer_cfg, use_raw,
                use_ch_a=False, use_ch_b=True)

            # A OR B
            det_ab, sa2, db2, _ = evaluate_trial(
                seq, model, scaler, thr_a, thr_b, infer_cfg, use_raw,
                use_ch_a=True, use_ch_b=True)

            results_all["A"]["y_true"].append(label)
            results_all["A"]["preds"].append(int(det_a))
            results_all["A"]["scores_a"].append(sa)

            results_all["B"]["y_true"].append(label)
            results_all["B"]["preds"].append(int(det_b))
            results_all["B"]["scores_b"].append(db)

            results_all["AB"]["y_true"].append(label)
            results_all["AB"]["preds"].append(int(det_ab))
            results_all["AB"]["scores_ab"].append(max(sa2, db2))

    # ── 통합 메트릭
    print(f"\n{'='*70}")
    print("  Ablation Study Results (combined phases)")
    print(f"{'='*70}")

    lat_a_mean = float(np.mean(lat_a_all))
    lat_b_mean = float(np.mean(lat_b_all))
    lat_ab_mean = lat_a_mean + lat_b_mean

    rows = []
    for ch_key, ch_name, lat in [
        ("A",  "Channel A only (GRU-AE)",    lat_a_mean),
        ("B",  "Channel B only (MRL delta)", lat_b_mean),
        ("AB", "Channel A OR Channel B",      lat_ab_mean),
    ]:
        r = results_all[ch_key]
        if ch_key == "A":
            scores = r["scores_a"]
        elif ch_key == "B":
            scores = r["scores_b"]
        else:
            scores = r["scores_ab"]

        prec, rec, f1, auroc, fp_pt = compute_metrics(
            r["y_true"], r["preds"], scores)

        rows.append((ch_name, prec, rec, f1, auroc, lat, fp_pt))

        n_pos = sum(r["y_true"])
        n_neg = len(r["y_true"]) - n_pos
        tp = sum(1 for t, p in zip(r["y_true"], r["preds"]) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(r["y_true"], r["preds"]) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(r["y_true"], r["preds"]) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(r["y_true"], r["preds"]) if t == 0 and p == 0)

        print(f"\n  [{ch_name}]")
        print(f"    TP={tp} FP={fp} FN={fn} TN={tn}  (pos={n_pos} neg={n_neg})")
        print(f"    Precision={prec:.1f}%  Recall={rec:.1f}%  F1={f1:.1f}%")
        print(f"    AUROC={auroc:.4f}  Latency={lat:.3f}ms  FP/trial={fp_pt:.3f}")

    # ── 테이블 출력
    print(f"\n{'='*100}")
    header = f"  {'Configuration':<38} {'Prec(%)':>8} {'Rec(%)':>8} {'F1(%)':>8} {'AUROC':>8} {'Lat(ms)':>10} {'FP/trial':>10}"
    print(header)
    print(f"  {'-'*98}")
    for ch_name, prec, rec, f1, auroc, lat, fp_pt in rows:
        print(f"  {ch_name:<38} {prec:>7.1f}% {rec:>7.1f}% {f1:>7.1f}% {auroc:>8.4f} {lat:>9.3f}ms {fp_pt:>10.3f}")
    print(f"{'='*100}")

    if args.latex:
        print("\n── LaTeX rows ──")
        for ch_name, prec, rec, f1, auroc, lat, fp_pt in rows:
            auroc_nan = "N/A" if np.isnan(auroc) else f"{auroc:.3f}"
            print(f"{ch_name} & {prec:.1f} & {rec:.1f} & {f1:.1f} & {auroc_nan} & {lat:.2f} & {fp_pt:.2f} \\\\")


if __name__ == "__main__":
    main()
