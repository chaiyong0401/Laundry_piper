"""
5-Method Anomaly Detection Comparison
======================================
Heuristic / K-means / FCM / IForest / GRU-AE (Ch-A / Ch-A+B)
동일 데이터 · 동일 consec 기준으로 TPR / FPR 비교.

사전 준비 (각 모델 학습 완료 후 실행):
  python train_baselines.py --model grasp        # kmeans, fcm
  python train_iforest.py   --model grasp [--use_raw]
  python train_gru_ae.py    --model grasp --use_raw --eval

Usage:
  python compare_methods.py --model grasp
  python compare_methods.py --model transport
  python compare_methods.py --model grasp --consec 1 2 3
"""

import argparse
import os
import sys
import glob
import numpy as np
import joblib

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)

# FCM 클래스 pickle 복원을 위해 반드시 import
from train_baselines import FuzzyCMeans  # noqa: F401


# ══════════════════════════════════════════════
#  공통: 데이터 로딩
# ══════════════════════════════════════════════
def load_seqs_6d(data_dir: str, prefix: str, label: str) -> list[np.ndarray]:
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}.npy")
    files = sorted([f for f in glob.glob(pattern) if not f.endswith("_raw.npy")])
    return [np.load(f).astype(np.float32) for f in files
            if np.load(f).ndim == 2 and np.load(f).shape[1] == 6 and np.load(f).shape[0] >= 5]


def load_seqs_raw(data_dir: str, prefix: str, label: str) -> list[np.ndarray]:
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}_raw.npy")
    files = sorted(glob.glob(pattern))
    return [np.load(f).astype(np.float32) for f in files
            if np.load(f).ndim == 2 and np.load(f).shape[0] >= 5]


# ══════════════════════════════════════════════
#  공통: consec 감지 헬퍼
# ══════════════════════════════════════════════
def _consec_detect(scores: np.ndarray, threshold: float, consec_needed: int) -> bool:
    """scores 중 threshold 초과가 consec_needed 연속으로 나타나면 True."""
    consec = 0
    for s in scores:
        if s > threshold:
            consec += 1
            if consec >= consec_needed:
                return True
        else:
            consec = 0
    return False


# ══════════════════════════════════════════════
#  공통: 시퀀스 단위 평가 (프레임 score 기반)
# ══════════════════════════════════════════════
def seq_level_eval(score_fn, seqs: list[np.ndarray],
                   threshold: float, consec_needed: int) -> tuple[int, int]:
    """score_fn(seq) → (T,) scores. score > threshold → 이상 프레임."""
    n_anom = sum(
        1 for seq in seqs if _consec_detect(score_fn(seq), threshold, consec_needed)
    )
    return n_anom, len(seqs)


# ══════════════════════════════════════════════
#  1. Heuristic
# ══════════════════════════════════════════════
def make_heuristic_scorer(z_threshold: float = 0.0, warmup: int = 50):
    """48D raw에서 16개 Z채널 평균 < z_threshold → 이상 (score=1, else 0).
    warmup 프레임은 0으로 채워 pre-contact 구간 false alarm 방지."""
    def score_fn(seq):  # seq: (T, 48)
        T = len(seq)
        scores = np.zeros(T, dtype=float)
        if T > warmup:
            mean_z = seq[warmup:, _Z_IDX].mean(axis=1)
            scores[warmup:] = (mean_z < z_threshold).astype(float)
        return scores
    return score_fn, 0.5


# ══════════════════════════════════════════════
#  공통: 윈도우 feature 추출 (train_baselines와 동일 로직)
# ══════════════════════════════════════════════
_Z_IDX = np.arange(2, 48, 3)  # 48D raw → 16 Z채널

def _build_win_feats(feature: str, seq_len: int, stride: int, warmup: int):
    """pkl에서 읽은 설정으로 윈도우 feature 추출 함수 반환."""
    def _z_channels(win):
        return win[:, _Z_IDX].astype(np.float64)

    def _mean_vec(win):
        return (win[:, 1:2] * win[:, 3:6]).astype(np.float64)

    def _mean_z(win):
        return win[:, _Z_IDX].mean(axis=1, keepdims=True).astype(np.float64)

    feat_fn = (_mean_z     if feature == "mean_z" else
               _z_channels if feature == "z_raw"  else _mean_vec)
    half = seq_len // 2

    def win_feats(seq):
        T = len(seq)
        feats = []
        for start in range(warmup, T - seq_len + 1, stride):
            w = seq[start:start + seq_len]
            f = feat_fn(w)
            mean_f  = f.mean(axis=0)
            delta_f = f[half:].mean(axis=0) - f[:half].mean(axis=0)
            feats.append(np.concatenate([mean_f, delta_f]))
        return np.array(feats, dtype=np.float64) if feats else np.array([])

    return win_feats


# ══════════════════════════════════════════════
#  2. K-means
# ══════════════════════════════════════════════
def make_kmeans_scorer(model_path: str):
    art = joblib.load(model_path)
    km, scaler, threshold = art["km"], art["scaler"], art["threshold"]
    feature = art.get("feature", "mrl")
    seq_len = art.get("seq_len", 50)
    stride  = art.get("stride_eval", 5)
    warmup  = art.get("warmup", 0)

    win_feats = _build_win_feats(feature, seq_len, stride, warmup)

    def score_fn(seq):
        feats = win_feats(seq)
        if len(feats) == 0:
            return np.array([])
        X = scaler.transform(feats)
        return np.min(
            np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2),
            axis=1)
    return score_fn, threshold


# ══════════════════════════════════════════════
#  3. FCM
# ══════════════════════════════════════════════
def make_fcm_scorer(model_path: str):
    art = joblib.load(model_path)
    fcm, scaler, threshold = art["fcm"], art["scaler"], art["threshold"]
    feature = art.get("feature", "mrl")
    seq_len = art.get("seq_len", 50)
    stride  = art.get("stride_eval", 5)
    warmup  = art.get("warmup", 0)

    win_feats = _build_win_feats(feature, seq_len, stride, warmup)

    def score_fn(seq):
        feats = win_feats(seq)
        if len(feats) == 0:
            return np.array([])
        return fcm.anomaly_score_dist(scaler.transform(feats))
    return score_fn, threshold


# ══════════════════════════════════════════════
#  4. IsolationForest
# ══════════════════════════════════════════════
def make_iforest_scorer(model_path: str, scaler_path: str, use_raw: bool,
                        z_only: bool = False, warmup: int = 50):
    art = joblib.load(model_path)
    if isinstance(art, dict):
        model   = art["model"]
        scaler  = art["scaler"]
        feature = art.get("feature", "mean_z" if z_only else ("raw" if use_raw else "6d"))
        warmup  = art.get("warmup", warmup)
    else:
        model   = art
        scaler  = joblib.load(scaler_path)
        feature = "mean_z" if z_only else ("raw" if use_raw else "6d")

    def score_fn(seq):
        T = len(seq)
        scores = np.zeros(T, dtype=float)
        if T <= warmup:
            return scores
        s = seq[warmup:]
        if feature == "mean_z":
            s = s[:, _Z_IDX].mean(axis=1, keepdims=True)
        elif feature != "raw":  # "6d" 등 이미 처리된 feature
            pass
        X = scaler.transform(s)
        scores[warmup:] = (model.predict(X) == -1).astype(float)
        return scores
    return score_fn, 0.5


# ══════════════════════════════════════════════
#  5. GRU-AE (채널 A / A+B)
# ══════════════════════════════════════════════
def make_gruae_scorers(data_dir: str, prefix: str, use_raw: bool, device):
    import torch
    import pickle
    from train_gru_ae import GRUAE, apply_relative_transform, build_hybrid_window
    from dual_channel_detector import FramewiseDeltaDetector

    suffix = "_raw" if use_raw else ""
    model_path  = os.path.join(data_dir, f"{prefix}_gru_ae{suffix}.pt")
    scaler_path = os.path.join(data_dir, f"{prefix}_scaler{suffix}.pkl")
    thresh_path = os.path.join(data_dir, f"{prefix}_threshold{suffix}.npy")
    thresh_b_path = os.path.join(data_dir, f"{prefix}_threshold_b{suffix}.npy")
    cfg_path    = os.path.join(data_dir, f"{prefix}_infer_cfg{suffix}.pkl")

    for p in [model_path, scaler_path, thresh_path, cfg_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"GRU-AE 모델 파일 없음: {p}")

    import pickle
    scaler = joblib.load(scaler_path)
    threshold_a = float(np.load(thresh_path).flat[0])
    threshold_b = float(np.load(thresh_b_path).flat[0]) if os.path.exists(thresh_b_path) else None
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)

    seq_len      = cfg.get("seq_len", 50)
    relative     = cfg.get("relative", "none")
    weight_alpha = cfg.get("weight_alpha", 0.0)
    n_rel_dims   = cfg.get("n_rel_dims", None)
    hybrid       = cfg.get("hybrid", False)

    # checkpoint에서 아키텍처 자동 추론 (infer_cfg에 저장 안 된 경우 대비)
    state = torch.load(model_path, map_location=device)
    input_dim  = state["encoder.gru.weight_ih_l0"].shape[1]
    hidden_dim = state["encoder.gru.weight_ih_l0"].shape[0] // 3  # GRU: 3 gates
    num_layers = sum(1 for k in state if k.startswith("encoder.gru.weight_ih_l"))
    dropout    = cfg.get("dropout", 0.2)

    model = GRUAE(input_dim, hidden_dim, num_layers, dropout, seq_len)
    model.load_state_dict(state)
    model.to(device).eval()

    # ── 채널 A: 윈도우 단위 score 시퀀스 반환 (consec = 연속 이상 윈도우 수)
    def win_scores_a(seq, infer_stride=5):
        """슬라이딩 윈도우별 reconstruction error 반환 → (n_windows,)"""
        T = len(seq)
        scores = []
        if T < seq_len:
            return np.array([], dtype=np.float32)
        with torch.no_grad():
            for start in range(0, T - seq_len + 1, infer_stride):
                window = seq[start:start + seq_len].copy()
                if hybrid:
                    window = build_hybrid_window(window)
                x = scaler.transform(window).astype(np.float32)
                x = apply_relative_transform(x, relative, n_rel_dims)
                x_t = torch.tensor(x).unsqueeze(0).to(device)
                scores.append(model.reconstruction_error(x_t, weight_alpha).item())
        return np.array(scores, dtype=np.float32)

    # ── 채널 B: 프레임 단위 delta score 반환
    def frame_scores_b(seq, smooth_window=3):
        from dual_channel_detector import FramewiseDeltaDetector
        det = FramewiseDeltaDetector(threshold_b, smooth_window)
        return np.array([det.update(f)[0] for f in seq], dtype=np.float32)

    return (win_scores_a, threshold_a,
            frame_scores_b, threshold_b)


# ══════════════════════════════════════════════
#  통합 평가 및 출력
# ══════════════════════════════════════════════
def evaluate_method(name: str, score_fn, threshold: float,
                    seqs_6d: dict, seqs_raw: dict,
                    use_raw_input: bool, consec_list: list[int]) -> dict:
    """
    반환: {consec: {label: (n_anom, n_total)}}
    """
    seqs = seqs_raw if use_raw_input else seqs_6d
    results = {}
    for consec in consec_list:
        results[consec] = {}
        for label, seq_list in seqs.items():
            if not seq_list:
                results[consec][label] = (0, 0)
                continue
            n_a, n_t = seq_level_eval(score_fn, seq_list, threshold, consec)
            results[consec][label] = (n_a, n_t)
    return results


def print_table(all_results: dict, labels: list[str], consec_list: list[int]):
    """
    all_results: {method_name: {consec: {label: (n_anom, n_total)}}}
    """
    success_labels = [l for l in labels if "success" in l]
    failure_labels = [l for l in labels if "success" not in l]

    col_w = 10
    method_w = 20

    # 헤더
    header = f"\n{'Method':<{method_w}} {'consec':>7}"
    for l in success_labels:
        header += f"  {'FPR('+l+')':>{col_w}}"
    for l in failure_labels:
        header += f"  {'TPR('+l+')':>{col_w}}"
    print(header)
    print("─" * (method_w + 8 + (col_w + 2) * (len(success_labels) + len(failure_labels))))

    for method, res in all_results.items():
        first = True
        for consec in consec_list:
            row = f"{'':>{method_w}} {consec:>7}" if not first else f"{method:<{method_w}} {consec:>7}"
            first = False
            for l in success_labels + failure_labels:
                n_a, n_t = res[consec].get(l, (0, 0))
                pct = f"{n_a/n_t*100:.1f}%" if n_t > 0 else " N/A"
                row += f"  {pct:>{col_w}}"
            print(row)
        print("─" * (method_w + 8 + (col_w + 2) * (len(success_labels) + len(failure_labels))))


# ══════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="5-method anomaly detection comparison")
    parser.add_argument("--model",    choices=["grasp", "transport"], default="grasp")
    parser.add_argument("--use_raw",  action="store_true",
                        help="IForest/GRU-AE에 48D raw feature 사용")
    parser.add_argument("--consec",   type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--consec_b", type=int, default=3,
                        help="GRU-AE Ch-B 고정 consec (default: 3)")
    parser.add_argument("--z_threshold", type=float, default=0.0,
                        help="Heuristic Z축 임계값 (default: 0.0, mean_Z < 임계값 → 이상)")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    default_dirs = {"grasp": "grasp_05", "transport": "transport_05"}
    data_dir = args.data_dir or os.path.join(WS, default_dirs[args.model])
    prefix   = args.model
    suffix   = "_raw" if args.use_raw else ""
    consec_list = args.consec

    consec_b = args.consec_b
    print(f"[CONFIG] model={prefix}, use_raw={args.use_raw}, consec={consec_list}, consec_b={consec_b}")
    print(f"[CONFIG] data_dir={data_dir}\n")

    # ── 데이터 로드
    labels = ["success", "failure"]
    seqs_6d  = {l: load_seqs_6d(data_dir, prefix, l)  for l in labels}
    seqs_raw = {l: load_seqs_raw(data_dir, prefix, l) for l in labels}
    for l in labels:
        print(f"[DATA] {l}: 6D={len(seqs_6d[l])}, raw={len(seqs_raw[l])}")

    all_results = {}

    # ── 1. Heuristic (z_raw 기반: mean_Z < z_threshold)
    print("\n[Heuristic] 평가 중...")
    score_fn, thr = make_heuristic_scorer(args.z_threshold)
    all_results["Heuristic"] = evaluate_method(
        "Heuristic", score_fn, thr, seqs_6d, seqs_raw, True, consec_list)

    # ── 2. K-means
    km_path = os.path.join(WS, f"{prefix}_kmeans.pkl")
    if os.path.exists(km_path):
        print("[K-means] 평가 중...")
        score_fn, thr = make_kmeans_scorer(km_path)
        km_art = joblib.load(km_path)
        km_use_raw = km_art.get("feature", "mrl") in ("z_raw", "mean_z")
        all_results["K-means"] = evaluate_method(
            "K-means", score_fn, thr, seqs_6d, seqs_raw, km_use_raw, consec_list)
    else:
        print(f"[K-means] 모델 없음 ({km_path}) → 스킵. train_baselines.py 먼저 실행하세요.")

    # ── 3. FCM
    fcm_path = os.path.join(WS, f"{prefix}_fcm.pkl")
    if os.path.exists(fcm_path):
        print("[FCM] 평가 중...")
        score_fn, thr = make_fcm_scorer(fcm_path)
        fcm_art = joblib.load(fcm_path)
        fcm_use_raw = fcm_art.get("feature", "mrl") in ("z_raw", "mean_z")
        all_results["FCM"] = evaluate_method(
            "FCM", score_fn, thr, seqs_6d, seqs_raw, fcm_use_raw, consec_list)
    else:
        print(f"[FCM] 모델 없음 ({fcm_path}) → 스킵. train_baselines.py 먼저 실행하세요.")

    # ── 4. IsolationForest (z_only 우선, 없으면 suffix 버전)
    iforest_path_z  = os.path.join(WS, f"{prefix}_iforest_z.pkl")
    iforest_scaler_z= os.path.join(WS, f"{prefix}_iforest_scaler_z.pkl")
    iforest_path    = os.path.join(WS, f"{prefix}_iforest{suffix}.pkl")
    iforest_scaler  = os.path.join(WS, f"{prefix}_iforest_scaler{suffix}.pkl")

    if os.path.exists(iforest_path_z) and os.path.exists(iforest_scaler_z):
        print("[IForest-Z] 평가 중... (z_only 모델 사용)")
        score_fn, thr = make_iforest_scorer(iforest_path_z, iforest_scaler_z,
                                            use_raw=True, z_only=True)
        all_results["IForest-Z"] = evaluate_method(
            "IForest-Z", score_fn, thr, seqs_6d, seqs_raw, True, consec_list)
    elif os.path.exists(iforest_path) and os.path.exists(iforest_scaler):
        print("[IForest] 평가 중...")
        score_fn, thr = make_iforest_scorer(iforest_path, iforest_scaler,
                                            use_raw=args.use_raw)
        all_results["IForest"] = evaluate_method(
            "IForest", score_fn, thr, seqs_6d, seqs_raw, args.use_raw, consec_list)
    else:
        print(f"[IForest] 모델 없음 → 스킵. train_iforest.py 먼저 실행하세요.")

    # ── 5. GRU-AE
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GRU-AE] 평가 중... (device={device})")
        score_fn_a, thr_a, score_fn_b, thr_b = make_gruae_scorers(
            data_dir, prefix, args.use_raw, device)

        seqs_for_a = seqs_raw if args.use_raw else seqs_6d  # Ch-A: raw or 6D
        seqs_for_b = seqs_raw if args.use_raw else seqs_6d  # Ch-B: train_gru_ae eval과 동일

        # 채널 A (윈도우 단위 consec)
        res_a = {}
        for consec in consec_list:
            res_a[consec] = {}
            for label, seq_list in seqs_for_a.items():
                n_anom = sum(
                    1 for seq in seq_list
                    if _consec_detect(score_fn_a(seq), thr_a, consec)
                )
                res_a[consec][label] = (n_anom, len(seq_list))
        all_results["GRU-AE (Ch-A)"] = res_a

        # 채널 A+B: Ch-A consec 가변 / Ch-B consec 고정(consec_b)
        if thr_b is not None:
            res_ab = {}
            for consec in consec_list:
                res_ab[consec] = {}
                for label in labels:
                    seqs_a = seqs_for_a.get(label, [])
                    seqs_b = seqs_for_b.get(label, [])
                    n_total = len(seqs_a)
                    n_anom = 0
                    for seq_a, seq_b in zip(seqs_a, seqs_b):
                        flag_a = _consec_detect(score_fn_a(seq_a), thr_a, consec)
                        flag_b = _consec_detect(score_fn_b(seq_b), thr_b, consec_b)
                        if flag_a or flag_b:
                            n_anom += 1
                    res_ab[consec][label] = (n_anom, n_total)
            all_results[f"GRU-AE (A|B,cB={consec_b})"] = res_ab

    except FileNotFoundError as e:
        print(f"[GRU-AE] {e} → 스킵. train_gru_ae.py 먼저 실행하세요.")
    except ImportError as e:
        print(f"[GRU-AE] import 실패: {e}")

    # ── 결과 출력
    if all_results:
        print_table(all_results, labels, consec_list)


if __name__ == "__main__":
    main()
