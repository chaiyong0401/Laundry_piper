"""
Heuristic / K-means / FCM Baseline 학습 및 평가 스크립트
=========================================================
--feature mrl   (기본값, 구 방식)
  6D MRL 기반 mean_vec_biased (3D) 윈도우 평균 특징.

--feature z_raw  (신규, Z축 부호 활용)
  48D raw taxel에서 16개 Z채널 추출 → 윈도우 평균 (16D).
  Success: Z > 0 (물체가 센서를 정상 압박)
  Failure: Z < 0 (센서끼리 맞닿아 반대 방향 변위)

  Heuristic: mean_Z < threshold  (frame-level, 학습 불필요)
  K-means/FCM: 16D Z 윈도우 특징으로 one-class 학습

K 선택:
  --n_clusters 0  → silhouette score로 자동 결정 (K=2..6 탐색)
  --n_clusters N  → N 고정 사용

Usage:
  python train_baselines.py --model grasp --feature mean_z
  python train_baselines.py --model grasp --feature mean_z --n_clusters 0
  python train_baselines.py --model grasp --feature mrl
  python train_baselines.py --model grasp --feature z_raw --method kmeans
"""

import argparse
import os
import glob
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EPS = 1e-9
Z_IDX = np.arange(2, 48, 3)   # 48D raw에서 Z채널 인덱스 (0-based: 2,5,8,...,47)


# ─────────────────────────────────────────────
#  데이터 로딩
# ─────────────────────────────────────────────
def load_sequences_6d(data_dir: str, prefix: str, label: str) -> list[np.ndarray]:
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}.npy")
    files = sorted([f for f in glob.glob(pattern) if not f.endswith("_raw.npy")])
    seqs = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[1] != 6 or arr.shape[0] < 5:
            print(f"[SKIP] {os.path.basename(f)}: shape={arr.shape}")
            continue
        seqs.append(arr.astype(np.float32))
        print(f"[LOAD] {os.path.basename(f)}: {arr.shape}")
    return seqs


def load_sequences_raw(data_dir: str, prefix: str, label: str) -> list[np.ndarray]:
    pattern = os.path.join(data_dir, f"{prefix}_*_{label}_raw.npy")
    files = sorted(glob.glob(pattern))
    seqs = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[1] != 48 or arr.shape[0] < 5:
            print(f"[SKIP] {os.path.basename(f)}: shape={arr.shape}")
            continue
        seqs.append(arr.astype(np.float32))
        print(f"[LOAD] {os.path.basename(f)}: {arr.shape}")
    return seqs


# ─────────────────────────────────────────────
#  Feature 추출
# ─────────────────────────────────────────────
def extract_mean_vec(seq6d: np.ndarray) -> np.ndarray:
    """6D → mean_vec_biased (T, 3)"""
    return (seq6d[:, 1:2] * seq6d[:, 3:6]).astype(np.float64)


def extract_z_channels(seq_raw: np.ndarray) -> np.ndarray:
    """48D raw → 16개 Z채널 (T, 16)"""
    return seq_raw[:, Z_IDX].astype(np.float64)


def extract_mean_z(seq_raw: np.ndarray) -> np.ndarray:
    """48D raw → 16개 Z채널 평균 (T, 1): 방향성 신호를 1D로 집약"""
    return seq_raw[:, Z_IDX].mean(axis=1, keepdims=True).astype(np.float64)


def window_mean_features(seq: np.ndarray, feat_fn,
                          seq_len: int, stride: int,
                          warmup: int = 0) -> np.ndarray:
    """seq → 윈도우별 feature (n_windows, D).
    feature = [mean (D), delta=last_half_mean - first_half_mean (D)]
    warmup: 초반 N프레임 스킵 (pre-contact 구간 제거)
    """
    T = len(seq)
    half = seq_len // 2
    feats = []
    for start in range(warmup, T - seq_len + 1, stride):
        window = seq[start:start + seq_len]
        f = feat_fn(window)
        mean_f  = f.mean(axis=0)                    # (D,) 평균
        delta_f = f[half:].mean(axis=0) - f[:half].mean(axis=0)  # (D,) 추세
        feats.append(np.concatenate([mean_f, delta_f]))
    return np.array(feats, dtype=np.float64) if feats else np.empty((0, 1))


# ─────────────────────────────────────────────
#  FCM (scikit-fuzzy 없이 직접 구현)
# ─────────────────────────────────────────────
class FuzzyCMeans:
    def __init__(self, n_clusters: int = 3, m: float = 2.0,
                 max_iter: int = 300, tol: float = 1e-6, random_state: int = 42):
        self.n_clusters   = n_clusters
        self.m            = m
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self.centers_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        U = rng.random((n, self.n_clusters))
        U /= U.sum(axis=1, keepdims=True)
        for _ in range(self.max_iter):
            Um = U ** self.m
            centers = (Um.T @ X) / (Um.sum(axis=0)[:, None] + EPS)
            d_mat = np.maximum(
                np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2), EPS)
            power = 2.0 / (self.m - 1.0)
            ratio = (d_mat[:, :, None] / d_mat[:, None, :]) ** power
            U_new = 1.0 / ratio.sum(axis=2)
            if np.max(np.abs(U_new - U)) < self.tol:
                U = U_new; break
            U = U_new
        self.centers_ = centers
        return self

    def membership(self, X: np.ndarray) -> np.ndarray:
        d_mat = np.maximum(
            np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2), EPS)
        power = 2.0 / (self.m - 1.0)
        ratio = (d_mat[:, :, None] / d_mat[:, None, :]) ** power
        return 1.0 / ratio.sum(axis=2)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return 1.0 - self.membership(X).max(axis=1)

    def anomaly_score_dist(self, X: np.ndarray) -> np.ndarray:
        """min-distance to cluster centers (K-means 방식 score, 바운드 없음)."""
        d_mat = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        return d_mat.min(axis=1)


# ─────────────────────────────────────────────
#  Silhouette 기반 최적 K 선택
# ─────────────────────────────────────────────
def select_k_by_silhouette(X_scaled: np.ndarray,
                            k_range: range = range(2, 7)) -> int:
    """
    success 윈도우 특징(X_scaled)에 대해 K=k_range 각각 KMeans를 피팅하고
    silhouette score가 가장 높은 K를 반환.
    """
    best_k, best_score = k_range.start, -1.0
    print(f"\n[Silhouette] K 자동 선택 (탐색 범위: K={k_range.start}..{k_range.stop - 1})")
    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_scaled)
        sc = silhouette_score(X_scaled, km.labels_)
        marker = " ← best" if sc > best_score else ""
        print(f"  K={k}: silhouette={sc:.4f}{marker}")
        if sc > best_score:
            best_score, best_k = sc, k
    print(f"[Silhouette] 선택된 K = {best_k}  (score={best_score:.4f})")
    return best_k


# ─────────────────────────────────────────────
#  평가 공통
# ─────────────────────────────────────────────
def consec_detect(scores: np.ndarray, threshold: float, consec_needed: int) -> bool:
    cnt = 0
    for s in scores:
        cnt = cnt + 1 if s > threshold else 0
        if cnt >= consec_needed:
            return True
    return False


def seq_eval(score_fn, seqs, threshold, consec_needed):
    return (sum(1 for s in seqs if consec_detect(score_fn(s), threshold, consec_needed)),
            len(seqs))


def print_results(method: str, seqs_by_label: dict,
                  score_fn, threshold: float, consec_list: list[int]):
    print(f"\n{'='*65}")
    print(f"  {method}  (threshold={threshold:.4f})")
    print(f"{'='*65}")
    print(f"{'consec':>8} | {'label':<18} | {'anom':>5}/{'total':>5} | {'rate':>7}")
    print("-" * 65)
    for consec in consec_list:
        for label, seqs in seqs_by_label.items():
            if not seqs:
                continue
            n_a, n_t = seq_eval(score_fn, seqs, threshold, consec)
            tag = "← FPR" if "success" in label else "← TPR"
            print(f"{consec:>8} | {label:<18} | {n_a:>5}/{n_t:>5} | "
                  f"{n_a/n_t*100:>6.1f}%  {tag}")
        print("-" * 65)


# ─────────────────────────────────────────────
#  Heuristic
# ─────────────────────────────────────────────
def run_heuristic(seqs_by_label: dict, feature: str,
                  consec_list: list[int], z_threshold: float = 0.0):
    if feature == "z_raw":
        # 프레임별 16개 Z 평균 < z_threshold → 이상
        def score_fn(seq_raw):
            return (extract_z_channels(seq_raw).mean(axis=1) < z_threshold).astype(float)
        label_str = f"mean_Z < {z_threshold}"
        threshold  = 0.5
    else:
        # 기존 MRL 기반
        def score_fn(seq6d):
            max_norm = seq6d[:, 2]
            norm     = seq6d[:, 1]
            return ((max_norm < 0.5) | (norm < 0.55)).astype(float)
        label_str = "max_norm<0.5 OR norm<0.55"
        threshold  = 0.5

    print(f"\n{'='*65}")
    print(f"  Heuristic [{feature}]  ({label_str})")
    print(f"{'='*65}")
    print(f"{'consec':>8} | {'label':<18} | {'anom':>5}/{'total':>5} | {'rate':>7}")
    print("-" * 65)
    for consec in consec_list:
        for label, seqs in seqs_by_label.items():
            if not seqs:
                continue
            n_a, n_t = seq_eval(score_fn, seqs, threshold, consec)
            tag = "← FPR" if "success" in label else "← TPR"
            print(f"{consec:>8} | {label:<18} | {n_a:>5}/{n_t:>5} | "
                  f"{n_a/n_t*100:>6.1f}%  {tag}")
        print("-" * 65)


# ─────────────────────────────────────────────
#  K-means
# ─────────────────────────────────────────────
def run_kmeans(seqs_by_label: dict, n_clusters: int,
               threshold_pct: int, consec_list: list[int],
               feature: str = "z_raw",
               seq_len: int = 50, stride_train: int = 10, stride_eval: int = 5,
               warmup: int = 50,
               save_path: str | None = None):
    success_seqs = seqs_by_label.get("success", [])
    if not success_seqs:
        print("[ERROR] K-means: success 데이터 없음"); return

    feat_fn  = (extract_mean_z    if feature == "mean_z" else
                extract_z_channels if feature == "z_raw"  else extract_mean_vec)
    feat_dim = (1 if feature == "mean_z" else 16 if feature == "z_raw" else 3) * 2

    _feats = [window_mean_features(s, feat_fn, seq_len, stride_train, warmup=warmup)
              for s in success_seqs]
    _feats = [f for f in _feats if len(f) > 0]
    X_success = np.vstack(_feats)
    scaler   = StandardScaler().fit(X_success)
    X_scaled = scaler.transform(X_success)

    if n_clusters == 0:
        n_clusters = select_k_by_silhouette(X_scaled)

    print(f"\n[K-means|{feature}|windowed] 학습: {X_scaled.shape[0]} 샘플 "
          f"({feat_dim}D), {n_clusters} 클러스터")
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    km.fit(X_scaled)

    def _km_dist(Xs):
        return np.min(
            np.linalg.norm(Xs[:, None, :] - km.cluster_centers_[None, :, :], axis=2),
            axis=1)

    def score_fn(seq):
        feats = window_mean_features(seq, feat_fn, seq_len, stride_eval, warmup=warmup)
        if len(feats) == 0:
            return np.array([])
        return _km_dist(scaler.transform(feats))

    all_train = _km_dist(X_scaled)
    threshold = float(np.percentile(all_train, threshold_pct))
    print(f"[K-means] threshold ({threshold_pct}th pct) = {threshold:.4f}")

    if save_path:
        joblib.dump({
            "km": km, "scaler": scaler, "threshold": threshold,
            "feature": feature, "windowed": True,
            "seq_len": seq_len, "stride_eval": stride_eval, "warmup": warmup,
        }, save_path)
        print(f"[K-means] 저장: {save_path}")

    print_results(f"K-means [{feature}]", seqs_by_label, score_fn, threshold, consec_list)


# ─────────────────────────────────────────────
#  FCM
# ─────────────────────────────────────────────
def run_fcm(seqs_by_label: dict, n_clusters: int, m: float,
            threshold_pct: int, consec_list: list[int],
            feature: str = "z_raw",
            seq_len: int = 50, stride_train: int = 10, stride_eval: int = 5,
            warmup: int = 50,
            save_path: str | None = None):
    success_seqs = seqs_by_label.get("success", [])
    if not success_seqs:
        print("[ERROR] FCM: success 데이터 없음"); return

    feat_fn  = (extract_mean_z    if feature == "mean_z" else
                extract_z_channels if feature == "z_raw"  else extract_mean_vec)
    feat_dim = (1 if feature == "mean_z" else 16 if feature == "z_raw" else 3) * 2

    _feats = [window_mean_features(s, feat_fn, seq_len, stride_train, warmup=warmup)
              for s in success_seqs]
    _feats = [f for f in _feats if len(f) > 0]
    X_success = np.vstack(_feats)
    scaler   = StandardScaler().fit(X_success)
    X_scaled = scaler.transform(X_success)

    if n_clusters == 0:
        n_clusters = select_k_by_silhouette(X_scaled)

    print(f"\n[FCM|{feature}|windowed] 학습: {X_scaled.shape[0]} 샘플 "
          f"({feat_dim}D), {n_clusters} 클러스터, m={m}")
    fcm = FuzzyCMeans(n_clusters=n_clusters, m=m, random_state=42)
    fcm.fit(X_scaled)

    def score_fn(seq):
        feats = window_mean_features(seq, feat_fn, seq_len, stride_eval, warmup=warmup)
        if len(feats) == 0:
            return np.array([])
        return fcm.anomaly_score_dist(scaler.transform(feats))

    all_train = fcm.anomaly_score_dist(X_scaled)
    threshold = float(np.percentile(all_train, threshold_pct))
    print(f"[FCM] threshold ({threshold_pct}th pct) = {threshold:.4f}")

    if save_path:
        joblib.dump({
            "fcm": fcm, "scaler": scaler, "threshold": threshold,
            "feature": feature, "windowed": True,
            "seq_len": seq_len, "stride_eval": stride_eval, "warmup": warmup,
        }, save_path)
        print(f"[FCM] 저장: {save_path}")

    print_results(f"FCM [{feature}]", seqs_by_label, score_fn, threshold, consec_list)


# ─────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Heuristic/K-means/FCM baseline")
    parser.add_argument("--model",   choices=["grasp", "transport"], default="grasp")
    parser.add_argument("--method",  choices=["heuristic", "kmeans", "fcm", "all"],
                        default="all")
    parser.add_argument("--feature", choices=["mrl", "z_raw", "mean_z"], default="mean_z",
                        help="mean_z: 16채널 평균 1D (default) / z_raw: 16채널 16D / mrl: 6D MRL mean_vec")
    parser.add_argument("--n_clusters",    type=int,   default=0,
                        help="클러스터 수 (0=silhouette로 자동 선택, default: 0)")
    parser.add_argument("--fcm_m",         type=float, default=2.0)
    parser.add_argument("--threshold_pct", type=int,   default=95)
    parser.add_argument("--z_threshold",   type=float, default=0.0,
                        help="Heuristic Z축 임계값 (default: 0.0, mean_Z < 임계값 → 이상)")
    parser.add_argument("--warmup",        type=int,   default=50,
                        help="학습 윈도우에서 스킵할 초반 프레임 수 (pre-contact 제거, default: 50)")
    parser.add_argument("--seq_len",       type=int,   default=50)
    parser.add_argument("--stride_train",  type=int,   default=10)
    parser.add_argument("--stride_eval",   type=int,   default=5)
    parser.add_argument("--data_dir",      type=str,   default=None)
    args = parser.parse_args()

    ws = os.path.dirname(os.path.abspath(__file__))
    default_dirs = {"grasp": "grasp_05", "transport": "transport_05"}
    data_dir = args.data_dir or os.path.join(ws, default_dirs[args.model])
    prefix   = args.model

    print(f"\n[CONFIG] model={prefix}, method={args.method}, feature={args.feature}")
    print(f"[CONFIG] n_clusters={args.n_clusters}, fcm_m={args.fcm_m}, "
          f"threshold_pct={args.threshold_pct}")
    print(f"[CONFIG] seq_len={args.seq_len}, stride_train={args.stride_train}, "
          f"stride_eval={args.stride_eval}")
    print(f"[CONFIG] data_dir={data_dir}")

    # feature에 따라 로딩 방식 분기
    seqs_by_label: dict[str, list[np.ndarray]] = {}
    for label in ["success", "failure"]:
        if args.feature in ("z_raw", "mean_z"):
            seqs = load_sequences_raw(data_dir, prefix, label)
        else:
            seqs = load_sequences_6d(data_dir, prefix, label)
        seqs_by_label[label] = seqs
        print(f"[DATA] {label}: {len(seqs)} sequences")

    consec_list = [1, 2, 3, 5]
    methods = ["heuristic", "kmeans", "fcm"] if args.method == "all" else [args.method]

    for method in methods:
        if method == "heuristic":
            run_heuristic(seqs_by_label, args.feature, consec_list, args.z_threshold)

        elif method == "kmeans":
            save_path = os.path.join(ws, f"{prefix}_kmeans.pkl")
            run_kmeans(seqs_by_label, args.n_clusters, args.threshold_pct, consec_list,
                       feature=args.feature, seq_len=args.seq_len,
                       stride_train=args.stride_train, stride_eval=args.stride_eval,
                       warmup=args.warmup, save_path=save_path)

        elif method == "fcm":
            save_path = os.path.join(ws, f"{prefix}_fcm.pkl")
            run_fcm(seqs_by_label, args.n_clusters, args.fcm_m,
                    args.threshold_pct, consec_list,
                    feature=args.feature, seq_len=args.seq_len,
                    stride_train=args.stride_train, stride_eval=args.stride_eval,
                    warmup=args.warmup, save_path=save_path)


if __name__ == "__main__":
    main()
