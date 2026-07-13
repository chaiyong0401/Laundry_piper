"""
IsolationForest Baseline 학습 및 평가 스크립트
================================================
프레임 단위 one-class anomaly detection (GRU-AE baseline 비교용).
success 프레임으로 학습 후, 시퀀스 단위 TPR/FPR 평가.

Usage:
  python train_iforest.py --model grasp
  python train_iforest.py --model grasp --use_raw
  python train_iforest.py --model transport
  python train_iforest.py --model transport --use_raw
"""

import argparse
import os
import glob
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
#  데이터 로딩
# ─────────────────────────────────────────────
def load_sequences_raw(data_dir: str, prefix: str, label: str,
                       use_raw: bool = False) -> list[np.ndarray]:
    if use_raw:
        pattern = os.path.join(data_dir, f"{prefix}_*_{label}_raw.npy")
        files = sorted(glob.glob(pattern))
    else:
        pattern = os.path.join(data_dir, f"{prefix}_*_{label}.npy")
        files = sorted([f for f in glob.glob(pattern) if not f.endswith("_raw.npy")])
    sequences = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[0] < 5:
            print(f"[SKIP] {os.path.basename(f)}: shape={arr.shape}")
            continue
        sequences.append(arr.astype(np.float32))
        print(f"[LOAD] {os.path.basename(f)}: {arr.shape}")
    return sequences


# ─────────────────────────────────────────────
#  시퀀스 단위 평가
# ─────────────────────────────────────────────
def evaluate_sequences(model: IsolationForest, scaler: StandardScaler,
                       sequences: list[np.ndarray],
                       consec_needed: int = 1) -> tuple[int, int]:
    """
    시퀀스별로 이상 판정.
    consec_needed 프레임 연속 이상 시 해당 시퀀스를 anomaly로 판정.
    반환: (anomaly_count, total_count)
    """
    anomaly_count = 0
    for seq in sequences:
        x = scaler.transform(seq)
        # IsolationForest: predict returns +1 (normal) or -1 (anomaly)
        preds = model.predict(x)   # (T,)
        consec = 0
        detected = False
        for p in preds:
            if p == -1:
                consec += 1
                if consec >= consec_needed:
                    detected = True
                    break
            else:
                consec = 0
        if detected:
            anomaly_count += 1
    return anomaly_count, len(sequences)


def print_eval(seqs_by_label: dict, model, scaler, consec_list: list[int]):
    labels = list(seqs_by_label.keys())
    success_labels = [l for l in labels if 'success' in l]
    failure_labels = [l for l in labels if 'success' not in l]

    print("\n" + "=" * 60)
    print(f"{'consec':>8} | {'label':<20} | {'anom':>5} / {'total':>5} | {'rate':>7}")
    print("-" * 60)

    for consec in consec_list:
        # FPR: success 시퀀스가 anomaly로 잘못 판정되는 비율
        for label in success_labels:
            seqs = seqs_by_label[label]
            if not seqs:
                continue
            n_anom, n_total = evaluate_sequences(model, scaler, seqs, consec)
            print(f"{consec:>8} | {label:<20} | {n_anom:>5} / {n_total:>5} | "
                  f"{n_anom/n_total*100:>6.1f}%  ← FPR")

        # TPR: failure 시퀀스가 anomaly로 올바르게 판정되는 비율
        for label in failure_labels:
            seqs = seqs_by_label[label]
            if not seqs:
                continue
            n_anom, n_total = evaluate_sequences(model, scaler, seqs, consec)
            print(f"{consec:>8} | {label:<20} | {n_anom:>5} / {n_total:>5} | "
                  f"{n_anom/n_total*100:>6.1f}%  ← TPR")
        print("-" * 60)


# ─────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IsolationForest baseline trainer")
    parser.add_argument("--model", choices=["grasp", "transport"], default="grasp")
    parser.add_argument("--use_raw", action="store_true",
                        help="48D raw taxel feature 사용 (default: 6D)")
    parser.add_argument("--z_only", action="store_true",
                        help="48D raw에서 16개 Z채널만 사용 (Z부호 반전 활용)")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="학습 데이터 내 이상치 비율 추정값 (default: 0.05)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="학습 시 success 시퀀스 앞 N 프레임 제외 (pre-contact 제거, default: 50)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="데이터 디렉토리 직접 지정 (default: 모델에 따라 자동)")
    args = parser.parse_args()

    ws = os.path.dirname(os.path.abspath(__file__))
    default_dirs = {"grasp": "grasp_05", "transport": "transport_05"}
    data_dir = args.data_dir or os.path.join(ws, default_dirs[args.model])
    prefix   = args.model

    # z_only는 raw 데이터에서 16개 Z채널의 mean → 1D feature
    use_raw = args.use_raw or args.z_only
    suffix  = "_z" if args.z_only else ("_raw" if args.use_raw else "")
    Z_IDX   = np.arange(2, 48, 3)  # 48D에서 Z채널 인덱스

    print(f"\n[CONFIG] model={args.model}, use_raw={use_raw}, z_only={args.z_only}, "
          f"n_estimators={args.n_estimators}, contamination={args.contamination}, "
          f"warmup={args.warmup}")
    print(f"[CONFIG] data_dir={data_dir}")

    # ── 데이터 로드
    seqs_by_label: dict[str, list[np.ndarray]] = {}
    for label in ["success", "failure"]:
        seqs = load_sequences_raw(data_dir, prefix, label, use_raw=use_raw)
        if args.z_only:
            # 16개 Z채널 평균 → (T, 1): 방향성(부호) 정보를 1D로 집약
            seqs = [s[:, Z_IDX].mean(axis=1, keepdims=True) for s in seqs]
        seqs_by_label[label] = seqs
        dim = seqs[0].shape[1] if seqs else "?"
        print(f"[DATA] {label}: {len(seqs)} sequences  (dim={dim})")

    success_seqs = seqs_by_label.get("success", [])
    if not success_seqs:
        print("[ERROR] success 데이터 없음. 종료.")
        return

    # ── Scaler fit (success 프레임, warmup 이후)
    warmup_seqs = [s[args.warmup:] for s in success_seqs if len(s) > args.warmup]
    all_success_frames = np.vstack(warmup_seqs)
    print(f"\n[TRAIN] success frames (warmup={args.warmup} 제외): {all_success_frames.shape}")
    scaler = StandardScaler()
    scaler.fit(all_success_frames)
    X_train = scaler.transform(all_success_frames)

    # ── IsolationForest 학습
    print(f"[TRAIN] IsolationForest 학습 중...")
    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # ── 학습 데이터 이상 비율 확인
    train_preds = model.predict(X_train)
    n_anom_train = (train_preds == -1).sum()
    print(f"[TRAIN] 학습 데이터 이상 판정: {n_anom_train}/{len(X_train)} "
          f"({n_anom_train/len(X_train)*100:.1f}%)")

    # ── 모델 저장
    feature_key = "mean_z" if args.z_only else ("raw" if args.use_raw else "6d")
    model_path  = os.path.join(ws, f"{prefix}_iforest{suffix}.pkl")
    scaler_path = os.path.join(ws, f"{prefix}_iforest_scaler{suffix}.pkl")
    joblib.dump({"model": model, "scaler": scaler, "feature": feature_key,
                 "warmup": args.warmup}, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[SAVE] {model_path}  (feature={feature_key})")
    print(f"[SAVE] {scaler_path}")

    # ── 시퀀스 단위 평가
    print("\n[EVAL] 시퀀스 단위 평가 (consec=연속 이상 프레임 수)")
    print_eval(seqs_by_label, model, scaler, consec_list=[1, 2, 3, 5])


if __name__ == "__main__":
    main()
