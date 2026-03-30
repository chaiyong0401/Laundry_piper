#!/usr/bin/env python3
# preprocess.py
# /data/ 디렉토리의 *.npz 파일들을 전처리하여 /data/preprocessed/ 디렉토리에 *.npz의 형태로 저장
#
# 기능:
# 1) --samplerate (default=500): 500/250/100 중 선택
# 2) samplerate=500: 전부 사용(step=1)
#    samplerate=250: 2프레임당 1개 사용(step=2)
#    samplerate=100: 5프레임당 1개 사용(step=5)
#    (실제 수집 Hz가 508~509여도 "인덱스 기준 건너뛰기"로 처리)
# 3) --k (default=0.1): total_ignore = k * samplerate
# 4) --pre (default=0.7): pre_default = pre * total_ignore
# 5) post_default = total_ignore - pre_default
# 6) --tail (default=0.5): tail_keep = tail * samplerate
# 7) --Nbase (default=1.5): N_base = Nbase * samplerate

import argparse
from pathlib import Path
import numpy as np
import json


def decimate_by_step(frames_list, step: int):
    """인덱스 기준으로 step마다 1개만 사용."""
    if step < 1:
        raise ValueError("step은 1 이상이어야 합니다.")
    return frames_list[::step]


def preprocess_data(
    data,
    sampling_rate: int = 500,
    step: int = 1,
    eps: float = 1e-6,
    use_first_C: int = 3,
    k: float = 0.1,
    pre: float = 0.7,
    tail: float = 0.5,
    Nbase: float = 1.5,
):
    # ---- 새 구조 로드 ----
    if not all(k in data for k in ["x", "label", "t"]):
        raise KeyError("npz에 x / label / t 키가 없습니다.")

    F = np.asarray(data["x"], dtype=np.float32)   # (N, C, 4, 4)
    labels_arr = np.asarray(data["label"], dtype=object)
    timestamps = np.asarray(data["t"], dtype=np.float64)

    if F.ndim != 4 or F.shape[2:] != (4, 4):
        raise ValueError(f"x shape이 예상과 다릅니다: {F.shape}")

    # ---- 다운샘플링 ----
    F = F[::step]
    labels_arr = labels_arr[::step]
    timestamps = timestamps[::step]

    N, C, H, W = F.shape

    if C < use_first_C:
        raise ValueError(f"채널 부족: {C} < {use_first_C}")

    F = F[:, :use_first_C]

    # ---- baseline ----
    N_base = int(max(1, round(Nbase * sampling_rate)))
    N_base_eff = min(N_base, N)
    F_base = F[:N_base_eff]

    mean = np.mean(F_base, axis=0)
    std = np.std(F_base, axis=0)
    std = np.maximum(std, eps)

    tilde_F = (F - mean) / std

    delta_tilde_F = np.zeros_like(tilde_F)
    delta_tilde_F[1:] = tilde_F[1:] - tilde_F[:-1]

    X = np.concatenate([tilde_F, delta_tilde_F], axis=1)

    # ---- ignore mask ----
    total_ignore = int(max(1, round(k * sampling_rate)))
    pre_default = int(max(1, round(pre * total_ignore)))
    post_default = int(max(1, total_ignore - pre_default))

    ignore_mask = np.zeros((N,), dtype=bool)

    min_keep = int(N_base_eff)
    tail_keep = int(max(1, round(tail * sampling_rate)))

    transition_idxs = np.where(labels_arr[1:] != labels_arr[:-1])[0] + 1

    for t0 in transition_idxs:
        s = max(min_keep, t0 - pre_default)
        e = min(N - tail_keep, t0 + post_default)
        if s < e:
            ignore_mask[s:e] = True

    train_labels = labels_arr.copy()
    train_labels[ignore_mask] = "IGNORE"

        # ---- meta 처리 ----
    scenario = -1
    tag = ""

    if "meta" in data:
        meta_raw = data["meta"]

        # npz에 저장된 방식에 따라 object/0-d array일 수 있음
        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.item()

        if isinstance(meta_raw, (str, bytes)):
            meta_dict = json.loads(meta_raw)
            scenario = int(meta_dict.get("scenario", -1))
            tag = meta_dict.get("tag", "")
        elif isinstance(meta_raw, dict):
            scenario = int(meta_raw.get("scenario", -1))
            tag = meta_raw.get("tag", "")

    return {
        "X": X,
        "labels": labels_arr,
        "train_labels": train_labels,
        "timestamps": timestamps,
        "scenario": scenario,
        "tag": tag,
        "samplerate": np.int32(sampling_rate),
        "step": np.int32(step),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samplerate", type=int, default=500, choices=[500, 250, 100],
                    help="학습 기준 Hz (마스킹/초->프레임 변환에 사용). 500/250/100만 지원.")
    ap.add_argument("--k", type=float, default=0.1,
                    help="ignore 총 길이(초). total_ignore = k * samplerate")
    ap.add_argument("--pre", type=float, default=0.7,
                    help="ignore 구간에서 이전방향 비율. pre_default = pre * total_ignore")
    ap.add_argument("--tail", type=float, default=0.5,
                    help="마지막 tail(초) 구간은 ignore 생성에서 제외. tail_keep = tail * samplerate")
    ap.add_argument("--Nbase", type=float, default=1.5,
                    help="baseline으로 사용할 앞부분 길이(초). N_base = Nbase * samplerate")
    ap.add_argument("--use_first_C", type=int, default=3,
                    help="feat 채널 중 앞에서부터 사용할 채널 수")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="std 하한 eps")
    return ap.parse_args()


def main():
    args = parse_args()

    # samplerate -> step (인덱스 기준 다운샘플링)
    rate_to_step = {500: 1, 250: 2, 100: 5}
    step = rate_to_step[args.samplerate]

    cwd = Path.cwd()
    out_dir = cwd / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(cwd.glob("*.npz"))
    if not npz_files:
        print(f"[WARN] {cwd} 에 .npz 파일이 없습니다.")
        return

    print(f"[INFO] Found {len(npz_files)} npz files in {cwd}")
    print(f"[INFO] Output dir: {out_dir}")
    print(
        f"[INFO] samplerate={args.samplerate}Hz (step={step}) | "
        f"k={args.k}s | pre={args.pre} | tail={args.tail}s | Nbase={args.Nbase}s | use_first_C={args.use_first_C}"
    )

    for fp in npz_files:
        try:
            data = np.load(fp, allow_pickle=True)
            processed = preprocess_data(
                data,
                sampling_rate=args.samplerate,
                step=step,
                eps=args.eps,
                use_first_C=args.use_first_C,
                k=args.k,
                pre=args.pre,
                tail=args.tail,
                Nbase=args.Nbase,
            )

            out_path = out_dir / fp.name
            np.savez_compressed(
                out_path,
                X=processed["X"],
                labels=processed["labels"],
                train_labels=processed["train_labels"],
                timestamps=processed["timestamps"],
                scenario=processed["scenario"],
                tag=processed["tag"],
                samplerate=processed["samplerate"],
                step=processed["step"],
            )
            print(f"[OK] {fp.name} -> preprocessed/{fp.name} | X shape={processed['X'].shape}")
        except Exception as e:
            print(f"[FAIL] {fp.name}: {e}")


if __name__ == "__main__":
    main()
