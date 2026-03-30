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
    k: float = 0.1,      # seconds
    pre: float = 0.7,    # ratio
    tail: float = 0.5,  # seconds
    Nbase: float = 1.5,  # seconds
):
    """
    :param data: np.load(..., allow_pickle=True)로 읽은 npz
      {
        "frames": [ (timestamp, label, feat_3x4x4), ... ],
        "scenario": integer,
        "tag": string
      }
    :param sampling_rate: 마스킹/초->프레임 변환에 사용할 "학습 기준 Hz"
    :param step: 인덱스 기준 다운샘플링 스텝 (500->1, 250->2, 100->5)
    """

    # frames 로드
    if "frames" not in data:
        raise KeyError("npz에 'frames' 키가 없습니다.")
    frames = data["frames"]
    if len(frames) == 0:
        raise ValueError("frames가 비어 있습니다.")

    frames_list = list(frames)

    # (2) 인덱스 기준 건너뛰기
    frames_list = decimate_by_step(frames_list, step=step)
    if len(frames_list) == 0:
        raise ValueError("downsample 이후 frames_list가 비었습니다.")

    timestamps = []
    labels = []
    feats = []

    for item in frames_list:
        t, lab, feat = item
        timestamps.append(float(t))
        labels.append(lab)

        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 3 or feat.shape[1:] != (4, 4):
            raise ValueError(f"feat shape이 예상과 다릅니다: {feat.shape} (expected: (C,4,4))")
        if feat.shape[0] < use_first_C:
            raise ValueError(f"feat 채널 수가 부족합니다: C={feat.shape[0]}, need {use_first_C}")
        feat = feat[:use_first_C]
        feats.append(feat)

    F = np.stack(feats, axis=0).astype(np.float32)  # (N, C, 4, 4)
    N, C, H, W = F.shape

    # (7) baseline 길이: N_base = Nbase * sampling_rate
    N_base = int(max(1, round(Nbase * sampling_rate)))
    N_base_eff = min(N_base, N)
    F_base = F[:N_base_eff]

    mean = np.mean(F_base, axis=0)
    std = np.std(F_base, axis=0)
    std = np.maximum(std, eps)

    tilde_F = (F - mean) / std

    delta_tilde_F = np.zeros_like(tilde_F, dtype=np.float32)
    delta_tilde_F[1:] = tilde_F[1:] - tilde_F[:-1]

    X = np.concatenate([tilde_F, delta_tilde_F], axis=1)  # (N, 2C, 4, 4)

    # 라벨/ignore 마스크 생성
    labels_arr = np.asarray(labels, dtype=object)

    # (3) total_ignore = k * sampling_rate
    total_ignore = int(max(1, round(k * sampling_rate)))

    # (4) pre_default = pre * total_ignore
    pre_default = int(max(1, round(pre * total_ignore)))

    # (5) post_default = total_ignore - pre_default (최소 1)
    post_default = int(max(1, total_ignore - pre_default))

    ignore_mask = np.zeros((N,), dtype=bool)

    # baseline 구간 보호: baseline 내부는 ignore 만들지 않음
    min_keep = int(N_base_eff)

    # (6) tail_keep = tail * sampling_rate
    tail_keep = int(max(1, round(tail * sampling_rate)))

    # 전이 인덱스 찾기
    transition_idxs = []
    for t in range(1, N):
        if labels_arr[t] != labels_arr[t - 1]:
            transition_idxs.append(t)

    # 전이마다 ignore 구간 마스킹
    for t0 in transition_idxs:
        pre_len = pre_default
        post_len = post_default

        s = max(0, t0 - pre_len)
        e = min(N, t0 + post_len)

        # baseline 보호
        if s < min_keep:
            s = min_keep
        if s >= e:
            print("[WARN] 너무 초기에 전이가 일어나 ignore mask 생성 불가 (baseline 보호).")
            continue

        # tail 보호
        max_e = N - tail_keep
        e = min(e, max_e)
        if e <= s:
            print("[WARN] 너무 후기에 전이가 일어나 ignore mask 생성 불가 (tail 보호).")
            continue

        ignore_mask[s:e] = True

    train_labels = labels_arr.copy()
    train_labels[ignore_mask] = "IGNORE"

    # 기타 메타
    scenario = int(data["scenario"]) if "scenario" in data else -1
    tag = str(data["tag"]) if "tag" in data else ""

    return {
        "X": X,
        "labels": labels_arr,
        "train_labels": train_labels,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
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