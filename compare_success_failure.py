#!/usr/bin/env python3
"""
compare_success_failure.py
===========================
성공(success) 및 실패(failure) 데이터셋을 6D feature 및 48D raw(taxel) 데이터 관점에서 비교 및 시각화합니다.
grasp_new 및 transport_new 폴더 내의 데이터를 자동으로 탐색하여 분석 플롯을 생성합니다.

Usage:
    python compare_success_failure.py [--dir grasp_new] [--dir transport_new]
    (인자가 없으면 grasp_new와 transport_new를 모두 처리)
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 16개 taxel 인덱스 정의 (INTERLEAVED 방식) ──
# x0,y0,z0, x1,y1,z1 ... x15,y15,z15
X_IDX = np.arange(0, 48, 3)   # [0, 3, ..., 45]
Y_IDX = np.arange(1, 48, 3)   # [1, 4, ..., 46]
Z_IDX = np.arange(2, 48, 3)   # [2, 5, ..., 47]

# 6D Feature 이름 정의
FEAT_NAMES_6D = [
    "MRL (Mean Resultant Length)",
    "Norm (Weighted Mean Norm)",
    "Max Norm (Max Taxel Norm)",
    "Mean Unit X",
    "Mean Unit Y",
    "Mean Unit Z"
]

def resample_trajectory(arr, num_steps=100):
    """
    서로 다른 길이의 시퀀스를 uniform한 [0, 1] normalized time grid 상의 고정된 스텝 수로 보간합니다.
    arr: (T, D) shape의 ndarray
    """
    T, D = arr.shape
    if T == 1:
        # 단일 프레임인 경우 복제
        return np.repeat(arr, num_steps, axis=0)
    
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, num_steps)
    resampled = np.zeros((num_steps, D), dtype=arr.dtype)
    for d in range(D):
        resampled[:, d] = np.interp(x_new, x_old, arr[:, d])
    return resampled

def load_data_from_dir(target_dir):
    """
    지정된 디렉토리에서 success / failure 데이터를 로드합니다.
    """
    print(f"\n[Loader] '{target_dir}' 데이터 로딩 중...")
    
    # 1) 6D 데이터 파일명 가져오기 (raw가 포함되지 않은 .npy)
    all_npy = glob.glob(os.path.join(target_dir, "*.npy"))
    
    success_6d_files = sorted([f for f in all_npy if "success" in f and not f.endswith("_raw.npy")])
    failure_6d_files = sorted([f for f in all_npy if "failure" in f and not f.endswith("_raw.npy")])
    
    # 2) 48D raw 데이터 파일명 가져오기
    success_raw_files = sorted([f for f in all_npy if "success" in f and f.endswith("_raw.npy")])
    failure_raw_files = sorted([f for f in all_npy if "failure" in f and f.endswith("_raw.npy")])
    
    # 3) 실제 로드
    s_6d = [np.load(f).astype(np.float32) for f in success_6d_files]
    f_6d = [np.load(f).astype(np.float32) for f in failure_6d_files]
    
    s_raw = [np.load(f).astype(np.float32) for f in success_raw_files]
    f_raw = [np.load(f).astype(np.float32) for f in failure_raw_files]
    
    print(f"  - Success: 6D={len(s_6d)}개, 48D Raw={len(s_raw)}개")
    print(f"  - Failure: 6D={len(f_6d)}개, 48D Raw={len(f_raw)}개")
    
    return {
        "success_6d": s_6d,
        "failure_6d": f_6d,
        "success_raw": s_raw,
        "failure_raw": f_raw,
        "success_6d_files": success_6d_files,
        "failure_6d_files": failure_6d_files
    }

def plot_6d_comparison(data, dir_name, out_path, num_steps=100):
    """
    6D features (MRL, Norm, Max Norm, Mean Unit X/Y/Z) 비교 그래프 생성
    """
    s_list = data["success_6d"]
    f_list = data["failure_6d"]
    
    if len(s_list) == 0 and len(f_list) == 0:
        print(f"[Warning] '{dir_name}'에 6D 데이터가 없어 6D 비교 플롯을 건너뜁니다.")
        return

    # 시퀀스 길이 정규화 및 배열 변환
    s_resampled = []
    for arr in s_list:
        s_resampled.append(resample_trajectory(arr, num_steps))
    
    f_resampled = []
    for arr in f_list:
        f_resampled.append(resample_trajectory(arr, num_steps))
        
    s_arr = np.array(s_resampled) if len(s_resampled) > 0 else np.empty((0, num_steps, 6))
    f_arr = np.array(f_resampled) if len(f_resampled) > 0 else np.empty((0, num_steps, 6))
    
    # 2x3 grid 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"6D Features Comparison (Success vs Failure) - {dir_name}", fontsize=16, fontweight="bold")
    
    t = np.linspace(0, 1, num_steps)
    
    for i in range(6):
        ax = axes[i // 3, i % 3]
        ax.set_title(FEAT_NAMES_6D[i], fontsize=12, fontweight="bold")
        
        # Success 데이터 플로팅
        if s_arr.shape[0] > 0:
            # 개별 에피소드 그리기 (투명도 조절)
            for j in range(s_arr.shape[0]):
                ax.plot(t, s_arr[j, :, i], color="steelblue", alpha=0.15, lw=1)
            # 평균선
            s_mean = s_arr[:, :, i].mean(axis=0)
            s_std = s_arr[:, :, i].std(axis=0)
            ax.plot(t, s_mean, color="darkblue", lw=2.5, label=f"Success Mean (n={s_arr.shape[0]})")
            ax.fill_between(t, s_mean - s_std, s_mean + s_std, color="steelblue", alpha=0.15)
            
        # Failure 데이터 플로팅
        if f_arr.shape[0] > 0:
            # 개별 에피소드
            for j in range(f_arr.shape[0]):
                ax.plot(t, f_arr[j, :, i], color="darkorange", alpha=0.2, lw=1)
            # 평균선
            f_mean = f_arr[:, :, i].mean(axis=0)
            f_std = f_arr[:, :, i].std(axis=0)
            ax.plot(t, f_mean, color="firebrick", lw=2.5, label=f"Failure Mean (n={f_arr.shape[0]})")
            ax.fill_between(t, f_mean - f_std, f_mean + f_std, color="darkorange", alpha=0.15)
            
        ax.set_xlabel("Normalized Time", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="best")
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved 6D Comparison to: {out_path}")

def plot_48d_taxel_norms(data, dir_name, out_path, num_steps=100):
    """
    16개 Taxel 각각의 L2 norm (접촉 강도)을 4x4 그리드로 비교하는 그래프 생성
    """
    s_list = data["success_raw"]
    f_list = data["failure_raw"]
    
    if len(s_list) == 0 and len(f_list) == 0:
        print(f"[Warning] '{dir_name}'에 48D Raw 데이터가 없어 48D Taxel Norm 플롯을 건너뜁니다.")
        return
        
    # 각 raw 에피소드에 대해 16개 taxel별 L2 norm 시계열 계산 후 리샘플링
    # raw shape: (T, 48) -> vecs shape: (T, 16, 3) -> norms shape: (T, 16)
    s_norms_resampled = []
    for arr in s_list:
        vecs = arr.reshape(-1, 16, 3)
        norms = np.linalg.norm(vecs, axis=2) # (T, 16)
        s_norms_resampled.append(resample_trajectory(norms, num_steps))
        
    f_norms_resampled = []
    for arr in f_list:
        vecs = arr.reshape(-1, 16, 3)
        norms = np.linalg.norm(vecs, axis=2) # (T, 16)
        f_norms_resampled.append(resample_trajectory(norms, num_steps))
        
    s_arr = np.array(s_norms_resampled) if len(s_norms_resampled) > 0 else np.empty((0, num_steps, 16))
    f_arr = np.array(f_norms_resampled) if len(f_norms_resampled) > 0 else np.empty((0, num_steps, 16))
    
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True, sharey=True)
    fig.suptitle(f"48D Raw: Individual Taxel Force Norm (Success vs Failure) - {dir_name}", fontsize=16, fontweight="bold")
    
    t = np.linspace(0, 1, num_steps)
    
    for taxel_i in range(16):
        ax = axes[taxel_i // 4, taxel_i % 4]
        ax.set_title(f"Taxel {taxel_i} (Dim {taxel_i*3}~{taxel_i*3+2})", fontsize=10, fontweight="bold")
        
        # Success
        if s_arr.shape[0] > 0:
            for j in range(s_arr.shape[0]):
                ax.plot(t, s_arr[j, :, taxel_i], color="steelblue", alpha=0.15, lw=0.8)
            s_mean = s_arr[:, :, taxel_i].mean(axis=0)
            ax.plot(t, s_mean, color="darkblue", lw=2, label="Success")
            
        # Failure
        if f_arr.shape[0] > 0:
            for j in range(f_arr.shape[0]):
                ax.plot(t, f_arr[j, :, taxel_i], color="darkorange", alpha=0.2, lw=0.8)
            f_mean = f_arr[:, :, taxel_i].mean(axis=0)
            ax.plot(t, f_mean, color="firebrick", lw=2, label="Failure")
            
        ax.grid(True, alpha=0.3, linestyle="--")
        if taxel_i == 0:
            ax.legend(fontsize=8, loc="upper left")
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved 48D Taxel Norm Comparison to: {out_path}")

def plot_48d_axis_averages(data, dir_name, out_path, num_steps=100):
    """
    48D Raw 데이터에서 X, Y, Z 축 각각의 평균 raw tactile 세기를 전체 taxel에 걸쳐 평균화하여 비교합니다.
    (동작에 따른 힘의 방향성 변화 분석용)
    """
    s_list = data["success_raw"]
    f_list = data["failure_raw"]
    
    if len(s_list) == 0 and len(f_list) == 0:
        return

    # 각 에피소드별로 X, Y, Z축 값을 각각 평균내고 리샘플링
    s_axis_resampled = []
    for arr in s_list:
        # arr: (T, 48)
        x_avg = arr[:, X_IDX].mean(axis=1) # (T,)
        y_avg = arr[:, Y_IDX].mean(axis=1) # (T,)
        z_avg = arr[:, Z_IDX].mean(axis=1) # (T,)
        stacked = np.stack([x_avg, y_avg, z_avg], axis=1) # (T, 3)
        s_axis_resampled.append(resample_trajectory(stacked, num_steps))
        
    f_axis_resampled = []
    for arr in f_list:
        x_avg = arr[:, X_IDX].mean(axis=1)
        y_avg = arr[:, Y_IDX].mean(axis=1)
        z_avg = arr[:, Z_IDX].mean(axis=1)
        stacked = np.stack([x_avg, y_avg, z_avg], axis=1)
        f_axis_resampled.append(resample_trajectory(stacked, num_steps))
        
    s_arr = np.array(s_axis_resampled) if len(s_axis_resampled) > 0 else np.empty((0, num_steps, 3))
    f_arr = np.array(f_axis_resampled) if len(f_axis_resampled) > 0 else np.empty((0, num_steps, 3))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"48D Raw: Spatial Axis Average (Success vs Failure) - {dir_name}", fontsize=14, fontweight="bold")
    
    t = np.linspace(0, 1, num_steps)
    axis_labels = ["X-Axis Average (Shear Force X)", "Y-Axis Average (Shear Force Y)", "Z-Axis Average (Normal Force Z)"]
    
    for i in range(3):
        ax = axes[i]
        ax.set_title(axis_labels[i], fontsize=11, fontweight="bold")
        
        # Success
        if s_arr.shape[0] > 0:
            for j in range(s_arr.shape[0]):
                ax.plot(t, s_arr[j, :, i], color="steelblue", alpha=0.15, lw=1)
            s_mean = s_arr[:, :, i].mean(axis=0)
            s_std = s_arr[:, :, i].std(axis=0)
            ax.plot(t, s_mean, color="darkblue", lw=2, label="Success Mean")
            ax.fill_between(t, s_mean - s_std, s_mean + s_std, color="steelblue", alpha=0.1)
            
        # Failure
        if f_arr.shape[0] > 0:
            for j in range(f_arr.shape[0]):
                ax.plot(t, f_arr[j, :, i], color="darkorange", alpha=0.2, lw=1)
            f_mean = f_arr[:, :, i].mean(axis=0)
            f_std = f_arr[:, :, i].std(axis=0)
            ax.plot(t, f_mean, color="firebrick", lw=2, label="Failure Mean")
            ax.fill_between(t, f_mean - f_std, f_mean + f_std, color="darkorange", alpha=0.1)
            
        ax.set_ylabel("Raw Sensor Value")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="best")
        
    axes[-1].set_xlabel("Normalized Time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Axis Average Comparison to: {out_path}")

def print_summary_stats(data, dir_name):
    """
    성공/실패 데이터셋의 간단한 통계 수치를 터미널에 깔끔하게 출력합니다.
    """
    print(f"\n" + "="*60)
    print(f" [Summary Statistics] Directory: {dir_name}")
    print("="*60)
    
    # 에피소드 길이 (프레임 수) 분석
    s_lens = [len(x) for x in data["success_6d"]]
    f_lens = [len(x) for x in data["failure_6d"]]
    
    print(f"{'Metric':<25} | {'Success':<15} | {'Failure':<15}")
    print("-"*60)
    print(f"{'Episode Count':<25} | {len(s_lens):<15d} | {len(f_lens):<15d}")
    
    if len(s_lens) > 0:
        s_len_str = f"{np.mean(s_lens):.1f} ({min(s_lens)}~{max(s_lens)})"
    else:
        s_len_str = "N/A"
        
    if len(f_lens) > 0:
        f_len_str = f"{np.mean(f_lens):.1f} ({min(f_lens)}~{max(f_lens)})"
    else:
        f_len_str = "N/A"
        
    print(f"{'Length Mean (Range)':<25} | {s_len_str:<15} | {f_len_str:<15}")
    
    # 전체 에피소드 MRL 평균 분석 (6D feature의 index 0)
    if len(data["success_6d"]) > 0:
        s_mrl_all = np.concatenate([x[:, 0] for x in data["success_6d"]])
        s_mrl_str = f"{s_mrl_all.mean():.4f} (std={s_mrl_all.std():.4f})"
    else:
        s_mrl_str = "N/A"
        
    if len(data["failure_6d"]) > 0:
        f_mrl_all = np.concatenate([x[:, 0] for x in data["failure_6d"]])
        f_mrl_str = f"{f_mrl_all.mean():.4f} (std={f_mrl_all.std():.4f})"
    else:
        f_mrl_str = "N/A"
        
    print(f"{'Overall MRL Mean':<25} | {s_mrl_str:<15} | {f_mrl_str:<15}")
    
    # 마지막 구간 (후반 20%) 접촉 세기 분석 (Norm: 6D feature의 index 1)
    if len(data["success_6d"]) > 0:
        s_norm_ends = [x[int(len(x)*0.8):, 1].mean() for x in data["success_6d"] if len(x) >= 5]
        s_norm_end_str = f"{np.mean(s_norm_ends):.4f}" if len(s_norm_ends) > 0 else "N/A"
    else:
        s_norm_end_str = "N/A"
        
    if len(data["failure_6d"]) > 0:
        f_norm_ends = [x[int(len(x)*0.8):, 1].mean() for x in data["failure_6d"] if len(x) >= 5]
        f_norm_end_str = f"{np.mean(f_norm_ends):.4f}" if len(f_norm_ends) > 0 else "N/A"
    else:
        f_norm_end_str = "N/A"
        
    print(f"{'End 20% Norm Mean':<25} | {s_norm_end_str:<15} | {f_norm_end_str:<15}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="성공 vs 실패 데이터 비교 시각화 툴")
    parser.add_argument("--dir", action="append", default=None,
                        help="분석할 디렉토리 지정 (예: --dir grasp_new --dir transport_new). 미지정시 둘 다 처리")
    args = parser.parse_args()
    
    # 기본 분석 대상 디렉토리 목록
    target_dirs = args.dir
    if target_dirs is None:
        target_dirs = []
        for d in ["grasp_new", "transport_new"]:
            if os.path.exists(d) and os.path.isdir(d):
                target_dirs.append(d)
        if not target_dirs:
            print("Error: 'grasp_new' or 'transport_new' directories not found in the current path.")
            print("Please specify the path using: python compare_success_failure.py --dir <path>")
            sys.exit(1)
            
    for t_dir in target_dirs:
        # 데이터 로드
        data = load_data_from_dir(t_dir)
        
        # 폴더 basename 획득 (출력 파일명용)
        base = os.path.basename(os.path.normpath(t_dir))
        
        # 1) 6D 비교 플롯 생성
        out_6d = f"{base}_compare_6d.png"
        plot_6d_comparison(data, base, out_6d)
        
        # 2) 48D Taxel Norm 비교 플롯 생성
        out_48d_taxels = f"{base}_compare_48d_taxels.png"
        plot_48d_taxel_norms(data, base, out_48d_taxels)
        
        # 3) 48D 축별 평균 플롯 생성
        out_48d_avg = f"{base}_compare_48d_avg.png"
        plot_48d_axis_averages(data, base, out_48d_avg)
        
        # 4) 통계 요약 텍스트 출력
        print_summary_stats(data, base)

if __name__ == "__main__":
    main()
