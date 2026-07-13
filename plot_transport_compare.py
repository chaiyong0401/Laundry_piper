"""
transport success vs failure 시각적 비교 플롯
Usage: python plot_transport_compare.py
"""
import numpy as np
import glob
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/home/dyros/mcy_ws/transport_new"
PREFIX   = "transport"
OUT_PATH = f"{DATA_DIR}/transport_success_vs_failure.png"

# ── 데이터 로드 ──────────────────────────────────────────
def load_raw(label):
    files = sorted(glob.glob(f"{DATA_DIR}/{PREFIX}_*_{label}_raw.npy"))
    return [np.load(f).astype(np.float32) for f in files], files

def compute_mrl(arr):
    """(T, 48) → (T,) MRL: 16 taxel별 norm 평균"""
    vecs = arr.reshape(arr.shape[0], 16, 3)
    return np.linalg.norm(vecs, axis=2).mean(axis=1)

succ_arrs, succ_files = load_raw("success")
fail_arrs, fail_files = load_raw("failure")
print(f"success: {len(succ_arrs)}개  failure: {len(fail_arrs)}개")

# ── 에피소드별 통계 ──────────────────────────────────────
def ep_stats(arrs):
    stats = []
    for a in arrs:
        mrl = compute_mrl(a)
        stats.append({
            "mrl_mean": mrl.mean(),
            "mrl_std":  mrl.std(),
            "mrl_min":  mrl.min(),
            "mrl_max":  mrl.max(),
            "length":   len(a),
        })
    return stats

s_stats = ep_stats(succ_arrs)
f_stats = ep_stats(fail_arrs)

# ── 플롯 ─────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Transport: Success vs Failure 비교", fontsize=14, fontweight="bold")

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# ── 1) MRL 시계열 (정규화 길이) ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_title("MRL 시계열 (에피소드별, 정규화 길이)")
for a in succ_arrs:
    mrl = compute_mrl(a)
    t = np.linspace(0, 1, len(mrl))
    ax1.plot(t, mrl, color="steelblue", alpha=0.25, lw=0.8)
for a in fail_arrs:
    mrl = compute_mrl(a)
    t = np.linspace(0, 1, len(mrl))
    ax1.plot(t, mrl, color="darkorange", alpha=0.4, lw=0.8)
from matplotlib.lines import Line2D
ax1.legend(handles=[
    Line2D([0],[0], color="steelblue",  lw=2, label=f"success (n={len(succ_arrs)})"),
    Line2D([0],[0], color="darkorange", lw=2, label=f"failure (n={len(fail_arrs)})"),
], fontsize=9)
ax1.set_xlabel("정규화 시간"); ax1.set_ylabel("MRL")
ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)

# ── 2) MRL 분포 (바이올린) ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title("MRL 분포 (전체 프레임)")
s_mrl_all = np.concatenate([compute_mrl(a) for a in succ_arrs])
f_mrl_all = np.concatenate([compute_mrl(a) for a in fail_arrs])
vp = ax2.violinplot([s_mrl_all, f_mrl_all], positions=[1, 2], showmedians=True)
vp["bodies"][0].set_facecolor("steelblue"); vp["bodies"][0].set_alpha(0.6)
vp["bodies"][1].set_facecolor("darkorange"); vp["bodies"][1].set_alpha(0.6)
ax2.set_xticks([1, 2]); ax2.set_xticklabels(["success", "failure"])
ax2.set_ylabel("MRL"); ax2.grid(True, axis="y", alpha=0.3)

# ── 3) 에피소드별 mrl_mean vs mrl_std 산점도 ─────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_title("에피소드별 MRL mean vs std  (노이즈 에피소드 식별)")
for i, st in enumerate(s_stats):
    ax3.scatter(st["mrl_mean"], st["mrl_std"], color="steelblue", alpha=0.6, s=40)
for i, st in enumerate(f_stats):
    ax3.scatter(st["mrl_mean"], st["mrl_std"], color="darkorange", alpha=0.8,
                s=60, marker="^")
# noisy success 표시 (mrl_std > 0.3)
for i, st in enumerate(s_stats):
    if st["mrl_std"] > 0.3:
        ax3.scatter(st["mrl_mean"], st["mrl_std"], color="red", s=80,
                    zorder=5, marker="x", linewidths=2)
ax3.axhline(0.3, color="red", lw=1, linestyle="--", alpha=0.6, label="mrl_std=0.3 (노이즈 기준)")
ax3.legend(handles=[
    Line2D([0],[0], color="steelblue",  marker="o", lw=0, markersize=7, label="success"),
    Line2D([0],[0], color="darkorange", marker="^", lw=0, markersize=8, label="failure"),
    Line2D([0],[0], color="red",        marker="x", lw=0, markersize=8,
           markeredgewidth=2, label="noisy success (std>0.3)"),
    Line2D([0],[0], color="red", lw=1, linestyle="--", label="std=0.3 기준선"),
], fontsize=8)
ax3.set_xlabel("MRL mean"); ax3.set_ylabel("MRL std"); ax3.grid(True, alpha=0.3)

# ── 4) 에피소드 길이 분포 ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_title("에피소드 길이 분포")
s_lens = [st["length"] for st in s_stats]
f_lens = [st["length"] for st in f_stats]
bins = np.linspace(100, 1000, 20)
ax4.hist(s_lens, bins=bins, color="steelblue", alpha=0.6, label="success")
ax4.hist(f_lens, bins=bins, color="darkorange", alpha=0.6, label="failure")
ax4.legend(fontsize=8); ax4.set_xlabel("프레임 수"); ax4.set_ylabel("에피소드 수")
ax4.grid(True, axis="y", alpha=0.3)

# ── 5) MRL 마지막 30% 구간 비교 (drop 감지) ──────────────
ax5 = fig.add_subplot(gs[2, :2])
ax5.set_title("에피소드 후반 30% MRL (drop 발생 구간)")
for a in succ_arrs:
    mrl = compute_mrl(a)
    end = mrl[int(len(mrl)*0.7):]
    t = np.linspace(0.7, 1.0, len(end))
    ax5.plot(t, end, color="steelblue", alpha=0.25, lw=0.8)
for a in fail_arrs:
    mrl = compute_mrl(a)
    end = mrl[int(len(mrl)*0.7):]
    t = np.linspace(0.7, 1.0, len(end))
    ax5.plot(t, end, color="darkorange", alpha=0.5, lw=1.0)
ax5.set_xlabel("정규화 시간 (후반 30%)"); ax5.set_ylabel("MRL")
ax5.set_ylim(0, 1.05); ax5.grid(True, alpha=0.3)
ax5.legend(handles=[
    Line2D([0],[0], color="steelblue",  lw=2, label="success"),
    Line2D([0],[0], color="darkorange", lw=2, label="failure"),
], fontsize=9)

# ── 6) MRL delta 분포 비교 ───────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_title("MRL delta 분포 (채널 B 신호)")
def mrl_deltas(arrs, smooth=3):
    ds = []
    for a in arrs:
        mrl = compute_mrl(a)
        for t in range(smooth, len(mrl)):
            ref = mrl[t-smooth:t].mean()
            ds.append(abs(mrl[t] - ref))
    return np.array(ds)

s_deltas = mrl_deltas(succ_arrs)
f_deltas = mrl_deltas(fail_arrs)
threshold_b = 0.027072
bins_d = np.linspace(0, 0.15, 40)
ax6.hist(s_deltas, bins=bins_d, color="steelblue", alpha=0.6, density=True, label="success")
ax6.hist(f_deltas, bins=bins_d, color="darkorange", alpha=0.6, density=True, label="failure")
ax6.axvline(threshold_b, color="red", lw=1.5, linestyle="--", label=f"thr_B={threshold_b:.3f}")
ax6.legend(fontsize=8); ax6.set_xlabel("|MRL delta|"); ax6.set_ylabel("밀도")
ax6.grid(True, axis="y", alpha=0.3)

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"[저장] {OUT_PATH}")

# ── 텍스트 요약 ──────────────────────────────────────────
print(f"\n{'='*55}")
print(f"{'':20s} {'success':>15} {'failure':>15}")
print(f"{'='*55}")
print(f"{'에피소드 수':<20} {len(succ_arrs):>15} {len(fail_arrs):>15}")
print(f"{'MRL mean (전체)':<20} {s_mrl_all.mean():>15.4f} {f_mrl_all.mean():>15.4f}")
print(f"{'MRL std (전체)':<20} {s_mrl_all.std():>15.4f} {f_mrl_all.std():>15.4f}")
print(f"{'MRL delta mean':<20} {s_deltas.mean():>15.4f} {f_deltas.mean():>15.4f}")
print(f"{'MRL delta 99pct':<20} {np.percentile(s_deltas,99):>15.4f} {np.percentile(f_deltas,99):>15.4f}")
n_noisy = sum(1 for st in s_stats if st["mrl_std"] > 0.3)
print(f"{'noisy success (>0.3)':<20} {n_noisy:>15} {'':>15}")
print(f"{'thr_B 초과 비율':<20} {(s_deltas>threshold_b).mean()*100:>14.1f}% {(f_deltas>threshold_b).mean()*100:>14.1f}%")
