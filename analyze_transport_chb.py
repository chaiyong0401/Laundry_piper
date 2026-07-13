"""
Transport Ch-B 분석: 성공 vs 실패 에피소드의 MRL / MRL delta 비교
"""
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

DATA_DIR = "/home/dyros/mcy_ws/transport_new"
SMOOTH_WINDOW = 3
WARMUP = 30
THRESHOLD_B = 0.212874  # 현재 calibrated 값


def compute_mrl(frame: np.ndarray) -> float:
    """48D raw → MRL (Mean Ridge Load)"""
    vecs = frame[:48].reshape(16, 3).astype(float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    unit = vecs / norms
    return float(np.linalg.norm(unit.mean(axis=0)))


def compute_force_mag(frame: np.ndarray) -> float:
    """48D raw → mean force magnitude (taxel norm 평균)"""
    vecs = frame[:48].reshape(16, 3).astype(float)
    return float(np.linalg.norm(vecs, axis=1).mean())


def compute_mrl_series(seq, smooth_window=SMOOTH_WINDOW):
    T = len(seq)
    mrl = np.array([compute_mrl(seq[t]) for t in range(T)])
    mag = np.array([compute_force_mag(seq[t]) for t in range(T)])
    deltas = np.full(T, np.nan)
    for t in range(smooth_window, T):
        ref = mrl[t - smooth_window:t].mean()
        deltas[t] = abs(mrl[t] - ref)
    return mrl, deltas, mag


# ─── 데이터 로드 ────────────────────────────────────────────────────────────
success_files = sorted(glob.glob(f"{DATA_DIR}/*success_raw.npy"))
failure_files = sorted(glob.glob(f"{DATA_DIR}/*failure_raw.npy"))

success_data = [np.load(f) for f in success_files]
failure_data = [np.load(f) for f in failure_files]

print(f"Success: {len(success_data)}, Failure: {len(failure_data)}")
print(f"Episode lengths - Success: {[len(s) for s in success_data]}")
print(f"Episode lengths - Failure: {[len(f) for f in failure_data]}")

# ─── MRL / delta 계산 ───────────────────────────────────────────────────────
s_series = [compute_mrl_series(s) for s in success_data]
f_series = [compute_mrl_series(f) for f in failure_data]

# ─── 에피소드별 max delta (warmup 이후) ────────────────────────────────────
s_max_delta = [np.nanmax(d[WARMUP:]) if len(d) > WARMUP else np.nan
               for _, d, _ in s_series]
f_max_delta = [np.nanmax(d[WARMUP:]) if len(d) > WARMUP else np.nan
               for _, d, _ in f_series]

# ─── 전체 delta 분포 ────────────────────────────────────────────────────────
all_s_deltas = np.concatenate([d[WARMUP:][~np.isnan(d[WARMUP:])]
                                for _, d, _ in s_series])
all_f_deltas = np.concatenate([d[WARMUP:][~np.isnan(d[WARMUP:])]
                                for _, d, _ in f_series])

print(f"\n=== MRL Delta 통계 (warmup={WARMUP} 이후) ===")
print(f"{'':15s}  {'Success':>10s}  {'Failure':>10s}")
print(f"{'mean':15s}  {all_s_deltas.mean():10.4f}  {all_f_deltas.mean():10.4f}")
print(f"{'std':15s}  {all_s_deltas.std():10.4f}  {all_f_deltas.std():10.4f}")
print(f"{'99%ile':15s}  {np.percentile(all_s_deltas, 99):10.4f}  {np.percentile(all_f_deltas, 99):10.4f}")
print(f"{'max':15s}  {all_s_deltas.max():10.4f}  {all_f_deltas.max():10.4f}")
print(f"{'> threshold_b':15s}  {(all_s_deltas > THRESHOLD_B).mean():10.3f}  {(all_f_deltas > THRESHOLD_B).mean():10.3f}")
print(f"\nEpisode max delta:")
print(f"  Success max_delta: {np.nanmean(s_max_delta):.4f} ± {np.nanstd(s_max_delta):.4f}")
print(f"  Failure max_delta: {np.nanmean(f_max_delta):.4f} ± {np.nanstd(f_max_delta):.4f}")
print(f"\n  Failure episode max deltas: {[f'{v:.3f}' for v in f_max_delta]}")

# ─── 그래프 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# [0,0~1] MRL 시계열 (성공 3개, 실패 전체 overlay)
ax0 = fig.add_subplot(gs[0, :2])
for i, (mrl, _, _) in enumerate(s_series[:8]):
    ax0.plot(mrl, color='steelblue', alpha=0.3, linewidth=0.8)
for i, (mrl, _, _) in enumerate(f_series):
    ax0.plot(mrl, color='tomato', alpha=0.7, linewidth=1.2,
             label='failure' if i == 0 else '')
ax0.axvline(WARMUP, color='gray', linestyle='--', linewidth=0.8, label=f'warmup={WARMUP}')
ax0.set_title('MRL Time Series (blue=success, red=failure)')
ax0.set_xlabel('Frame')
ax0.set_ylabel('MRL')
ax0.legend()
ax0.set_ylim(0, 1.1)

# [0,2] MRL 분포 box
ax1 = fig.add_subplot(gs[0, 2])
s_mrl_all = np.concatenate([mrl for mrl, _, _ in s_series])
f_mrl_all = np.concatenate([mrl for mrl, _, _ in f_series])
ax1.boxplot([s_mrl_all, f_mrl_all], labels=['Success', 'Failure'])
ax1.set_title('MRL Distribution')
ax1.set_ylabel('MRL')

# [1,0~1] MRL delta 시계열
ax2 = fig.add_subplot(gs[1, :2])
for i, (_, delta, _) in enumerate(s_series[:8]):
    ax2.plot(delta, color='steelblue', alpha=0.3, linewidth=0.8)
for i, (_, delta, _) in enumerate(f_series):
    ax2.plot(delta, color='tomato', alpha=0.7, linewidth=1.2,
             label='failure' if i == 0 else '')
ax2.axhline(THRESHOLD_B, color='darkred', linestyle='--', linewidth=1.2,
            label=f'threshold_b={THRESHOLD_B:.3f}')
ax2.axvline(WARMUP, color='gray', linestyle='--', linewidth=0.8)
ax2.set_title(f'MRL Delta (smooth_window={SMOOTH_WINDOW}) — blue=success, red=failure')
ax2.set_xlabel('Frame')
ax2.set_ylabel('|delta MRL|')
ax2.legend()

# [1,2] delta 분포 box (warmup 이후)
ax3 = fig.add_subplot(gs[1, 2])
ax3.boxplot([all_s_deltas, all_f_deltas], labels=['Success\n(all)', 'Failure\n(all)'])
ax3.axhline(THRESHOLD_B, color='darkred', linestyle='--', linewidth=1.2,
            label=f'thresh={THRESHOLD_B:.3f}')
ax3.set_title('MRL Delta Distribution\n(post-warmup)')
ax3.set_ylabel('|delta MRL|')
ax3.legend()

# [2,0] force magnitude 시계열
ax4 = fig.add_subplot(gs[2, :2])
for i, (_, _, mag) in enumerate(s_series[:8]):
    ax4.plot(mag, color='steelblue', alpha=0.3, linewidth=0.8)
for i, (_, _, mag) in enumerate(f_series):
    ax4.plot(mag, color='tomato', alpha=0.7, linewidth=1.2,
             label='failure' if i == 0 else '')
ax4.axvline(WARMUP, color='gray', linestyle='--', linewidth=0.8)
ax4.set_title('Force Magnitude (mean taxel norm) — blue=success, red=failure')
ax4.set_xlabel('Frame')
ax4.set_ylabel('Mean ||taxel||')
ax4.legend()

# [2,2] episode max delta scatter
ax5 = fig.add_subplot(gs[2, 2])
f_lens = [len(f) for f in failure_data]
ax5.scatter(f_lens, f_max_delta, color='tomato', zorder=3, label='failure')
ax5.scatter([len(s) for s in success_data[:8]], s_max_delta[:8],
            color='steelblue', alpha=0.5, label='success')
ax5.axhline(THRESHOLD_B, color='darkred', linestyle='--', linewidth=1.2,
            label=f'thresh={THRESHOLD_B:.3f}')
ax5.set_title('Max MRL Delta vs Episode Length')
ax5.set_xlabel('Episode Length (frames)')
ax5.set_ylabel('Max delta (post-warmup)')
ax5.legend()

plt.suptitle('Transport Channel B Analysis: Success vs Failure', fontsize=14, fontweight='bold')
plt.savefig('/home/dyros/mcy_ws/transport_chb_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n그래프 저장: /home/dyros/mcy_ws/transport_chb_analysis.png")
