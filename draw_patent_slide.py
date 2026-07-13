"""
특허 제출용 촉각 센서 핵심 아이디어 슬라이드 (단일 페이지)
Dual-Channel Tactile Anomaly Detection

python draw_patent_slide.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import os

OUT_DIR = "/home/dyros/mcy_ws/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 0.6,
})

# ── 색상 팔레트 ──────────────────────────────────────────────────────
C = dict(
    sensor   = '#D4680A',   # orange
    feat     = '#B7950B',   # gold
    win      = '#7D6608',   # dark gold
    enc      = '#1A5276',   # dark blue
    dec      = '#1F618D',   # blue
    chA      = '#1ABC9C',   # teal
    chB      = '#C0392B',   # red
    or_gate  = '#2C3E50',   # dark
    abort    = '#922B21',   # dark red
    normal   = '#1E8449',   # green
    claim    = '#4A235A',   # purple
    arrow    = '#2C3E50',
    bg       = '#FAFAFA',
    panel_bg = '#EAF2FF',
    chB_bg   = '#FDEDEC',
    chA_bg   = '#E8F8F5',
)


def rbox(ax, x, y, w, h, color, text, fs=8.5, tc='white', bold=False,
         radius=0.02, alpha=1.0, ec=None, ls='-', lw=1.2, ha='center', va='center'):
    ec = ec or color
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f'round,pad={radius}',
                           facecolor=color, edgecolor=ec,
                           linewidth=lw, linestyle=ls, alpha=alpha, zorder=3)
    ax.add_patch(patch)
    fw = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha=ha, va=va,
            fontsize=fs, color=tc, fontweight=fw, zorder=4,
            linespacing=1.4)


def arr(ax, x0, y0, x1, y1, color=C['arrow'], lw=1.6, style='->', head=12):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, mutation_scale=head), zorder=5)


def darr(ax, x0, y0, x1, y1, color=C['arrow'], lw=1.4):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=10,
                                linestyle='dashed'), zorder=5)


# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 10.125))  # 16:9 ratio
ax.set_xlim(0, 18); ax.set_ylim(0, 10.125)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── 배경 헤더 바 ─────────────────────────────────────────────────────
header = FancyBboxPatch((0, 9.3), 18, 0.825,
                        boxstyle='square,pad=0', facecolor=C['or_gate'],
                        edgecolor='none', zorder=1)
ax.add_patch(header)
ax.text(9, 9.72,
        'Dual-Channel Tactile Anomaly Detection for Robotic Laundry Grasping',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color='white', zorder=2)
ax.text(16.5, 9.72, '[Patent Key Idea]',
        ha='center', va='center', fontsize=8.5, color='#A9CCE3', zorder=2)

# ════════════════════════════════════════════════════════════════════
# SECTION A: 왼쪽 — Sensor Input Pipeline  (x: 0.3 ~ 3.3)
# ════════════════════════════════════════════════════════════════════
ax.text(1.8, 9.1, 'Tactile Input & Feature', ha='center', fontsize=8.5,
        fontweight='bold', color=C['sensor'], zorder=2)

rbox(ax, 0.3, 7.75, 3.0, 1.0, C['sensor'],
     'XELA uSkin Sensor\n16 taxels × 3-axis (x, y, z)\n→ 48-D raw force signal',
     fs=8, bold=True)

arr(ax, 1.8, 7.75, 1.8, 7.2)

rbox(ax, 0.3, 6.2, 3.0, 0.95, C['feat'],
     '6D Feature Extraction\n'
     'MRL · norm · max_norm\n'
     'mean_unit_x/y/z',
     fs=7.8)

# MRL 수식 설명 박스
rbox(ax, 0.1, 4.9, 3.4, 1.1, C['feat'],
     'MRL  =  ‖ mean( unit_force_vectors ) ‖  ∈ [0, 1]\n'
     '→  High MRL: consistent contact direction (firm grip)\n'
     '→  Low MRL : scattered / lost contact (slip / drop)',
     fs=7.5, tc='white', radius=0.015, alpha=0.9,
     ec='#F39C12', lw=1.5)
arr(ax, 1.8, 6.2, 1.8, 6.0)

arr(ax, 1.8, 4.9, 1.8, 4.5)

rbox(ax, 0.3, 3.5, 3.0, 0.9, C['win'],
     'Sliding Window  (T = 50 frames, stride = 1)\n'
     '+ StandardScaler → normalized sequence',
     fs=7.8)

# 분기 화살표 → Ch-A, Ch-B
ax.plot([1.8, 1.8, 5.2], [3.5, 3.1, 3.1], color=C['chA'], lw=1.8, zorder=3)
ax.annotate('', xy=(5.2, 3.1), xytext=(4.8, 3.1),
            arrowprops=dict(arrowstyle='->', color=C['chA'], lw=1.8,
                            mutation_scale=12), zorder=5)

ax.plot([1.8, 1.8, 5.2], [3.5, 2.1, 2.1], color=C['chB'], lw=1.8, zorder=3)
ax.annotate('', xy=(5.2, 2.1), xytext=(4.8, 2.1),
            arrowprops=dict(arrowstyle='->', color=C['chB'], lw=1.8,
                            mutation_scale=12), zorder=5)

ax.text(3.8, 3.3, 'Ch-A', fontsize=7.5, color=C['chA'], fontweight='bold')
ax.text(3.8, 2.3, 'Ch-B', fontsize=7.5, color=C['chB'], fontweight='bold')

# ════════════════════════════════════════════════════════════════════
# SECTION B: 중앙 — Channel A  (x: 5.2 ~ 10.8)
# ════════════════════════════════════════════════════════════════════

# Ch-A 패널 배경
chA_panel = FancyBboxPatch((5.1, 2.55), 5.9, 5.5,
                            boxstyle='round,pad=0.05',
                            facecolor=C['chA_bg'], edgecolor=C['chA'],
                            linewidth=1.5, linestyle='--', zorder=1)
ax.add_patch(chA_panel)
ax.text(8.05, 8.15, 'Channel A  —  Slow Detector  (Sequence-level)',
        ha='center', fontsize=9.5, fontweight='bold', color=C['chA'], zorder=2)
ax.text(8.05, 7.82, 'GRU Autoencoder (One-class: trained on normal grasps only)',
        ha='center', fontsize=7.8, color='#1A5276', style='italic', zorder=2)

# Encoder
rbox(ax, 5.3, 6.35, 2.4, 1.25, C['enc'],
     'GRU Encoder\nD=6, H=128, L=2\n\nInput: (B, T, 6)\n→  h_n: (2, B, 128)',
     fs=7.5)

# h_n 레이블
ax.text(8.3, 7.0, 'h_n\n(context\nvector)', ha='center', fontsize=7.5,
        color=C['enc'], style='italic', zorder=4)

# Decoder
rbox(ax, 9.0, 6.35, 2.4, 1.25, C['dec'],
     'GRU Decoder\nH=128 → D=6, L=2\n\nrepeat h_n[-1] T×\n→  recon: (B, T, 6)',
     fs=7.5)

arr(ax, 7.7, 7.0, 8.7, 7.0, color=C['enc'], lw=1.6)
arr(ax, 5.3+2.4, 7.0-0.05, 9.0, 6.97, color=C['arrow'], lw=1.6)

# MSE score
rbox(ax, 5.3, 5.0, 2.4, 0.9, '#566573',
     'Reconstruction Error\nMSE( input, recon )\n→  score_A',
     fs=7.5)
arr(ax, 10.2, 7.0, 10.2, 5.55, color=C['arrow'], lw=1.6)
ax.plot([10.2, 10.2, 7.7], [5.55, 5.55, 5.55],
        color=C['arrow'], lw=1.6, zorder=3)
ax.annotate('', xy=(7.7, 5.55), xytext=(8.0, 5.55),
            arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5,
                            mutation_scale=10), zorder=5)

# Threshold A
rbox(ax, 5.3, 3.75, 2.4, 0.95, C['chA'],
     'Threshold θ_A\n(consecutive k windows)\n→  anomaly_A  [True/False]',
     fs=7.5)
arr(ax, 6.5, 5.0, 6.5, 4.7, color=C['chA'], lw=1.6)

# 학습 방식 노트
rbox(ax, 8.5, 3.75, 2.4, 0.95, '#154360',
     'One-Class Training\nOnly SUCCESS grasps\nθ_A = mean + 3σ (val set)',
     fs=7.5, tc='#AED6F1', ec='#2E86C1', lw=1.2, alpha=0.9)

# Ch-A → OR
arr(ax, 6.5, 3.75, 6.5, 3.15, color=C['chA'], lw=1.7)
ax.plot([6.5, 6.5, 11.3], [3.15, 3.1, 3.1],
        color=C['chA'], lw=1.7, zorder=3)
ax.annotate('', xy=(11.3, 3.1), xytext=(11.0, 3.1),
            arrowprops=dict(arrowstyle='->', color=C['chA'], lw=1.7,
                            mutation_scale=12), zorder=5)

# ════════════════════════════════════════════════════════════════════
# SECTION C: 중앙하단 — Channel B  (x: 5.2 ~ 10.8)
# ════════════════════════════════════════════════════════════════════

chB_panel = FancyBboxPatch((5.1, 0.6), 5.9, 1.9,
                            boxstyle='round,pad=0.05',
                            facecolor=C['chB_bg'], edgecolor=C['chB'],
                            linewidth=1.5, linestyle='--', zorder=1)
ax.add_patch(chB_panel)
ax.text(8.05, 2.52, 'Channel B  —  Fast Detector  (Frame-level)',
        ha='center', fontsize=9.5, fontweight='bold', color=C['chB'], zorder=2)

rbox(ax, 5.3, 1.4, 2.3, 0.85, '#922B21',
     'MRL Extraction\n(frame-wise, per timestep)',
     fs=7.8)

rbox(ax, 7.8, 1.4, 2.3, 0.85, C['chB'],
     '|ΔMRL_t|  =  |MRL_t  −  MRL_ref|\n(smoothed  w = 3)',
     fs=7.6)
arr(ax, 7.6, 1.82, 7.8, 1.82, color=C['chB'], lw=1.6)

rbox(ax, 7.8, 0.65, 2.3, 0.7, C['chB'],
     'Threshold θ_B  (auto-calibrated)\n→  anomaly_B  [True/False]',
     fs=7.5)
arr(ax, 8.95, 1.4, 8.95, 1.35, color=C['chB'], lw=1.6)

# Ch-B → OR
arr(ax, 8.95, 0.65, 8.95, 0.45, color=C['chB'], lw=1.7)
ax.plot([8.95, 11.5, 11.5], [0.45, 0.45, 2.0],
        color=C['chB'], lw=1.7, zorder=3)
ax.annotate('', xy=(11.5, 2.0), xytext=(11.5, 2.3),
            arrowprops=dict(arrowstyle='->', color=C['chB'], lw=1.7,
                            mutation_scale=12), zorder=5)
ax.plot([11.3, 11.5], [2.3, 2.3],
        color=C['chB'], lw=1.7, zorder=3)

# ════════════════════════════════════════════════════════════════════
# SECTION D: 오른쪽 — OR Gate + Decision  (x: 11.3 ~ 14.5)
# ════════════════════════════════════════════════════════════════════

# OR Gate 박스
rbox(ax, 11.3, 2.05, 2.0, 1.4, C['or_gate'],
     'OR\nGate\n\nanomaly_A\nOR\nanomaly_B',
     fs=8.5, bold=True)

# ABORT 경로
rbox(ax, 13.7, 3.05, 3.0, 1.1, C['abort'],
     'ABORT\nHalt transport → Drop Recovery\n(Re-localize & Re-grasp)',
     fs=8, bold=True)
arr(ax, 13.3, 3.25, 13.7, 3.55, color=C['abort'], lw=2.0)

# NORMAL 경로
rbox(ax, 13.7, 1.7, 3.0, 0.9, C['normal'],
     'NORMAL  —  Continue Transport',
     fs=8.5, bold=False)
darr(ax, 13.3, 2.75, 13.7, 2.15, color=C['normal'], lw=1.8)
ax.text(13.35, 2.47, 'no anomaly', fontsize=7, color=C['normal'],
        ha='center', style='italic')

# z_aligned 보조 설명
rbox(ax, 13.7, 0.65, 3.0, 0.85, '#566573',
     'Grasp Success Check  (z_aligned)\nCumulative ΔZ after retreat > θ_z\n→  confirms object held',
     fs=7.5, tc='#D5D8DC', ec='#ABB2B9', lw=1.0)

# ════════════════════════════════════════════════════════════════════
# SECTION E: 하단 — 핵심 특허 클레임  (x: 0.2 ~ 18)
# ════════════════════════════════════════════════════════════════════

claims = [
    ("① MRL Feature",
     "Direction-consistency of 16-taxel\nforce vectors; robust to spatial variation"),
    ("② One-Class GRU-AE  (Ch-A)",
     "Sequence-level reconstruction error\ndetects gradual slip / force attenuation"),
    ("③ MRL Delta Detector  (Ch-B)",
     "Frame-level |ΔMRL| catches instantaneous\ndrop events (complement to Ch-A)"),
    ("④ OR-Gate Fusion",
     "Complementary coverage: low false-negative\nrate without sensor redundancy"),
]

# 하단 전체 배경
full_claim_bg = FancyBboxPatch((0.2, 0.05), 17.6, 1.45,
                               boxstyle='round,pad=0.03',
                               facecolor='#F4ECF7', edgecolor=C['claim'],
                               linewidth=1.5, zorder=1)
ax.add_patch(full_claim_bg)

ax.text(9.0, 1.42, 'Key Patent Claims',
        ha='center', fontsize=9.5, fontweight='bold', color=C['claim'], zorder=3)

for i, (title, body) in enumerate(claims):
    bx = 0.4 + i * 4.4
    by = 0.08
    rbox(ax, bx, by, 4.2, 1.1, C['claim'],
         f'{title}\n{body}',
         fs=7.6, tc='white', radius=0.02, alpha=0.88)

# ════════════════════════════════════════════════════════════════════
# 구분선 (입력 파이프라인 / 채널 사이)
ax.plot([3.45, 3.45], [0.05, 9.25], color='#BDC3C7', lw=0.8,
        linestyle=':', zorder=2)
ax.plot([11.2, 11.2], [0.05, 9.25], color='#BDC3C7', lw=0.8,
        linestyle=':', zorder=2)
ax.plot([13.6, 13.6], [0.05, 9.25], color='#BDC3C7', lw=0.8,
        linestyle=':', zorder=2)

ax.text(1.8, 9.2, 'INPUT', ha='center', fontsize=7, color='#7F8C8D')
ax.text(8.05, 9.2, 'DETECTION  CHANNELS', ha='center', fontsize=7, color='#7F8C8D')
ax.text(12.4, 9.2, 'DECISION', ha='center', fontsize=7, color='#7F8C8D')

# ════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.2)
path = os.path.join(OUT_DIR, 'patent_slide_tactile.pdf')
plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] {path}")
print(f"[OK] {path.replace('.pdf', '.png')}")
