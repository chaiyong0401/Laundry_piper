"""
TinyUNet 아키텍처 다이어그램 생성 스크립트
train_unet_basket.py의 TinyUNet(c_in=6, base=32, use_se=True) 기준
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# ── 색상 정의 ──────────────────────────────────────────────
C_CONV  = '#4A90D9'   # DoubleConv (파랑)
C_DOWN  = '#E8734A'   # Down/MaxPool (주황)
C_UP    = '#5CB85C'   # Up/ConvTranspose (초록)
C_SE    = '#9B59B6'   # SE block (보라)
C_OUT   = '#E74C3C'   # Output conv (빨강)
C_INPUT = '#2C3E50'   # Input (어두운)
C_SKIP  = '#F39C12'   # Skip connection (노랑)
C_ARROW = '#555555'

def box(ax, x, y, w, h, color, label, sublabel='', fontsize=9, alpha=0.9, radius=0.15):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold',
            color='white', zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel,
                ha='center', va='center', fontsize=7.5,
                color='white', alpha=0.9, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2, style='->', zorder=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'),
                zorder=zorder)

def curved_arrow(ax, x1, y1, x2, y2, rad=0.3, color=C_SKIP, lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)

# ── 타이틀 ─────────────────────────────────────────────────
ax.text(9, 11.5, 'TinyUNet Architecture  (c_in=6, base=32, use_se=True)',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#2C3E50')

# ── 인코더 / 디코더 레이블 ──────────────────────────────────
ax.text(1.2, 10.7, 'Encoder', ha='center', fontsize=10, color='#E8734A', fontstyle='italic', fontweight='bold')
ax.text(14.5, 10.7, 'Decoder', ha='center', fontsize=10, color='#5CB85C', fontstyle='italic', fontweight='bold')

# ── 입력 ───────────────────────────────────────────────────
# 6채널: diff(1) + cur(1) + RGB(3) + domain(1)
box(ax, 3.0, 10.2, 2.8, 0.65, C_INPUT,
    'Input  (B, 6, H, W)',
    'diff | cur | R | G | B | domain', fontsize=9)

# ── inc: DoubleConv(6→32) ─────────────────────────────────
arrow(ax, 3.0, 9.87, 3.0, 9.27)
box(ax, 3.0, 9.0, 2.6, 0.55, C_CONV,
    'inc : DoubleConv', 'ch: 6 → 32  [H×W]')

# ── d1: Down(32→64) ───────────────────────────────────────
arrow(ax, 3.0, 8.72, 3.0, 8.07)
box(ax, 3.0, 7.8, 2.6, 0.55, C_DOWN,
    'd1 : MaxPool↓ + DoubleConv', 'ch: 32 → 64  [H/2×W/2]')

# ── d2: Down(64→128) ──────────────────────────────────────
arrow(ax, 3.0, 7.52, 3.0, 6.87)
box(ax, 3.0, 6.6, 2.6, 0.55, C_DOWN,
    'd2 : MaxPool↓ + DoubleConv', 'ch: 64 → 128  [H/4×W/4]')

# ── d3: Down(128→256) — Bottleneck ────────────────────────
arrow(ax, 3.0, 6.32, 3.0, 5.67)
box(ax, 3.0, 5.4, 2.6, 0.55, C_DOWN,
    'd3 : MaxPool↓ + DoubleConv', 'ch: 128 → 256  [H/8×W/8]')

# ── u1: Up(256→128) ───────────────────────────────────────
arrow(ax, 4.3, 5.4, 11.7, 5.4)
box(ax, 13.0, 5.4, 2.6, 0.55, C_UP,
    'u1 : ConvTranspose↑ + cat + DoubleConv', 'ch: 256+128 → 128  [H/4×W/4]')

# ── u2: Up(128→64) ────────────────────────────────────────
arrow(ax, 13.0, 5.68, 13.0, 6.32)
box(ax, 13.0, 6.6, 2.6, 0.55, C_UP,
    'u2 : ConvTranspose↑ + cat + DoubleConv', 'ch: 128+64 → 64  [H/2×W/2]')

# ── u3: Up(64→32) ─────────────────────────────────────────
arrow(ax, 13.0, 6.88, 13.0, 7.52)
box(ax, 13.0, 7.8, 2.6, 0.55, C_UP,
    'u3 : ConvTranspose↑ + cat + DoubleConv', 'ch: 64+32 → 32  [H×W]')

# ── outc: Conv2d(32→1) ────────────────────────────────────
arrow(ax, 13.0, 8.08, 13.0, 8.72)
box(ax, 13.0, 9.0, 2.6, 0.55, C_OUT,
    'outc : Conv2d 1×1', 'ch: 32 → 1  [H×W]')

# ── Sigmoid / Output ──────────────────────────────────────
arrow(ax, 13.0, 9.28, 13.0, 9.87)
box(ax, 13.0, 10.2, 2.6, 0.55, C_INPUT,
    'Output  (B, 1, H, W)',
    'Binary Mask  (Sigmoid)')

# ── Skip Connections ──────────────────────────────────────
# skip1: inc(32) → u3
ax.annotate('', xy=(11.7, 7.8), xytext=(4.3, 9.0),
            arrowprops=dict(arrowstyle='->', color=C_SKIP, lw=2.0,
                            connectionstyle='arc3,rad=-0.25'), zorder=2)
ax.text(8.0, 8.85, 'skip (32ch)', ha='center', fontsize=8,
        color=C_SKIP, fontweight='bold')

# skip2: d1(64) → u2
ax.annotate('', xy=(11.7, 6.6), xytext=(4.3, 7.8),
            arrowprops=dict(arrowstyle='->', color=C_SKIP, lw=2.0,
                            connectionstyle='arc3,rad=-0.2'), zorder=2)
ax.text(8.0, 7.6, 'skip (64ch)', ha='center', fontsize=8,
        color=C_SKIP, fontweight='bold')

# skip3: d2(128) → u1
ax.annotate('', xy=(11.7, 5.4), xytext=(4.3, 6.6),
            arrowprops=dict(arrowstyle='->', color=C_SKIP, lw=2.0,
                            connectionstyle='arc3,rad=-0.15'), zorder=2)
ax.text(8.0, 5.85, 'skip (128ch)', ha='center', fontsize=8,
        color=C_SKIP, fontweight='bold')

# ── SE Block 상세 (오른쪽 하단) ──────────────────────────────
se_x, se_y0 = 9.0, 3.8
ax.text(se_x, se_y0 + 0.85, 'SE Block (Squeeze-and-Excitation)',
        ha='center', fontsize=10, fontweight='bold', color='#2C3E50')

se_blocks = [
    (8.1, se_y0, 1.4, 'AdaptiveAvgPool2d\n(B,C,1,1)', '#7F8C8D'),
    (9.55, se_y0, 1.4, 'Linear\nC → C//r', C_SE),
    (11.0, se_y0, 1.0, 'ReLU', '#27AE60'),
    (12.2, se_y0, 1.4, 'Linear\nC//r → C', C_SE),
    (13.65, se_y0, 1.0, 'Sigmoid', '#E74C3C'),
    (15.1, se_y0, 1.4, '× Channel\nScale', '#F39C12'),
]
for (sx, sy, sw, slabel, sc) in se_blocks:
    rect = FancyBboxPatch((sx - sw/2, sy - 0.28), sw, 0.56,
                          boxstyle="round,pad=0.04,rounding_size=0.1",
                          facecolor=sc, edgecolor='white', linewidth=1.2, alpha=0.88, zorder=3)
    ax.add_patch(rect)
    ax.text(sx, sy, slabel, ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='white', zorder=4)

for i in range(len(se_blocks) - 1):
    x_from = se_blocks[i][0] + se_blocks[i][2]/2
    x_to   = se_blocks[i+1][0] - se_blocks[i+1][2]/2
    ax.annotate('', xy=(x_to, se_y0), xytext=(x_from, se_y0),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5), zorder=2)

# SE 입력 레이블
ax.text(7.0, se_y0, 'Feature\nMap x', ha='center', va='center', fontsize=8, color='#2C3E50')
ax.annotate('', xy=(7.4, se_y0), xytext=(7.7, se_y0),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))

# SE 괄호
rect_se_bg = FancyBboxPatch((6.8, se_y0 - 0.5), 10.0, 1.0,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=1.0,
                             alpha=0.4, zorder=1)
ax.add_patch(rect_se_bg)

# ── DoubleConv 상세 (왼쪽 하단) ──────────────────────────────
dc_x0, dc_y0 = 1.0, 3.8
ax.text(dc_x0 + 2.2, dc_y0 + 0.85, 'DoubleConv Block',
        ha='center', fontsize=10, fontweight='bold', color='#2C3E50')

dc_blocks = [
    (dc_x0,       dc_y0, 1.2, 'Conv2d\n3×3', C_CONV),
    (dc_x0 + 1.5, dc_y0, 1.2, 'BatchNorm\n+ ReLU', '#3498DB'),
    (dc_x0 + 3.0, dc_y0, 1.2, 'Conv2d\n3×3', C_CONV),
    (dc_x0 + 4.5, dc_y0, 1.2, 'BatchNorm\n+ ReLU', '#3498DB'),
    (dc_x0 + 6.0, dc_y0, 1.0, 'SE\nBlock', C_SE),
]
for (dx, dy, dw, dlabel, dc) in dc_blocks:
    rect = FancyBboxPatch((dx - dw/2, dy - 0.28), dw, 0.56,
                          boxstyle="round,pad=0.04,rounding_size=0.1",
                          facecolor=dc, edgecolor='white', linewidth=1.2, alpha=0.88, zorder=3)
    ax.add_patch(rect)
    ax.text(dx, dy, dlabel, ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='white', zorder=4)

for i in range(len(dc_blocks) - 1):
    x_from = dc_blocks[i][0] + dc_blocks[i][2]/2
    x_to   = dc_blocks[i+1][0] - dc_blocks[i+1][2]/2
    ax.annotate('', xy=(x_to, dc_y0), xytext=(x_from, dc_y0),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5), zorder=2)

rect_dc_bg = FancyBboxPatch((dc_x0 - 0.8, dc_y0 - 0.5), 7.5, 1.0,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=1.0,
                             alpha=0.4, zorder=1)
ax.add_patch(rect_dc_bg)

# ── 범례 ───────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_CONV,  label='DoubleConv'),
    mpatches.Patch(color=C_DOWN,  label='Down (MaxPool + DoubleConv)'),
    mpatches.Patch(color=C_UP,    label='Up (ConvTranspose + cat + DoubleConv)'),
    mpatches.Patch(color=C_SE,    label='SE Block'),
    mpatches.Patch(color=C_OUT,   label='Output Conv 1×1'),
    mpatches.Patch(color=C_SKIP,  label='Skip Connection'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=3,
          framealpha=0.85, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout()
out_path = '/home/dyros/mcy_ws/Laundry_decision/run_bf/tinyunet_architecture.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.show()
