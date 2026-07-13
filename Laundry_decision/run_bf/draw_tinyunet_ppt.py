"""
TinyUNet 네트워크 구조 다이어그램 - PPT용 (수정3)
"""
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(26, 13))
ax.set_xlim(0, 26)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

C_INPUT  = '#1A5276'
C_ENC    = '#1E8449'
C_BOTTLE = '#7D6608'
C_DEC    = '#922B21'
C_OUT    = '#5B2C6F'
C_SKIP   = '#2E86C1'
C_SE     = '#E67E22'

BW = 2.8    # 블록 너비
BH = 1.45   # 블록 높이 (3줄)

def block(ax, cx, cy, color, name, desc, ch, alpha=0.88):
    """name(크게) / desc / ch 세 줄"""
    rect = FancyBboxPatch((cx - BW/2, cy - BH/2), BW, BH,
                          boxstyle="round,pad=0.07,rounding_size=0.12",
                          facecolor=color, edgecolor='white', linewidth=2.2,
                          alpha=alpha, zorder=3, clip_on=False)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.36, name, ha='center', va='center',
            fontsize=15, fontweight='bold', color='white', zorder=4, clip_on=False)
    ax.text(cx, cy + 0.00, desc, ha='center', va='center',
            fontsize=10.5, color='#EAECEE', zorder=4, clip_on=False)
    ax.text(cx, cy - 0.36, ch,   ha='center', va='center',
            fontsize=10.5, color='#FDFEFE', zorder=4, clip_on=False)

def block2(ax, cx, cy, color, name, ch, alpha=0.88):
    """name(크게) / ch 두 줄 (input, output용)"""
    rect = FancyBboxPatch((cx - BW/2, cy - BH/2), BW, BH,
                          boxstyle="round,pad=0.07,rounding_size=0.12",
                          facecolor=color, edgecolor='white', linewidth=2.2,
                          alpha=alpha, zorder=3, clip_on=False)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.18, name, ha='center', va='center',
            fontsize=15, fontweight='bold', color='white', zorder=4, clip_on=False)
    ax.text(cx, cy - 0.22, ch,   ha='center', va='center',
            fontsize=10.5, color='#EAECEE', zorder=4, clip_on=False)

def l_arrow_down(ax, sx, sy, dx, dy, color='#444', lw=2.2):
    """인코더용 ㄴ자: src 하단 → 수직↓ → 수평→ → dst 좌측 중심 진입"""
    ax.plot([sx, sx, dx - BW/2],
            [sy - BH/2, dy, dy],
            color=color, lw=lw,
            solid_capstyle='round', solid_joinstyle='round', zorder=5)
    ax.annotate('', xy=(dx - BW/2, dy),
                xytext=(dx - BW/2 - 0.001, dy),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=6)

def l_arrow_up(ax, sx, sy, dx, dy, color='#444', lw=2.2):
    """디코더용: src 우측 → 수평→ → 수직↑ → dst 하단 중심 진입"""
    ax.plot([sx + BW/2, dx, dx],
            [sy,         sy, dy - BH/2],
            color=color, lw=lw,
            solid_capstyle='round', solid_joinstyle='round', zorder=5)
    # 화살표 머리: 위쪽 방향 (dst 하단 중심으로 진입)
    ax.annotate('', xy=(dx, dy - BH/2),
                xytext=(dx, dy - BH/2 - 0.001),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=6)

# ════════════════════════════════════════════════════
# 블록 위치 (U자 배치)
# ════════════════════════════════════════════════════
pos = {
    'input':  ( 2.2, 10.8),
    'inc':    ( 2.2,  9.0),
    'd1':     ( 5.5,  7.6),
    'd2':     ( 8.8,  6.2),
    'd3':     (12.1,  4.8),
    'u1':     (15.4,  6.2),
    'u2':     (18.7,  7.6),
    'u3':     (22.0,  9.0),
    'output': (22.0, 10.8),
}

# ── 블록
block2(ax, *pos['input'],  C_INPUT,
       'Input', '6 × H × W')

block(ax,  *pos['inc'],    C_ENC,
      'inc', 'DoubleConv + SE', '6 → 32')

block(ax,  *pos['d1'],     C_ENC,
      'd1', 'DoubleConv + SE', '32 → 64')

block(ax,  *pos['d2'],     C_ENC,
      'd2', 'DoubleConv + SE', '64 → 128')

block(ax,  *pos['d3'],     C_BOTTLE,
      'd3  (Bottleneck)', 'DoubleConv + SE', '128 → 256', alpha=0.93)

block(ax,  *pos['u1'],     C_DEC,
      'u1', 'ConvTranspose + DoubleConv + SE', '(256+128) → 128')

block(ax,  *pos['u2'],     C_DEC,
      'u2', 'ConvTranspose + DoubleConv + SE', '(128+64) → 64')

block(ax,  *pos['u3'],     C_DEC,
      'u3', 'ConvTranspose + DoubleConv + SE', '(64+32) → 32')

block2(ax, *pos['output'], C_OUT,
       'outc', 'Conv 1×1  |  32 → 1  (Binary Mask)')

# ════════════════════════════════════════════════════
# 순방향 화살표
# ════════════════════════════════════════════════════
sx0, sy0 = pos['input']
sx1, sy1 = pos['inc']
# input → inc (수직)
ax.annotate('', xy=(sx1, sy1 + BH/2),
            xytext=(sx0, sy0 - BH/2),
            arrowprops=dict(arrowstyle='->', color='#444', lw=2.2), zorder=5)

# 인코더 ㄴ자
for src, dst in [('inc','d1'), ('d1','d2'), ('d2','d3')]:
    sx, sy = pos[src]; dx, dy = pos[dst]
    l_arrow_down(ax, sx, sy, dx, dy)

# 디코더 ㄴ자
for src, dst in [('d3','u1'), ('u1','u2'), ('u2','u3')]:
    sx, sy = pos[src]; dx, dy = pos[dst]
    l_arrow_up(ax, sx, sy, dx, dy)

# u3 → output (수직)
sx3, sy3 = pos['u3']
sxo, syo = pos['output']
ax.annotate('', xy=(sxo, syo - BH/2),
            xytext=(sx3, sy3 + BH/2),
            arrowprops=dict(arrowstyle='->', color='#444', lw=2.2), zorder=5)

# ════════════════════════════════════════════════════
# Skip Connections (점선 호)
# ════════════════════════════════════════════════════
for src, dst in [('inc','u3'), ('d1','u2'), ('d2','u1')]:
    sx, sy = pos[src]; dx, dy = pos[dst]
    ax.annotate('', xy=(dx - BW/2, dy),
                xytext=(sx + BW/2, sy),
                arrowprops=dict(arrowstyle='->', color=C_SKIP, lw=2.0,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'), zorder=4)

# 라벨: inc↔u3 스킵 연결선(y=9.0) 위 중앙에 배치
ax.text(12.1, 9.55, 'Skip Connections  (Concatenate)',
        ha='center', va='center', fontsize=14, color=C_SKIP, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  edgecolor=C_SKIP, alpha=0.95, linewidth=2.2), zorder=6)

# ════════════════════════════════════════════════════
# SE Block 인셋 — 하단 독립
# ════════════════════════════════════════════════════
IX, IY = 13.0, 1.8
IW, IH = 25.0, 2.6

bg = FancyBboxPatch((IX - IW/2, IY - IH/2), IW, IH,
                    boxstyle="round,pad=0.12",
                    facecolor='#F0F2F5', edgecolor=C_SE,
                    linewidth=2.0, alpha=0.97, zorder=2)
ax.add_patch(bg)
ax.text(IX, IY + IH/2 - 0.32,
        'SE Block  (Squeeze-and-Excitation)',
        ha='center', fontsize=11.5, fontweight='bold', color=C_SE, zorder=5)

SBW, SBH = 3.2, 0.8
se_items = [
    (IX - 9.0, IY - 0.15, '#566573', 'Feature Map',      '(C × H × W)'),
    (IX - 5.2, IY - 0.15, '#566573', 'Global Avg Pool',   'C × 1 × 1'),
    (IX - 0.8, IY - 0.15, C_SE,     'FC → ReLU → FC',    '→ Sigmoid'),
    (IX + 4.0, IY - 0.15, C_SE,     '채널 가중치',          'w ∈ (0,1)^C'),
    (IX + 8.5, IY - 0.15, '#566573', 'Recalibrated Map',  'Feature × w'),
]
for (x, y, col, l1, l2) in se_items:
    r = FancyBboxPatch((x - SBW/2, y - SBH/2), SBW, SBH,
                       boxstyle="round,pad=0.06,rounding_size=0.1",
                       facecolor=col, edgecolor='white', linewidth=1.4,
                       alpha=0.85, zorder=4)
    ax.add_patch(r)
    ax.text(x, y + 0.14, l1, ha='center', va='center',
            fontsize=10.0, fontweight='bold', color='white', zorder=5)
    ax.text(x, y - 0.19, l2, ha='center', va='center',
            fontsize=9.0, color='#EAECEE', zorder=5)

se_xs = [x for (x,y,c,l1,l2) in se_items]
for i in range(len(se_xs) - 1):
    x1 = se_xs[i] + SBW/2
    x2 = se_xs[i+1] - SBW/2
    y0 = IY - 0.15
    if i == 2:
        ax.text((x1+x2)/2, y0 + 0.05, '×', ha='center', va='center',
                fontsize=22, fontweight='bold', color=C_SE, zorder=6)
        ax.text((x1+x2)/2, y0 - 0.38, 'element-wise', ha='center', va='center',
                fontsize=8.5, color=C_SE, zorder=6)
    else:
        ax.annotate('', xy=(x2, y0), xytext=(x1, y0),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.6), zorder=5)

# ════════════════════════════════════════════════════
# 타이틀 & 범례
# ════════════════════════════════════════════════════
ax.text(12.1, 12.75, 'TinyUNet Architecture',
        ha='center', fontsize=16, fontweight='bold', color='#1C2833')

legend_items = [
    mpatches.Patch(color=C_INPUT,  label='Input'),
    mpatches.Patch(color=C_ENC,    label='Encoder'),
    mpatches.Patch(color=C_BOTTLE, label='Bottleneck'),
    mpatches.Patch(color=C_DEC,    label='Decoder'),
    mpatches.Patch(color=C_OUT,    label='Output'),
    mpatches.Patch(color=C_SKIP,   label='Skip Connection'),
]
ax.legend(handles=legend_items, loc='upper right', ncol=2,
          framealpha=0.92, fontsize=10,
          bbox_to_anchor=(1.0, 1.02))

plt.tight_layout()
out = '/home/dyros/mcy_ws/Laundry_decision/run_bf/tinyunet_ppt.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
