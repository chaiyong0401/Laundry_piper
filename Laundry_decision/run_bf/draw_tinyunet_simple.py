"""TinyUNet 간략 다이어그램 (SE/DoubleConv 내부 생략)"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

C_IN   = '#2C3E50'
C_ENC  = '#E8734A'
C_BOT  = '#C0392B'
C_DEC  = '#27AE60'
C_OUT  = '#8E44AD'
C_SKIP = '#F39C12'
C_ARR  = '#555555'

def box(ax, x, y, w, h, color, line1, line2='', fs=9):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle="round,pad=0.06,rounding_size=0.12",
                          facecolor=color, edgecolor='white', linewidth=1.8,
                          alpha=0.92, zorder=3)
    ax.add_patch(rect)
    dy = 0.13 if line2 else 0
    ax.text(x, y+dy, line1, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='white', zorder=4)
    if line2:
        ax.text(x, y-dy, line2, ha='center', va='center',
                fontsize=fs-1, color='white', alpha=0.88, zorder=4)

def arr(ax, x1, y1, x2, y2, color=C_ARR, lw=2.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'), zorder=2)

# ── 타이틀
ax.text(7, 8.6, 'TinyUNet  (c_in=6, base=32)',
        ha='center', fontsize=13, fontweight='bold', color='#2C3E50')

# ── 열 x좌표
EX, DX = 3.5, 10.5   # Encoder / Decoder
# y 좌표 (위→아래)
Y = [7.7, 6.5, 5.3, 4.1]   # inc, d1, d2, d3(bottleneck)
DY = [5.3, 6.5, 7.7]        # u1, u2, u3

# ── Input
box(ax, EX, 8.4, 3.2, 0.55, C_IN,
    'Input  (B, 6, H, W)',
    'diff | cur | R|G|B | domain')
arr(ax, EX, 8.12, EX, 7.97)

# ── Encoder
enc_labels = [
    ('inc', '6 → 32',  'H × W'),
    ('d1',  '32 → 64', 'H/2 × W/2'),
    ('d2',  '64 → 128','H/4 × W/4'),
    ('d3',  '128 → 256','H/8 × W/8  (Bottleneck)'),
]
colors_enc = [C_ENC, C_ENC, C_ENC, C_BOT]

for i, (name, ch, res) in enumerate(enc_labels):
    c = colors_enc[i]
    box(ax, EX, Y[i], 3.0, 0.60, c, f'{name}  {ch}', res)
    if i < len(Y)-1:
        arr(ax, EX, Y[i]-0.30, EX, Y[i+1]+0.30)

# ── Decoder
dec_labels = [
    ('u1', '256+128 → 128', 'H/4 × W/4'),
    ('u2', '128+64 → 64',   'H/2 × W/2'),
    ('u3', '64+32 → 32',    'H × W'),
]
for i, (name, ch, res) in enumerate(dec_labels):
    box(ax, DX, DY[i], 3.0, 0.60, C_DEC, f'{name}  {ch}', res)
    if i > 0:
        arr(ax, DX, DY[i-1]+0.30, DX, DY[i]-0.30)

# Bottleneck → u1
arr(ax, EX+1.5, Y[3], DX-1.5, DY[0])

# ── Skip connections
skip_pairs = [
    (Y[2], DY[0], '128ch'),  # d2 → u1
    (Y[1], DY[1], '64ch'),   # d1 → u2
    (Y[0], DY[2], '32ch'),   # inc → u3
]
for (ey, dy_, label) in skip_pairs:
    ax.annotate('', xy=(DX-1.5, dy_), xytext=(EX+1.5, ey),
                arrowprops=dict(arrowstyle='->', color=C_SKIP, lw=2.0,
                                connectionstyle='arc3,rad=-0.25'), zorder=2)
    mx = (EX+1.5 + DX-1.5)/2
    my = (ey + dy_)/2 + 0.25
    ax.text(mx, my, f'skip  {label}', ha='center', fontsize=8,
            color=C_SKIP, fontweight='bold')

# ── Output conv
arr(ax, DX, DY[2]+0.30, DX, 8.12)
box(ax, DX, 8.4, 3.2, 0.55, C_OUT,
    'outc  Conv2d 1×1  →  32→1',
    'Output (B, 1, H, W)  Binary Mask')

# ── 범례
legend = [
    mpatches.Patch(color=C_ENC, label='Encoder (DoubleConv)'),
    mpatches.Patch(color=C_BOT, label='Bottleneck'),
    mpatches.Patch(color=C_DEC, label='Decoder (ConvTranspose + cat)'),
    mpatches.Patch(color=C_SKIP, label='Skip Connection'),
    mpatches.Patch(color=C_OUT,  label='Output Conv'),
]
ax.legend(handles=legend, loc='lower center', ncol=5,
          framealpha=0.85, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))

# ── Encoder / Decoder 레이블
ax.text(EX, 3.65, 'Encoder ▼', ha='center', fontsize=9,
        color=C_ENC, fontstyle='italic', fontweight='bold')
ax.text(DX, 4.9, '▲ Decoder', ha='center', fontsize=9,
        color=C_DEC, fontstyle='italic', fontweight='bold')

plt.tight_layout()
out = '/home/dyros/mcy_ws/Laundry_decision/run_bf/tinyunet_simple.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
