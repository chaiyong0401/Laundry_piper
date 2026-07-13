"""
TinyUNet 상세 아키텍처 다이어그램
c_in=6, base=32, use_se=True, input=384×384

python draw_tinyunet_detailed.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = "/home/dyros/mcy_ws/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})

C_ENC  = '#2471A3'   # Encoder (blue)
C_BOT  = '#6C3483'   # Bottleneck (purple)
C_DEC  = '#BA4A00'   # Decoder (orange)
C_SKIP = '#1E8449'   # Skip connection (green)
C_SE   = '#7D3C98'   # SE block (violet)
C_IO   = '#1C2833'   # Input/Output
C_OP   = '#444444'   # Operation label

# ── helpers ─────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, color, text='', fs=8, tc='white',
         radius=0.06, alpha=0.93, ec='white', lw=1.4):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle=f'round,pad={radius}',
                       facecolor=color, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=3)
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, color=tc, zorder=4, linespacing=1.45)


def arr(ax, x0, y0, x1, y1, color=C_OP, lw=1.6, head=11, conn='arc3,rad=0.0'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=head,
                                connectionstyle=conn), zorder=6)


def arr_angle(ax, x0, y0, x1, y1, aA, aB, color=C_OP, lw=1.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=11,
                                connectionstyle=f'angle,angleA={aA},angleB={aB},rad=3'),
                zorder=6)


# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(20, 12.5))
ax.set_xlim(0, 20); ax.set_ylim(0, 12.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Header ───────────────────────────────────────────────────────────
hdr = FancyBboxPatch((0, 11.9), 20, 0.6, boxstyle='square,pad=0',
                     facecolor=C_IO, edgecolor='none', zorder=1)
ax.add_patch(hdr)
ax.text(10, 12.2,
        'TinyUNet Architecture   (c_in = 6, base = 32, use_se = True,  input = 384 × 384)',
        ha='center', va='center', fontsize=12.5, fontweight='bold',
        color='white', zorder=2)

# ════════════════════════════════════════════════════════════════════
# Layout constants
# ════════════════════════════════════════════════════════════════════
ENC_X  = 7.5    # encoder column x
DEC_X  = 12.5   # decoder column x
BOT_X  = 10.0   # bottleneck x
BW     = 1.1    # box width (encoder/decoder)
BOT_BW = 1.4    # bottleneck box width

# Feature map box heights ∝ spatial resolution (384→2.6, 192→1.45, 96→0.85, 48→0.52)
H1, H2, H3, H4 = 2.6, 1.45, 0.85, 0.52

# Y centers for each resolution level
Y1 = 9.6    # 384×384 level
Y2 = 7.0    # 192×192 level
Y3 = 5.15   # 96×96  level
Y4 = 3.55   # 48×48  bottleneck

# ════════════════════════════════════════════════════════════════════
# ENCODER boxes
# ════════════════════════════════════════════════════════════════════
rbox(ax, ENC_X, Y1, BW, H1, C_ENC,
     'inc\n\nDoubleConv + SE\n\n6 → 32 ch\n384 × 384', fs=8.2)

rbox(ax, ENC_X, Y2, BW, H2, C_ENC,
     'd1\n32 → 64 ch\n192 × 192', fs=8)

rbox(ax, ENC_X, Y3, BW, H3, C_ENC,
     'd2\n64 → 128 ch\n96 × 96', fs=7.8)

rbox(ax, BOT_X, Y4, BOT_BW, H4, C_BOT,
     'd3  (Bottleneck)     128 → 256 ch   |   48 × 48', fs=8.2)

# ════════════════════════════════════════════════════════════════════
# DECODER boxes
# ════════════════════════════════════════════════════════════════════
rbox(ax, DEC_X, Y3, BW, H3, C_DEC,
     'u1\n256→128 ch\n96 × 96', fs=7.8)

rbox(ax, DEC_X, Y2, BW, H2, C_DEC,
     'u2\n128 → 64 ch\n192 × 192', fs=8)

rbox(ax, DEC_X, Y1, BW, H1, C_DEC,
     'u3\n\nConvT + cat\n+ DoubleConv + SE\n\n64 → 32 ch\n384 × 384', fs=8)

# outc (1×1 conv)
rbox(ax, 14.35, Y1, 0.65, H1, '#922B21',
     'outc\nConv2d\n1×1\n\n32→1 ch\n384×384', fs=7.8)

# ════════════════════════════════════════════════════════════════════
# INPUT / OUTPUT boxes
# ════════════════════════════════════════════════════════════════════
rbox(ax, 5.5, Y1, 1.05, H1, C_IO,
     'INPUT\n6 ch × 384×384\n─────\nΔdepth\nz-depth\nR, G, B\ndomain', fs=8)

rbox(ax, 16.1, Y1, 1.05, H1, '#1E8449',
     'OUTPUT\n1 ch × 384×384\n─────\nBinary Mask\n(sigmoid)\n↓\nGrasp pts', fs=8)

# ════════════════════════════════════════════════════════════════════
# ARROWS — Encoder flow (downward)
# ════════════════════════════════════════════════════════════════════

# Input → inc
arr(ax, 5.5+0.53, Y1, ENC_X-BW/2, Y1, lw=1.8)

# inc → d1 (vertical down)
arr(ax, ENC_X, Y1-H1/2, ENC_X, Y2+H2/2, lw=1.6)
ax.text(ENC_X-0.62, (Y1-H1/2 + Y2+H2/2)/2,
        'MaxPool2d(2)', ha='center', va='center', fontsize=7.5,
        color=C_ENC, rotation=90, style='italic')

# d1 → d2 (vertical down)
arr(ax, ENC_X, Y2-H2/2, ENC_X, Y3+H3/2, lw=1.6)
ax.text(ENC_X-0.62, (Y2-H2/2 + Y3+H3/2)/2,
        'MaxPool2d(2)', ha='center', va='center', fontsize=7.5,
        color=C_ENC, rotation=90, style='italic')

# d2 → d3 (L-shape: down then right to bottleneck)
arr_angle(ax, ENC_X, Y3-H3/2,
          BOT_X-BOT_BW/2, Y4,
          aA=-90, aB=180, color=C_ENC)
ax.text((ENC_X+BOT_X)/2 - 0.5, (Y3-H3/2 + Y4)/2 - 0.1,
        'MaxPool2d(2)', ha='center', va='center', fontsize=7.5,
        color=C_ENC, style='italic')

# ════════════════════════════════════════════════════════════════════
# ARROWS — Bottleneck → Decoder
# ════════════════════════════════════════════════════════════════════

# d3 → u1 (L-shape: right then up)
arr_angle(ax, BOT_X+BOT_BW/2, Y4,
          DEC_X, Y3-H3/2,
          aA=0, aB=-90, color=C_DEC)
ax.text((BOT_X+DEC_X)/2 + 0.4, (Y4+Y3-H3/2)/2,
        'ConvTranspose2d\n(256→128, ×2)', ha='center', va='center',
        fontsize=7.5, color=C_DEC, style='italic')

# u1 → u2 (vertical up)
arr(ax, DEC_X, Y3+H3/2, DEC_X, Y2-H2/2, lw=1.6, color=C_DEC)
ax.text(DEC_X+0.62, (Y3+H3/2 + Y2-H2/2)/2,
        'ConvTranspose2d\n(128→64, ×2)', ha='center', va='center',
        fontsize=7.5, color=C_DEC, rotation=90, style='italic')

# u2 → u3 (vertical up)
arr(ax, DEC_X, Y2+H2/2, DEC_X, Y1-H1/2, lw=1.6, color=C_DEC)
ax.text(DEC_X+0.62, (Y2+H2/2 + Y1-H1/2)/2,
        'ConvTranspose2d\n(64→32, ×2)', ha='center', va='center',
        fontsize=7.5, color=C_DEC, rotation=90, style='italic')

# u3 → outc
arr(ax, DEC_X+BW/2, Y1, 14.35-0.33, Y1, lw=1.6, color='#922B21')

# outc → output
arr(ax, 14.35+0.33, Y1, 16.1-0.53, Y1, lw=1.8, color='#1E8449')
ax.text((14.35+0.33+16.1-0.53)/2, Y1+0.22,
        'sigmoid', ha='center', fontsize=7.5, color='#1E8449', style='italic')

# ════════════════════════════════════════════════════════════════════
# SKIP CONNECTIONS  (encoder right edge → decoder left edge)
# ════════════════════════════════════════════════════════════════════
skip_levels = [
    (Y1, 'x1 = 32 ch',  'cat: 32+32 → 64 ch  →  DoubleConv+SE  →  32 ch'),
    (Y2, 'x2 = 64 ch',  'cat: 64+64 → 128 ch  →  DoubleConv+SE  →  64 ch'),
    (Y3, 'x3 = 128 ch', 'cat: 128+128 → 256 ch  →  DoubleConv+SE  →  128 ch'),
]
for (y_lv, skip_name, cat_label) in skip_levels:
    arr(ax, ENC_X+BW/2, y_lv, DEC_X-BW/2, y_lv,
        color=C_SKIP, lw=1.9, head=12)
    mid = (ENC_X+BW/2 + DEC_X-BW/2) / 2
    ax.text(mid, y_lv+0.20, skip_name,
            ha='center', va='bottom', fontsize=7.8,
            color=C_SKIP, fontweight='bold')
    ax.text(mid, y_lv-0.20, cat_label,
            ha='center', va='top', fontsize=7, color=C_SKIP, style='italic')

# ── Encoder / Decoder labels ──────────────────────────────────────
ax.text(ENC_X, 11.5, '▼  ENCODER', ha='center', fontsize=10.5,
        color=C_ENC, fontweight='bold')
ax.text(DEC_X, 11.5, 'DECODER  ▲', ha='center', fontsize=10.5,
        color=C_DEC, fontweight='bold')

# ════════════════════════════════════════════════════════════════════
# DETAIL INSET A: DoubleConv + SE block  (bottom-left)
# ════════════════════════════════════════════════════════════════════
A_X0, A_Y0 = 0.2, 0.15
A_W,  A_H  = 7.0, 2.85
bg_a = FancyBboxPatch((A_X0, A_Y0), A_W, A_H, boxstyle='round,pad=0.08',
                      facecolor='#EBF5FB', edgecolor='#AED6F1', linewidth=1.2, zorder=1)
ax.add_patch(bg_a)
ax.text(A_X0+A_W/2, A_Y0+A_H+0.02,
        'DoubleConv + SE Block  (applied at every encoder/decoder stage)',
        ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1A5276')

# Row 1: two Conv layers
CY = A_Y0 + 1.9
for (cx, txt, clr) in [
    (A_X0+0.95, 'Conv2d\n3×3, pad=1\nC_in → C_out', C_ENC),
    (A_X0+2.25, 'BatchNorm2d\n+ ReLU',               '#1ABC9C'),
    (A_X0+3.55, 'Conv2d\n3×3, pad=1\nC_out → C_out', C_ENC),
    (A_X0+4.85, 'BatchNorm2d\n+ ReLU',               '#1ABC9C'),
]:
    rbox(ax, cx, CY, 1.05, 0.60, clr, txt, fs=7.5)
for x0, x1 in [(A_X0+1.48, A_X0+1.73), (A_X0+2.78, A_X0+3.03),
               (A_X0+4.08, A_X0+4.33)]:
    arr(ax, x0, CY, x1, CY, lw=1.2)
arr(ax, A_X0+0.42, CY, A_X0+0.43, CY, lw=1.2)  # input arrow stub

# Row 2: SE block
SY = A_Y0 + 0.90
ax.text(A_X0+A_W/2, SY+0.58, 'SE Block (Squeeze-and-Excitation, r=8)',
        ha='center', fontsize=8, color=C_SE, style='italic')
for (cx, txt, clr) in [
    (A_X0+0.95, 'AdaptiveAvg\nPool2d(1)',     C_SE),
    (A_X0+2.3,  'Linear\nC→C/8\nReLU',       '#8E44AD'),
    (A_X0+3.65, 'Linear\nC/8→C\nSigmoid',    '#8E44AD'),
    (A_X0+5.0,  '× channel\nscaling',         '#6C3483'),
]:
    rbox(ax, cx, SY, 1.05, 0.60, clr, txt, fs=7.5)
for x0, x1 in [(A_X0+1.48, A_X0+1.78), (A_X0+2.83, A_X0+3.13),
               (A_X0+4.18, A_X0+4.48)]:
    arr(ax, x0, SY, x1, SY, lw=1.2, color=C_SE)

# Arrow from BN+ReLU (row1) down to SE (row2)
arr(ax, A_X0+4.85, CY-0.30, A_X0+4.85, SY+0.30, color=C_SE, lw=1.2)
arr(ax, A_X0+0.42, SY, A_X0+0.43, SY, lw=1.2, color=C_SE)

# Output arrow from SE
arr(ax, A_X0+5.53, SY, A_X0+6.7, SY, lw=1.4, color='#2C3E50')
ax.text(A_X0+6.75, SY+0.02, '→ out\n(C_out ch)', va='center', fontsize=7.5, color='#2C3E50')

# Input label
ax.text(A_X0+0.15, CY+0.02, '→\n(C_in ch)', va='center', ha='right', fontsize=7.5, color='#2C3E50')

# ════════════════════════════════════════════════════════════════════
# DETAIL INSET B: Up block  (bottom-center)
# ════════════════════════════════════════════════════════════════════
B_X0, B_Y0 = 7.6, 0.15
B_W,  B_H  = 5.8, 2.85
bg_b = FancyBboxPatch((B_X0, B_Y0), B_W, B_H, boxstyle='round,pad=0.08',
                      facecolor='#FEF9E7', edgecolor='#F9E79F', linewidth=1.2, zorder=1)
ax.add_patch(bg_b)
ax.text(B_X0+B_W/2, B_Y0+B_H+0.02,
        'Up Block  (u1 / u2 / u3)',
        ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#784212')

UY = B_Y0 + 1.85
for (cx, txt, clr) in [
    (B_X0+0.85, 'ConvTranspose2d\n2×2, stride=2\nC_in → C_in/2', C_DEC),
    (B_X0+2.5,  'concat\n(skip)',   '#E67E22'),
    (B_X0+4.1,  'DoubleConv\n+ SE', C_DEC),
]:
    rbox(ax, cx, UY, 1.25, 0.70, clr, txt, fs=7.8)
arr(ax, B_X0+1.48, UY, B_X0+1.88, UY, lw=1.3)
arr(ax, B_X0+3.13, UY, B_X0+3.48, UY, lw=1.3)

# skip input from above into concat
arr(ax, B_X0+2.5, UY+0.35+0.5, B_X0+2.5, UY+0.35, color=C_SKIP, lw=1.5)
ax.text(B_X0+2.5, UY+0.35+0.55, 'skip\n(C_in/2 ch)',
        ha='center', va='bottom', fontsize=7.5, color=C_SKIP, fontweight='bold')

# Size labels below blocks
for (cx, lbl) in [(B_X0+0.85, 'H×W → 2H×2W\nC_in/2 ch'),
                  (B_X0+2.5, '→ C_in ch\n2H×2W'),
                  (B_X0+4.1, '→ C_out ch\n2H×2W')]:
    ax.text(cx, UY-0.42, lbl, ha='center', va='top', fontsize=7, color='#5D6D7E', style='italic')

# Example table
EX_Y = B_Y0 + 0.72
ax.text(B_X0+B_W/2, EX_Y+0.12, 'Concrete sizes  (base=32):', ha='center',
        fontsize=8, fontweight='bold', color='#2C3E50')
rows = [
    ('u1', 'C_in=256', 'skip=128ch (from d2)', 'concat=256ch', 'C_out=128', '48→96'),
    ('u2', 'C_in=128', 'skip=64ch  (from d1)', 'concat=128ch', 'C_out=64',  '96→192'),
    ('u3', 'C_in=64',  'skip=32ch  (from inc)', 'concat=64ch',  'C_out=32',  '192→384'),
]
col_xs = [B_X0+0.35, B_X0+1.25, B_X0+2.55, B_X0+3.95, B_X0+4.95, B_X0+5.5]
for ri, row in enumerate(rows):
    ry = EX_Y - 0.04 - ri * 0.28
    for ci, val in enumerate(row):
        clr = C_DEC if ci == 0 else ('#922B21' if ci==3 else '#2C3E50')
        fw = 'bold' if ci == 0 else 'normal'
        ax.text(col_xs[ci], ry, val, ha='left', va='center',
                fontsize=7.3, color=clr, fontweight=fw)

# ════════════════════════════════════════════════════════════════════
# LEGEND  (bottom-right)
# ════════════════════════════════════════════════════════════════════
L_X0, L_Y0 = 13.8, 0.15
L_W,  L_H  = 5.9, 2.85
bg_l = FancyBboxPatch((L_X0, L_Y0), L_W, L_H, boxstyle='round,pad=0.08',
                      facecolor='#F8F9FA', edgecolor='#BDC3C7', linewidth=1.1, zorder=1)
ax.add_patch(bg_l)
ax.text(L_X0+L_W/2, L_Y0+L_H+0.02,
        'Legend', ha='center', va='bottom',
        fontsize=9.5, fontweight='bold', color='#2C3E50')

legend_items = [
    (C_ENC,    'Encoder block  (DoubleConv + SE)'),
    (C_BOT,    'Bottleneck  (DoubleConv + SE)'),
    (C_DEC,    'Decoder block  (ConvTranspose + cat + DoubleConv + SE)'),
    (C_SKIP,   'Skip connection  (feature map concatenation)'),
    (C_SE,     'SE = Squeeze-and-Excitation  (channel attention)'),
    ('#1ABC9C', 'BN + ReLU  (Batch Normalization + Activation)'),
    ('#922B21', 'Output conv  (1×1, sigmoid → binary mask)'),
]
for i, (clr, lbl) in enumerate(legend_items):
    ly = L_Y0 + L_H - 0.30 - i * 0.36
    rbox(ax, L_X0+0.35, ly, 0.42, 0.24, clr, alpha=0.92, radius=0.03)
    ax.text(L_X0+0.62, ly, lbl, va='center', fontsize=7.8, color='#2C3E50')

# ════════════════════════════════════════════════════════════════════
path = os.path.join(OUT_DIR, 'tinyunet_detailed.pdf')
plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] {path}")
print(f"[OK] {path.replace('.pdf', '.png')}")
