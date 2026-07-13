"""
논문용 Figure 3종 생성
1. Full System Pipeline
2. TinyUNet Architecture
3. GRU-AE + Dual-Channel (MRL delta) Detector

python draw_paper_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT_DIR = "/home/dyros/mcy_ws/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 공통 스타일 ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.8,
})

C = dict(
    vision   = '#4A90D9',
    motion   = '#27AE60',
    tactile  = '#E67E22',
    failure  = '#E74C3C',
    relocal  = '#8E44AD',
    regrasp  = '#2980B9',
    bg       = '#F4F6F7',
    arrow    = '#2C3E50',
    skip     = '#95A5A6',
    enc      = '#3498DB',
    dec      = '#E67E22',
    chA      = '#2ECC71',
    chB      = '#E74C3C',
    unet_enc = '#2980B9',
    unet_dec = '#E67E22',
    unet_skip= '#27AE60',
    se       = '#8E44AD',
    gray     = '#BDC3C7',
)

def fancy_box(ax, xy, w, h, color, text, fontsize=9, text_color='white',
              radius=0.03, bold=False, alpha=1.0, linestyle='-', edgecolor=None):
    x, y = xy
    ec = edgecolor or color
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.01,rounding_size={radius}",
                         facecolor=color, edgecolor=ec,
                         linewidth=1.2, alpha=alpha, linestyle=linestyle,
                         zorder=3)
    ax.add_patch(box)
    fw = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=fw, zorder=4,
            wrap=True)

def arrow(ax, x0, y0, x1, y1, color='#2C3E50', lw=1.5, style='->', lc=None):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=lc or color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=5)

def dashed_arrow(ax, x0, y0, x1, y1, color='#95A5A6', lw=1.2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'),
                zorder=5)


# ═══════════════════════════════════════════════════════════════════
#  Figure 1: Full System Pipeline
# ═══════════════════════════════════════════════════════════════════
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    ax.text(8, 6.65, 'Laundry Transfer System Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#2C3E50')

    # ── 메인 파이프라인 박스 정의 ─────────────────────────────────
    bw, bh = 2.0, 0.8
    gap = 0.35
    phases = [
        # (x,   y,    label,                              color,          sub)
        (0.5,  4.5,  'Vision\n(RGB Gate + TinyUNet)',    C['vision'],   ''),
        (3.0,  4.5,  'Motion Planning\n(MoveIt2 OMPL)', C['motion'],   ''),
        (5.5,  4.5,  'Grasp\n(Close Gripper)',           C['motion'],   ''),
        (8.0,  4.5,  'Tactile\nMonitoring',              C['tactile'],  ''),
        (10.5, 4.5,  'Failure\nDetection',               C['failure'],  ''),
        (13.0, 4.5,  'Transport\n(h→ba_init→ba_end)',    C['motion'],   ''),
    ]
    for x, y, label, col, _ in phases:
        fancy_box(ax, (x, y), bw, bh, col, label, fontsize=8.5, bold=True)

    # 메인 화살표
    xs = [p[0] for p in phases]
    for i in range(len(xs)-1):
        arrow(ax, xs[i]+bw, phases[i][1]+bh/2,
                  xs[i+1], phases[i+1][1]+bh/2, lw=2.0)

    # ── 서브 텍스트 (각 박스 아래) ───────────────────────────────
    subs = [
        'depth_diff + RGB\n→ grasp target (x,y,z)',
        'Pre-grasp pose\n(OMPL, 20s timeout)',
        'Push → close\nreset taxel buffer',
        'GRU-AE (Ch-A)\n+ MRL delta (Ch-B)',
        'z_aligned check\n(after retreat)',
        'Anomaly → abort\nDrop recovery',
    ]
    for (x, y, *_), sub in zip(phases, subs):
        ax.text(x + bw/2, y - 0.18, sub, ha='center', va='top',
                fontsize=7.2, color='#5D6D7E', style='italic')

    # ── 파지 판정 분기 ────────────────────────────────────────────
    # z_aligned=True → transport
    # z_aligned=False → re-localization

    # Failure Detection (index 4) 하단 → Re-localization
    fx, fy = phases[4][0], phases[4][1]
    fancy_box(ax, (fx+0.2, 2.5), 1.6, 0.75, C['relocal'],
              'Re-localization\n(Vision retry)', fontsize=8)
    arrow(ax, fx+bw/2, fy, fx+bw/2, 2.5+0.75, color=C['relocal'], lw=1.8)
    ax.text(fx+bw/2+0.12, fy-0.28, 'z_aligned=False', fontsize=7.5,
            color=C['failure'], ha='center')

    # Re-localization → Motion Planning (back)
    rx = fx + 0.2 + 1.6/2
    dashed_arrow(ax, rx, 2.5, phases[1][0]+bw/2, phases[1][1],
                 color=C['relocal'], lw=1.5)
    ax.text((rx + phases[1][0]+bw/2)/2, 2.1,
            'retry', ha='center', fontsize=7.5, color=C['relocal'], style='italic')

    # Transport → Release (end)
    tx, ty = phases[5][0], phases[5][1]
    fancy_box(ax, (tx+bw+0.25, 4.5), 1.5, 0.8, '#7F8C8D',
              'Release\n(Open Gripper)', fontsize=8.5)
    arrow(ax, tx+bw, ty+bh/2, tx+bw+0.25, ty+bh/2, lw=2.0)

    # ── Drop recovery box ─────────────────────────────────────────
    dashed_arrow(ax, phases[5][0]+bw/2, phases[5][1], phases[5][0]+bw/2, 2.5,
                 color=C['chB'], lw=1.5)
    fancy_box(ax, (phases[5][0]-0.3, 1.85), 2.6, 0.75, C['chB'],
              'Drop Recovery\n(basket re-detect → re-grasp)', fontsize=7.8)
    ax.text(phases[5][0]+bw/2+0.15, 2.2, 'abort\n(Ch-A/B)', fontsize=7,
            color=C['chB'], ha='left')
    dashed_arrow(ax, phases[5][0]+0.5, 1.85, phases[1][0]+bw/2, phases[1][1],
                 color=C['chB'], lw=1.3)

    # ── 범례 ─────────────────────────────────────────────────────
    legend_items = [
        (C['vision'],  'Vision / Detection'),
        (C['motion'],  'Motion Control'),
        (C['tactile'], 'Tactile Monitoring'),
        (C['failure'], 'Failure Detection'),
        (C['relocal'], 'Re-localization'),
        (C['chB'],     'Drop Recovery'),
    ]
    for i, (col, lbl) in enumerate(legend_items):
        x0 = 0.5 + i * 2.55
        ax.add_patch(mpatches.Rectangle((x0, 0.3), 0.35, 0.25,
                                        facecolor=col, edgecolor='none', zorder=3))
        ax.text(x0+0.45, 0.42, lbl, fontsize=7.8, va='center', color='#2C3E50')

    # ── 입력 센서 표시 ────────────────────────────────────────────
    ax.text(0.5+bw/2, 5.5, 'RealSense D435i\n(RGB + Depth)',
            ha='center', fontsize=7.5, color='#1A5276',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8',
                      edgecolor='#2980B9', linewidth=0.8))
    ax.text(phases[3][0]+bw/2, 5.5, 'XELA uSkin\n(16 taxels × 3D)',
            ha='center', fontsize=7.5, color='#7D6608',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7',
                      edgecolor='#F39C12', linewidth=0.8))

    arrow(ax, 0.5+bw/2, 5.35, 0.5+bw/2, 5.3, lw=1.2, color='#2980B9')
    arrow(ax, phases[3][0]+bw/2, 5.35, phases[3][0]+bw/2, 5.3,
          lw=1.2, color='#F39C12')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_pipeline.pdf')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {path}")


# ═══════════════════════════════════════════════════════════════════
#  Figure 2: TinyUNet Architecture
# ═══════════════════════════════════════════════════════════════════
def draw_tinyunet():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(7.5, 7.6, 'TinyUNet Architecture  (c_in=6, base=32, SE block)',
            ha='center', fontsize=13, fontweight='bold', color='#2C3E50')

    # ── 레이어 박스 배치 ─────────────────────────────────────────
    # Encoder: inc, d1, d2, d3
    # Bottleneck: d3 output
    # Decoder: u1, u2, u3 + skip
    # Output: 1×1 conv → sigmoid

    enc_cols  = [C['unet_enc']] * 4
    dec_cols  = [C['unet_dec']] * 3
    bw, bh = 1.55, 0.72

    # x positions (encoder left → right, decoder right → left mirror)
    enc_x = [0.4, 2.4, 4.4, 6.4]   # inc, d1, d2, d3
    dec_x = [9.0, 10.7, 12.4]       # u1, u2, u3
    enc_y = [4.0, 2.9, 1.8, 0.7]    # 계단식
    dec_y = [0.7, 1.8, 2.9]         # 거울

    enc_labels = [
        'inc\nDoubleConv(6→32)',
        'd1\nMaxPool+DC\n(32→64)',
        'd2\nMaxPool+DC\n(64→128)',
        'd3  [Bottleneck]\nMaxPool+DC\n(128→256)',
    ]
    dec_labels = [
        'u1\nConvT+DC\n(256→128)',
        'u2\nConvT+DC\n(128→64)',
        'u3\nConvT+DC\n(64→32)',
    ]

    # Encoder boxes
    for i, (x, y, lbl) in enumerate(zip(enc_x, enc_y, enc_labels)):
        col = '#1A5276' if i == 3 else C['unet_enc']
        fancy_box(ax, (x, y), bw, bh, col, lbl, fontsize=7.5)

    # Decoder boxes
    for x, y, lbl in zip(dec_x, dec_y, dec_labels):
        fancy_box(ax, (x, y), bw, bh, C['unet_dec'], lbl, fontsize=7.5)

    # Output box
    fancy_box(ax, (14.0, 2.9), 0.85, 0.72, '#717D7E',
              'outc\n1×1 Conv\n→ 1ch', fontsize=7.5)

    # ── 인코더 화살표 (down path) ─────────────────────────────────
    for i in range(len(enc_x)-1):
        x0 = enc_x[i] + bw/2
        x1 = enc_x[i+1] + bw/2
        y0 = enc_y[i]
        y1 = enc_y[i+1] + bh
        # 오른쪽 → 아래
        arrow(ax, x0+bw/2, y0+bh/2, x0+bw/2+0.3, y0+bh/2,
              color=C['arrow'], lw=1.5)
        ax.annotate('', xy=(x1, y1+bh/2), xytext=(x0+bw/2+0.3, y0+bh/2),
                    arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5,
                                    connectionstyle='arc3,rad=0.0'), zorder=5)

    # bottleneck → u1
    arrow(ax, enc_x[3]+bw, enc_y[3]+bh/2, dec_x[0], dec_y[0]+bh/2,
          color=C['arrow'], lw=1.8)

    # 디코더 화살표
    for i in range(len(dec_x)-1):
        arrow(ax, dec_x[i]+bw, dec_y[i]+bh/2,
                  dec_x[i+1], dec_y[i+1]+bh/2, color=C['arrow'], lw=1.5)

    # u3 → outc
    arrow(ax, dec_x[2]+bw, dec_y[2]+bh/2, 14.0, dec_y[2]+bh/2,
          color=C['arrow'], lw=1.5)

    # ── Skip connections (점선) ───────────────────────────────────
    skip_pairs = [(0, 2), (1, 1), (2, 0)]  # enc_idx → dec_idx
    skip_color = C['unet_skip']
    for ei, di in skip_pairs:
        ex = enc_x[ei] + bw/2
        ey = enc_y[ei] + bh
        dx_ = dec_x[di] + bw/2
        dy_ = dec_y[di] + bh
        top_y = 6.5 - ei * 0.55
        # 수직 상승 → 수평 → 수직 하강
        ax.plot([ex, ex], [ey, top_y], color=skip_color, lw=1.4, linestyle='--', zorder=3)
        ax.plot([ex, dx_], [top_y, top_y], color=skip_color, lw=1.4, linestyle='--', zorder=3)
        ax.annotate('', xy=(dx_, dy_), xytext=(dx_, top_y),
                    arrowprops=dict(arrowstyle='->', color=skip_color, lw=1.4,
                                    linestyle='dashed'), zorder=4)
        ax.text((ex+dx_)/2, top_y+0.1,
                f'skip (cat)', fontsize=6.8, ha='center', color=skip_color)

    # ── DoubleConv 설명 박스 ──────────────────────────────────────
    dc_x, dc_y = 0.4, 6.0
    fancy_box(ax, (dc_x, dc_y), 4.0, 1.1, '#EBF5FB',
              '', fontsize=8, text_color='#2C3E50', radius=0.04)
    ax.text(dc_x+2.0, dc_y+0.9, 'DoubleConv block',
            ha='center', fontsize=8.5, fontweight='bold', color='#154360')
    # mini sub-boxes
    sub = ['Conv3×3\n+BN+ReLU', 'Conv3×3\n+BN+ReLU', 'SE block\n(r=8)']
    sub_c = [C['unet_enc'], C['unet_enc'], C['se']]
    for j, (s, sc) in enumerate(zip(sub, sub_c)):
        sx = dc_x + 0.15 + j * 1.3
        fancy_box(ax, (sx, dc_y+0.1), 1.15, 0.65, sc, s, fontsize=7.0)
        if j < len(sub)-1:
            arrow(ax, sx+1.15, dc_y+0.42, sx+1.3, dc_y+0.42,
                  color=C['arrow'], lw=1.2)

    # ── SE block 설명 ─────────────────────────────────────────────
    se_x, se_y = 5.3, 5.9
    fancy_box(ax, (se_x, se_y), 4.5, 1.25, '#F9EBEA',
              '', fontsize=7.5, radius=0.04)
    ax.text(se_x+2.25, se_y+1.05, 'Squeeze-and-Excitation (SE, r=8)',
            ha='center', fontsize=8.0, fontweight='bold', color='#641E16')
    se_subs = ['AvgPool\n(global)', 'FC\nch→ch//8', 'ReLU', 'FC\nch//8→ch', 'Sigmoid', '× input']
    se_cols = ['#AED6F1','#5DADE2','#2E86C1','#5DADE2','#AED6F1','#7FB3D3']
    for j, (s, sc) in enumerate(zip(se_subs, se_cols)):
        sx = se_x + 0.1 + j * 0.73
        fancy_box(ax, (sx, se_y+0.1), 0.65, 0.72, sc, s, fontsize=6.0, text_color='#1A252F')
        if j < len(se_subs)-1:
            arrow(ax, sx+0.65, se_y+0.46, sx+0.73, se_y+0.46,
                  color='#922B21', lw=1.0)

    # ── 입력/출력 표시 ────────────────────────────────────────────
    ax.text(enc_x[0]+bw/2, enc_y[0]+bh+0.15,
            'Input: (B, 6, 384, 384)\n[Δdepth, z, R, G, B, domain]',
            ha='center', fontsize=7.8, color='#1A5276',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8',
                      edgecolor='#2980B9', linewidth=0.8))
    ax.text(14.42, dec_y[2]+bh+0.15,
            'Output:\n(B,1,384,384)\nBinary mask\n(Sigmoid)',
            ha='center', fontsize=7.5, color='#4A235A',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8DAEF',
                      edgecolor='#8E44AD', linewidth=0.8))

    # ── 범례 ─────────────────────────────────────────────────────
    items = [(C['unet_enc'],'Encoder (Down)'), (C['unet_dec'],'Decoder (Up)'),
             (C['unet_skip'],'Skip connection (cat)'), (C['se'],'SE block')]
    for i, (col, lbl) in enumerate(items):
        ax.add_patch(mpatches.Rectangle((0.4+i*3.5, 0.1), 0.35, 0.22,
                                        facecolor=col, edgecolor='none'))
        ax.text(0.85+i*3.5, 0.21, lbl, fontsize=8, va='center', color='#2C3E50')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_tinyunet.pdf')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.pdf','.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {path}")


# ═══════════════════════════════════════════════════════════════════
#  Figure 3: GRU-AE + Dual-Channel Detector
# ═══════════════════════════════════════════════════════════════════
def draw_gru_ae():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(8, 8.6, 'Dual-Channel Tactile Anomaly Detector',
            ha='center', fontsize=13, fontweight='bold', color='#2C3E50')

    # ── XELA 입력 ─────────────────────────────────────────────────
    fancy_box(ax, (0.2, 6.2), 2.0, 1.1, '#F39C12',
              'XELA uSkin\n16 taxels × (x,y,z)\n→ 48D raw', fontsize=8,
              text_color='white', bold=True)

    # 6D feature extraction
    fancy_box(ax, (0.2, 4.5), 2.0, 1.1, '#D4AC0D',
              '6D Feature\nExtraction\n(ΔX,ΔY,ΔZ,MRL,θ,norm)', fontsize=7.8,
              text_color='white')
    arrow(ax, 1.2, 6.2, 1.2, 5.6, lw=1.8)

    # sliding window
    fancy_box(ax, (0.2, 3.0), 2.0, 0.9, '#B7950B',
              'Sliding Window\nT=50 frames\n(stride=1)', fontsize=7.8,
              text_color='white')
    arrow(ax, 1.2, 4.5, 1.2, 3.9, lw=1.8)

    # StandardScaler
    fancy_box(ax, (0.2, 1.8), 2.0, 0.85, '#7F8C8D',
              'StandardScaler\n+ RelTransform', fontsize=8, text_color='white')
    arrow(ax, 1.2, 3.0, 1.2, 2.65, lw=1.8)

    # ── Channel A: GRU-AE ─────────────────────────────────────────
    ax.text(7.2, 8.1, 'Channel A  (slow: sequence-level)',
            ha='center', fontsize=9.5, fontweight='bold', color=C['chA'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#EAFAF1', edgecolor=C['chA']))

    # Encoder
    fancy_box(ax, (2.8, 6.0), 2.2, 1.3, C['enc'],
              'GRU Encoder\n(D=6, H=128, L=2)\n\nInput: (B,T,6)\n→ h_n: (2,B,128)',
              fontsize=7.5, text_color='white')

    # Bottleneck label
    ax.text(4.7, 6.65, 'h_n\n(context)', ha='center', fontsize=7.5,
            color='#1A5276', style='italic')

    # Decoder
    fancy_box(ax, (5.3, 6.0), 2.2, 1.3, C['dec'],
              'GRU Decoder\n(H=128→D=6, L=2)\n\nrepeat h_n[-1] T×\n→ recon: (B,T,6)',
              fontsize=7.5, text_color='white')

    # arrows enc→dec
    arrow(ax, 2.2, 3.75, 3.9, 6.0, lw=1.8, color=C['enc'])
    arrow(ax, 4.0+0.8, 6.65+0.0, 5.3, 6.65, lw=1.8, color=C['arrow'])

    # MSE
    fancy_box(ax, (8.0, 6.2), 1.9, 0.9, '#566573',
              'MSE\nReconstruction\nError  (score_A)',
              fontsize=7.5, text_color='white')
    arrow(ax, 5.3+2.2, 6.65, 8.0, 6.65, lw=1.8, color=C['arrow'])

    # threshold A
    fancy_box(ax, (8.0, 4.8), 1.9, 0.85, C['chA'],
              'Threshold θ_A\n(consecutive k)\n→ anomaly_A',
              fontsize=7.5, text_color='white')
    arrow(ax, 8.95, 6.2, 8.95, 5.65, lw=1.8, color=C['chA'])

    # Training note
    ax.text(4.1, 5.5, 'Trained on\nnormal grasp only\n(one-class)',
            ha='center', fontsize=7.5, color='#1A5276', style='italic',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#D6EAF8',
                      edgecolor='#2980B9', linewidth=0.8))

    # ── Channel B: MRL Delta ──────────────────────────────────────
    ax.text(7.2, 4.2, 'Channel B  (fast: frame-level)',
            ha='center', fontsize=9.5, fontweight='bold', color=C['chB'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC', edgecolor=C['chB']))

    fancy_box(ax, (2.8, 2.9), 2.2, 1.0, '#C0392B',
              'MRL Extraction\nMRL = mean‖taxel_vec‖\n(frame-wise)', fontsize=7.8,
              text_color='white')
    arrow(ax, 2.2, 2.25, 3.9, 2.9, lw=1.8, color=C['chB'])

    fancy_box(ax, (5.3, 2.9), 2.2, 1.0, C['chB'],
              'Frame Delta\n|MRL_t − ref|\n(smooth w=3)',
              fontsize=7.8, text_color='white')
    arrow(ax, 5.0, 3.4, 5.3, 3.4, lw=1.8, color=C['arrow'])

    fancy_box(ax, (8.0, 2.9), 1.9, 1.0, C['chB'],
              'Threshold θ_B\n(auto-calibrated)\n→ anomaly_B',
              fontsize=7.8, text_color='white')
    arrow(ax, 7.5, 3.4, 8.0, 3.4, lw=1.8, color=C['chB'])

    # MRL input
    arrow(ax, 2.2, 2.25, 2.8, 3.4, lw=1.8, color=C['chB'])

    # ── OR gate ───────────────────────────────────────────────────
    fancy_box(ax, (10.4, 4.0), 1.8, 1.5, '#2C3E50',
              'OR Gate\n\nanomaly_A\nOR\nanomaly_B',
              fontsize=8.5, text_color='white', bold=True)

    arrow(ax, 8.0+1.9, 4.8+0.42, 10.4, 4.75, lw=1.8, color=C['chA'])
    arrow(ax, 8.0+1.9, 3.4,       10.4, 4.25, lw=1.8, color=C['chB'])

    # ── Output ────────────────────────────────────────────────────
    fancy_box(ax, (12.6, 4.15), 2.0, 1.2, '#922B21',
              'ABORT\nTransport\n→ Drop Recovery',
              fontsize=8.5, text_color='white', bold=True)
    arrow(ax, 10.4+1.8, 4.75, 12.6, 4.75, lw=2.2, color='#922B21')

    # No anomaly path
    fancy_box(ax, (12.6, 5.8), 2.0, 0.8, C['chA'],
              'NORMAL\n(continue)', fontsize=8.5, text_color='white')
    dashed_arrow(ax, 11.3, 5.5, 12.6, 6.2, color=C['chA'])
    ax.text(11.5, 5.7, 'no\nanomaly', fontsize=7, color=C['chA'], ha='center')

    # ── 입력 분기 화살표 ─────────────────────────────────────────
    arrow(ax, 1.2, 1.8, 1.2, 1.4, lw=1.5, color=C['arrow'])
    # 왼쪽 → Ch-A
    ax.plot([1.2, 2.0, 2.0, 2.8], [1.4, 1.4, 6.65, 6.65],
            color=C['enc'], lw=1.6, linestyle='-', zorder=3)
    ax.annotate('', xy=(2.8, 6.65), xytext=(2.6, 6.65),
                arrowprops=dict(arrowstyle='->', color=C['enc'], lw=1.5), zorder=4)
    # 왼쪽 → Ch-B (MRL)
    ax.plot([1.2, 2.0, 2.0, 2.8], [1.4, 1.4, 3.4, 3.4],
            color=C['chB'], lw=1.6, linestyle='-', zorder=3)
    ax.annotate('', xy=(2.8, 3.4), xytext=(2.6, 3.4),
                arrowprops=dict(arrowstyle='->', color=C['chB'], lw=1.5), zorder=4)

    ax.text(2.15, 2.4, 'to Ch-A', fontsize=7, color=C['enc'], rotation=90, va='center')
    ax.text(2.45, 2.4, 'to Ch-B', fontsize=7, color=C['chB'], rotation=90, va='center')

    # ── 설명 텍스트 ───────────────────────────────────────────────
    desc = ("Ch-A detects gradual force loss (sequence-level reconstruction error).\n"
            "Ch-B detects sudden drop events (frame-to-frame MRL delta).\n"
            "Final decision: anomaly if Ch-A OR Ch-B triggered.")
    ax.text(0.3, 0.55, desc, fontsize=8, color='#2C3E50', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA',
                      edgecolor='#BDC3C7', linewidth=0.8))

    # ── 범례 ─────────────────────────────────────────────────────
    items = [(C['enc'],'GRU Encoder'), (C['dec'],'GRU Decoder'),
             (C['chA'],'Channel A (GRU-AE)'), (C['chB'],'Channel B (MRL delta)'),
             ('#922B21','Abort / Drop Recovery')]
    for i, (col, lbl) in enumerate(items):
        ax.add_patch(mpatches.Rectangle((0.3+i*3.1, 0.1), 0.35, 0.22,
                                        facecolor=col, edgecolor='none'))
        ax.text(0.75+i*3.1, 0.21, lbl, fontsize=7.8, va='center', color='#2C3E50')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_gru_ae_dual_channel.pdf')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.pdf','.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {path}")


if __name__ == '__main__':
    print("Generating figures...")
    draw_pipeline()
    draw_tinyunet()
    draw_gru_ae()
    print(f"\nAll figures saved to: {OUT_DIR}/")
    print("  fig1_pipeline.pdf / .png")
    print("  fig2_tinyunet.pdf / .png")
    print("  fig3_gru_ae_dual_channel.pdf / .png")
