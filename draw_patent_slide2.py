"""
특허 제출용 슬라이드 2: 촉각 센서 신호·피처 시각화 (그래프/그림 중심)
- 왼쪽: XELA 16 taxel 힘 벡터 시각화 (Firm Grip vs Slip)
- 오른쪽: MRL·Ch-A 재구성오차·Ch-B delta 시계열

python draw_patent_slide2.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

OUT_DIR = "/home/dyros/mcy_ws/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

# ── Colors ─────────────────────────────────────────────────────────
HEADER = '#2C3E50'
C_chA  = '#1ABC9C'
C_chB  = '#E74C3C'
C_mrl  = '#2471A3'
C_thr  = '#7D3C98'

PHASE_CLR = {
    'grip':     '#D5F5E3',
    'hold':     '#EBF5FB',
    'slip':     '#FDEBD0',
    'drop':     '#FADBD8',
    'empty':    '#EAECEE',   # gripper empty
    'regrip':   '#D5F5E3',   # re-grasping
    'rehold':   '#A9DFBF',   # re-hold success
}

# ── 합성 데이터 생성 ────────────────────────────────────────────────
np.random.seed(2024)
T = 345
t = np.arange(T)

# Phase boundaries (frame index)
PH_GRIP   = (0,   45)
PH_HOLD   = (45,  140)
PH_SLIP   = (140, 178)
PH_DROP   = (178, 215)
PH_EMPTY  = (215, 255)   # 그리퍼 비어있음 / re-localize
PH_REGRIP = (255, 295)   # 재파지
PH_REHOLD = (295, 345)   # 재파지 성공 후 이송 재개

PHASES = [
    (PH_GRIP,   PHASE_CLR['grip'],    'Grip'),
    (PH_HOLD,   PHASE_CLR['hold'],    'Hold\n(Transport)'),
    (PH_SLIP,   PHASE_CLR['slip'],    'Slip\n(Ch-A ↑)'),
    (PH_DROP,   PHASE_CLR['drop'],    'Drop\n(Ch-B spike)'),
    (PH_EMPTY,  PHASE_CLR['empty'],   'Empty\n(Re-locate)'),
    (PH_REGRIP, PHASE_CLR['regrip'],  'Re-grasp'),
    (PH_REHOLD, PHASE_CLR['rehold'],  'Re-Hold\n(Transport resumed)'),
]

# MRL signal
mrl = np.zeros(T)
mrl[:45]    = np.linspace(0.22, 0.87, 45)   # grip 시작: 증가
mrl[45:140] = 0.87                            # hold: 안정
mrl[140:178]= np.linspace(0.87, 0.44, 38)   # slip: 점차 감소
mrl[178:215]= 0.20                            # drop: 급격히 낮음
mrl[215:255]= 0.07                            # empty: 거의 0 (비접촉)
mrl[255:295]= np.linspace(0.07, 0.87, 40)   # re-grasp: 다시 증가
mrl[295:]   = 0.87                            # re-hold: 안정 회복
mrl += np.random.normal(0, 0.016, T)
mrl = np.clip(mrl, 0.02, 1.0)
mrl = np.convolve(mrl, np.ones(4)/4, 'same')

# Ch-A: GRU-AE reconstruction error
# - normal: low (~0.022)
# - slip: 서서히 증가 (패턴이 학습 분포에서 이탈)
# - drop/empty: 높지만 안정 (empty 신호도 OOD → 오차 유지)
# - re-grasp: 점차 감소 (정상 파지 패턴 재확립)
# - re-hold: 다시 낮아짐 (학습 분포 복귀)
recon = np.ones(T) * 0.022 + np.random.normal(0, 0.003, T)
recon[140:178] += np.linspace(0, 0.055, 38)          # slip: 점진적 증가
recon[178:215]  = 0.105 + np.random.normal(0, 0.007, 37)  # drop: 높고 불안정
recon[215:255]  = 0.112 + np.random.normal(0, 0.005, 40)  # empty: 높지만 안정
recon[255:295]  = np.linspace(0.112, 0.024, 40) + np.random.normal(0, 0.004, 40)
recon[295:]     = 0.022 + np.random.normal(0, 0.003, 50)  # re-hold: 정상 복귀
recon = np.clip(recon, 0, None)
recon = np.convolve(recon, np.ones(5)/5, 'same')
theta_A = 0.072

# Ch-B: |ΔMRL|
# - drop event에서만 spike, 나머지는 조용
mrl_delta = np.abs(np.diff(mrl, prepend=mrl[0]))
mrl_delta = np.convolve(mrl_delta, np.ones(3)/3, 'same')
mrl_delta[178:185] = [0.014, 0.042, 0.105, 0.161, 0.119, 0.072, 0.038]  # drop spike
# re-grasp 시 완만한 delta (작은 파동, threshold 미달)
mrl_delta[255:295] = np.abs(np.diff(mrl[255:295], prepend=mrl[255])) * 0.6
theta_B = 0.075

# Trigger frames
chA_trig = next((i for i in range(140, T) if recon[i] > theta_A), None)
chB_trig = next((i for i in range(175, 200) if mrl_delta[i] > theta_B), None)
# ABORT = min(chA_trig, chB_trig)
abort_frame = min(f for f in [chA_trig, chB_trig] if f is not None)
resume_frame = PH_REHOLD[0]   # 재파지 성공, 이송 재개


# ── Taxel 벡터 생성 ─────────────────────────────────────────────────
def make_taxels(state='grip'):
    """
    3D 힘 벡터 생성 (fx, fy: 전단력 XY, fz: 법선력 Z)
    - grip: Z 법선력 지배적, XY 전단력 작음
    - drop: Z 작음, XY 무작위 (산개)
    반환: px, py, fx, fy, fz, mags_3d
    """
    np.random.seed(12 if state == 'grip' else 77)
    n = 4
    px = np.array([i for i in range(n) for _ in range(n)], dtype=float)
    py = np.array([j for _ in range(n) for j in range(n)], dtype=float)
    if state == 'grip':
        fz = np.random.uniform(0.72, 0.95, n * n)   # 지배적 법선력
        fx = np.random.normal(0, 0.07, n * n)         # 소 전단력 X
        fy = np.random.normal(0, 0.07, n * n)         # 소 전단력 Y
    else:
        fz = np.random.uniform(0.04, 0.25, n * n)    # 약한 법선력
        fx = np.random.normal(0, 0.30, n * n)         # 무작위 전단 X
        fy = np.random.normal(0, 0.30, n * n)         # 무작위 전단 Y
    mags = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    return px, py, fx, fy, fz, mags


def taxel_mrl(fx, fy, fz):
    """3D 단위벡터 평균의 크기 = MRL."""
    mags = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2) + 1e-9
    mu = np.mean(fx / mags)
    mv = np.mean(fy / mags)
    mw = np.mean(fz / mags)
    return np.sqrt(mu ** 2 + mv ** 2 + mw ** 2), mu, mv, mw


# ── Phase band helper ───────────────────────────────────────────────
def draw_phases(ax, ymin, ymax):
    for (a, b), color, label in PHASES:
        ax.axvspan(a, b, alpha=0.45, color=color, zorder=0)
        ax.text((a + b) / 2, ymin + (ymax - ymin) * 0.97, label,
                ha='center', va='top', fontsize=7, color='#555555', style='italic')


# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10.125), facecolor='white')

# Header
ax_hdr = fig.add_axes([0, 0.934, 1, 0.066])
ax_hdr.set_facecolor(HEADER)
ax_hdr.axis('off')
ax_hdr.text(0.5, 0.52,
    'Tactile Anomaly Detection — Sensor Signal & Feature Visualization  '
    '[Patent Key Idea · Slide 2]',
    ha='center', va='center', fontsize=12.5, fontweight='bold',
    color='white', transform=ax_hdr.transAxes)

# Main GridSpec  (top=0.93 to avoid header)
gs = gridspec.GridSpec(3, 2, figure=fig,
    left=0.04, right=0.975,
    top=0.925, bottom=0.065,
    wspace=0.20, hspace=0.58,
    width_ratios=[0.72, 1.28])

gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], hspace=0.65)
ax_grip = fig.add_subplot(gs_left[0])
ax_slip = fig.add_subplot(gs_left[1])

ax_mrl = fig.add_subplot(gs[0, 1])
ax_cha = fig.add_subplot(gs[1, 1])
ax_chb = fig.add_subplot(gs[2, 1])

# ── Taxel circle helper ─────────────────────────────────────────────
TAXEL_CLR    = '#B8F5A0'   # 밝은 연두색 (fill)
TAXEL_EC     = '#3CB371'   # 연두색 테두리
MEAN_CLR_A   = '#1A5276'   # Panel A mean vector color
MEAN_CLR_B   = '#922B21'   # Panel B mean vector color

def draw_taxel_circles(ax, px, py, fx, fy, mags,
                       disp=0.30, r_min=0.09, r_max=0.26):
    """
    각 taxel을 원으로 표현:
    - 원 크기(반지름) ∝ 3D 힘 크기 (fx²+fy²+fz² 포함)
    - 원 중심 이동 = XY 전단력 방향 (shear displacement)
    - Z 법선력은 원 크기에만 반영 (시각적으로 XY 이동 없음)
    """
    max_mag = max(mags) if max(mags) > 0 else 1.0
    for i in range(len(px)):
        gx, gy = px[i], py[i]
        ax.plot(gx, gy, '+', color='#AAB7B8', markersize=5, lw=0.9, zorder=2)
        # XY 전단력으로 이동 (Z는 원 크기에만 기여)
        cx = gx + fx[i] * disp
        cy = gy + fy[i] * disp
        r = r_min + (mags[i] / max_mag) * (r_max - r_min)
        circ = Circle((cx, cy), r,
                      facecolor=TAXEL_CLR, edgecolor=TAXEL_EC,
                      linewidth=0.9, alpha=0.90, zorder=3)
        ax.add_patch(circ)


def draw_mean_vector(ax, mu, mv, color):
    """평균 합력 벡터 XY 투영 (화살표만, 수치 없음)."""
    cx, cy = 1.5, 1.5
    ax.annotate('', xy=(cx + mu * 1.4, cy + mv * 1.4), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.5, mutation_scale=16), zorder=6)


# ── Panel A: Firm Grip Taxels ───────────────────────────────────────
px, py, fx, fy, fz, mags = make_taxels('grip')
mrl_val, mu, mv, mw = taxel_mrl(fx, fy, fz)

draw_taxel_circles(ax_grip, px, py, fx, fy, mags)

ax_grip.set_xlim(-0.65, 3.65); ax_grip.set_ylim(-0.90, 3.65)
ax_grip.set_aspect('equal'); ax_grip.axis('off')
ax_grip.set_title(f'(A)  Firm Grip   —   MRL = {mrl_val:.2f}',
                  fontsize=10.5, fontweight='bold', color='#1E8449', pad=5)

# ── Panel B: Slip Taxels ────────────────────────────────────────────
px2, py2, fx2, fy2, fz2, mags2 = make_taxels('drop')
mrl_val2, mu2, mv2, mw2 = taxel_mrl(fx2, fy2, fz2)

draw_taxel_circles(ax_slip, px2, py2, fx2, fy2, mags2)

ax_slip.set_xlim(-0.65, 3.65); ax_slip.set_ylim(-0.90, 3.65)
ax_slip.set_aspect('equal'); ax_slip.axis('off')
ax_slip.set_title(f'(B)  Slip / Drop   —   MRL = {mrl_val2:.2f}',
                  fontsize=10.5, fontweight='bold', color='#922B21', pad=5)

# MRL formula between the two panels (figure-level text)
fig.text(0.215, 0.478,
    r'MRL $=$ $\left\|\frac{1}{N}\sum_{i=1}^{N}\hat{f}_i\right\|$   $\in$ [0, 1]',
    ha='center', va='center', fontsize=9.5, color='#2C3E50',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDFEFE',
              edgecolor='#AEB6BF', linewidth=1.0))

# ── 공통 이벤트 라인 helper ─────────────────────────────────────────
def draw_event_vlines(ax):
    # ABORT 라인
    ax.axvline(abort_frame, color='#922B21', lw=1.5, ls='-.', alpha=0.8, zorder=6)
    # Resume transport 라인
    ax.axvline(resume_frame, color='#1E8449', lw=1.5, ls='-.', alpha=0.8, zorder=6)

# ── Panel C: MRL time series ────────────────────────────────────────
ax_mrl.fill_between(t, mrl, alpha=0.18, color=C_mrl)
ax_mrl.plot(t, mrl, color=C_mrl, lw=1.9, zorder=3)
draw_phases(ax_mrl, 0.0, 1.06)
draw_event_vlines(ax_mrl)
ax_mrl.text(abort_frame + 2, 0.70, 'ABORT', fontsize=7, color='#922B21', fontweight='bold')
ax_mrl.text(resume_frame + 2, 0.70, 'RESUME', fontsize=7, color='#1E8449', fontweight='bold')
ax_mrl.set_xlim(0, T); ax_mrl.set_ylim(0, 1.06)
ax_mrl.set_ylabel('MRL', fontsize=9)
ax_mrl.set_title('(C)  Mean Resultant Length (MRL) over Time',
                 fontsize=10, fontweight='bold', color=C_mrl)
ax_mrl.set_xlabel('Time (frames)', fontsize=9)
ax_mrl.tick_params(labelsize=8)
ax_mrl.set_facecolor('#FCFCFC')
ax_mrl.spines['top'].set_visible(False)
ax_mrl.spines['right'].set_visible(False)

# ── Panel D: Ch-A reconstruction error ─────────────────────────────
ax_cha.fill_between(t, recon, alpha=0.25, color=C_chA)
ax_cha.plot(t, recon, color=C_chA, lw=1.9, zorder=3)
ax_cha.axhline(theta_A, color=C_thr, lw=1.6, ls='--', zorder=4)
draw_phases(ax_cha, 0.0, 0.158)
draw_event_vlines(ax_cha)

if chA_trig is not None:
    ax_cha.axvline(chA_trig, color=C_chA, lw=1.6, ls=':', alpha=0.9, zorder=5)
    ax_cha.annotate(f'Ch-A triggered\n(t={chA_trig})',
                    xy=(chA_trig, theta_A + 0.001),
                    xytext=(chA_trig - 45, theta_A + 0.042),
                    fontsize=7.5, color=C_chA, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_chA, lw=1.3))

ax_cha.set_xlim(0, T); ax_cha.set_ylim(0, 0.158)
ax_cha.set_ylabel('Recon. Error', fontsize=9)
ax_cha.set_title('(D)  Ch-A: GRU-AE Reconstruction Error  '
                 '(slip → high / empty → stable high / re-hold → restored)',
                 fontsize=9.5, fontweight='bold', color=C_chA)
ax_cha.set_xlabel('Time (frames)', fontsize=9)
ax_cha.tick_params(labelsize=8)
ax_cha.set_facecolor('#FCFCFC')
ax_cha.spines['top'].set_visible(False)
ax_cha.spines['right'].set_visible(False)

# ── Panel E: Ch-B |ΔMRL| ───────────────────────────────────────────
ax_chb.fill_between(t, mrl_delta, alpha=0.25, color=C_chB)
ax_chb.plot(t, mrl_delta, color=C_chB, lw=1.9, zorder=3)
ax_chb.axhline(theta_B, color=C_thr, lw=1.6, ls='--', zorder=4)
draw_phases(ax_chb, 0.0, 0.21)
draw_event_vlines(ax_chb)

if chB_trig is not None:
    ax_chb.axvline(chB_trig, color=C_chB, lw=1.6, ls=':', alpha=0.9, zorder=5)
    ax_chb.annotate(f'Ch-B triggered\n(t={chB_trig})',
                    xy=(chB_trig, mrl_delta[chB_trig] * 0.9),
                    xytext=(chB_trig + 12, mrl_delta[chB_trig] + 0.02),
                    fontsize=7.5, color=C_chB, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_chB, lw=1.3))

# Re-grasp 동안 small delta 주석
ax_chb.text((PH_REGRIP[0]+PH_REGRIP[1])//2, theta_B * 0.55,
            'gradual Δ < θ_B\n(no false trigger)',
            ha='center', fontsize=7, color='#1E8449', style='italic')

ax_chb.set_xlim(0, T); ax_chb.set_ylim(0, 0.21)
ax_chb.set_xlabel('Time  (frames)', fontsize=9)
ax_chb.set_ylabel('|ΔMRL|', fontsize=9)
ax_chb.set_title('(E)  Ch-B: MRL Frame Delta  '
                 '(instantaneous drop spike only — quiet during re-grasp)',
                 fontsize=9.5, fontweight='bold', color=C_chB)
ax_chb.tick_params(labelsize=8)
ax_chb.set_facecolor('#FCFCFC')
ax_chb.spines['top'].set_visible(False)
ax_chb.spines['right'].set_visible(False)

# ── Footer note ─────────────────────────────────────────────────────
fig.text(0.5, 0.012,
    'Taxel arrows: XY-plane projection of 16 force vectors (4×4 layout).  '
    'Color = force magnitude.  Bold arrow = mean resultant vector (length ∝ MRL).  '
    'Time-series signals are illustrative, based on typical experimental patterns.',
    ha='center', va='bottom', fontsize=7, color='#7F8C8D', style='italic')

# ── Save ────────────────────────────────────────────────────────────
path = os.path.join(OUT_DIR, 'patent_slide2_signals.pdf')
plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] {path}")
print(f"[OK] {path.replace('.pdf', '.png')}")
