"""
시스템 아키텍처 - 작업 순서 기반 정보 흐름 다이어그램
"""
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(20, 11))
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('#F5F6FA')

C_VIS  = '#1A5276'
C_TAC  = '#6C3483'
C_TASK = '#1E8449'
C_CTRL = '#7B241C'
C_HW   = '#2C3E50'

# ── 블록 그리기 ────────────────────────────────────────────────
def block(ax, x, y, w, h, color, title, items, tfs=10.5, ifs=8.8):
    # 외곽
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1,rounding_size=0.2",
                          facecolor=color, edgecolor='white', linewidth=2.5,
                          alpha=0.13, zorder=1)
    ax.add_patch(rect)
    # 타이틀 바
    tb = FancyBboxPatch((x, y+h-0.72), w, 0.72,
                        boxstyle="round,pad=0.05,rounding_size=0.15",
                        facecolor=color, edgecolor='none', alpha=0.88, zorder=2)
    ax.add_patch(tb)
    ax.text(x+w/2, y+h-0.36, title, ha='center', va='center',
            fontsize=tfs, fontweight='bold', color='white', zorder=3)
    # 아이템
    n = len(items)
    step = (h - 0.72 - 0.3) / max(n, 1)
    for i, item in enumerate(items):
        iy = y + h - 0.72 - 0.2 - i*step - step/2
        ir = FancyBboxPatch((x+0.18, iy-0.24), w-0.36, 0.48,
                            boxstyle="round,pad=0.04,rounding_size=0.08",
                            facecolor=color, edgecolor='white', linewidth=1.0,
                            alpha=0.50, zorder=2)
        ax.add_patch(ir)
        ax.text(x+w/2, iy, item, ha='center', va='center',
                fontsize=ifs, color='white', fontweight='bold', zorder=3)

# ── 번호 있는 화살표 ───────────────────────────────────────────
def seq_arrow(ax, x1, y1, x2, y2, color, num, label,
              lw=2.2, rad=0.0, num_side='top'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'), zorder=5)
    mx, my = (x1+x2)/2, (y1+y2)/2
    off = 0.22 if num_side == 'top' else -0.22
    # 라벨
    ax.text(mx, my+off, label, ha='center', va='center', fontsize=8.0,
            color=color, fontweight='bold', zorder=7,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor=color, alpha=0.92, linewidth=1.3))
    # 번호 원
    circ = Circle((x1 + (x2-x1)*0.18, my + off*0.5 + (0.35 if num_side=='top' else -0.35)),
                  0.19, color=color, zorder=8)
    ax.add_patch(circ)
    ax.text(x1 + (x2-x1)*0.18, my + off*0.5 + (0.35 if num_side=='top' else -0.35),
            str(num), ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=9)

# ════════════════════════════════════════════════════
# 4개 블록
# ════════════════════════════════════════════════════
BX = [0.3, 5.1, 9.9, 14.7]   # 각 블록 x 시작
BW = 4.5                        # 폭
BH = 9.8                        # 높이
BY = 0.6                        # y 시작

block(ax, BX[0], BY, BW, BH, C_VIS,
      'Vision Pipeline',
      ['RealSense D435\n(pyrealsense2)',
       'Background Sub\n→ 6ch input 생성',
       'TinyUNet\n→ Binary Mask',
       '3D Localization\n→ (x, y, z)',
       'JSON 출력\n→ node2.py'])

block(ax, BX[1], BY, BW, BH, C_TAC,
      'Tactile Pipeline',
      ['XELA Sensor\n/xServTopic',
       'Force Processing\nmean_vec_biased',
       'State Detection\nNo Grasp / Stable / Perturb',
       'Slip Monitor\nnorm↓ / theta↑',
       'State Flags\nstop_requested'])

block(ax, BX[2], BY, BW, BH, C_TASK,
      'Task Manager  (basket2.py)',
      ['Stage 1\n의류 위치 감지 요청',
       'Stage 2\n파지 위치 이동',
       'Stage 2\n그리퍼 닫기',
       'Stage 3\n바구니로 이동',
       'Stage 3\n그리퍼 열기 / 완료'])

block(ax, BX[3], BY, BW, BH, C_CTRL,
      'Robot Control',
      ['pose_goal_server\nMode 9 / OMPL / IK',
       'MoveIt2 + KDL IK\nTOTG 경로 생성',
       'JointTrajectoryController\narm / gripper',
       'piper_ctrl_single_node\nCAN → SDK',
       'Piper Robot\n6-DOF + Gripper'])

# ════════════════════════════════════════════════════
# 작업 순서별 화살표
# ════════════════════════════════════════════════════

# 오른쪽 경계 / 왼쪽 경계
V_R  = BX[0] + BW        # Vision 우측   4.8
T_L  = BX[1]             # Tactile 좌측  5.1
T_R  = BX[1] + BW        # Tactile 우측  9.6
M_L  = BX[2]             # Manager 좌측  9.9
M_R  = BX[2] + BW        # Manager 우측 14.4
C_L  = BX[3]             # Control 좌측 14.7

# ① Task Manager → Vision: 감지 요청
seq_arrow(ax, M_L, 8.9, V_R+0.01, 8.9, C_VIS, 1,
          '① 감지 요청', rad=-0.25, num_side='top')

# ② Vision → Task Manager: 파지 위치 반환
seq_arrow(ax, V_R, 8.1, M_L-0.01, 8.1, C_VIS, 2,
          '② Grasp Pose (x,y,z)', rad=0.25, num_side='top')

# ③ Task Manager → Robot Control: 파지 위치로 이동
seq_arrow(ax, M_R, 7.3, C_L, 7.3, C_TASK, 3,
          '③ PoseGoal (pre-grasp)', num_side='top')

# ④ Robot Control → Task Manager: 완료 응답
seq_arrow(ax, C_L, 6.5, M_R-0.01, 6.5, C_CTRL, 4,
          '④ success / fail', num_side='top')

# ⑤ Task Manager → Robot Control: 그리퍼 닫기
seq_arrow(ax, M_R, 5.5, C_L, 5.5, C_TASK, 5,
          '⑤ Gripper Close (pos=0)', num_side='top')

# ⑥ Tactile → Task Manager: 상태 플래그 (지속)
seq_arrow(ax, T_R, 4.3, M_L-0.01, 4.3, C_TAC, 6,
          '⑥ State Flags (연속)', num_side='top')

# ⑦ Task Manager → Robot Control: 바구니 이동
seq_arrow(ax, M_R, 3.3, C_L, 3.3, C_TASK, 7,
          '⑦ PoseGoal (basket)', num_side='top')

# ⑧ Tactile → Task Manager: 슬립 감지 (이상 시)
seq_arrow(ax, T_R, 2.2, M_L-0.01, 2.2, C_TAC, 8,
          '⑧ stop_requested (슬립)', num_side='top', rad=0.0)

# ⑨ Task Manager → Robot Control: 그리퍼 열기
seq_arrow(ax, M_R, 1.4, C_L, 1.4, C_TASK, 9,
          '⑨ Gripper Open (pos=0.04)', num_side='top')

# ════════════════════════════════════════════════════
# 타이틀 & 범례
# ════════════════════════════════════════════════════
ax.text(10.0, 10.72, 'Piper Robot Laundry Task — Information Flow',
        ha='center', va='center', fontsize=14,
        fontweight='bold', color='#1C2833')

legend_items = [
    mpatches.Patch(color=C_VIS,  label='Vision Pipeline'),
    mpatches.Patch(color=C_TAC,  label='Tactile Pipeline'),
    mpatches.Patch(color=C_TASK, label='Task Manager'),
    mpatches.Patch(color=C_CTRL, label='Robot Control'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=4,
          framealpha=0.92, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout()
out = '/home/dyros/mcy_ws/system_architecture.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
