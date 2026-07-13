"""
Piper 로봇의 바구니 내부 도달 가능 영역(workspace) 분석 스크립트
- pinocchio로 IK 가능 여부를 샘플링
- 바구니 내부 3D 그리드를 검사하여 도달 가능/불가능 시각화
"""

import subprocess, tempfile, os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ===== 설정 =====
XACRO_PATH = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_description/urdf/piper_description.xacro"
URDF_PATH  = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_description/urdf/piper_description.urdf"
TIP_LINK   = "DEEP"        # MoveIt과 동일한 tip (link6 +0.11m 앞)
BASE_LINK  = "base_link"

# ba_init: hover 위치 근처의 알려진 도달 가능 자세 (초기값으로 사용)
BA_INIT = np.array([-1.6379, 1.5724, -1.2721, -0.0566, 1.2838, 0.2908, 0.0, 0.0])

# xacro 모델은 world frame 기준 (base_link = world + z_offset)
# basket2.py 좌표는 base_link frame → world frame 변환: z += 0.445
BASE_Z_OFFSET = 0.445

# 바구니 내부 범위 (base_link frame → world frame 변환)
X_MIN, X_MAX = -0.24,        0.12          # x offset 없음
Y_MIN, Y_MAX = -0.50,       -0.26          # y offset 없음
Z_MIN, Z_MAX = -0.32 + BASE_Z_OFFSET, 0.02 + BASE_Z_OFFSET  # 0.125 ~ 0.465

# 샘플링 해상도 (높을수록 정밀하지만 느림)
N_X, N_Y, N_Z = 10, 10, 12

# IK 설정
IK_MAX_ITER = 500
IK_STEP     = 0.3
IK_EPS_POS  = 5e-3   # 5mm
IK_EPS_ORI  = 0.15   # rad

# ===== 모델 로드 =====
def load_model(xacro_path):
    import xacro
    xml = xacro.process_file(xacro_path).toxml()
    model = pin.buildModelFromXML(xml)
    data  = model.createData()
    print(f"[INFO] 모델 로드: {model.name}, nq={model.nq}, njoints={model.njoints}")
    frames = [model.frames[i].name for i in range(len(model.frames))]
    print(f"[INFO] frames: {frames}")
    return model, data

# ===== IK (pinocchio CLIK) =====
def check_ik(model, data, target_pos, target_rot_mat, tip_id, q_init):
    """
    target_pos: (3,) base frame
    target_rot_mat: (3,3) base frame
    q_init: 초기 joint 값 (ba_init 권장)
    return: (success: bool, q: np.ndarray or None)
    """
    target_se3 = pin.SE3(target_rot_mat, target_pos)

    # 여러 초기값으로 시도 (ba_init + 약간의 노이즈)
    np.random.seed(42)
    q_inits = [q_init.copy()] + [
        np.clip(q_init + np.random.uniform(-0.3, 0.3, len(q_init)),
                model.lowerPositionLimit, model.upperPositionLimit)
        for _ in range(2)
    ]

    for q0 in q_inits:
        q = q0.copy()
        for _ in range(IK_MAX_ITER):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            cur = data.oMf[tip_id]
            err = pin.log6(cur.inverse() * target_se3).vector
            if np.linalg.norm(err[:3]) < IK_EPS_POS and np.linalg.norm(err[3:]) < IK_EPS_ORI:
                return True, q
            J  = pin.computeFrameJacobian(model, data, q, tip_id, pin.LOCAL)
            dq = np.linalg.lstsq(J, err, rcond=None)[0]
            q  = pin.integrate(model, q, dq * IK_STEP)
            q  = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    return False, None

# ===== 메인 분석 =====
def analyze():
    model, data = load_model(XACRO_PATH)

    # tip frame id 찾기
    tip_id = None
    for fid in range(len(model.frames)):
        if model.frames[fid].name == TIP_LINK:
            tip_id = fid
            break
    if tip_id is None:
        # 가장 가까운 이름으로 fallback
        names = [model.frames[i].name for i in range(len(model.frames))]
        print(f"[WARN] '{TIP_LINK}' 없음. 사용 가능한 frame: {names}")
        tip_id = len(model.frames) - 1
        print(f"[WARN] 마지막 frame 사용: {model.frames[tip_id].name}")

    print(f"[INFO] tip frame: '{model.frames[tip_id].name}' (id={tip_id})")

    # ba_init orientation을 기준 orientation으로 사용
    pin.forwardKinematics(model, data, BA_INIT)
    pin.updateFramePlacements(model, data)
    base_rot = data.oMf[tip_id].rotation.copy()

    # 테스트할 orientation: ba_init 기준 + roll/pitch 변형
    orientations = [base_rot]
    for roll_deg in [-30, -15, 15, 30]:
        for pitch_deg in [-30, -15, 15, 30]:
            r = R.from_euler('xyz', [np.radians(roll_deg), np.radians(pitch_deg), 0])
            orientations.append(r.as_matrix() @ base_rot)

    xs = np.linspace(X_MIN, X_MAX, N_X)
    ys = np.linspace(Y_MIN, Y_MAX, N_Y)
    zs = np.linspace(Z_MIN, Z_MAX, N_Z)

    reachable   = []
    unreachable = []
    total = N_X * N_Y * N_Z
    count = 0

    print(f"[INFO] 총 {total}개 포인트 검사 시작 (orientation {len(orientations)}가지)...")

    for x in xs:
        for y in ys:
            for z in zs:
                pos = np.array([x, y, z])
                reached = False
                for rot in orientations:
                    ok, _ = check_ik(model, data, pos, rot, tip_id, BA_INIT)
                    if ok:
                        reached = True
                        break
                if reached:
                    reachable.append(pos)
                else:
                    unreachable.append(pos)
                count += 1
                if count % 50 == 0:
                    print(f"  {count}/{total} ({100*count//total}%)")

    reachable   = np.array(reachable)   if reachable   else np.empty((0, 3))
    unreachable = np.array(unreachable) if unreachable else np.empty((0, 3))

    print(f"\n[결과] 도달 가능: {len(reachable)} / {total} ({100*len(reachable)//total}%)")
    print(f"[결과] 도달 불가: {len(unreachable)} / {total}")

    # Z별 도달률 출력
    print("\n[Z 구간별 도달률] (base frame 기준)")
    for z in zs:
        r_cnt = np.sum(np.abs(reachable[:, 2]   - z) < 0.01) if len(reachable)   else 0
        u_cnt = np.sum(np.abs(unreachable[:, 2] - z) < 0.01) if len(unreachable) else 0
        tot   = r_cnt + u_cnt
        pct   = 100 * r_cnt // tot if tot > 0 else 0
        z_base = z - BASE_Z_OFFSET
        print(f"  z_base={z_base:+.3f}m (z_world={z:+.3f}m): {r_cnt}/{tot} ({pct}%) 도달 가능")

    # ===== 시각화 =====
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    if len(reachable) > 0:
        ax.scatter(reachable[:, 0], reachable[:, 1], reachable[:, 2],
                   c='green', alpha=0.6, s=30, label=f'도달 가능 ({len(reachable)})')
    if len(unreachable) > 0:
        ax.scatter(unreachable[:, 0], unreachable[:, 1], unreachable[:, 2],
                   c='red', alpha=0.3, s=30, label=f'도달 불가 ({len(unreachable)})')

    # 바구니 경계 박스
    bx = [X_MIN, X_MAX, X_MAX, X_MIN, X_MIN]
    by_bot = [Y_MIN, Y_MIN, Y_MAX, Y_MAX, Y_MIN]
    for z_edge in [Z_MIN, Z_MAX]:
        ax.plot(bx, by_bot, [z_edge]*5, 'b--', alpha=0.4, linewidth=1)
    for xi, yi in zip([X_MIN, X_MAX, X_MAX, X_MIN], [Y_MIN, Y_MIN, Y_MAX, Y_MAX]):
        ax.plot([xi, xi], [yi, yi], [Z_MIN, Z_MAX], 'b--', alpha=0.4, linewidth=1)

    # 실패 포인트 표시 (base frame → world frame 변환)
    failed_base = np.array([-0.005, -0.484, -0.260])
    failed_world = failed_base + np.array([0, 0, BASE_Z_OFFSET])
    ax.scatter(*failed_world, c='black', s=100, marker='X', zorder=5,
               label=f'실패 포인트 (base={failed_base})')

    ax.set_xlabel('X (world)')
    ax.set_ylabel('Y (world)')
    ax.set_zlabel('Z (world)  [base = world - 0.445]')
    ax.set_title('Piper 바구니 내부 Workspace 분석 (DEEP 기준)\n(초록=도달가능, 빨강=불가, 검정X=실제 실패 포인트)')
    ax.legend(fontsize=9)

    # 우측에 통계 텍스트
    fig.text(0.75, 0.15,
             f"총 {total}점\n도달: {len(reachable)} ({100*len(reachable)//total}%)\n불가: {len(unreachable)}",
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('/tmp/piper_workspace.png', dpi=150, bbox_inches='tight')
    print("\n[INFO] 그래프 저장: /tmp/piper_workspace.png")
    plt.show()

if __name__ == "__main__":
    analyze()
