#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, math, threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from piper_sdk import C_PiperInterface_V2

DEG_PER_RAD = 180.0 / math.pi
SCALE = 1000  # 0.001 단위(0.001deg/0.001mm)

# === 업↔다운(왕복) 파라미터 ===
BOUNCE_JOINT   = 2        # 0-based index (J3)
BOUNCE_AMP_DEG = 6.0      # ±6 deg
BOUNCE_PERIOD  = 1.0      # 1초 주기

class FTJArbiter(Node):
    def __init__(self):
        super().__init__('arbiter_ftj')

        # Piper SDK: CAN은 여기서만 OPEN
        self.p = C_PiperInterface_V2('can0')
        self.p.ConnectPort()
        self.p.EnableArm(7)
        time.sleep(0.05)
        try:
            # 기본: P/J 모드 중 하나로 속도만 세팅 (필요 시 조정)
            self.p.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        except Exception:
            pass

        # === FTJ 액션 서버 ===
        self.server = ActionServer(
            self,
            FollowJointTrajectory,
            '/arbiter_controller/follow_joint_trajectory',
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
        )

        # === FTJ/바운스 상태 ===
        self._cancel = threading.Event()
        self._bounce_enable = threading.Event()
        self._bounce_th = threading.Thread(target=self._bounce_worker, daemon=True)
        self._bounce_th.start()
        self.get_logger().info('ARBiter FTJ ActionServer up.')

    # ---------- 유틸 ----------
    def _get_current_q_md(self):
        """
        현재 조인트(0.001deg int) 6개 리턴. 실패 시 None.
        """
        try:
            aj = self.p.GetArmJointMsgs()
            # aj.joint_state 안에 Joint 1..6 값이 0.001deg 단위로 들어있음
            s = str(aj.joint_state)
            # 간단 파싱: 'Joint 1:####' 형태
            vals = []
            for j in range(1,7):
                key = f"Joint {j}:"
                idx = s.find(key)
                if idx == -1:
                    return None
                rest = s[idx+len(key):].strip().splitlines()[0]
                vals.append(int(rest.split()[0]))
            return vals
        except Exception:
            return None

    def _try_set_joint_mode(self, speed=100):
        try:
            self.p.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # 시도1: J모드 가정
            return True
        except Exception:
            pass
        try:
            # self.p.MotionCtrl_2(0x01, 0x00, speed, 0x00)  # 시도2
            return True
        except Exception:
            pass
        return False

    def _send_jointctrl(self, q_md):
        """
        JointCtrl(*q_md) 호출. 실패 시 False
        """
        try:
            # JointCtrl 쓰는 걸 확인했다고 했으니 바로 호출
            self.p.JointCtrl(*q_md)
            return True
        except TypeError:
            # 혹시 시그니처가 리스트 하나면 이걸로
            try:
                self.p.JointCtrl(q_md)
                return True
            except Exception:
                return False
        except Exception:
            return False

    # ---------- 바운스 쓰레드 ----------
    def _bounce_worker(self):
        """
        _bounce_enable set() 되면:
        - 현재 q_md를 읽어서 J3 기준으로 ±BOUNCE_AMP_DEG 왕복
        - JointCtrl 로 0.5주기마다 토글
        """
        amp_md = int(round(BOUNCE_AMP_DEG * SCALE))  # 0.001deg
        halfT = BOUNCE_PERIOD / 2.0

        while rclpy.ok():
            if not self._bounce_enable.is_set():
                time.sleep(0.02)
                continue

            base = self._get_current_q_md()
            if not base:
                time.sleep(0.05)
                continue

            up = base[:]  # up/down 목표 생성
            dn = base[:]
            up[BOUNCE_JOINT] = base[BOUNCE_JOINT] + amp_md
            dn[BOUNCE_JOINT] = base[BOUNCE_JOINT] - amp_md

            # 조인트 모드 보장
            self._try_set_joint_mode(speed=100)

            # 토글 루프 (다음 FTJ goal이 오면 goal_cb가 enable을 끔)
            target_up = True
            while self._bounce_enable.is_set():
                # if target_up:
                #     # self._send_jointctrl(up)
                # else:
                #     self._send_jointctrl(dn)
                # target_up = not target_up
                time.sleep(halfT)

    # ---------- FTJ 콜백 ----------
    def goal_cb(self, goal_request):
        # 새 FTJ 들어오면 바운스 중지
        if self._bounce_enable.is_set():
            self.get_logger().info("FTJ goal in → stop bounce")
        self._bounce_enable.clear()
        self._cancel.clear()
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        self.get_logger().info('FTJ cancel requested.')
        self._cancel.set()
        return CancelResponse.ACCEPT

    def execute_cb(self, goal_handle):
        goal = goal_handle.request
        traj = goal.trajectory
        points = traj.points
        if not points:
            goal_handle.abort()
            return FollowJointTrajectory.Result(
                error_code=FollowJointTrajectory.Result.INVALID_GOAL,
                error_string='Empty trajectory'
            )

        # 조인트 모드 보장
        if not self._try_set_joint_mode(speed=100):
            self.get_logger().warn("failed to set joint mode; continuing")

        t0 = time.time()
        names = list(traj.joint_names)
        # 이름이 비어있지 않다면 joint1..6 순서 매핑
        name_to_idx = {n:i for i,n in enumerate(names)} if names else {}

        for pt in points:
            if self._cancel.is_set():
                goal_handle.canceled()
                return FollowJointTrajectory.Result(
                    error_code=FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                    error_string='Canceled'
                )

            t_rel = pt.time_from_start.sec + pt.time_from_start.nanosec*1e-9
            now = time.time()
            if t_rel > 0 and now - t0 < t_rel:
                time.sleep(max(0.0, t_rel - (now - t0)))

            # positions → 라디안 6개 → 0.001deg int
            q_rad = [0.0]*6
            if pt.positions:
                if names:  # 이름 기반 매핑
                    # 기대 이름: joint1..joint6
                    for j in range(6):
                        key = f"joint{j+1}"
                        if key in name_to_idx and name_to_idx[key] < len(pt.positions):
                            q_rad[j] = float(pt.positions[name_to_idx[key]])
                else:
                    q_rad = list(pt.positions[:6])
            q_md = [int(round(v * DEG_PER_RAD * SCALE)) for v in q_rad]

            self._send_jointctrl(q_md)

            # 피드백(옵션)
            fb = FollowJointTrajectory.Feedback()
            fb.joint_names = traj.joint_names
            fb.desired = JointTrajectoryPoint(positions=q_rad,
                                              time_from_start=pt.time_from_start)
            goal_handle.publish_feedback(fb)

        goal_handle.succeed()

        # === 실행 끝나면 바운스 시작 ===
        self.get_logger().info("FTJ done → start bounce")
        self._bounce_enable.set()

        return FollowJointTrajectory.Result(
            error_code=FollowJointTrajectory.Result.SUCCESSFUL
        )

def main():
    rclpy.init()
    rclpy.spin(FTJArbiter())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
