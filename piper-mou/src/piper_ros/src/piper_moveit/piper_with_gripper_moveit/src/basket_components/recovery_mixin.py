import time
import threading
import numpy as np
from rclpy.duration import Duration
from builtin_interfaces.msg import Duration as DurationMsg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from basket_components.constants import CTRL_JOINT_NAMES


class RecoveryMixin:
    """토크/낙하 이상 감지 시 정지·복구 메서드."""

    def _request_stop(self, reason: str):
        with self._lock:
            if self.stop_requested:
                return
            self.stop_requested = True
            self.stop_reason = reason

        self.get_logger().warn(f" 정지 요청: {reason}")
        threading.Thread(target=self._send_stop_override, daemon=True).start()

    def _send_stop_override(self):
        """현재 joint_states 기반으로 6축 arm을 즉시 정지하는 trajectory 전송."""
        if self.last_js_msg is None:
            self.get_logger().error("stop override 실패: joint_states 없음")
            return

        name_to_pos = {n: v for n, v in zip(self.last_js_msg.name, self.last_js_msg.position)}
        try:
            q_now = [float(name_to_pos[n]) for n in CTRL_JOINT_NAMES]
        except KeyError as e:
            self.get_logger().error(f"stop override 실패: 누락 관절 {e}")
            return

        jt = JointTrajectory()
        jt.joint_names = CTRL_JOINT_NAMES
        jt.header.stamp = self.get_clock().now().to_msg()

        p1 = JointTrajectoryPoint()
        p1.positions = q_now
        p1.velocities = [0.0] * len(CTRL_JOINT_NAMES)
        p1.time_from_start = Duration(sec=0, nanosec=10_000_000)

        p2 = JointTrajectoryPoint()
        p2.positions = q_now
        p2.velocities = [0.0] * len(CTRL_JOINT_NAMES)
        p2.time_from_start = Duration(sec=0, nanosec=20_000_000)

        jt.points = [p1, p2]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt

        self.get_logger().warn("stop override trajectory 전송 (비동기)")
        future = self._arm_client.send_goal_async(goal)

        def _goal_resp_cb(fut):
            try:
                gh = fut.result()
            except Exception as e:
                self.get_logger().error(f"stop override goal 응답 예외: {e!r}")
                return

            if not gh or not gh.accepted:
                self.get_logger().error("stop override goal 거부됨")
                return

            def _result_cb(rfut):
                try:
                    wrapped = rfut.result()
                    self.get_logger().info(
                        f"stop override 완료: status={wrapped.status} "
                    )
                except Exception as e:
                    self.get_logger().error(f"stop override 결과 콜백 예외: {e!r}")

            gh.get_result_async().add_done_callback(_result_cb)

        future.add_done_callback(_goal_resp_cb)

    def _handle_stop_and_recover(self):
        if not self.stop_requested:
            return False

        self.get_logger().warn(f"🛑 STOP 처리 시작: {self.stop_reason}")
        self._send_stop_override()

        with self._lock:
            self.stop_requested = False
            self.stop_reason = ""

        self.get_logger().info("✅ STOP 처리 완료 → main에서 close/back 진행")
        return True

    def _trigger_slip_stop(self):
        if not self.contacted:
            self.contacted = True
            self.get_logger().warn(f"🚨 [MONITOR] Slip Recorded! Will recover after motion.")
            self.monitor_slip = False

    def _floor_recovery_grasp(self) -> bool:
        """이송 중 낙하한 세탁물을 바닥에서 찾아 재파지. True=성공."""
        import copy

        if self.current_joint_state is None:
            self.get_logger().error("[RECOVERY] joint_state 미수신")
            return False

        j1_now = float(self.current_joint_state[0])
        recovery_pose = [j1_now] + list(self.ba_init[1:6]) + [0.0, 0.0]
        self.get_logger().info(f"[RECOVERY] 복구 자세 이동 (j1={j1_now:.4f}, joints2-6=ba_init)")

        for _ in range(3):
            if self.goal(recovery_pose, 'keep'):
                break
            self.get_logger().warn("[RECOVERY] 복구 자세 이동 실패 → 재시도")
        else:
            self.get_logger().error("[RECOVERY] 복구 자세 이동 3회 실패")
            return False

        self.reset_sensor_data()
        self.get_logger().info("[RECOVERY] 바닥 세탁물 감지 중...")
        target = self.target_get_ba(mode="basket")

        if target.pose.position.z == -99.0:
            self.get_logger().warn("[RECOVERY] 세탁물 미감지 → 재파지 포기")
            return False

        hover_pose = copy.deepcopy(target)
        hover_pose.pose.position.z = 0.0

        pre_grasp_pose = copy.deepcopy(target)
        pre_grasp_pose.pose.position.z += 0.05

        res = self.move_forward(hover_pose, fallback_to_ik=True, ik_mode=10)
        if res not in ('success', 'stop'):
            self.get_logger().warn(f"[RECOVERY] Hover 이동 실패: {res}")
            return False

        res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, ik_mode=6)
        if res not in ('success', 'stop'):
            self.get_logger().warn(f"[RECOVERY] Pre-Grasp 이동 실패: {res}")
            return False

        res = self.move_forward(target, fallback_to_ik=True, ik_mode=6)
        if res not in ('success', 'stop'):
            self.get_logger().warn(f"[RECOVERY] Push 이동 실패: {res}")
            return False

        self.close_gripper()
        time.sleep(0.5)

        for _ in range(3):
            if self.move_backward(target):
                break
            self.get_logger().warn("[RECOVERY] Retreat 실패 → 재시도")

        while not self.goal(recovery_pose, 'keep'):
            self.get_logger().warn("[RECOVERY] 복구 자세 복귀 실패 → 재시도")
            time.sleep(0.5)

        is_grasp = self.max_norm > 0.45
        self.get_logger().info(
            f"[RECOVERY] norm={self.norm:.3f} max_norm={self.max_norm:.3f} "
            f"→ {'재파지 성공' if is_grasp else '재파지 실패'}"
        )

        if not is_grasp:
            self.open_gripper()
            return False

        return True
