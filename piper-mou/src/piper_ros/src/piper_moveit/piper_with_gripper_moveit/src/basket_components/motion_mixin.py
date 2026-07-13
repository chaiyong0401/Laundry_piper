import numpy as np
import time
from rclpy.duration import Duration
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from piper_with_gripper_moveit.action import PoseGoal
from scipy.spatial.transform import Rotation as R


class MotionMixin:
    """로봇 모션 및 그리퍼 제어 메서드."""

    def ee_pose_callback(self, msg):
        pos = msg.position
        ori = msg.orientation
        r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = [pos.x, pos.y, pos.z]
        self.ee_pose = T

        # Save as PoseStamped for alignment reference
        st = PoseStamped()
        st.header.frame_id = "base_link"
        st.header.stamp = self.get_clock().now().to_msg()
        st.pose = msg
        self.ee_pose_st = st

    # 사용자님의 원본 _send_goal 함수
    def _send_goal(self, client, joint_names, joint_values, wait_for_result=True, timeout_sec=30.0):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in joint_values]

        # if set(joint_names) == set(self.gripper_joint_names):
        #     point.time_from_start = Duration(sec=2, nanosec=0)
        # else:
        #     point.time_from_start = Duration(sec=5, nanosec=0)

        # Dynamic Duration Calculation to prevent "Status 6" (Too Slow/Fast)
        duration_sec = 3.0 # Default fallback

        try:
            # Calculate max joint difference if current state is available
            # current_joint_state is always 8-DOF [j1..j6, j7, j8]; slice to match joint_values length
            if hasattr(self, 'current_joint_state') and self.current_joint_state is not None:
                n = len(joint_values)
                if len(self.current_joint_state) >= n:
                    cur_slice = self.current_joint_state[:n]
                    diffs = [abs(t - c) for t, c in zip(joint_values, cur_slice)]
                    max_diff = max(diffs)

                    # Speed Policy:
                    # Small move (rescue): 0.05 rad -> want ~0.5s (0.1 rad/s)
                    # Large move (home): 3.0 rad -> want ~5.0s (0.6 rad/s)

                    # 속도 정책: 0.5 rad/s 기준 (이전 0.25 → 2배 빠르게)
                    req_duration = (max_diff / 0.5) + 0.3
                    duration_sec = max(0.8, req_duration)

                    if max_diff < 0.05:
                        duration_sec = 0.4
                    elif max_diff > 4.0:  # Very large
                        duration_sec = 5.0

                    # self.get_logger().info(f"  [Move] Max Diff={max_diff:.3f} rad, Duration set to {duration_sec:.2f}s")
        except Exception as e:
            self.get_logger().warn(f"Duration calc failed: {e}")
            duration_sec = 3.0

        if set(joint_names) == set(self.gripper_joint_names):
            point.time_from_start = Duration(sec=1, nanosec=0)
        else:
            s = int(duration_sec)
            ns = int((duration_sec - s) * 1e9)
            point.time_from_start = Duration(sec=s, nanosec=ns)

        goal_msg.trajectory.points.append(point)

        # wait_for_server는 __init__에서 이미 확인했으므로 여기서는 생략 가능
        # if not client.wait_for_server(timeout_sec=timeout_sec):
        #     self.get_logger().error(f"Action server '{client._action_name}' not available.")
        #     return False

        send_goal_future = client.send_goal_async(goal_msg)
        # executor가 메인 스레드에서 spin 중 → future가 자동 완료됨, polling으로 대기
        _send_start = time.time()
        while not send_goal_future.done():
            if time.time() - _send_start > 5.0:
                self.get_logger().error(
                    f"[_send_goal] TIMEOUT: goal 전송 응답 없음 ({client._action_name})")
                return False
            time.sleep(0.005)

        goal_handle = send_goal_future.result()
        self.current_goal_handle = goal_handle  # Store handle for cancellation
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn(f"Goal for '{client._action_name}' rejected.")
            return False
        if not wait_for_result:
            return True

        # 결과 대기 타임아웃: 그리퍼는 3s, 팔은 timeout_sec (기본 30s)
        is_gripper = set(joint_names) == set(self.gripper_joint_names)
        result_timeout = 3.0 if is_gripper else timeout_sec

        get_result_future = goal_handle.get_result_async()
        # executor가 spin 중 → polling으로 대기
        _result_start = time.time()
        while not get_result_future.done():
            if time.time() - _result_start > result_timeout:
                self.get_logger().error(
                    f"[_send_goal] TIMEOUT ({result_timeout:.0f}s): 결과 미수신 ({client._action_name}) "
                    f"→ goal 취소 시도")
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                return False
            if getattr(self, '_transport_abort', None) and self._transport_abort.is_set():
                try:
                    goal_handle.cancel_goal_async()
                    # 취소 처리가 완료될 때까지 대기 (새 goal reject 방지)
                    _c = time.time()
                    while not get_result_future.done() and time.time() - _c < 3.0:
                        time.sleep(0.01)
                except Exception:
                    pass
                return False
            time.sleep(0.005)

        result = get_result_future.result()
        if not result:
            self.get_logger().error("결과를 받지 못했습니다.")
            return False

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            return True
        elif result.status == 6:
            # [FIX] Status 6 (ABORTED) often happens due to strict path/goal tolerances.
            # If we are physically close enough, we can consider it a success.
            # threshold 0.15 rad (~8.5 deg): 물리적으로 도달했으나 controller tolerance로
            # abort되는 정상 케이스를 success로 처리.
            # Rescue에서의 미이동 오판은 rescue 루프 내 before/after 비교로 별도 처리.
            n = len(joint_values)
            if hasattr(self, 'current_joint_state') and self.current_joint_state is not None and len(self.current_joint_state) >= n:
                diffs = [abs(t - c) for t, c in zip(joint_values, self.current_joint_state[:n])]
                max_diff = max(diffs)
                if max_diff < 0.15: # Within ~8.5 degrees
                    self.get_logger().info(f"Action aborted (Status 6) but near goal (diff={max_diff:.3f} rad). Treating as success.")
                    return True
            self.get_logger().warn(f"Action '{client._action_name}' failed with status: {result.status}")
            return False
        else:
            self.get_logger().warn(f"Action '{client._action_name}' failed with status: {result.status}")
            return False

    def goal(self, joint_values_8dof, gripper_action='keep'):
        self.get_logger().info(f"[Goal 명령] Pose: {joint_values_8dof}, Gripper: {gripper_action}")
        arm_target = joint_values_8dof[:6]

        success = False
        # [Added] Check if already at goal to prevent MoveIt Status 6 Abort
        if hasattr(self, 'current_joint_state') and self.current_joint_state is not None and len(self.current_joint_state) >= 6:
            diffs = [abs(a - c) for a, c in zip(arm_target, self.current_joint_state[:6])]
            if all(d < 0.05 for d in diffs): # ~2.8 degrees tolerance
                self.get_logger().info("  >> 이미 목표 관절 각도에 도달해 있습니다. (명령 스킵)")
                success = True
            else:
                success = self._send_goal(self._arm_client, self.arm_joint_names, arm_target, wait_for_result=True)
        else:
            success = self._send_goal(self._arm_client, self.arm_joint_names, arm_target, wait_for_result=True)

        # MoveIt planning 실패(status 6) 시 direct JTC로 fallback
        # transport abort 후 MoveGroupInterface 내부 상태가 corrupted되면 plan()이 계속 실패함
        if not success:
            self.get_logger().warn("  >> MoveIt planning 실패 → direct JTC fallback 시도")
            success = self._send_goal(self._arm_direct_client, self.arm_joint_names, arm_target, wait_for_result=True)

        if not success:
            self.get_logger().error("  >> 팔 이동 실패")
            return False

        if gripper_action == 'open':
            return self.open_gripper()
        elif gripper_action == 'close':
            return self.close_gripper()

        return True

    def send_gripper_command(self, joint7, joint8, wait_for_result=True):
        gripper_target = [joint7, joint8]
        return self._send_goal(
            self._gripper_client,
            self.gripper_joint_names,
            gripper_target,
            wait_for_result=wait_for_result
        )

    def open_gripper(self):
        self.get_logger().info("그리퍼 오픈")
        return self.send_gripper_command(0.03, 0.03, wait_for_result=True)


    def close_gripper(self):
        self.get_logger().info("그리퍼 클로즈")
        return self.send_gripper_command(0.0, -0.0, wait_for_result=True)

    def _wait_future(self, future, timeout_sec=None, label: str = ""):
        """executor.spin() 중일 때 future를 polling으로 대기.
        rclpy.spin_until_future_complete()는 이미 spin 중인 executor와 충돌하므로 사용 불가.
        """
        import traceback as _tb
        caller = _tb.extract_stack()[-2]
        tag = label or f"{caller.filename.split('/')[-1]}:{caller.lineno}"
        start = time.time()
        warned_1s = False
        while not future.done():
            elapsed = time.time() - start
            if not warned_1s and elapsed > 1.0:
                self.get_logger().warn(f"[_wait_future] 1s 경과 미완료: {tag}")
                warned_1s = True
            if timeout_sec is not None and elapsed > timeout_sec:
                self.get_logger().error(
                    f"[_wait_future] TIMEOUT ({timeout_sec}s) 초과: {tag} → future 미완료")
                return False
            time.sleep(0.005)
        elapsed = time.time() - start
        if elapsed > 0.5:
            self.get_logger().info(f"[_wait_future] 완료: {tag} ({elapsed:.2f}s)")
        return True

    def recover_robot(self):
        """Re-enable robot to clear Safety/Torque limits."""
        try:
            from piper_msgs.srv import Enable
            if not self._enable_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("/enable_srv not available.")
                return False

            req = Enable.Request()
            req.enable_request = True

            future = self._enable_client.call_async(req)
            # Synchronous wait for simplicity in this logic
            self._wait_future(future, timeout_sec=1.0)

            if future.result() is not None:
                self.get_logger().info("✅ Robot Re-Enabled (Fault Cleared)")
                import time
                time.sleep(1.0) # [CRITICAL] Stabilize controller state
                return True
            else:
                self.get_logger().warn("Failed to enable robot (No response)")
                return False
        except Exception as e:
            self.get_logger().warn(f"recover_robot check failed: {e}")
            return False

    def move_forward(self, pose_st: PoseStamped, fallback_to_ik=True, disable_torque=False, ik_mode=6, rpy_tol=[0.0, 0.0, 0.0], skip_mode9=False):
        """
        Hybrid Move Forward (Mode 9 -> Mode ik_mode)
        with 4-sigma Torque Stop Protection.
        Returns: "success", "stop", "fail"
        ik_mode: 6 for self-only, 10 for env-aware, 1 for planning with rpy_tol (added)
        skip_mode9: True → skip Cartesian Mode 9, go directly to ik_mode
        """
        ik_name = "Mode 6 (Joint IK)" if ik_mode == 6 else (f"Mode 10 (Env IK)" if ik_mode == 10 else f"Mode {ik_mode}")
        if skip_mode9:
            self.get_logger().info(f"[forward] Starting Direct {ik_name} (Mode 9 skipped)")
        else:
            self.get_logger().info(f"[forward] Starting Hybrid Move (Mode 9 -> {ik_name})")

        # --- Internal Helper to run a specific mode ---
        def run_mode(mode_val):
            # 1. Reset Stop Flags & Enable Torque Monitor
            with self._lock:
                self.stop_requested = False
                self.stop_reason = ""
                self.torque_stop = False
                self.monitor_torque = not disable_torque

            goal_msg = PoseGoal.Goal()
            goal_msg.target_pose = pose_st
            goal_msg.mode = mode_val
            if len(rpy_tol) == 3:
                goal_msg.roll_tol_rad  = float(rpy_tol[0])
                goal_msg.pitch_tol_rad = float(rpy_tol[1])
                goal_msg.yaw_tol_rad   = float(rpy_tol[2])

            # [NEW] 속도 제한 (Pilz Planner 지시 시 적용)
            goal_msg.velocity_scaling = 0.6
            goal_msg.acceleration_scaling = 0.6

            # 2. Send Goal
            send_future = self._pose_client.send_goal_async(goal_msg)
            self._wait_future(send_future, timeout_sec=5.0)

            if not send_future.done():
                self.get_logger().error(f"[forward] mode={mode_val} goal send timeout")
                return "timeout"

            goal_handle = send_future.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().warn(f"[forward] mode={mode_val} goal rejected")
                return "rejected"

            # 3. Wait Result (OMPL 기반 Mode 1은 planning + execution 시간 포함)
            result_timeout = 50.0 if mode_val == 1 else 15.0
            result_future = goal_handle.get_result_async()
            self._wait_future(result_future, timeout_sec=result_timeout)

            if not result_future.done():
                self.get_logger().error(f"[forward] mode={mode_val} result wait timeout")
                goal_handle.cancel_goal_async()  # MoveIt server 정리
                return "timeout"

            result = result_future.result().result

            # 4. Check Stop Flag (Torque Collision)
            if self.stop_requested:
                self.get_logger().warn(f"[forward] 4σ stop flag detected during mode={mode_val} → stop")
                return "stop"

            if result.success:
                self.get_logger().info(f"[forward] mode={mode_val} success: {result.message}")
                return "success"
            else:
                self.get_logger().warn(f"[forward] mode={mode_val} failed: {result.message}")
                return "fail"

        # --- Step 1. Try Mode 9 (Cartesian) --- skip if skip_mode9=True
        if not skip_mode9:
            self.get_logger().info("Trying Mode 9 (Cartesian)...")
            res9 = run_mode(9)

            if res9 == "success":
                return "success"
            if res9 == "stop":
                return "stop"

        # If Mode 9 failed (or rejected/timeout) -> Fallback
        if skip_mode9 or fallback_to_ik:
            ik_name = "Mode 6 (Joint IK)" if ik_mode == 6 else f"Mode {ik_mode} (Env IK)"
            if skip_mode9:
                self.get_logger().info(f"Directly running {ik_name}...")
            else:
                self.get_logger().warn(f"Mode 9 result='{res9}'. Retrying with {ik_name}...")
            # --- Step 2. Try IK Mode ---
            res_ik = run_mode(ik_mode)
            return res_ik
        else:
            self.get_logger().error(f"Mode 9 result='{res9}'. Fallback disabled -> Abort.")
            return "fail"

    def move_backward(self, pose: PoseStamped):
        """잡은 후 안전하게 후진(Retreat)하는 함수 (Z + 15cm, Mode 6 직행)"""
        self.get_logger().info("(Retreat) 시작: Z축 +15cm (Mode 6 직행)")

        # 깊은 복사 대신 새로운 객체 생성
        retreat_pose = PoseStamped()
        retreat_pose.header = pose.header
        retreat_pose.pose.orientation = pose.pose.orientation

        retreat_pose.pose.position.x = pose.pose.position.x
        retreat_pose.pose.position.y = pose.pose.position.y
        retreat_pose.pose.position.z = pose.pose.position.z + 0.15

        # Retreat 시에는 그리퍼가 바구니/세탁물 모델과 겹쳐 있으므로 Mode 6 (자가충돌만 체크) 직행.
        # Mode 9 (Cartesian Pilz)는 작업 공간 경계에서 실패해 9초 타임아웃이 발생하므로 skip.
        res = self.move_forward(retreat_pose, fallback_to_ik=True, ik_mode=6, skip_mode9=True)

        if res == "success":
            return True
        else:
            self.get_logger().warn(f"(Retreat) move_forward result: {res}")
            return False

    def transform_pose_with_offset(self, pose: Pose, HT: np.ndarray):
        # 1. position 추출
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z

        # 2. 동차좌표 벡터 생성
        point_vec = np.array([x, y, z, 1.0])  # 4x1 벡터

        # 3. 동차변환행렬 곱
        transformed = HT @ point_vec  # (4x4) @ (4x1) → (4x1)

        # 4. 새로운 Pose로 반환
        new_pose = Pose()
        new_pose.position.x = transformed[0]
        new_pose.position.y = transformed[1]
        new_pose.position.z = transformed[2]

        # orientation은 그대로 유지하거나 필요시 수정
        new_pose.orientation = pose.orientation

        return new_pose

    def align_joint6(self):
        """joint 1~5 유지, joint 6만 회전하여 wm_init_ee_orientation 기준 수평 정렬.
        MoveIt collision 우회를 위해 /arm_controller 직접 전송."""
        from scipy.spatial.transform import Rotation as R_scipy
        if self.current_joint_state is None:
            self.get_logger().warn("[align_joint6] joint state 미수신, 스킵")
            return False
        if self.ee_pose is None or self.wm_init_ee_orientation is None:
            self.get_logger().warn("[align_joint6] EE pose 또는 wm_init_ee_orientation 미수신, 스킵")
            return False

        R_current = R_scipy.from_matrix(self.ee_pose[:3, :3])
        R_desired  = self.wm_init_ee_orientation
        R_diff = (R_current.inv() * R_desired).as_matrix()
        delta6 = float(np.arctan2(R_diff[1, 0], R_diff[0, 0]))
        self.get_logger().info(f"[align_joint6] delta6={np.degrees(delta6):.2f}°")

        j6_new = float(self.current_joint_state[5]) + delta6
        j6_new = float(np.clip(j6_new, -2.0944, 2.0944))
        target = list(self.current_joint_state[:6])
        target[5] = j6_new
        return self._send_goal(self._arm_direct_client, self.arm_joint_names, target, wait_for_result=True)
