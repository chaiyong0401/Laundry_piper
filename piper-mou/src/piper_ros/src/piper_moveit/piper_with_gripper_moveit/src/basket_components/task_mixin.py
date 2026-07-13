import time
import copy
import numpy as np
import datetime
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as ScipyRotation


class TaskMixin:
    """고수준 태스크 시퀀스: 세탁물 파지(overcome2) + 전체 작업 루프(execute_sequence)."""

    def overcome2(self):
        while True:
            self.open_gripper()
            self.get_logger().info("Step 1: View Pose로 이동 (Empty Check)")
            while not self.goal(self.wm_view, 'keep'):
                self.get_logger().warn("  >> wm_view 이동 실패 → 재시도 중")
                time.sleep(0.5)

            chk_pose = self.target_get_wm(mode="failure_eh")

            if chk_pose is None:
                self.get_logger().warn("⚠️ [Overcome] 감지 결과 없음 (인식 실패). 루프 재시도.")
                continue

            if chk_pose.pose.position.z == -99.0:
                self.get_logger().info("🛑 [Overcome] 세탁물 없음 (RGB Gate). 작업 종료.")
                break

            self.get_logger().info("✅ [Overcome] 의류 감지됨. 파지 자세로 진입.")
            self.get_logger().info("Step 2: 파지 자세(wm_mid)로 이동 및 좌표 계산")

            while not self.goal(self.wm_mid, 'keep'):
                self.get_logger().warn("  >> wm_mid 이동 실패 → 재시도 중")
                time.sleep(0.5)
            time.sleep(1.0)

            time.sleep(0.05)
            if self.ee_pose is not None:
                from scipy.spatial.transform import Rotation as R_scipy
                self.wm_init_ee_orientation = R_scipy.from_matrix(self.ee_pose[:3, :3])
                self.get_logger().info("[overcome2] wm_mid EE orientation 캡처 완료")

            target_wm = self.target_get_wm(mode="failure_eh_strict")
            self.reset_sensor_data()

            if target_wm is None:
                self.get_logger().info("결과 없음")
                break

            self.get_logger().info(f"[DEBUG] target_wm frame_id is: '{target_wm.header.frame_id}'")
            is_fixed = (target_wm.header.frame_id == "base_link|fixed")
            target_wm.header.frame_id = "base_link"

            t_r, t_g, t_b = (1.0, 0.0, 0.0) if is_fixed else (0.0, 0.0, 1.0)
            target_name = "Fixed WM Target" if is_fixed else "Target WM"
            self.publish_target_marker(target_wm.pose, id=2, r=t_r, g=t_g, b=t_b)
            self.get_logger().info(f"  [OVERCOME] {target_name}: \n{target_wm.pose}")

            from scipy.spatial.transform import Rotation as R_scipy
            pre_grasp_pose = copy.deepcopy(target_wm)

            q = [target_wm.pose.orientation.x, target_wm.pose.orientation.y, target_wm.pose.orientation.z, target_wm.pose.orientation.w]
            rot = R_scipy.from_quat(q)
            retreat_vec = rot.apply([0, 0, -0.03])

            pre_grasp_pose.pose.position.x += retreat_vec[0]
            pre_grasp_pose.pose.position.y += retreat_vec[1]
            pre_grasp_pose.pose.position.z += retreat_vec[2]

            p_r, p_g, p_b = (1.0, 0.5, 0.0) if is_fixed else (0.0, 1.0, 1.0)
            self.get_logger().info(f" Pre-Grasp Approach ... ")
            self.publish_target_marker(pre_grasp_pose.pose, id=3, r=p_r, g=p_g, b=p_b)
            self.get_logger().info("  [OVERCOME] Valid Target found. Moving to Pre-Grasp (EE -3cm)...")
            res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=11, rpy_tol=[6.28, 6.28, 6.28])
            if res != "success":
                self.get_logger().warn("  [OVERCOME] Pre-Grasp Mode 11 실패 → Mode 6 재시도")
                res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=6, rpy_tol=[6.28, 6.28, 6.28])
            if res != "success":
                self.get_logger().warn("  [OVERCOME] Pre-Grasp Mode 6 실패 → Mode 0 (OMPL) 재시도")
                res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=0, rpy_tol=[6.28, 6.28, 6.28])
            approach_success = (res == "success")
            if approach_success:
                self.get_logger().info(f"  [OVERCOME] Approach Success!")

                self.get_logger().info("[overcome2] joint6 align 중...")
                for _aj6_try in range(3):
                    if self.align_joint6():
                        break
                    self.get_logger().warn(f"  >> joint6 정렬 실패 → 재시도 ({_aj6_try+1}/3)")

                time.sleep(0.05)
                if self.ee_pose is not None:
                    curr_rot = self.ee_pose[:3, :3]
                    q_cur = R_scipy.from_matrix(curr_rot).as_quat()
                    for ps in [target_wm, pre_grasp_pose]:
                        ps.pose.orientation.x = float(q_cur[0])
                        ps.pose.orientation.y = float(q_cur[1])
                        ps.pose.orientation.z = float(q_cur[2])
                        ps.pose.orientation.w = float(q_cur[3])
                    self.get_logger().info(
                        f"[align_joint6 후] orientation 갱신 (위치 유지): "
                        f"pos=[{target_wm.pose.position.x:.3f}, {target_wm.pose.position.y:.3f}, {target_wm.pose.position.z:.3f}] "
                        f"q=[{q_cur[0]:.3f}, {q_cur[1]:.3f}, {q_cur[2]:.3f}, {q_cur[3]:.3f}]"
                    )
            if not approach_success:
                self.get_logger().warn("  >> [OVERCOME] Pre-Grasp failed (All Angles). Skipping this target.")
                continue

            self.get_logger().info("  [OVERCOME] Moving to Grasp Target...")
            p = target_wm.pose.position
            o = target_wm.pose.orientation
            self.get_logger().info(
                f"  [OVERCOME] target_wm pose:\n"
                f"    position   : x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}\n"
                f"    orientation: x={o.x:.4f}  y={o.y:.4f}  z={o.z:.4f}  w={o.w:.4f}"
            )
            forward_state = None
            for i in range(2):
                forward_state = self.move_forward(target_wm, fallback_to_ik=True, disable_torque=True, ik_mode=11, rpy_tol=[6.28, 6.28, 6.28])

                if forward_state == "success":
                    break

                if forward_state == "stop":
                    self._handle_stop_and_recover()
                    self.recover_robot()
                    break

                self.get_logger().warn(f"forward 실패 ({i+1}/2) → 재시도 중")

            else:
                self.get_logger().warn("forward 2회 실패 → 새 target")

                self.get_logger().warn("  [DIAGNOSE] Mode 6 (IK only) 로 도달 가능성 테스트...")
                diag_result = self.move_forward(target_wm, fallback_to_ik=True, disable_torque=True, ik_mode=6, rpy_tol=[6.28, 6.28, 6.28])
                if diag_result == "success":
                    self.get_logger().warn(
                        f"  [DIAGNOSE] Mode 6 성공 → IK 존재. 직선 경로 문제 (joint limit / singularity).\n"
                        f"    position   : x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}\n"
                        f"    orientation: x={o.x:.4f}  y={o.y:.4f}  z={o.z:.4f}  w={o.w:.4f}"
                    )
                    forward_state = "success"
                else:
                    self.get_logger().error(
                        f"  [DIAGNOSE] Mode 6 도 실패 → IK 해 없음. pose 자체가 workspace 밖이거나 충돌 상태.\n"
                        f"    position   : x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}\n"
                        f"    orientation: x={o.x:.4f}  y={o.y:.4f}  z={o.z:.4f}  w={o.w:.4f}"
                    )
                    self.get_logger().warn("  [FALLBACK] Mode 0 (OMPL) 로 push 시도...")
                    ompl_result = self.move_forward(target_wm, fallback_to_ik=True, disable_torque=True, ik_mode=0, rpy_tol=[6.28, 6.28, 6.28])
                    if ompl_result == "success":
                        self.get_logger().info("  [FALLBACK] Mode 0 (OMPL) 성공!")
                        forward_state = "success"
                    else:
                        self.get_logger().error(
                            f"  [FALLBACK] Mode 0 도 실패. 이 target 포기.\n"
                            f"    position   : x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}"
                        )
                        continue

            self.grasp_buffer = []
            self.grasp_raw_buffer = []
            self.is_recording = True
            self.close_gripper()
            self.get_logger().info("  [OVERCOME] Retreating to pre-grasp...")
            ret = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=11, rpy_tol=[6.28, 6.28, 6.28])
            if ret != "success":
                self.get_logger().warn("  [OVERCOME] pre_grasp 복귀 실패 → Mode 6 재시도")
                ret = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=6, rpy_tol=[6.28, 6.28, 6.28])
            if ret != "success":
                self.get_logger().warn("  [OVERCOME] pre_grasp 복귀 실패 → Mode 0 (OMPL) 재시도")
                self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=0, rpy_tol=[6.28, 6.28, 6.28])

            # 4. 촉각 기반 파지 성공 판단 (rule-based only)
            time.sleep(0.05)
            self.is_recording = False
            is_grasp      = self.max_norm > 0.35
            is_valid_norm = self.norm > 0.30
            is_not_edge   = self.theta < 85.0
            grasp_success = is_grasp and is_valid_norm and is_not_edge
            result_src = "rule-based"
            if grasp_success:
                self.get_logger().info(
                    f"  [OVERCOME] 파지 성공 [{result_src}] (norm={self.norm:.3f}, max_norm={self.max_norm:.3f}, theta={self.theta:.1f}°)"
                )
            else:
                self.get_logger().warn(
                    f"  [OVERCOME] 파지 실패 [{result_src}] (norm={self.norm:.3f}, max_norm={self.max_norm:.3f}, theta={self.theta:.1f}°) → 그리퍼 오픈 후 재탐색"
                )
                self.open_gripper()
                continue

            self.get_logger().info("  >> [Overcome] Force Retreat 5cm (Blind)")

            current_pose = self.ee_pose
            back_vec = current_pose[:3, :3] @ np.array([0, 0, -0.05])
            target_back_mat = current_pose.copy()
            target_back_mat[:3, 3] += back_vec

            r = ScipyRotation.from_matrix(target_back_mat[:3, :3])
            qx, qy, qz, qw = r.as_quat()

            retreat_pose = PoseStamped()
            retreat_pose.header.frame_id = "base_link"
            retreat_pose.pose.position.x = target_back_mat[0, 3]
            retreat_pose.pose.position.y = target_back_mat[1, 3]
            retreat_pose.pose.position.z = target_back_mat[2, 3]
            retreat_pose.pose.orientation.x = qx
            retreat_pose.pose.orientation.y = qy
            retreat_pose.pose.orientation.z = qz
            retreat_pose.pose.orientation.w = qw

            # pre_grasp → ff → wm_end
            while not self.goal(self.ff, 'keep'):
                self.get_logger().warn("  >> ff 이동 실패 → 재시도 중")
                time.sleep(0.5)
            while not self.goal(self.wm_end, 'keep'):
                self.get_logger().warn("  >> wm_end 이동 실패 → 재시도 중")
                time.sleep(0.5)
            self.get_logger().info("Z-축 정렬 감지는 생략합니다!!")

            self.z_aligned = True

            self.get_logger().info("  >> [Loop] 세탁기(wm_end)로 이동하여 놓기")

            while not self.goal(self.wm_end, 'keep'):
                self.get_logger().warn("  >> wm_end 이동 실패 → 재시도 중")
                time.sleep(0.5)

            self.open_gripper()

            self.contacted = False
            time.sleep(1.0)
            continue

    def execute_sequence(self):
        self.overcome2()

        while True:
            import rclpy
            if not rclpy.ok():
                break

            while True:
                self.open_gripper()
                self.z_aligned = False
                self.get_logger().info("Step 1: 이니셜 포즈(ba_init)로 이동 중")
                while not self.goal(self.ba_init, 'keep'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                    time.sleep(0.5)
                self.reset_sensor_data()
                self.get_logger().info("Step 2: 잡기 시도 중...")
                target_ba = self.target_get_ba(mode="basket")
                if target_ba.pose.position.z == -99.0:
                    self.get_logger().info("🛑 [Overcome] 빈 바구니 감지 (RGB Gate). overcome2 호출.")
                    self.overcome2()
                    break

                pre_grasp_pose = copy.deepcopy(target_ba)
                pre_grasp_pose.pose.position.z += 0.05

                self.get_logger().info("\n==================================================")
                self.get_logger().info(f"  [TARGET] BA (Grasp): \n{target_ba.pose}")
                target_ba_world = copy.deepcopy(target_ba.pose)
                target_ba_world.position.z += 0.445
                self.get_logger().info(f"  [WORLD] BA (Grasp) (Base+0.445): \n{target_ba_world}")
                self.get_logger().info("==================================================")
                self.publish_target_marker(target_ba.pose, id=0, r=1.0, g=0.0, b=0.0)

                self.get_logger().info("\n==================================================")
                self.get_logger().info(f"  [PRE-TARGET] Pre-Grasp Pose (+5cm): \n{pre_grasp_pose.pose}")
                pre_grasp_world = copy.deepcopy(pre_grasp_pose.pose)
                pre_grasp_world.position.z += 0.445
                self.get_logger().info(f"  [WORLD] Pre-Grasp (Base+0.445): \n{pre_grasp_world}")
                self.get_logger().info("==================================================")
                self.publish_target_marker(pre_grasp_pose.pose, id=1, r=0.0, g=1.0, b=0.0)

                self.get_logger().info("Step 2-1: 접근(Pre-Grasp) 하강 중... (Collision Check ON)")
                approach_success = False

                for i in range(3):
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=False, disable_torque=True)
                    if res == "success":
                        approach_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Linear)\033[0m")
                        break
                    self.get_logger().info(f"\033[94m  >> [Stage 1] 직선 하강 실패 ({i+1}/3) result={res} → 재시도\033[0m")

                if not approach_success:
                    self.get_logger().info("\033[94m [Stage 2] 엄격한 수직 IK(Mode 10) 시도\033[0m")
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, ik_mode=10, disable_torque=True)
                    if res == "success":
                        approach_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Strict IK)\033[0m")

                if not approach_success:
                    self.get_logger().info("\033[94m [Stage 4] Orientation 완전 자유(Mode 1, no path constraint) 시도\033[0m")
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, ik_mode=1, rpy_tol=[6.28, 6.28, 6.28], disable_torque=True)
                    if res == "success":
                        approach_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Free Orientation)\033[0m")

                if approach_success:
                    tip_pose = np.eye(4)
                    tip_pose[:3, :3] = self.ee_pose[:3, :3]
                    offset_vec = self.ee_pose[:3, :3] @ np.array([0, 0, 0.11])
                    tip_pose[:3, 3] = self.ee_pose[:3, 3] + offset_vec
                    self.get_logger().info(f"  >> Current Wrist : {self.ee_pose[:3,3]}")
                    self.get_logger().info(f"\033[92m  >> Computed Tip  : {tip_pose[:3,3]}\033[0m")

                    target_ba.pose.orientation = self.ee_pose_st.pose.orientation

                if not approach_success:
                    self.get_logger().warn("  >> Pre-Grasp 전부 실패 → 새 target 요청")
                    continue

                self.get_logger().info("Step 2-3: 잡기(Push) 이동 중...")
                forward_state = None
                for i in range(3):
                    forward_state = self.move_forward(target_ba, ik_mode=11)
                    if forward_state == "success":
                        self.get_logger().info("\n\033[92m[SUCCESS] Target BA 도달!\033[0m")
                        tip_pos_ba = self.ee_pose[:3, 3] + (self.ee_pose[:3, :3] @ np.array([0, 0, 0.11]))
                        self.get_logger().info(f"  >> Current Wrist : {self.ee_pose[:3,3]}")
                        self.get_logger().info(f"\033[92m  >> Computed Tip  : {tip_pos_ba}\033[0m")
                        break

                    if forward_state == "stop":
                        self._handle_stop_and_recover()
                        break

                    self.get_logger().warn(f"forward 실패 ({i+1}/5) result={forward_state} → 재시도 중")

                else:
                    self.get_logger().warn("forward 5회 실패 → 새 target")
                    continue

                self.close_gripper()

                if self.run_mode in ('collect', 'inference'):
                    self.reset_sensor_data()
                    self.grasp_buffer = []
                    self.grasp_raw_buffer = []
                    self.realtime_buffer.clear()
                    self.realtime_raw_buffer.clear()
                    self.inference_frame_cnt          = 0
                    self.infer_consec_normal          = 0
                    self.infer_consec_anomaly         = 0
                    self.infer_latest_score           = None
                    self.infer_grasp_anomaly_detected = False
                    self.grasp_channel_b_flag        = False
                    self.grasp_channel_b_consec      = 0
                    self.grasp_delta_b_latest        = 0.0
                    if self.grasp_channel_b is not None:
                        self.grasp_channel_b.reset()
                    if self.grasp_api_detector is not None:
                        self.grasp_api_detector.reset()
                    self.is_recording = True
                    self.get_logger().info(f"[DATA] 녹화 시작 (mode={self.run_mode})")

                for i in range(5):
                    if self.move_backward(target_ba):
                        break
                    self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")

                while not self.goal(self.ba_init, 'keep'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                    time.sleep(0.5)

                if self.run_mode in ('collect', 'inference'):
                    self.is_recording = False
                    n_frames = len(self.grasp_buffer)
                    if n_frames > 1:
                        arr = np.array(self.grasp_buffer)
                        frozen = int(np.sum(np.all(np.abs(np.diff(arr, axis=0)) < 1e-5, axis=1)))
                        frozen_pct = 100.0 * frozen / (n_frames - 1)
                        quality = "양호" if frozen_pct < 10 else ("경고" if frozen_pct < 50 else "불량-센서끊김의심")
                        self.get_logger().info(
                            f"[DATA] 녹화 종료: {n_frames} 프레임 "
                            f"| frozen={frozen_pct:.1f}% [{quality}]"
                        )
                    else:
                        self.get_logger().info(f"[DATA] 녹화 종료: {n_frames} 프레임")

                    if self.run_mode == 'inference' and len(self.grasp_raw_buffer) > 0:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        dbg_path = f"/home/dyros/mcy_ws/grasp_05/infer_debug_{ts}_raw.npy"
                        np.save(dbg_path, np.array(self.grasp_raw_buffer, dtype=np.float32))
                        self.get_logger().info(f"[DEBUG] inference raw 저장: {dbg_path} shape={np.array(self.grasp_raw_buffer).shape}")

                self.get_logger().info(" Z-축 정렬 (Slip Check) 감지")

                grab_norm = self.norm
                grab_theta = self.theta
                current_theta = self.theta
                theta_diff = abs(current_theta - grab_theta)

                is_grasp = self.max_norm > 0.45
                is_stable = theta_diff < 6.0
                is_valid_norm = self.norm > 0.35
                is_not_edge = current_theta < 85.0

                rule_success = is_stable and is_valid_norm and is_not_edge and is_grasp

                if self.run_mode == 'collect':
                    held = self._ask_gripper_held()
                    if held:
                        self._save_grasp_data(label='success')
                        self.z_aligned = True
                        break
                    elif held is False:
                        self._save_grasp_data(label='failure')
                        while not self.open_gripper():
                            self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
                    else:
                        if rule_success:
                            self.z_aligned = True
                            break
                        else:
                            while not self.open_gripper():
                                self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")

                elif self.run_mode == 'inference':
                    with self._infer_lock:
                        anom_detected = self.infer_grasp_anomaly_detected
                        score         = self.infer_latest_score
                    score_str = f"{score:.4f}" if score is not None else "None"
                    self.get_logger().info(
                        f"[INFERENCE] anomaly_detected={anom_detected} "
                        f"(consec_needed={self.infer_consec_anomaly_needed}), "
                        f"최종 score={score_str}")
                    if not anom_detected:
                        self.get_logger().info("[INFERENCE] 성공 판정")
                        self.z_aligned = True
                        break
                    else:
                        self.get_logger().warn("[INFERENCE] 실패 판정 → 재시도")
                        while not self.open_gripper():
                            self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")

                else:
                    if rule_success:
                        self.z_aligned = True
                        break
                    else:
                        self.get_logger().warn(f" 잡기 실패: Δθ={theta_diff:.1f}°, θ={current_theta:.1f}°, norm={self.norm:.3f})")
                        while not self.open_gripper():
                            self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")

            if target_ba.pose.position.z == -99.0:
                self.get_logger().info("🛑 [Overcome] 빈 바구니 감지 (RGB Gate). 작업 종료.")
                break

            self.get_logger().info("Step 3: 엔드 포즈(wm_end)로 이동 중")
            self.contacted = False

            self.monitor_initial_norm = self.norm
            self.monitor_initial_theta = self.theta
            self.monitor_slip = (self.run_mode != 'inference')
            self.get_logger().info(f"👀 Slip Monitoring START (Ref: norm={self.norm:.2f}, theta={self.theta:.1f})")

            if self.run_mode in ('collect', 'inference'):
                self.reset_sensor_data()
                self.insert_buffer = []
                self.insert_raw_buffer = []
                self.insert_realtime_buffer.clear()
                self.insert_infer_frame_cnt           = 0
                self.insert_infer_consec_normal       = 0
                self.insert_infer_consec_anomaly      = 0
                self.insert_infer_latest_score        = None
                self.insert_anomaly_ever_detected     = False
                self.transport_channel_b_flag         = False
                self.transport_channel_b_consec       = 0
                self.transport_delta_b_latest         = 0.0
                if self.transport_channel_b is not None:
                    self.transport_channel_b.reset()
                self.is_insert_recording = True
                self.get_logger().info("[DATA] 이송 구간 녹화 시작 (ba_init → open_gripper)")

            self._transport_abort.clear()

            while not self.goal(self.wm_init, 'keep'):
                if self._transport_abort.is_set():
                    break
                self.get_logger().warn("  >> 실패 → 재시도 중")
                time.sleep(0.5)

            if not self._transport_abort.is_set():
                while not self.goal(self.wm_end, 'keep'):
                    if self._transport_abort.is_set():
                        break
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                    time.sleep(0.5)

            if self._transport_abort.is_set() and self.run_mode == 'inference':
                self.is_insert_recording = False
                self.get_logger().warn("[INFERENCE] 낙하 감지 → 바닥 재파지 시도")
                time.sleep(0.3)
                self.open_gripper()
                self._transport_abort.clear()

                recovered = self._floor_recovery_grasp()
                if recovered:
                    self.get_logger().info("[RECOVERY] 재파지 성공 → 이송 재개")
                    self._transport_abort.clear()
                    while not self.goal(self.wm_init, 'keep'):
                        if self._transport_abort.is_set():
                            break
                        self.get_logger().warn("  >> [RECOVERY] wm_init 이동 실패 → 재시도")
                        time.sleep(0.5)
                    if not self._transport_abort.is_set():
                        while not self.goal(self.wm_end, 'keep'):
                            if self._transport_abort.is_set():
                                break
                            self.get_logger().warn("  >> [RECOVERY] wm_end 이동 실패 → 재시도")
                            time.sleep(0.5)
                    if not self._transport_abort.is_set():
                        self.open_gripper()
                        time.sleep(0.3)
                else:
                    self.get_logger().warn("[RECOVERY] 재파지 실패 → 바구니로 복귀")
                    while not self.goal(self.ba_init, 'keep'):
                        self.get_logger().warn("  >> ba_init 복귀 실패 → 재시도 중")
                        time.sleep(0.5)
                continue

            if self.run_mode in ('collect', 'inference'):
                self.is_insert_recording = False
                self.get_logger().info(f"[DATA] 이송 구간 녹화 종료: {len(self.insert_buffer)} 프레임")

            if self.run_mode == 'collect':
                held = self._ask_insertion_result()
                if held is not None:
                    self._save_insert_data(label='success' if held else 'failure')

            if self.run_mode == 'inference':
                with self._infer_lock:
                    insert_consec  = self.insert_infer_consec_normal
                    insert_needed  = self.insert_infer_consec_needed
                    insert_score   = self.insert_infer_latest_score
                    anomaly_ever   = self.insert_anomaly_ever_detected
                insert_ok = (not anomaly_ever) and (insert_consec >= insert_needed)
                score_str = (f"{insert_score:.6f}"
                             if insert_score is not None else "N/A")
                self.get_logger().info(
                    f"[INFERENCE] 이송 판정: {'정상' if insert_ok else '이상'} "
                    f"(anomaly_ever={anomaly_ever}, consec={insert_consec}/{insert_needed}, "
                    f"latest_score={score_str})"
                )
                if not insert_ok:
                    self.get_logger().warn("[INFERENCE] 이송 중 낙하 감지 → 극복 동작 트리거")
                    self.contacted = True

            self.monitor_slip = False
            self.get_logger().info("👀 Slip Monitoring END")

            self.get_logger().info("Step 4: 접촉 감지 결과 확인 중")
            current_theta = self.theta
            theta_diff = abs(current_theta - grab_theta)

            is_stable = theta_diff < 6.0
            is_valid_norm = self.norm > 0.35
            is_not_edge = current_theta < 85.0

            if self.contacted:
                self.get_logger().warn(f" Slip Recorded during Motion! (Overcome Trigger)")
            elif is_stable and is_valid_norm and is_not_edge:
                self.get_logger().warn(f" No slip: Δθ={theta_diff:.1f}°, θ={current_theta:.1f}°), norm={self.norm:.3f}")

            self.open_gripper()

            if self.contacted:
                self.get_logger().info("  >> 접촉 감지됨 → 실패 극복")
                self.overcome2()
            else:
                self.get_logger().info("  >> 접촉 없음")

            while not self.goal(self.wm_init, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
                time.sleep(0.5)
            continue
