import rclpy
import time
import os
import argparse
import datetime
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from builtin_interfaces.msg import Duration
from xela_server_ros2.msg import SensStream
import numpy as np
import threading
from piper_with_gripper_moveit.action import TargetPose
from piper_with_gripper_moveit.action import PoseGoal
import sys
from scipy.spatial.transform import Rotation as R
from rclpy.clock import Clock
from collections import deque
from sensor_msgs.msg import JointState

########################## 비지도 클러스터링 ####################################################

# import joblib

# EPS = 1e-6

# # fcm.joblib 언피클용 최소 클래스(이 이름이 없으면 로드 에러)
# class FCMModel:
#     def __init__(self, prep=None, centers=None, m: float = 2.0) -> None:
#         self.prep = prep
#         self.centers_ = centers
#         self.m = float(m)

# def _np64(x):
#     return np.require(np.asarray(x, dtype=np.float64), dtype=np.float64, requirements=["C"])

# def _extract_imputer_scaler(prep):
#     """ColumnTransformer('num' 파이프라인)에서 imputer/scaler 파라미터 추출"""
#     num = prep.named_transformers_["num"]
#     imputer = num.named_steps["impute"]
#     scaler  = num.named_steps["scale"]
#     imp = _np64(imputer.statistics_)     # (3,)
#     mean = _np64(scaler.mean_)           # (3,)
#     scale = _np64(scaler.scale_)         # (3,)
#     scale[scale == 0.0] = 1.0
#     return imp, mean, scale

# def _preprocess_mb(mb_vec3, imp_stats, mean, scale):
#     """mb(3,) → impute+standardize → (1,3) float64"""
#     X = _np64(mb_vec3).reshape(1, 3)
#     if np.isnan(X).any():
#         X = np.where(np.isnan(X), imp_stats.reshape(1, 3), X)
#     Xs = (X - mean.reshape(1, 3)) / scale.reshape(1, 3)
#     return _np64(Xs)

# def _fcm_membership(Xs, centers, m):
#     """Xs: (n,3), centers: (C,3) → U: (C,n)"""
#     Xs = _np64(Xs); centers = _np64(centers)
#     d = np.linalg.norm(Xs[:, None, :] - centers[None, :, :], axis=2)
#     d = np.maximum(d, EPS)
#     power = 2.0 / (m - 1.0)
#     U = 1.0 / ((d[:, :, None] / d[:, None, :]) ** power).sum(axis=2)
#     return U.T

##############################################################################################


class ControlTower(Node):

    def __init__(self, run_mode: str = 'normal'):
        super().__init__('control_tower_node')
        self.run_mode = run_mode
        self.get_logger().info(f"[MODE] 실행 모드: {self.run_mode}")
        self._timing_log = []

        self.ee_pose = None
        self.wm_init_ee_orientation = None  # wm_init 도달 시 캡처한 수평 그리퍼 orientation
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        self.cum_x = [0.0] * 16
        self.cum_y = [0.0] * 16
        self.cum_z = [0.0] * 16

        self.z_aligned = False
        self.contacted = False

        self.z_aligned_k = False
        self.contacted_k = False
        self.contacted_kk = False
        self.z_aligned_f = False
        self.contacted_f = False
        self.contacted_ff = False

        self.norm = None
        self.max_norm = 0.0
        self.theta = None

########################## 비지도 클러스터링 ####################################################

        # # === KMeans/FCM 모델 로드 및 전처리 파라미터 준비 ===
        self.kmeans_available = False
        self.fcm_available = False

        self.kmeans = None
        self.fcm_centers = None
        self.fcm_m = 2.0

        self.imp_stats = None
        self.scaler_mean = None
        self.scaler_scale = None

        self.minmax_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/taxel_minmax_norm.npy"
        self.gmin, self.grng = self._load_minmax(self.minmax_path)
        self.get_logger().info(f"MinMax loaded: {self.minmax_path}")

        # [CALIBRATION] Camera Extrinsics (EE to RGB Optical Center)
        # These are used for BOTH coordinate transformation and RViz visualization
        # [D435i]
        # self.R_cal = np.array([
        #     [0.025154, 0.997896, 0.059755],
        #     [-0.999639, 0.025669, -0.007874],
        #     [-0.009391, -0.059535, 0.998182]
        # ])
        # self.t_cal = np.array([[-0.058162],
        #                        [0.035760],
        #                        [0.07954]])

        # [D405]
        self.R_cal = np.array([
            [-0.084547, 0.992441, -0.088956],
            [-0.995873, -0.081206, 0.040533],
            [0.033003, 0.092016, 0.995210]
        ])
        self.t_cal = np.array([[-0.050437],
                               [0.007361],
                               [0.061211]])
        # self.R_cal = np.array([
        #     [-0.07403140416428758,  0.9970253999896115, -0.02144068499412377],
        #     [-0.997213265121594,   -0.07381230005954229,  0.010837353249242954],
        #     [0.009222530183886607,  0.022183239967900933,  0.9997113827508088]])
        # self.t_cal = np.array([[-0.05615831961252831],
        #                      [ 0.035761448900299364],
        #                      [ 0.06590399326489311]]) 

        # # 1) kmeans.joblib (sklearn Pipeline: prep + kmeans)
        # try:
        #     km_art = joblib.load("/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/kmeans_norm.joblib")
        #     if hasattr(km_art, "named_steps") and "prep" in km_art.named_steps and "kmeans" in km_art.named_steps:
        #         prep = km_art.named_steps["prep"]
        #         self.kmeans = km_art.named_steps["kmeans"]
        #         self.imp_stats, self.scaler_mean, self.scaler_scale = _extract_imputer_scaler(prep)
        #         self.kmeans_available = True
        #         self.get_logger().info("Loaded kmeans.joblib")
        # except Exception as e:
        #     self.get_logger().warn(f"kmeans.joblib load failed: {e}")

        # # 2) fcm.joblib (dict 또는 FCMModel)
        # try:
        #     fc_art = joblib.load("/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/fcm_norm.joblib")
        #     if isinstance(fc_art, dict) and fc_art.get("type", "") == "fcm":
        #         prep = fc_art["prep"]
        #         self.imp_stats, self.scaler_mean, self.scaler_scale = _extract_imputer_scaler(prep)
        #         self.fcm_centers = _np64(fc_art["centers"])  # (C,3) 표준화 공간
        #         self.fcm_m = float(fc_art.get("m", 2.0))
        #         self.fcm_available = True
        #         self.get_logger().info("Loaded fcm.joblib (dict)")
        #     elif isinstance(fc_art, FCMModel):
        #         prep = fc_art.prep
        #         self.imp_stats, self.scaler_mean, self.scaler_scale = _extract_imputer_scaler(prep)
        #         self.fcm_centers = _np64(fc_art.centers_)
        #         self.fcm_m = float(fc_art.m)
        #         self.fcm_available = True
        #         self.get_logger().info("Loaded fcm.joblib (FCMModel)")
        # except Exception as e:
        #     self.get_logger().warn(f"fcm.joblib load failed: {e}")

#######################################################################################################################################################


        self.subscription = self.create_subscription(
            SensStream,
            '/xServTopic',
            self.sensor_callback,
            10
        )
        self.create_subscription(Pose, '/end_pose', self.ee_pose_callback, 10)
        self.current_joint_state = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self._arm_client = ActionClient(self, FollowJointTrajectory, '/moveit_action/arm_controller/follow_joint_trajectory')
        self._arm_direct_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        self._pose_client = ActionClient(self, PoseGoal, 'pose_goal')
        self._target_client = ActionClient(self, TargetPose, 'target_pose')

        self.marker_pub = self.create_publisher(Marker, 'grasp_target_marker', 10)

        # [REAL-TIME] Camera Marker Timer (2Hz)
        self.camera_timer = self.create_timer(0.5, self.publish_camera_center_callback)

        servers_ready = True
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Arm action server not available.")
            servers_ready = False
        if not self._arm_direct_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn("Arm direct controller not available (MoveIt-only fallback disabled).")
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available.")
            servers_ready = False
        if not self._pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Pose Goal action server not available.")
            servers_ready = False
        if not self._target_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Target Pose action server not available.")
            servers_ready = False

        if not servers_ready:
            self.destroy_node()
            rclpy.shutdown()
            raise RuntimeError("필수 액션 서버를 찾을 수 없습니다.")


        self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.gripper_joint_names = ['joint7', 'joint8']

        self.ba_init = [-1.6379, 1.5724, -1.2721, -0.0566, 1.2838, 0.2908, 0.0, 0.0]
        # self.wm_init = [1.569855, -0.004553, -1.050652, -0.021037, 1.275889, 0.186441, 0.0, 0.0]
        self.wm_init = [1.569855, 0.505457344, -0.91650776, 0.038010476, 0.664895504, 0.178504452, 0.0, 0.0]
        self.wm_pre_grasp = [1.557995249, 0.693442225, -0.887973870, 0.099570951, 0.397515720, 0.201273613, 0.0, 0.0]
        # self.ba_end = [-1.5527, 2.6613, -1.5133, -0.2642, 0.6119, 0.3951, 0.0, 0.0]
        self.ba_end = [-1.668361, 1.906768, -1.42798, -0.076177, 1.1511, 0.0429471, 0.0, 0.0]
        self.wm_end = [1.571, 2.146, -2.268, 0.000, 0.594, 0.000, 0.0, 0.0]
        self.wm_mid = [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -1.357858404, 0.0, 0.0]
        self.home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.f = [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -0.1, 0.0, 0.0]

        self.ff = [1.5533, 0.1396, -0.4014, 0.0349, 0.3665, 0.1745, 0.0, 0.0]
        self.h = [-1.7977, 1.5184, -2.1468, 0.6109, 0.4887, -0.4014, 0, 0]

        # ── Tactile inference (wm task) ────────────────────────────────
        self.wm_data_dir = "/home/dyros/mcy_ws/wm_05"
        os.makedirs(self.wm_data_dir, exist_ok=True)

        # Grasp sliding window buffer (50 frames)
        self.realtime_raw_buffer = deque(maxlen=50)
        self.is_grasp_recording  = False
        self.grasp_raw_save_buf  = []
        self.grasp_frame_cnt     = 0

        # Transport sliding window buffer
        self.transport_raw_buffer   = deque(maxlen=50)
        self.is_transport_recording = False
        self.transport_raw_save_buf = []
        self.transport_frame_cnt    = 0

        # GRU-AE grasp model state
        self.gru_ae_model     = None
        self.gru_ae_scaler    = None
        self.gru_ae_threshold = 0.04
        self.gru_ae_infer_cfg = {"relative": "none", "weight_alpha": 0.0, "hybrid": False, "n_rel_dims": None}
        self.infer_latest_score      = None
        self.infer_consec_anomaly    = 0
        self.infer_consec_anomaly_needed = 2
        self.grasp_anomaly_detected  = False

        # GRU-AE transport model state
        self.transport_gru_model     = None
        self.transport_gru_scaler    = None
        self.transport_gru_threshold   = 0.04
        self.insert_gru_ae_threshold_b = 0.5
        self.transport_gru_infer_cfg = {"relative": "none", "weight_alpha": 0.0, "hybrid": False, "n_rel_dims": None}
        self.transport_latest_score        = None
        self.transport_consec_anomaly      = 0
        self.transport_consec_anomaly_needed = 3
        self.transport_anomaly_detected    = False
        self.transport_anomaly_ever        = False
        self._transport_abort              = threading.Event()
        self.grasp_use_raw                 = False   # True → realtime_raw_buffer에 48D raw 적재
        self.grasp_warmup_frames_needed    = 60      # 파지 초기 과도 구간 skip (≈660ms @ 90Hz)
        self.transport_use_raw             = False   # True → transport_raw_buffer에 48D raw 적재
        # channel B (FramewiseDeltaDetector)
        self.transport_channel_b             = None
        self.transport_channel_b_flag        = False
        self.transport_channel_b_consec      = 0
        self.transport_channel_b_consec_needed = 5
        self.transport_channel_b_warmup_needed = 30
        self.transport_infer_frame_cnt       = 0

        # Inference thread
        self._infer_lock        = threading.Lock()
        self._infer_trigger     = threading.Event()
        self._transport_trigger = threading.Event()
        self._infer_thread_running = True
        self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)

        if self.run_mode == 'inference':
            self._load_wm_grasp_model()
            self._load_wm_transport_model()
        self._infer_thread.start()
        # ───────────────────────────────────────────────────────────────

    def ee_pose_callback(self, msg):
        pos = msg.position
        ori = msg.orientation
        r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = [pos.x, pos.y, pos.z]
        self.ee_pose = T

    def joint_state_callback(self, msg):
        current_map = {n: p for n, p in zip(msg.name, msg.position)}
        pos = [current_map.get(f"joint{i}", 0.0) for i in range(1, 7)]
        pos.append(current_map.get("joint7", 0.0))
        pos.append(current_map.get("joint8", 0.0))
        self.current_joint_state = pos

    def align_joint6(self):
        """현재 joint 1~5 유지, joint 6만 회전하여 world 좌표계에서 그리퍼가 수평이 되도록 조정.
        목표 orientation은 wm_init 도달 시 캡처한 EE orientation.
        joint6 axis = EE frame z축이므로 R_diff의 z축 회전량(delta)을 현재 joint6에 더함.
        드럼 안에 있어 MoveIt collision 검사를 우회해야 하므로
        하드웨어 컨트롤러(/arm_controller)에 직접 전송."""
        if self.current_joint_state is None:
            self.get_logger().warn("[align_joint6] joint state 미수신, 스킵")
            return False
        if self.ee_pose is None or self.wm_init_ee_orientation is None:
            self.get_logger().warn("[align_joint6] EE pose 또는 wm_init orientation 미수신, 스킵")
            return False

        from scipy.spatial.transform import Rotation as R_scipy
        R_current = R_scipy.from_matrix(self.ee_pose[:3, :3])
        R_desired = self.wm_init_ee_orientation

        # R_desired = R_current @ Rz(delta)  →  Rz(delta) = R_current.inv() @ R_desired
        R_diff = (R_current.inv() * R_desired).as_matrix()
        delta6 = float(np.arctan2(R_diff[1, 0], R_diff[0, 0]))
        self.get_logger().info(f"[align_joint6] delta6={np.degrees(delta6):.2f}°")

        j6_new = float(self.current_joint_state[5]) + delta6
        j6_new = float(np.clip(j6_new, -2.0944, 2.0944))

        target = list(self.current_joint_state[:6])
        target[5] = j6_new
        return self._send_goal(self._arm_direct_client, self.arm_joint_names, target, wait_for_result=True)

    # 사용자님의 원본 _send_goal 함수
    def _send_goal(self, client, joint_names, joint_values, wait_for_result=True, timeout_sec=10.0,
                   result_timeout=30.0, time_sec=5.0):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in joint_values]

        if set(joint_names) == set(self.gripper_joint_names):
            point.time_from_start = Duration(sec=1, nanosec=0)
        else:
            point.time_from_start = Duration(sec=int(time_sec), nanosec=0)

        goal_msg.trajectory.points.append(point)

        send_goal_future = client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn(f"Goal for '{client._action_name}' rejected.")
            return False
        if not wait_for_result:
            return True

        get_result_future = goal_handle.get_result_async()

        # basket_ori.py와 동일: 5ms polling으로 _transport_abort 즉시 감지 → cancel_goal_async()
        _result_start = time.time()
        while not get_result_future.done():
            if time.time() - _result_start > result_timeout:
                self.get_logger().error(
                    f"[_send_goal] TIMEOUT ({result_timeout:.0f}s) → goal 취소 시도")
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                return False
            if getattr(self, '_transport_abort', None) and self._transport_abort.is_set():
                try:
                    goal_handle.cancel_goal_async()
                    # 취소 완료 대기 (새 goal reject 방지)
                    _c = time.time()
                    while not get_result_future.done() and time.time() - _c < 3.0:
                        rclpy.spin_once(self, timeout_sec=0.01)
                except Exception:
                    pass
                return False
            rclpy.spin_once(self, timeout_sec=0.005)

        result = get_result_future.result()
        if not result:
            self.get_logger().error("결과를 받지 못했습니다.")
            return False

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            return True
        elif result.status == 6:
            # Status 6 (ABORTED): tolerance 초과 abort → 물리적으로 도달했으면 성공으로 처리
            n = len(joint_values)
            if hasattr(self, 'current_joint_state') and self.current_joint_state is not None and len(self.current_joint_state) >= n:
                max_diff = max(abs(t - c) for t, c in zip(joint_values, self.current_joint_state[:n]))
                if max_diff < 0.15:
                    return True
            self.get_logger().warn(f"Action '{client._action_name}' failed with status: {result.status}")
            return False
        else:
            self.get_logger().warn(f"Action '{client._action_name}' failed with status: {result.status}")
            return False

    def send_joint_command(self, joint_values_8dof):
        self.get_logger().info(f"명령 수신: {joint_values_8dof}")
        arm_target = joint_values_8dof[0:6]
        gripper_target = joint_values_8dof[6:8]

        arm_success = self._send_goal(self._arm_client, self.arm_joint_names, arm_target, wait_for_result=True)
        if not arm_success:
            self.get_logger().error("팔 이동 실패. 다음 동작을 취소합니다.")
            # return False

        gripper_success = self._send_goal(self._gripper_client, self.gripper_joint_names, gripper_target, wait_for_result=True)
        if not gripper_success:
            self.get_logger().error("그리퍼 이동 실패.")
            # return False

        self.get_logger().info(f"명령 완료: {joint_values_8dof}")
        return True

    def goal(self, joint_values_8dof, gripper_action='keep', fast=False):
        time_sec = 3.0 if fast else 5.0
        self.get_logger().info(f"[Goal 명령] Pose: {joint_values_8dof}, Gripper: {gripper_action}, fast={fast}")
        arm_target = joint_values_8dof[:6]

        success = self._send_goal(self._arm_client, self.arm_joint_names, arm_target, wait_for_result=True, time_sec=time_sec)
        if not success:
            self.get_logger().warn("  >> MoveIt 실패 → 직접 컨트롤러로 재시도")
            success = self._send_goal(self._arm_direct_client, self.arm_joint_names, arm_target, wait_for_result=True, time_sec=time_sec)
        if not success:
            self.get_logger().error("  >> 팔 이동 실패 (MoveIt + 직접 컨트롤러 모두 실패)")
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
        return self.send_gripper_command(-0.0, -0.0, wait_for_result=True)

    def send_pose(self, pose: PoseStamped, mode: int = 1, vel_scale: float = 0.3):
        goal_msg = PoseGoal.Goal()
        goal_msg.target_pose = pose
        goal_msg.mode = mode
        goal_msg.velocity_scaling = vel_scale
        goal_msg.acceleration_scaling = vel_scale

        if mode == 1:
            # 위치만 목표, orientation constraint 비활성화
            goal_msg.roll_tol_rad  = 6.28
            goal_msg.pitch_tol_rad = 6.28
            goal_msg.yaw_tol_rad   = 6.28

        send_goal_future = self._pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("pose_goal이 거부되었습니다.")
            return False
        self.get_logger().info("pose_goal 명령 전송 완료.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f"\033[94m✅ pose 명령 성공: {result.message}\033[0m")
            return True
        else:
            self.get_logger().warn(f"⚠️ pose 명령 실패: {result.message}")
            return False


    def move_forward(self, pose: PoseStamped, fallback_to_ik=True):
        """
        Hybrid Move Forward (Mode 11 -> Mode 6)
        Mode 11: Cartesian 직선 경로 + 직접 JTC (충돌 무시)
        Mode 6 : pure IK fallback
        Returns: True if success, False otherwise
        """
        self.get_logger().info("[forward] Starting Hybrid Move (Mode 11 -> Mode 6)")

        def run_mode(mode_val):
            goal_msg = PoseGoal.Goal()
            goal_msg.target_pose = pose
            goal_msg.mode = mode_val

            send_goal_future = self._pose_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

            if not send_goal_future.done(): return "timeout"
            goal_handle = send_goal_future.result()
            if not goal_handle or not goal_handle.accepted: return "rejected"

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=20.0)
            if not result_future.done(): return "timeout"

            return "success" if result_future.result().result.success else "fail"

        # 1. Try Mode 11 (Cartesian 직선, 충돌 무시, 직접 JTC)
        res11 = run_mode(11)
        if res11 == "success": return True

        # 2. Fallback to Mode 6 (Joint IK, 충돌 무시)
        if fallback_to_ik:
            self.get_logger().warn(f"Mode 11 failed ({res11}). Retrying with Mode 6 (Joint IK)...")
            return run_mode(6) == "success"

        return False

    def reset_sensor_data(self):
        self.prev_x, self.prev_y, self.prev_z = None, None, None
        self.cum_x, self.cum_y, self.cum_z = [0.0] * 16, [0.0] * 16, [0.0] * 16
        self.z_aligned, self.contacted = False, False

        self.z_aligned_k, self.contacted_k = False, False
        self.contacted_ff, self.contacted_kk = False, False
        self.z_aligned_f, self.contacted_f = False, False

        self.get_logger().info("센서 기준값 및 상태 초기화 완료.")

    def sensor_callback(self, msg):
        try:
            x_vals = [t.x for t in msg.sensors[0].taxels]
            y_vals = [t.y for t in msg.sensors[0].taxels]
            z_vals = [t.z for t in msg.sensors[0].taxels]

            x_vals, y_vals, z_vals = self._normalize_taxels(x_vals, y_vals, z_vals)  # 모두 float64

            if self.prev_x is None:
                self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals
                # 첫 프레임엔 예측/표시 안 함
                self.z_aligned = self.contacted = False
                self.z_aligned_k = self.contacted_k = False
                self.z_aligned_f = self.contacted_f = False
                self.contacted_ff = self.contacted_kk = False
                # print("처음입니다.")
                return

            dx = [c - p for c, p in zip(x_vals, self.prev_x)]
            dy = [c - p for c, p in zip(y_vals, self.prev_y)]
            dz = [c - p for c, p in zip(z_vals, self.prev_z)]

            for i in range(16):
                self.cum_x[i] += dx[i]
                self.cum_y[i] += dy[i]
                self.cum_z[i] += dz[i]

            vecs = np.array([self.cum_x, self.cum_y, self.cum_z]).T
            norms = np.linalg.norm(vecs, axis=1)
            max_norm = np.max(norms)
            if max_norm == 0:
                max_norm = 1e-6
            scaled_vecs = vecs / max_norm
            weights = norms
            weights[weights == 0] = 1e-6

            mean_vec_biased = np.sum(vecs, axis=0) / np.sum(weights)
            current_features = np.concatenate([scaled_vecs.flatten(), mean_vec_biased])

            norm = np.linalg.norm(mean_vec_biased)
            z_axis = np.array([0, 0, 1])
            cos_theta = np.dot(mean_vec_biased, z_axis) / (norm + 1e-8)
            theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            self.norm = norm
            self.max_norm = max_norm
            self.theta = theta_deg

            # 1. Noise Filter: 개별 taxel 최대값이 너무 작으면 노이즈로 무시
            if max_norm < 0.45:
                self.z_aligned = False
                self.contacted = False
            elif norm < 0.35:
                # 힘 크기 부족 → 미파지
                self.z_aligned = False
            else:
                # 세탁물 방향이 제각각이라 theta(힘 방향) 조건 불필요
                # norm이 충분하면 그리퍼 안에 물체가 있는 것으로 판단
                self.z_aligned = True

########################## 비지도 클러스터링 ####################################################

            # # --- KMeans/FCM --- COLOR_MAP = {0: "green", 1: "red", 2: "gray"}
            # LABEL_STATE_MAP_k = {0: (False, True), 1: (True, False), 2: (False, False)}    
            # LABEL_STATE_MAP_f = {0: (False, True), 1: (False, False), 2: (True, False)}          

            # # --- KMeans ---
            # if getattr(self, "kmeans_available", False) and self.kmeans is not None and self.imp_stats is not None:
            #     try:
            #         Xs_k = _preprocess_mb(mean_vec_biased, self.imp_stats, self.scaler_mean, self.scaler_scale)
            #         k_label = int(self.kmeans.predict(Xs_k)[0])
            #         za_k, ct_k = LABEL_STATE_MAP_k.get(k_label, (False, False))
            #         self.z_aligned_k = bool(za_k)
            #         self.contacted_k = bool(ct_k)
            #         if self.contacted_k == True:
            #             # print("kkkkkkkkkkkkkkkkkk")
            #             self.contacted_kk = True

            #     except Exception as e:
            #         self.get_logger().warn(f"KMeans predict skipped: {e}")
            #         # 이전 값 유지

            # # --- FCM ---
            # if getattr(self, "fcm_available", False) and self.fcm_centers is not None and self.imp_stats is not None:
            #     try:
            #         Xs_f = _preprocess_mb(mean_vec_biased, self.imp_stats, self.scaler_mean, self.scaler_scale)
            #         U = _fcm_membership(Xs_f, self.fcm_centers, self.fcm_m)  # (C,1)
            #         f_label = int(np.argmax(U[:, 0]))
            #         za_f, ct_f = LABEL_STATE_MAP_f.get(f_label, (False, False))
            #         self.z_aligned_f = bool(za_f)
            #         self.contacted_f = bool(ct_f)
            #         if self.contacted_f == True:
            #             # print("ffffffffffffffff")
            #             self.contacted_ff = True

            #     except Exception as e:
            #         self.get_logger().warn(f"FCM predict skipped: {e}")
            #         # 이전 값 유지

   #############################################################################################################################             

            # ── basket_ori.py 동일: 6D MRL feature 계산 ────────────────
            vecs_arr = np.array([self.cum_x, self.cum_y, self.cum_z]).T  # (16,3)
            valid_mask = norms > 0.1
            if valid_mask.sum() >= 2:
                unit_vecs = vecs_arr[valid_mask] / norms[valid_mask, np.newaxis]
                mean_unit = np.mean(unit_vecs, axis=0)
                mrl = float(np.linalg.norm(mean_unit))
            else:
                mean_unit = np.zeros(3)
                mrl = 0.0
            mrl_frame = np.array([
                mrl,
                float(norm),
                float(max_norm),
                float(mean_unit[0]),
                float(mean_unit[1]),
                float(mean_unit[2]),
            ], dtype=np.float32)
            # ────────────────────────────────────────────────────────────

            # 48D raw frame (basket_ori.py transport와 동일: cum_x/y/z flatten)
            raw_48 = vecs_arr.flatten().astype(np.float32)  # (48,)

            # 슬라이딩 윈도우 버퍼
            if self.grasp_use_raw:
                self.realtime_raw_buffer.append(raw_48)         # grasp: 48D raw
            else:
                self.realtime_raw_buffer.append(mrl_frame)      # grasp: 6D MRL (fallback)
            if self.transport_use_raw:
                self.transport_raw_buffer.append(raw_48)        # transport: 48D raw
            else:
                self.transport_raw_buffer.append(mrl_frame)     # transport: 6D MRL (fallback)

            if self.is_grasp_recording:
                self.grasp_raw_save_buf.append(
                    raw_48.copy() if self.grasp_use_raw else mrl_frame.copy())
                self.grasp_frame_cnt += 1
                if self.grasp_frame_cnt % 5 == 0:
                    self._infer_trigger.set()

            if self.is_transport_recording:
                self.transport_raw_save_buf.append(
                    raw_48.copy() if self.transport_use_raw else mrl_frame.copy())
                self.transport_frame_cnt += 1
                self.transport_infer_frame_cnt += 1
                if self.transport_frame_cnt % 5 == 0:
                    self._transport_trigger.set()
                # channel B: FramewiseDeltaDetector 퍼프레임 체크 (6D MRL frame 사용)
                if (self.transport_channel_b is not None and
                        self.transport_infer_frame_cnt > self.transport_channel_b_warmup_needed and
                        self.max_norm > 0.3):
                    delta_b, flag_b = self.transport_channel_b.update(mrl_frame)
                    if flag_b:
                        self.transport_channel_b_consec += 1
                        if self.transport_channel_b_consec >= self.transport_channel_b_consec_needed:
                            with self._infer_lock:
                                if not self.transport_anomaly_ever:
                                    self.transport_anomaly_ever = True
                                    self.transport_anomaly_detected = True
                                    self._transport_abort.set()
                                    self.get_logger().warn(
                                        f"🚨 [WM-TRANSPORT] 채널 B 낙하 감지! delta={delta_b:.4f}")
                    else:
                        self.transport_channel_b_consec = 0

            # prev 갱신 (항상 마지막에)
            self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals

        except Exception as e:
            self.get_logger().error(f"Sensor callback error: {e}")

    # ── Tactile model loading ──────────────────────────────────────────

    def _load_wm_grasp_model(self):
        import joblib, pickle, torch
        sys.path.insert(0, "/home/dyros/mcy_ws")
        from train_gru_ae import GRUAE
        data_dir = "/home/dyros/mcy_ws/grasp_new"
        use_raw  = False  # basket_ori.py와 동일: grasp는 MRL 모델 강제
        self.grasp_use_raw = use_raw
        suffix   = "_raw" if use_raw else ""
        paths = {
            "model":  os.path.join(data_dir, f"grasp_gru_ae{suffix}.pt"),
            "scaler": os.path.join(data_dir, f"grasp_scaler{suffix}.pkl"),
            "thr_a":  os.path.join(data_dir, f"grasp_threshold{suffix}.npy"),
        }
        for k, p in paths.items():
            if not os.path.exists(p):
                self.get_logger().warn(f"[WM-GRASP] 파일 없음: {p} → inference 비활성")
                return
        try:
            checkpoint = torch.load(paths["model"], map_location="cpu")
            scaler     = joblib.load(paths["scaler"])
            input_dim  = scaler.n_features_in_
            hidden_dim = checkpoint['encoder.gru.weight_hh_l0'].shape[1]
            model = GRUAE(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=2, dropout=0.2, seq_len=50)
            model.load_state_dict(checkpoint)
            model.eval()
            self.gru_ae_model     = model
            self.gru_ae_scaler    = scaler
            self.gru_ae_threshold = float(np.load(paths["thr_a"])[0])
            cfg_path = os.path.join(data_dir, f"grasp_infer_cfg{suffix}.pkl")
            if os.path.exists(cfg_path):
                with open(cfg_path, "rb") as f:
                    self.gru_ae_infer_cfg = pickle.load(f)
            self.get_logger().info(
                f"[WM-GRASP] 모델 로드 완료 (dim={input_dim}, thr={self.gru_ae_threshold:.4f})")
        except Exception as e:
            self.get_logger().error(f"[WM-GRASP] 로드 실패: {e}")

    def _load_wm_transport_model(self):
        import joblib, pickle, torch
        sys.path.insert(0, "/home/dyros/mcy_ws")
        from train_gru_ae import GRUAE
        from dual_channel_detector import FramewiseDeltaDetector
        data_dir = "/home/dyros/mcy_ws/transport_new"
        use_raw  = os.path.exists(os.path.join(data_dir, "transport_gru_ae_raw.pt"))
        self.transport_use_raw = use_raw
        suffix   = "_raw" if use_raw else ""
        paths = {
            "model":  os.path.join(data_dir, f"transport_gru_ae{suffix}.pt"),
            "scaler": os.path.join(data_dir, f"transport_scaler{suffix}.pkl"),
            "thr_a":  os.path.join(data_dir, f"transport_threshold{suffix}.npy"),
        }
        for k, p in paths.items():
            if not os.path.exists(p):
                self.get_logger().warn(f"[WM-TRANSPORT] 파일 없음: {p} → inference 비활성")
                return
        try:
            checkpoint = torch.load(paths["model"], map_location="cpu")
            scaler     = joblib.load(paths["scaler"])
            input_dim  = scaler.n_features_in_
            hidden_dim = checkpoint['encoder.gru.weight_hh_l0'].shape[1]
            model = GRUAE(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=2, dropout=0.2, seq_len=50)
            model.load_state_dict(checkpoint)
            model.eval()
            self.transport_gru_model   = model
            self.transport_gru_scaler  = scaler
            self.transport_gru_threshold = float(np.load(paths["thr_a"])[0])
            thr_b_path = os.path.join(data_dir, f"transport_threshold_b{suffix}.npy")
            if os.path.exists(thr_b_path):
                self.insert_gru_ae_threshold_b = float(np.load(thr_b_path)[0])
            cfg_path = os.path.join(data_dir, f"transport_infer_cfg{suffix}.pkl")
            if os.path.exists(cfg_path):
                with open(cfg_path, "rb") as f:
                    self.transport_gru_infer_cfg = pickle.load(f)
            self.transport_channel_b = FramewiseDeltaDetector(
                delta_threshold=self.insert_gru_ae_threshold_b, smooth_window=3)
            self.get_logger().info(
                f"[WM-TRANSPORT] 모델 로드 완료 (dim={input_dim}, "
                f"thr_A={self.transport_gru_threshold:.4f}, thr_B={self.insert_gru_ae_threshold_b:.4f})")
        except Exception as e:
            self.get_logger().error(f"[WM-TRANSPORT] 로드 실패: {e}")

    # ── Inference ─────────────────────────────────────────────────────

    def _run_grasp_inference(self):
        if self.gru_ae_model is None or len(self.realtime_raw_buffer) < 50:
            return None
        if self.grasp_frame_cnt < self.grasp_warmup_frames_needed:
            return None  # 그리퍼 닫힘 초기 과도 구간 제외
        try:
            import torch
            from train_gru_ae import apply_relative_transform, build_hybrid_window
            cfg = self.gru_ae_infer_cfg
            seq = np.array(list(self.realtime_raw_buffer), dtype=np.float32)
            if cfg.get("hybrid", False):
                seq = build_hybrid_window(seq)
            seq_s = self.gru_ae_scaler.transform(seq).astype(np.float32)
            seq_r = apply_relative_transform(seq_s, cfg.get("relative", "none"),
                                             n_rel_dims=cfg.get("n_rel_dims", None))
            x = torch.tensor(seq_r).unsqueeze(0)
            with torch.no_grad():
                return float(self.gru_ae_model.reconstruction_error(
                    x, weight_alpha=cfg.get("weight_alpha", 0.0)).item())
        except Exception as e:
            self.get_logger().error(f"[WM-GRASP infer] {e}")
            return None

    def _run_transport_inference(self):
        if self.transport_gru_model is None or len(self.transport_raw_buffer) < 50:
            return None
        try:
            import torch
            from train_gru_ae import apply_relative_transform, build_hybrid_window
            cfg = self.transport_gru_infer_cfg
            seq = np.array(list(self.transport_raw_buffer), dtype=np.float32)
            if cfg.get("hybrid", False):
                seq = build_hybrid_window(seq)
            seq_s = self.transport_gru_scaler.transform(seq).astype(np.float32)
            seq_r = apply_relative_transform(seq_s, cfg.get("relative", "none"),
                                             n_rel_dims=cfg.get("n_rel_dims", None))
            x = torch.tensor(seq_r).unsqueeze(0)
            with torch.no_grad():
                return float(self.transport_gru_model.reconstruction_error(
                    x, weight_alpha=cfg.get("weight_alpha", 0.0)).item())
        except Exception as e:
            self.get_logger().error(f"[WM-TRANSPORT infer] {e}")
            return None

    def _inference_worker(self):
        while self._infer_thread_running:
            g = self._infer_trigger.wait(timeout=0.05)
            t = self._transport_trigger.is_set()

            if g:
                self._infer_trigger.clear()
                score = self._run_grasp_inference()
                if score is not None:
                    with self._infer_lock:
                        self.infer_latest_score = score
                        if score > self.gru_ae_threshold:
                            self.infer_consec_anomaly += 1
                            if self.infer_consec_anomaly >= self.infer_consec_anomaly_needed:
                                self.grasp_anomaly_detected = True
                        else:
                            self.infer_consec_anomaly = 0
                    self.get_logger().info(
                        f"[WM-GRASP] score={score:.4f} thr={self.gru_ae_threshold:.4f} "
                        f"consec={self.infer_consec_anomaly} "
                        f"{'ANOMALY' if self.grasp_anomaly_detected else 'normal'}")

            if t:
                self._transport_trigger.clear()
                score = self._run_transport_inference()
                if score is not None:
                    with self._infer_lock:
                        self.transport_latest_score = score
                        if score > self.transport_gru_threshold:
                            self.transport_consec_anomaly += 1
                            if (self.transport_consec_anomaly >= self.transport_consec_anomaly_needed
                                    and not self.transport_anomaly_detected):
                                self.transport_anomaly_detected = True
                                self.transport_anomaly_ever    = True
                                self._transport_abort.set()
                                self.get_logger().warn("[WM-TRANSPORT] 채널 A Slip 감지! 복구 필요")
                        else:
                            self.transport_consec_anomaly = 0
                    self.get_logger().info(
                        f"[WM-TRANSPORT] score={score:.4f} thr={self.transport_gru_threshold:.4f} "
                        f"consec={self.transport_consec_anomaly}")

    # ── Data collection helpers ────────────────────────────────────────

    def _save_wm_grasp_data(self, label: str):
        if not self.grasp_raw_save_buf:
            return
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.wm_data_dir, f"grasp_{ts}_{label}_raw.npy")
        np.save(path, np.array(self.grasp_raw_save_buf, dtype=np.float32))
        self.get_logger().info(f"[COLLECT] 저장: {path} ({len(self.grasp_raw_save_buf)}프레임)")
        self.grasp_raw_save_buf = []

    def _save_wm_transport_data(self, label: str):
        if not self.transport_raw_save_buf:
            return
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.wm_data_dir, f"transport_{ts}_{label}_raw.npy")
        np.save(path, np.array(self.transport_raw_save_buf, dtype=np.float32))
        self.get_logger().info(f"[COLLECT] 저장: {path} ({len(self.transport_raw_save_buf)}프레임)")
        self.transport_raw_save_buf = []

    def close_door(self):
        pass

    def open_door(self):
        pass

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


    def target_get(self, mode="wm"): ### basket, wm, mid
        goal_msg = TargetPose.Goal()
        goal_msg.start = True  # 요청 트리거
        goal_msg.mode = mode

        self.get_logger().info('target_pose 요청 전송')
        future = self._target_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        if not future.done():
            self.get_logger().error('Goal 요청 실패')
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal이 액션 서버에서 거부됨')
            return None

        # 결과 대기
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f"받은 EA: {result.target_pose}") # EA
            self.get_logger().info(f"현재 Pose: {self.ee_pose}") # EE

        
            ee2cam = np.vstack((np.hstack((self.R_cal, self.t_cal)), [0, 0, 0, 1]))

            base2cam = self.ee_pose @ ee2cam ## O2C
            

            # [Two-Stage Check] Sentinel Passthrough
            if result.target_pose.position.z == -99.0:
                self.get_logger().info("  >> [Sentinel] Empty Detected (-99.0). Returning immediately.")
                pose_stamped = PoseStamped()
                pose_stamped.pose = result.target_pose
                return pose_stamped

            pose = self.transform_pose_with_offset(result.target_pose, base2cam) ## O2t
            self.get_logger().info(f"\033[94m변환한 bA: {pose}\033[0m") # Blue

            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.frame_id = 'base_link'
            pose_stamped.header.stamp = Clock().now().to_msg()
            
            # [VISUALIZATION] Publish Camera Center Marker (Light Blue, ID=100)
            cam_center_pose = Pose()
            cam_center_pose.position.x = base2cam[0,3]
            cam_center_pose.position.y = base2cam[1,3]
            cam_center_pose.position.z = base2cam[2,3]
            self.publish_target_marker(cam_center_pose, id=100, r=0.0, g=0.5, b=1.0) # Light Blue 

            # [VISUALIZATION] Publish Target Marker (Blue)
            self.publish_target_marker(pose_stamped.pose, id=1, r=0.0, g=0.0, b=1.0)

            return pose_stamped
        else:

            return None
        
    def eunho_sequence(self):
        self.open_gripper()
        self.get_logger().info("Step 1 : 이니셜 포즈(wm_init)로 이동 중")
        while not self.goal(self.wm_init, 'keep'):
            self.get_logger().warn("  >> 실패 → 재시도 중")
        self.reset_sensor_data()

        self.get_logger().info("Step 2: 잡기 시도 중...")
        target_wm = self.target_get(mode="wm")
        if target_wm:
            self.get_logger().info("[SUCCESS] Target WM이 존재합니다")
        else:
            self.get_logger().info("[FAILED] Target WM이 존재하지 않습니다")

    def _normalize_taxels(self, x_vals, y_vals, z_vals):
        """
        raw 16×(x,y,z) -> 48축 Min–Max 정규화 결과를 분해해서 (x,y,z) 반환 (모두 float64)
        """
        A = np.empty(48, dtype=np.float64)
        A[0:16] = np.asarray(x_vals, dtype=np.float64)
        A[16:32] = np.asarray(y_vals, dtype=np.float64)
        A[32:48] = np.asarray(z_vals, dtype=np.float64)
        An = (A - self.gmin) / self.grng
        return An[0:16].copy(), An[16:32].copy(), An[32:48].copy()
    
    def _load_minmax(self, path: str):
        """taxel_minmax_norm.npy -> (gmin(48,), grng(48,)) as float64"""
        try:
            mm = np.load(path, allow_pickle=True).item()
            gmin = np.asarray(mm["min"], dtype=np.float64)
            grng = np.asarray(mm["range"], dtype=np.float64)
            # 안전가드
            if gmin.shape[0] != 48 or grng.shape[0] != 48:
                raise ValueError(f"min/range shape must be 48, got {gmin.shape}, {grng.shape}")
            grng[~np.isfinite(grng) | (grng <= 0.0)] = 1.0
            return gmin, grng
        except Exception as e:
            self.get_logger().warn(f"MinMax load failed ({e}); using identity (no-op)")
            gmin = np.zeros(48, dtype=np.float64)
            grng = np.ones(48, dtype=np.float64)
            return gmin, grng

    def _step_time(self, label: str, t0: float) -> float:
        dt = time.time() - t0
        self.get_logger().info(f"[TIMER] {label:<45s} {dt:6.2f}s")
        self._timing_log.append((label, dt))
        return time.time()

    def _print_timing_summary(self):
        if not self._timing_log:
            return
        sep = "=" * 64
        lines = [sep, f"{'[ TIMING SUMMARY ]':^64}"]
        total = 0.0
        for label, dt in self._timing_log:
            lines.append(f"  {label:<48s} {dt:6.2f}s")
            if not label.startswith("==="):
                total += dt
        lines.append("-" * 64)
        lines.append(f"  {'합계 (사이클 내 측정 구간)':<48s} {total:6.2f}s")
        lines.append(sep)
        self.get_logger().info("\n" + "\n".join(lines))
        self._timing_log = []

    def execute_sequence(self):
        self.get_logger().info("Step 0: 문 여는 중")
        self.open_door()
        self.open_gripper()

        while rclpy.ok():
            _t_cycle = time.time()
            self._timing_log = []
            while True:
                _t = time.time()
                self.get_logger().info("Step 1 : 이니셜 포즈(wm_init)로 이동 중")
                while not self.goal(self.wm_init, 'keep', fast=True):
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                _t = self._step_time("wm_init 이동", _t)

                # wm_init 도달 직후 EE orientation 캡처 (수평 그리퍼 기준 orientation)
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.ee_pose is not None:
                    from scipy.spatial.transform import Rotation as R_scipy
                    self.wm_init_ee_orientation = R_scipy.from_matrix(self.ee_pose[:3, :3])
                    self.get_logger().info("[wm_init] EE orientation 캡처 완료")

                self.reset_sensor_data()

                self.get_logger().info("Step 2: 잡기 시도 중...")

                target_wm = self.target_get(mode="wm") # EA
                _t = self._step_time("target_get_wm 비전 처리", _t)

                if target_wm is None:
                    continue

                if target_wm.pose.position.z == -99.0:
                    break

                # --- 드럼 표면 z 계산 (base_link 기준) ---
                # world→base_link z offset = 0.445m (URDF: world→base_link joint xyz="0 0 0.445")
                import math as _math
                import copy
                from scipy.spatial.transform import Rotation as R_scipy
                _DRUM_CX        = -0.01
                _DRUM_CZ_world  =  0.88
                _DRUM_R         =  0.29
                _WORLD_TO_BASE_Z = 0.445          # base_link is 0.445m above world origin
                _DRUM_CZ_base   = _DRUM_CZ_world - _WORLD_TO_BASE_Z  # 0.435

                _tx = target_wm.pose.position.x
                _ty = target_wm.pose.position.y
                _tz_orig = target_wm.pose.position.z
                _dx = _tx - _DRUM_CX
                _r_eff = _math.sqrt(max(0.0, _DRUM_R**2 - _dx**2))
                _z_bottom_base = _DRUM_CZ_base - _r_eff   # bottom of drum in base_link
                _z_top_base    = _DRUM_CZ_base + _r_eff   # top of drum in base_link

                # 드럼 표면 1cm 위를 target_wm.z 로 설정
                _z_new = _z_bottom_base + 0.015
                target_wm.pose.position.z = _z_new
                self.get_logger().info(
                    f"[DrumSurface]\n"
                    f"  target_wm original : x={_tx:.4f}, y={_ty:.4f}, z={_tz_orig:.4f}\n"
                    f"  drum (base_link)   : center_x={_DRUM_CX}, center_z={_DRUM_CZ_base:.3f}, radius={_DRUM_R}\n"
                    f"  dx={_dx:.4f}  →  z_bottom={_z_bottom_base:.4f}, z_top={_z_top_base:.4f}\n"
                    f"  target_wm.z set to z_bottom+1cm = {_z_new:.4f}"
                )

                # 1. Calculate and Move to Pre-Grasp (2cm back from Target)
                pre_grasp_pose = copy.deepcopy(target_wm)
                pre_grasp_pose.pose.position.z += 0.05  # push: 3cm 위에서 접근

                retreat_pose = copy.deepcopy(target_wm)
                retreat_pose.pose.position.z += 0.05    # retreat: 3cm 위로 후퇴

                # pre_grasp, retreat, target 모두 wm_init 수평 orientation으로 통일
                if self.wm_init_ee_orientation is not None:
                    q = self.wm_init_ee_orientation.as_quat()  # [x, y, z, w]
                    for ps in [pre_grasp_pose, retreat_pose, target_wm]:
                        ps.pose.orientation.x = float(q[0])
                        ps.pose.orientation.y = float(q[1])
                        ps.pose.orientation.z = float(q[2])
                        ps.pose.orientation.w = float(q[3])

                self.get_logger().info(f"\033[94mMoving to Pre-Grasp (5cm above target)...\033[0m")
                self.publish_target_marker(pre_grasp_pose.pose, id=2, r=1.0, g=1.0, b=0.0) # Yellow

                # Step 1a: 위치만 먼저 이동 (orientation 무시)
                while not self.send_pose(pre_grasp_pose, mode=1, vel_scale=0.5):  # 물체 없음 → 빠른 접근
                    self.get_logger().warn("  >> pre_grasp 이동 실패 → 재시도 중")
                _t = self._step_time("Pre-Grasp 접근 (send_pose)", _t)

                # Step 1b: joint 6만 wm_init 값으로 회전 → 그리퍼를 바닥과 수평하게
                self.get_logger().info(f"\033[94mAligning joint6 to horizontal...\033[0m")
                while not self.align_joint6():
                    self.get_logger().warn("  >> joint6 정렬 실패 → 재시도 중")
                _t = self._step_time("joint6 정렬", _t)

                # align_joint6 후 실제 EE orientation으로 target_wm / retreat_pose 갱신
                # wm_init_ee_orientation은 다른 arm 구성에서 캡처된 것이라
                # pre_grasp 위치에서 push/retreat 시 다른 orientation 목표를 주면 joints가 펴지는 문제 발생
                rclpy.spin_once(self, timeout_sec=0.15)
                if self.ee_pose is not None:
                    from scipy.spatial.transform import Rotation as R_scipy
                    q_cur = R_scipy.from_matrix(self.ee_pose[:3, :3]).as_quat()
                    for ps in [target_wm, retreat_pose]:
                        ps.pose.orientation.x = float(q_cur[0])
                        ps.pose.orientation.y = float(q_cur[1])
                        ps.pose.orientation.z = float(q_cur[2])
                        ps.pose.orientation.w = float(q_cur[3])
                    self.get_logger().info(
                        f"[align_joint6 후] push/retreat orientation 갱신: "
                        f"q=[{q_cur[0]:.3f}, {q_cur[1]:.3f}, {q_cur[2]:.3f}, {q_cur[3]:.3f}]"
                    )

                # 2. Push to Actual Target (Hybrid Move)
                self.get_logger().info(f"\033[94mPushing to Target WM...\033[0m")
                self.publish_target_marker(target_wm.pose, id=3, r=0.0, g=1.0, b=0.0) # Green (drum-adjusted target)
                
                if not self.move_forward(target_wm):
                    self._step_time("Push 이동 (실패-재시도)", _t)
                    self.get_logger().warn("  >> push 이동 실패 → 새 target 요청")
                    continue
                _t = self._step_time("Push 이동 (성공)", _t)

                self.close_gripper()
                _t = self._step_time("그리퍼 닫기", _t)

                # ── Grasp recording 시작 (basket_ori.py와 동일: close 직후 reset → 접촉+retreat만 포착) ──
                self.reset_sensor_data()          # 접근 경로 누적값 제거
                self.grasp_raw_save_buf  = []
                self.grasp_frame_cnt     = 0
                self.grasp_anomaly_detected = False
                self.realtime_raw_buffer.clear()
                with self._infer_lock:
                    self.infer_consec_anomaly = 0
                self.is_grasp_recording = True
                # ────────────────────────────────────────────────────

                # [RETREAT] 파지 후 2cm 위로 후퇴
                self.get_logger().info("\033[94mRetreating (2cm above target)...\033[0m")
                self.move_forward(retreat_pose, fallback_to_ik=True)
                _t = self._step_time("후퇴 (retreat)", _t)

                # ── Grasp recording 종료 ────────────────────────────
                self.is_grasp_recording = False
                # ────────────────────────────────────────────────────

                # [파지 판정] retreat 직후 센서 확인
                # wm_init까지 돌아가기 전에 판정 → 실패 시 불필요한 이동 방지
                self.get_logger().info("🔍 파지 판정 (retreat 후)")
                if self.z_aligned:
                    self.get_logger().info(
                        f"✅ 잡기 성공 (max_norm={self.max_norm:.2f}, norm={self.norm:.3f}, theta={self.theta:.1f}°)"
                    )
                    # ── collect mode: 성공 데이터 저장 ────────────────
                    if self.run_mode == 'collect':
                        self._save_wm_grasp_data('success')
                    # ── inference mode: GRU-AE 추가 판단 (retreat 후 현재 점수 기준) ──
                    elif self.run_mode == 'inference':
                        cur_score = self.infer_latest_score
                        if cur_score is not None and cur_score > self.gru_ae_threshold:
                            self.get_logger().warn(
                                f"[WM-GRASP] rule=OK but AE anomaly (score={cur_score:.4f} "
                                f"thr={self.gru_ae_threshold:.4f}) → 재시도")
                            while not self.open_gripper():
                                self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
                            while not self.goal(self.wm_init, 'keep', fast=True):
                                self.get_logger().warn("  >> 실패 → 재시도 중")
                            continue
                    # ────────────────────────────────────────────────────
                    while not self.goal(self.wm_init, 'keep'):  # 물체 파지 중 → 기본 속도
                        self.get_logger().warn("  >> 실패 → 재시도 중")
                    _t = self._step_time("파지 성공 → wm_init 복귀", _t)
                    break
                else:
                    _mn  = self.max_norm if self.max_norm is not None else 0.0
                    _n   = self.norm     if self.norm     is not None else 0.0
                    _th  = self.theta    if self.theta    is not None else 0.0
                    if _mn < 0.5:
                        _reason = f"max_norm={_mn:.4f} < 0.5 (noise)"
                    elif _n < 0.5:
                        _reason = f"norm={_n:.4f} < 0.5 (힘 부족)"
                    elif _th >= 70.0:
                        _reason = f"theta={_th:.1f}° >= 70° (방향 불일치)"
                    else:
                        _reason = f"z_aligned=False (max_norm={_mn:.4f}, norm={_n:.4f}, theta={_th:.1f}°)"
                    self.get_logger().warn(f"❌ 잡기 실패 → 재시도  원인: {_reason}")
                    # ── collect mode: 실패 데이터 저장 ────────────────
                    if self.run_mode == 'collect':
                        self._save_wm_grasp_data('failure')
                    # ────────────────────────────────────────────────────
                    while not self.open_gripper():
                        self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
                    while not self.goal(self.wm_init, 'keep', fast=True):
                        self.get_logger().warn("  >> 실패 → 재시도 중")
                    _t = self._step_time("파지 실패 → wm_init 복귀", _t)

            if target_wm.pose.position.z == -99.0:
                self._print_timing_summary()
                break

            self.get_logger().info("Step 3: 엔드 포즈로 이동 중")

            # ── Transport recording 시작 ────────────────────────────
            self.transport_raw_save_buf      = []
            self.transport_frame_cnt         = 0
            self.transport_infer_frame_cnt   = 0
            self.transport_anomaly_detected  = False
            self.transport_anomaly_ever      = False
            self.transport_channel_b_flag    = False
            self.transport_channel_b_consec  = 0
            if self.transport_channel_b is not None:
                self.transport_channel_b.reset()
            self._transport_abort.clear()
            self.reset_sensor_data()           # basket_ori.py와 동일: cum_x/y/z 리셋 (이전 파지 누적값 제거)
            self.transport_raw_buffer.clear()  # 이전 데이터 제거 → 이송 데이터만 inference
            with self._infer_lock:
                self.transport_consec_anomaly = 0
            self.is_transport_recording = True
            # ────────────────────────────────────────────────────────

            _t = time.time()
            while not self.goal(self.h, 'keep'):
                if self._transport_abort.is_set():
                    break
                self.get_logger().warn("  >> 실패 → 재시도 중")
                time.sleep(0.2)
            _t = self._step_time("h 이동 (이송 1단계)", _t)

            # ── Transport recording 종료 (이송 1단계까지만) ────────
            self.is_transport_recording = False

            if not self._transport_abort.is_set():
                while not self.goal(self.ba_init, 'keep'):
                    if self._transport_abort.is_set():
                        break
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                    time.sleep(0.2)
                _t = self._step_time("ba_init 이동 (이송 2단계)", _t)

            if not self._transport_abort.is_set():
                while not self.goal(self.ba_end, 'keep'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                    time.sleep(0.2)
                _t = self._step_time("ba_end 이동 (이송 3단계)", _t)
            abort_detected = self._transport_abort.is_set()
            transport_ok   = not abort_detected
            if self.run_mode == 'collect':
                self._save_wm_transport_data('success' if transport_ok else 'failure')
            # ────────────────────────────────────────────────────────

            # ── 낙하 감지 recovery (inference 모드) ──────────────────
            if abort_detected and self.run_mode == 'inference':
                self.get_logger().warn("[WM-TRANSPORT] 낙하 감지 → 재파지 조건 확인")
                time.sleep(0.5)
                self._transport_abort.clear()
                for _retry in range(3):
                    if self.open_gripper():
                        break
                    self.get_logger().warn(f"[RECOVERY] 그리퍼 오픈 실패 → 재시도 ({_retry+1}/3)")
                    time.sleep(0.3)

                j1_now = self.current_joint_state[0] if self.current_joint_state else 999.0
                if j1_now <= 0.5:
                    self.get_logger().warn(f"[RECOVERY] joint1={j1_now:.3f} ≤ 0.5 → 바닥 재파지 시도")
                    # 현재 joint1 유지 + ba_init 자세로 복구 위치 이동
                    recovery_pose = [j1_now] + list(self.ba_init[1:6]) + [0.0, 0.0]
                    for _ in range(3):
                        if self.goal(recovery_pose, 'keep', fast=True):  # 그리퍼 오픈 상태 → 빠른 이동
                            break
                        self.get_logger().warn("[RECOVERY] 복구 자세 이동 실패 → 재시도")
                    # 바닥 감지 → basket_ori.py와 동일: mode="basket"
                    self.reset_sensor_data()
                    target_floor = self.target_get(mode="basket")
                    if target_floor is not None and target_floor.pose.position.z != -99.0:
                        import copy as _copy
                        # basket_ori.py와 동일: MIN_SAFE_Z_BASE 클램핑 (바닥 이하 방지)
                        MIN_SAFE_Z_BASE = -0.310
                        if target_floor.pose.position.z < MIN_SAFE_Z_BASE:
                            self.get_logger().warn(
                                f"[RECOVERY] Target Z ({target_floor.pose.position.z:.3f}) 클램핑 → {MIN_SAFE_Z_BASE}")
                            target_floor.pose.position.z = MIN_SAFE_Z_BASE
                        pre = _copy.deepcopy(target_floor)
                        pre.pose.position.z += 0.05
                        ok = self.move_forward(pre, fallback_to_ik=True)
                        if ok:
                            ok = self.move_forward(target_floor, fallback_to_ik=True)
                        if not ok:
                            self.get_logger().warn("[RECOVERY] 이동 실패 (IK) → wm_init 복귀")
                            while not self.goal(self.wm_init, 'keep', fast=True):
                                self.get_logger().warn("  >> wm_init 복귀 실패 → 재시도")
                                time.sleep(0.2)
                        else:
                            self.close_gripper()
                            time.sleep(0.3)
                            self.get_logger().info("[RECOVERY] 재파지 성공 → 이송 재개 (h → ba_init → ba_end → open)")
                            while not self.goal(self.h, 'keep'):  # 물체 파지 중 → 기본 속도
                                self.get_logger().warn("  >> [RECOVERY] h 이동 실패 → 재시도")
                                time.sleep(0.2)
                            while not self.goal(self.ba_init, 'keep'):  # 물체 파지 중 → 기본 속도
                                self.get_logger().warn("  >> [RECOVERY] ba_init 이동 실패 → 재시도")
                                time.sleep(0.2)
                            while not self.goal(self.ba_end, 'keep'):  # 물체 파지 중 → 기본 속도
                                self.get_logger().warn("  >> [RECOVERY] ba_end 이동 실패 → 재시도")
                                time.sleep(0.2)
                            self.open_gripper()
                            self.get_logger().info("[RECOVERY] 이송 완료 → wm_init 복귀")
                            while not self.goal(self.wm_init, 'keep', fast=True):
                                self.get_logger().warn("  >> [RECOVERY] wm_init 복귀 실패 → 재시도")
                                time.sleep(0.2)
                    else:
                        self.get_logger().warn("[RECOVERY] 바닥 세탁물 미감지 → wm_init 복귀")
                        while not self.goal(self.wm_init, 'keep', fast=True):
                            self.get_logger().warn("  >> wm_init 복귀 실패 → 재시도")
                            time.sleep(0.2)
                else:
                    self.get_logger().warn(f"[RECOVERY] joint1={j1_now:.3f} > 0.5 → 재파지 생략, wm_init 복귀")
                    while not self.goal(self.wm_init, 'keep', fast=True):
                        self.get_logger().warn("  >> wm_init 복귀 실패 → 재시도")
                        time.sleep(0.2)
                continue
            # ────────────────────────────────────────────────────────

            self.open_gripper()
            while not self.goal(self.wm_init, 'keep', fast=True):  # 물체 내려놓은 후 → 빠른 복귀
                self.get_logger().warn("  >> wm_init 복귀 실패 → 재시도 중")
                time.sleep(0.2)
            _t = self._step_time("wm_init 최종 복귀", _t)
            self._step_time("=== 전체 사이클 완료", _t_cycle)
            self._print_timing_summary()
            continue


# ================================================================= #
# ===================== 핵심 변경 부분: Marker ===================== #
# ================================================================= #

    def publish_target_marker(self, pose, id=0, r=1.0, g=0.0, b=0.0):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "grasp_target"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.03 # diameter
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color.a = 0.8
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.lifetime = Duration(sec=0, nanosec=0) # Keeps marker until replaced
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published Marker ID {id} to {marker.header.frame_id}")

    def publish_camera_center_callback(self):
        """
        주기적으로 현재 ee_pose와 ee2cam 행렬을 사용하여 카메라 중심 위치를 Rviz에 표시
        """
        if self.ee_pose is None:
            # self.get_logger().info("ee_pose is None, skipping camera marker")
            return

        ee2cam = np.vstack((np.hstack((self.R_cal, self.t_cal)), [0, 0, 0, 1]))
        
        # 2. Base 기준 카메라 위치 계산
        base2cam = self.ee_pose @ ee2cam
        
        cam_center_pose = Pose()
        cam_center_pose.position.x = base2cam[0,3]
        cam_center_pose.position.y = base2cam[1,3]
        cam_center_pose.position.z = base2cam[2,3]
        
        # 3. 마커 발행 (ID=100, Light Blue)
        # Note: self.publish_target_marker log는 너무 자주 찍히면 노이즈가 되므로 여기서는 수동 구성 고려
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "camera_center"
        marker.id = 100
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = cam_center_pose
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color.a = 0.8
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.lifetime = Duration(sec=1, nanosec=0) # 1 sec lifetime for 0.5 sec timer
        
        self.marker_pub.publish(marker)


# ================================================================= #
# ===================== 핵심 변경 부분: main 함수 =================== #
# ================================================================= #

def main(args=None):
    parser = argparse.ArgumentParser(description='Logic_inte_wm 실행 모드')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'collect', 'inference'],
                        help='normal: rule-based | collect: wm 데이터 수집 | inference: GRU-AE 판단')
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args if ros_args else args)

    try:
        node = ControlTower(run_mode=parsed.mode)
    except RuntimeError as e:
        print(f"노드 초기화 중 오류 발생: {e}")
        return

    # MultiThreadedExecutor는 여전히 사용합니다.
    # 콜백들이 블로킹 함수(spin_until_future_complete)에 의해 중단되지 않도록 하기 위함입니다.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    # 메인 로직을 실행할 작업 스레드
    sequence_thread = threading.Thread(target=node.execute_sequence, daemon=True)
    sequence_thread.start()

    # =================================================================================
    # 중요: 메인 스레드에서는 executor.spin()을 호출하지 않습니다!
    # 대신, 작업 스레드가 끝날 때까지 (또는 Ctrl+C가 눌릴 때까지) 기다리기만 합니다.
    # 모든 ROS 메시지 처리는 sequence_thread 내부의 rclpy.spin_until_future_complete가
    # 전담하게 됩니다.
    # =================================================================================
    try:
        # 작업 스레드가 살아있는 동안 메인 스레드는 여기서 대기합니다.
        sequence_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 중단 요청됨.')
    finally:
        # 추론 스레드를 먼저 종료 (destroy_node 전에 해야 get_logger segfault 방지)
        if hasattr(node, '_infer_thread_running'):
            node._infer_thread_running = False
            node._infer_trigger.set()
            node._transport_trigger.set()
            node._infer_thread.join(timeout=2.0)
        node.get_logger().info('노드 종료 중...')
        executor.shutdown()
        if rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()