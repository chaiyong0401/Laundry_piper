
import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.clock import Clock
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from action_msgs.msg import GoalStatus
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool, Float32MultiArray # Added Float32MultiArray
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
from control_msgs.msg import JointTrajectoryControllerState

import math
from pathlib import Path
import pinocchio as pin
import os
import datetime
import argparse


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


URDF_OR_XACRO = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_description/urdf/piper_description.xacro"
URDF_JOINT_NAMES = [f"joint{i+1}" for i in range(8)]
CTRL_JOINT_NAMES = ['joint1','joint2','joint3','joint4','joint5','joint6']

DISPLAY_IDX = [3, 4]   # joint4, joint5
GRAVITY = np.array([0.0, 0.0, -9.81])
PRED_RATE_HZ = 100.0

SIGMA = 0.180  # Increased from 0.0856 to prevent false-positive torque violations during Mode 9 Cartesian descent
FOUR_SIGMA = 4.0 * SIGMA

A = np.array([
    [0.336031, -0.046468,  0.258372,  0.180103, -0.022022, -1.702219],
    [0.11016,  -0.00126,  -0.015487,  1.4827,    0.00906,   0.558599],
    [-0.009071, -0.001866, -0.026797, -0.026905,  1.37266,   0.943661]
])

B = np.array([0.24018, 0.031745, 0.161793])

def build_pin_model(urdf_or_xacro: str):
    p = Path(urdf_or_xacro).expanduser().resolve()
    if p.suffix == ".xacro":
        import xacro
        xml = xacro.process_file(str(p)).toxml()
        mdl = pin.buildModelFromXML(xml)
    else:
        mdl = pin.buildModelFromUrdf(str(p))
    mdl.gravity.linear = np.array(GRAVITY, dtype=float)
    return mdl


class ControlTower(Node):

    def __init__(self, run_mode: str = 'normal'):
        super().__init__('control_tower_node')
        # run_mode: 'normal'   → 기존 동작 (rule-based 판단, 데이터 수집 없음)
        #           'collect'  → 데이터 수집 모드 (human labeling)
        #           'inference'→ 학습된 GRU-AE로 판단
        self.run_mode = run_mode
        self.get_logger().info(f"[MODE] 실행 모드: {self.run_mode}")

        self.ee_pose = None
        self.wm_init_ee_orientation = None  # overcome2 시 wm_mid 도달 후 캡처
        self.ee_pose_st = PoseStamped() # Latest EE Pose as PoseStamped
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        self.cum_x = [0.0] * 16
        self.cum_y = [0.0] * 16
        self.cum_z = [0.0] * 16

        self.z_aligned = False
        self.contacted = False

        self.z_aligned = False
        self.contacted = False

        self.z_aligned_k = False
        self.contacted_k = False
        self.contacted_kk = False
        self.z_aligned_f = False
        self.contacted_f = False
        self.contacted_ff = False

        # self.norm = None
        # self.theta = None
        self.norm = 0.0
        self.theta = 0.0
        self.max_norm = 0.0

        self.minmax_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/taxel_minmax_norm.npy"
        self.minmax_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/taxel_minmax_norm.npy"
        self.gmin, self.grng = self._load_minmax(self.minmax_path)
        self.get_logger().info(f"MinMax loaded: {self.minmax_path}")
        
        # [USER REQUEST] Continuous Monitoring Variables
        self.monitor_slip = False
        self.monitor_initial_norm = 0.0
        self.monitor_initial_theta = 0.0
        self.current_goal_handle = None

        # [DATA COLLECTION] GRU-AE 학습용 데이터 수집
        self.is_recording = False
        self.grasp_buffer = []          # Model 1: 6D feature 시계열
        self.grasp_raw_buffer = []      # Model 1: raw taxel 48D 시계열
        self.mrl = 0.0
        self.data_save_dir = "/home/dyros/mcy_ws/grasp_data"
        os.makedirs(self.data_save_dir, exist_ok=True)

        # [INFERENCE] GRU-AE 모델 로드
        self.gru_ae_model = None
        self.gru_ae_scaler = None
        self.gru_ae_threshold = 0.02
        if self.run_mode == 'inference':
            self._load_gru_ae()

        # [SLIDING WINDOW] 실시간 inference 상태
        self.realtime_buffer     = deque(maxlen=50)
        self.inference_stride    = 5
        self.inference_frame_cnt = 0
        self.infer_consec_normal = 0
        self.infer_consec_needed = 3
        self.infer_latest_score  = None

########################## 비지도 클러스터링 ####################################################

        # # === KMeans/FCM 모델 로드 및 전처리 파라미터 준비 ===
        # self.kmeans_available = False
        # self.fcm_available = False

        # self.kmeans = None
        # self.fcm_centers = None
        # self.fcm_m = 2.0

        # self.imp_stats = None
        # self.scaler_mean = None
        # self.scaler_scale = None

        # self.minmax_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/taxel_minmax_norm.npy"
        # self.gmin, self.grng = self._load_minmax(self.minmax_path)
        # self.get_logger().info(f"MinMax loaded: {self.minmax_path}")

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
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.current_joint_state = None

        self._arm_client = ActionClient(self, FollowJointTrajectory, '/moveit_action/arm_controller/follow_joint_trajectory')
        self._arm_direct_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        self._pose_client = ActionClient(self, PoseGoal, 'pose_goal')
        self._target_client = ActionClient(self, TargetPose, 'target_pose')

        # TF buffer: DEEP 프레임의 실제 위치를 lookup하기 위함
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # [Controller Reset Client]
        from piper_msgs.srv import Enable
        self._enable_client = self.create_client(Enable, '/enable_srv')
        
        self.marker_pub = self.create_publisher(Marker, 'grasp_target_marker', 10)

        # 서버가 뜰 때까지 기다리는 로직을 __init__에 추가하면 더 안정적입니다.
        servers_ready = True
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Arm action server not available.")
            servers_ready = False
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

        if not servers_ready:
            self.destroy_node()
            rclpy.shutdown()
            raise RuntimeError("필수 액션 서버를 찾을 수 없습니다.")


        self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.gripper_joint_names = ['joint7', 'joint8']

        # [CALIBRATION] Camera Extrinsics (EE to RGB Optical Center)
        # These are used for BOTH coordinate transformation and RViz visualization
        self.R_cal = np.array([
            [0.025154, 0.997896, 0.059755],
            [-0.999639, 0.025669, -0.007874],
            [-0.009391, -0.059535, 0.998182]
        ])
        # self.t_cal = np.array([[-0.068162],
        #                        [0.030760],
        #                        [0.116954]])
        self.t_cal = np.array([[-0.058162],
                               [0.035760],
                               [0.07954]])
        
        # self.R_cal = np.array([
        #     [-0.07403140416428758,  0.9970253999896115, -0.02144068499412377],
        #     [-0.997213265121594,   -0.07381230005954229,  0.010837353249242954],
        #     [0.009222530183886607,  0.022183239967900933,  0.9997113827508088]])
        # self.t_cal = np.array([[-0.05615831961252831],
        #                      [ 0.035761448900299364],
        #                      [ 0.06590399326489311]]) 


        # original Rotation, translation 
        # R = np.array([
        #     [-0.07403140416428758,  0.9970253999896115, -0.02144068499412377],
        #     [-0.997213265121594,   -0.07381230005954229,  0.010837353249242954],
        #     [0.009222530183886607,  0.022183239967900933,  0.9997113827508088]
        # ])
        # t = np.array([[-0.04615831961252831],
        #                 [ 0.035761448900299364],
        #                 [ 0.06690399326489311]]) ######## 원래 07077399326489311 ############
        # ee2cam = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

        self.ba_init = [-1.6379, 1.5724, -1.2721, -0.0566, 1.2838, 0.2908, 0.0, 0.0]
        #origianl
        self.wm_init = [1.5640, 0.4538, -1.5319, -0.0488, 1.2797, 0.3169, 0.0, 0.0]
        # self.wm_init = [1.566641, 0.402892, -1.852798, -0.049529, 1.651095, 0.299209, 0.0, 0.0]
        self.ba_end = [-1.5527, 2.6613, -1.5133, -0.2642, 0.6119, 0.3951, 0.0, 0.0]
        self.wm_end = [1.571, 2.146, -2.268, 0.000, 0.594, 0.000, 0.0, 0.0]
        self.wm_mid_original = [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -1.357858404, 0.0, 0.0]
        #origianl
        # self.wm_mid= [1.5058533, 0.304101252, -0.156088912, -0.158199636, 0.0684677, -1.267376376, 0.0, 0.0]
        self.wm_mid= [1.569035, 0.043279, 0.026288, -0.075341, 0.159264, 0.256235, 0.0, 0.0]

        #origianl
        self.wm_view = [1.567378, 0.584147, -0.976707, -0.132923, 1.276936, -1.211015, 0.0, 0.0]
        # self.wm_view = [1.562546, 0.641172, -1.202764, -0.089156, 1.286792, -1.478571, 0.0, 0.0]

        self.home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.f = [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -0.1, 0.0, 0.0]

        self.ff = [1.5533, 0.1396, -0.4014, 0.0349, 0.3665, 0.1745, 0.0, 0.0]
        # self.ff = [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -1.357858404, 0.0, 0.0]


         # ======== (추가) Pinocchio 세팅 ========
        self.model = build_pin_model(URDF_OR_XACRO)
        self.data = self.model.createData()
        self.nj = len(URDF_JOINT_NAMES)

        self.qd = np.zeros(self.nj)
        self.qdotd = np.zeros(self.nj)
        self.qddotd = np.zeros(self.nj)
        self.have_desired = False

        self.effort = np.zeros(self.nj)
        self.effort_last_valid = np.zeros(self.nj)
        self.last_js_msg = None

        # ======== (추가) torque용 desired/effort 구독 ========
        self.create_subscription(
            JointTrajectoryControllerState,
            "/arm_controller/state",
            self.state_cb,
            10
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_cb,
            10
        )

        # ======== (추가) 100Hz 안전 체크 타이머 ========
        self.create_timer(1.0 / PRED_RATE_HZ, self.tick)

        # [REAL-TIME] Camera Marker Timer (2Hz)
        self.camera_timer = self.create_timer(0.5, self.publish_camera_center_callback)

        # paused 플래그/락
        self._lock = threading.Lock()
        self.paused = False

        self.stop_requested = False
        self.stop_reason = ""
        self.last_forward_pose = None   # move_forward에서 저장할 pose
        self.recover_pending = False
        self.monitor_torque = False   # ✅ forward 구간에서만 True
        self.torque_stop = False


    def get_deep_pose(self):
        """TF lookup으로 DEEP 프레임의 base_link 기준 4x4 pose를 반환.
        /end_pose(하드웨어 FK, link6 위치)와 다르므로 반드시 이 함수를 사용해야 함.
        실패 시 None 반환."""
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'DEEP',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            tr = t.transform.translation
            ro = t.transform.rotation
            T = np.eye(4)
            T[:3, :3] = R.from_quat([ro.x, ro.y, ro.z, ro.w]).as_matrix()
            T[:3, 3] = [tr.x, tr.y, tr.z]
            return T
        except Exception as e:
            self.get_logger().error(f"[get_deep_pose] TF lookup 실패: {e}")
            return None

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
    def _send_goal(self, client, joint_names, joint_values, wait_for_result=True, timeout_sec=10.0):
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
        # 여기서 spin은 작업 스레드에서만 독점적으로 사용됩니다.
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        self.current_goal_handle = goal_handle # Store handle for cancellation
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn(f"Goal for '{client._action_name}' rejected.")
            # return False
        if not wait_for_result:
            return True

        get_result_future = goal_handle.get_result_async()
        # 여기서 spin은 작업 스레드에서만 독점적으로 사용됩니다.
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result()
        if not result:
            self.get_logger().error("결과를 받지 못했습니다.")
            # return False

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
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
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

    def send_pose_wm(self, pose: PoseStamped):
        goal_msg = PoseGoal.Goal()
        goal_msg.target_pose = pose

        goal_msg.mode = 0 # pose

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
            self.get_logger().info(f"✅ pose 명령 성공: {result.message}")
            return True
        else:
            self.get_logger().warn(f"⚠️ pose 명령 실패: {result.message}")
            # return False

    def send_pose_ba(self, pose: PoseStamped):
        goal_msg = PoseGoal.Goal()
        self.get_logger().info(f"pose in send_pose_ba: {pose}")
        goal_msg.target_pose = pose

        # .action 파일에 정의된 이름과 정확히 일치시켜야 합니다.
        goal_msg.mode = 1 # rpy
        goal_msg.roll_tol_rad = 1.57/2            # 'tol_roll'/'tol_pitch' -> 'rp_tol_rad'
        goal_msg.pitch_tol_rad = 1.57/2
        goal_msg.yaw_tol_rad = 6.28              # 'tol_yaw' -> 'yaw_tol_rad'
        
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
            self.get_logger().info(f"✅ pose 명령 성공: {result.message}")
            return True
        else:
            self.get_logger().warn(f"⚠️ pose 명령 실패: {result.message}, Yaw 회전 재시도 진입...")

        # --- Yaw Search Strategy (from Logic_inte_ba.py) ---
        # IK failed for default orientation. Try rotating yaw (wrist).
        import copy
        yaw_candidates = [90, -90, 180, 45, -45, 135, -135] # degrees
        
        # Current Target Orientation (Quaternion)
        orig_q = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
        
        for yaw in yaw_candidates:
            self.get_logger().info(f"  >> Yaw {yaw}도 회전하여 재시도 중...")
            
            # Create rotated pose
            from scipy.spatial.transform import Rotation as ScipyRotation
            
            rot_z = ScipyRotation.from_euler('z', yaw, degrees=True)
            orig_rot = ScipyRotation.from_quat(orig_q)
            new_rot = orig_rot * rot_z # Local rotation
            
            nx, ny, nz, nw = new_rot.as_quat()
            
            new_pose = copy.deepcopy(pose)
            new_pose.pose.orientation.x = nx
            new_pose.pose.orientation.y = ny
            new_pose.pose.orientation.z = nz
            new_pose.pose.orientation.w = nw
            
            # Try Mode 6 (IK)
            run_goal = PoseGoal.Goal()
            run_goal.target_pose = new_pose
            run_goal.mode = 6 # IK
            
            future = self._pose_client.send_goal_async(run_goal)
            rclpy.spin_until_future_complete(self, future)
            
            gh = future.result()
            if not gh or not gh.accepted:
                continue
                
            res_future = gh.get_result_async()
            rclpy.spin_until_future_complete(self, res_future)
            res = res_future.result().result
            
            if res.success:
                self.get_logger().info(f"  >> Yaw {yaw}도 회전으로 IK 해결 및 이동 성공!")
                pose.pose.orientation = new_pose.pose.orientation
                return True
                
        self.get_logger().error("pose 명령 최종 실패: 모든 Yaw 각도 시도 실패")
        return False


    def move_forward_pre(self, pose: PoseStamped):
        goal_msg = PoseGoal.Goal()
        goal_msg.target_pose = pose

        goal_msg.mode = 9

        send_goal_future = self._pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        try:
            goal_handle = send_goal_future.result()
        except Exception as e:
            self.get_logger().error(f"pose_goal 전송 예외: {e!r}")
            return False

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("pose_goal이 거부되었습니다.")
            return False
        self.get_logger().info("pose_goal 명령 전송 완료.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f"✅ pose 명령 성공: {result.message}")
            return True
        else:
            self.get_logger().warn(f"⚠️ pose 명령 실패: {result.message}")
            return False
        

    def move_forward(self, pose_st: PoseStamped, fallback_to_ik=True, disable_torque=False, ik_mode=6, rpy_tol=[0.0, 0.0, 0.0]):
        """
        Hybrid Move Forward (Mode 9 -> Mode ik_mode)
        with 4-sigma Torque Stop Protection.
        Returns: "success", "stop", "fail"
        ik_mode: 6 for self-only, 10 for env-aware, 1 for planning with rpy_tol (added)
        """
        ik_name = "Mode 6 (Joint IK)" if ik_mode == 6 else (f"Mode 10 (Env IK)" if ik_mode == 10 else f"Mode {ik_mode}")
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
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
            
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
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=result_timeout)
            
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

        # --- Step 1. Try Mode 9 (Cartesian) ---
        self.get_logger().info("Trying Mode 9 (Cartesian)...")
        res9 = run_mode(9)
        
        if res9 == "success":
            return "success"
        if res9 == "stop":
            return "stop"
            

        # If Mode 9 failed (or rejected/timeout) -> Fallback
        if fallback_to_ik:
            ik_name = "Mode 6 (Joint IK)" if ik_mode == 6 else f"Mode {ik_mode} (Env IK)"
            self.get_logger().warn(f"Mode 9 result='{res9}'. Retrying with {ik_name}...")
            # --- Step 2. Try IK Mode ---
            res_ik = run_mode(ik_mode)
            return res_ik
        else:
            self.get_logger().error(f"Mode 9 result='{res9}'. Fallback disabled -> Abort.")
            return "fail"



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

    def move_back(self, pose: PoseStamped) -> bool:
        goal_msg = PoseGoal.Goal()
        goal_msg.target_pose = pose
        goal_msg.mode = 7   # back 모드

        # 1) goal 전송
        send_goal_future = self._pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("move_back goal rejected")
            return False

        # 2) 결과 대기
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f"✅ move_back 성공: {result.message}")
            return True
        else:
            self.get_logger().warn(f"⚠️ move_back 실패: {result.message}")
            return False
        
    
    def state_cb(self, msg: JointTrajectoryControllerState):
        if msg.joint_names:
            jmap = {n: i for i, n in enumerate(msg.joint_names)}
        else:
            jmap = {n: i for i, n in enumerate(URDF_JOINT_NAMES)}

        desired = msg.desired
        q  = np.zeros(self.nj)
        qd = np.zeros(self.nj)
        qdd= np.zeros(self.nj)

        for j, n in enumerate(URDF_JOINT_NAMES):
            if n in jmap:
                k = jmap[n]
                if len(desired.positions) > k:     q[j]  = desired.positions[k]
                if len(desired.velocities) > k:    qd[j] = desired.velocities[k]
                if len(desired.accelerations) > k:
                    a = desired.accelerations[k]
                    qdd[j] = a if math.isfinite(a) else 0.0

        self.qd, self.qdotd, self.qddotd = q, qd, qdd
        self.have_desired = True


    def joint_cb(self, msg: JointState):
        self.last_js_msg = msg
        idx_map = {n: i for i, n in enumerate(msg.name)}

        for j, n in enumerate(URDF_JOINT_NAMES):
            if n in idx_map:
                k = idx_map[n]
                if len(msg.effort) > k:
                    val = msg.effort[k]
                    if math.isfinite(val):
                        self.effort[j] = val
                        self.effort_last_valid[j] = val
                    else:
                        self.effort[j] = self.effort_last_valid[j]


    def tick(self):
        # forward 아닐 땐 아예 계산/판단 안 함
        if not self.monitor_torque:
            return
        if not self.have_desired or self.paused:
            return
        if self.stop_requested:   # 이미 stop 걸렸으면 더 안 봄
            return

        try:
            tau_pred = pin.rnea(self.model, self.data, self.qd, self.qdotd, self.qddotd)
            tau_pred = np.abs(tau_pred)
        except Exception as e:
            self.get_logger().warn(f"rnea failed: {e}")
            return

        tau_real = self.effort.copy()
        tau_pre = A @ tau_pred[0:6] + B

        for j in DISPLAY_IDX:
            pred_val = float(tau_pre[j - 2])
            real_val = float(tau_real[j])
            err = abs(pred_val - real_val)

            if err > FOUR_SIGMA:
                reason = f"4σ torque violation ({URDF_JOINT_NAMES[j]} err={err:.4f} > {FOUR_SIGMA:.4f})"
                self.get_logger().warn(f"[4σ 위반] {reason}")

                # ✅ 여기서 stop + 플래그 세팅까지 한 번에
                self._request_stop(reason)
                self.torque_stop = True         # 토크 기반 stop 발생 표시
                self.recover_pending = True
                break


    def _request_stop(self, reason: str):
        with self._lock:
            if self.stop_requested:   # 중복 stop 방지
                return
            self.stop_requested = True
            self.stop_reason = reason
            # self.paused = True        # tick 중지

        self.get_logger().warn(f" 정지 요청: {reason}")

        #  즉시 멈춤: 현재 자세를 한 점 JT로 쏨
        threading.Thread(target=self._send_stop_override, daemon=True).start()


    def _send_stop_override(self):
        """
        현재 joint_states 기반으로 6축 arm을 '지금 자세 그대로, 속도 0'으로
        아주 짧은 trajectory를 보내서 즉시 멈추는 함수.
        어떤 곳에서도 spin_until_future_complete를 호출하지 않는다.
        """

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
        p1.time_from_start = Duration(sec=0, nanosec=10_000_000)  # 0.01s

        p2 = JointTrajectoryPoint()
        p2.positions = q_now
        p2.velocities = [0.0] * len(CTRL_JOINT_NAMES)
        p2.time_from_start = Duration(sec=0, nanosec=20_000_000)  # 0.02s

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

        # (옵션) 혹시 stop thread가 씹혔을 때 대비해서 한 번 더 쏴도 됨
        self._send_stop_override()

        # ✅ STOP 플래그/이유만 내림. paused는 main이 back 끝난 뒤 풀 것
        with self._lock:
            self.stop_requested = False
            self.stop_reason = ""

        self.get_logger().info("✅ STOP 처리 완료 → main에서 close/back 진행")
        return True
    
    def _clear_recover_state(self):
        with self._lock:
            self.recover_pending = False
            self.last_forward_pose = None
            self.paused = False      # ✅ 이제 다시 tick 허용
            
    
    def move_backward(self, pose: PoseStamped):
        """잡은 후 안전하게 후진(Retreat)하는 함수 (Z + 15cm)"""
        self.get_logger().info("(Retreat) 시작: Z축 +15cm")
        
        # 깊은 복사 대신 새로운 객체 생성
        retreat_pose = PoseStamped()
        retreat_pose.header = pose.header
        retreat_pose.pose.orientation = pose.pose.orientation
        
        retreat_pose.pose.position.x = pose.pose.position.x
        retreat_pose.pose.position.y = pose.pose.position.y
        retreat_pose.pose.position.z = pose.pose.position.z + 0.15 
        
        # [IMPORTANT] Retreat 시에는 이미 그리퍼가 바구니나 세탁물 모델과 겹쳐(Collision) 있는 경우가 많습니다.
        # 따라서 충돌을 체크하는 Mode 9 대신, 자가 충돌만 체크하는 Mode 6를 사용하여 강제로 빠져나옵니다.
        res = self.move_forward(retreat_pose, fallback_to_ik=True, ik_mode=6)
        
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

    def reset_sensor_data(self):
        self.prev_x, self.prev_y, self.prev_z = None, None, None
        self.cum_x, self.cum_y, self.cum_z = [0.0] * 16, [0.0] * 16, [0.0] * 16
        self.z_aligned, self.contacted = False, False
        self.get_logger().info("센서 기준값 및 상태 초기화 완료.")

    def _spin_input(self, prompt: str) -> str:
        """input()을 별도 스레드에서 실행하고, 그 동안 ROS2 spin을 유지.
        collect 모드에서 input() 블로킹 시 센서 콜백이 멈추는 문제를 방지."""
        result = [None]
        done = threading.Event()

        def _ask():
            try:
                result[0] = input(prompt).strip().lower()
            except EOFError:
                result[0] = ''
            finally:
                done.set()

        t = threading.Thread(target=_ask, daemon=True)
        t.start()
        while not done.is_set():
            rclpy.spin_once(self, timeout_sec=0.05)
        t.join(timeout=1.0)
        return result[0] if result[0] is not None else ''

    def _save_grasp_data(self, label: str):
        """수집된 파지 구간 시계열을 레이블과 함께 저장."""
        if len(self.grasp_buffer) < 10:
            self.get_logger().warn(f"[DATA] 프레임 수 부족 ({len(self.grasp_buffer)}), 저장 스킵")
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 6D feature 저장
        seq = np.array(self.grasp_buffer, dtype=np.float32)       # [T, 6]
        fname = os.path.join(self.data_save_dir, f"grasp_{ts}_{label}.npy")
        np.save(fname, seq)
        # raw taxel 저장 (48D)
        raw = np.array(self.grasp_raw_buffer, dtype=np.float32)   # [T, 48]
        fname_raw = os.path.join(self.data_save_dir, f"grasp_{ts}_{label}_raw.npy")
        np.save(fname_raw, raw)
        self.get_logger().info(f"[DATA] 저장 완료: {fname} (shape={seq.shape}), {fname_raw} (shape={raw.shape}), label={label}")

    def _load_gru_ae(self):
        """학습된 GRU-AE 모델 + scaler + threshold 로드."""
        data_dir    = self.data_save_dir
        model_path  = os.path.join(data_dir, "grasp_gru_ae.pt")
        scaler_path = os.path.join(data_dir, "grasp_scaler.pkl")
        thresh_path = os.path.join(data_dir, "grasp_threshold.npy")
        for p in [model_path, scaler_path, thresh_path]:
            if not os.path.exists(p):
                self.get_logger().warn(f"[INFERENCE] 파일 없음: {p}")
                return
        try:
            import torch, joblib, sys
            sys.path.insert(0, "/home/dyros/mcy_ws")
            from train_gru_ae import GRUAE
            scaler    = joblib.load(scaler_path)
            input_dim = scaler.n_features_in_
            model = GRUAE(input_dim=input_dim, hidden_dim=64,
                          num_layers=2, dropout=0.2, seq_len=50)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            self.gru_ae_model     = model
            self.gru_ae_scaler    = scaler
            self.gru_ae_threshold = float(np.load(thresh_path)[0])
            self.get_logger().info(
                f"[INFERENCE] GRU-AE 로드 완료 "
                f"(input_dim={input_dim}, threshold={self.gru_ae_threshold:.6f})")
        except Exception as e:
            self.get_logger().error(f"[INFERENCE] 모델 로드 실패: {e}")

    def _run_windowed_inference(self) -> float | None:
        """realtime_buffer 의 현재 50프레임 window 로 즉시 anomaly score 계산."""
        if self.gru_ae_model is None or self.gru_ae_scaler is None:
            return None
        if len(self.realtime_buffer) < 50:
            return None
        try:
            import torch
            seq = np.array(list(self.realtime_buffer), dtype=np.float32)   # [50, 6]
            seq_s = self.gru_ae_scaler.transform(seq).astype(np.float32)
            x = torch.tensor(seq_s).unsqueeze(0)                           # [1, 50, 6]
            with torch.no_grad():
                recon = self.gru_ae_model(x)
            return float(((recon - x) ** 2).mean().item())
        except Exception as e:
            self.get_logger().error(f"[SLIDING] inference error: {e}")
            return None

    def _run_inference(self):
        """수집된 grasp_buffer로 GRU-AE anomaly score 계산."""
        if self.gru_ae_model is None:
            self.get_logger().warn("[INFERENCE] 모델 미로드, rule-based로 fallback")
            return None
        if len(self.grasp_buffer) < 10:
            self.get_logger().warn("[INFERENCE] 프레임 수 부족")
            return None
        try:
            import torch
            from scipy.signal import resample
            seq = np.array(self.grasp_buffer, dtype=np.float32)  # [T, 4]
            seq_fixed = resample(seq, 50, axis=0).astype(np.float32)  # [50, 4]
            x = torch.tensor(seq_fixed).unsqueeze(0)                  # [1, 50, 4]
            with torch.no_grad():
                recon = self.gru_ae_model(x)
            score = float(((recon - x) ** 2).mean().item())
            return score
        except Exception as e:
            self.get_logger().error(f"[INFERENCE] 추론 실패: {e}")
            return None

    def _request_human_label(self):
        """사람이 직접 파지 결과를 확인하고 레이블을 입력.
        Returns: 'success' | 'failure' | 'perturbation' | None (skip)
        """
        print("\n" + "="*50)
        print("[DATA COLLECTION] 파지 결과를 직접 확인하고 입력하세요")
        print("  s = success      (물체가 안정적으로 잡힘)  → 이송 진행")
        print("  f = failure      (물체가 잡히지 않음)      → 그리퍼 오픈, 재시도")
        print("  p = perturbation (약하게 걸렸거나 불안정)  → 그리퍼 오픈, 재시도")
        print("  x = skip         (저장 안 함, rule-based 사용)")
        print(f"  [참고] rule-based: norm={self.norm:.3f}, theta={self.theta:.1f}, max_norm={self.max_norm:.3f}")
        print("="*50)
        while True:
            key = self._spin_input("레이블 입력 (s/f/p/x): ")
            if key in ('s', 'f', 'p', 'x'):
                break
            print("  s / f / p / x 중 하나를 입력하세요")
        label_map = {'s': 'success', 'f': 'failure', 'p': 'perturbation'}
        if key == 'x':
            self.get_logger().info("[DATA] 저장 스킵")
            return None
        label = label_map[key]
        self._save_grasp_data(label=label)
        return label

    def _trigger_slip_stop(self):
        # [USER REQUEST] Record Slip but CONTINUE motion
        if not self.contacted:
            self.contacted = True
            self.get_logger().warn(f"🚨 [MONITOR] Slip Recorded! Will recover after motion.")
            # self.monitor_slip = False # Keep monitoring? Or stop to avoid spam?
            # Stop monitoring to avoid spamming logs for the same slip
            self.monitor_slip = False 
        
        # Do NOT cancel goal
        # if self.current_goal_handle: ...

    
    def joint_state_callback(self, msg):
        # Filter for arm joints 1-6 + gripper 7-8 if needed
        # We assume self.arm_joint_names matches msg.name or we just map by index if order is fixed.
        # usually msg.name contains all.
        
        # Let's just store the message for now, or map to our 8-dof vector
        # self.current_joint_state = msg
        
        # Or parse immediately to 8-dof list:
        # Assuming joint1..6, joint7, joint8 are present
        
        current_map = {n: p for n, p in zip(msg.name, msg.position)}
        # Piper usually has 'joint1'...'joint8'
        
        pos = []
        # Arm
        for i in range(1, 7):
            jn = f"joint{i}"
            pos.append(current_map.get(jn, 0.0))
        # Gripper
        pos.append(current_map.get("joint7", 0.0)) # joint7
        pos.append(current_map.get("joint8", 0.0)) # joint8
        
        self.current_joint_state = pos

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
            self.theta = theta_deg
            self.max_norm = max_norm

            # [DATA COLLECTION] MRL 계산 + 버퍼링
            valid_mask = norms > 0.1
            if valid_mask.sum() >= 2:
                unit_vecs = vecs[valid_mask] / norms[valid_mask, np.newaxis]
                mean_unit = np.mean(unit_vecs, axis=0)           # (3,)
                mrl = float(np.linalg.norm(mean_unit))           # [0, 1]
            else:
                mean_unit = np.zeros(3)
                mrl = 0.0
            self.mrl = mrl
            if self.is_recording:
                frame = [mrl, float(norm), float(max_norm),
                         float(mean_unit[0]), float(mean_unit[1]), float(mean_unit[2])]
                raw_frame = vecs.flatten().tolist()   # [48]
                self.grasp_buffer.append(frame)
                self.grasp_raw_buffer.append(raw_frame)

                # [SLIDING WINDOW] 실시간 inference
                if self.run_mode == 'inference':
                    self.realtime_buffer.append(frame)
                    self.inference_frame_cnt += 1
                    if (len(self.realtime_buffer) >= 50 and
                            self.inference_frame_cnt % self.inference_stride == 0):
                        score = self._run_windowed_inference()
                        if score is not None:
                            self.infer_latest_score = score
                            if score < self.gru_ae_threshold:
                                self.infer_consec_normal += 1
                            else:
                                self.infer_consec_normal = 0

            # [Logic_inte_ba Logic Check]
            
            # [USER REQUEST] Continuous Slip Monitoring
            if getattr(self, "monitor_slip", False):
                # 1. Norm Check: Drop by 50%
                if self.monitor_initial_norm > 0 and norm < (self.monitor_initial_norm * 0.5):
                    self.get_logger().error(f"🚨 [MONITOR] Slip Detected! Norm Dropped: {self.monitor_initial_norm:.3f} -> {norm:.3f}")
                    self._trigger_slip_stop()
                
                # 2. Theta Check: Change > 20 deg (if significant)
                elif abs(theta_deg - self.monitor_initial_theta) > 20.0:
                    self.get_logger().error(f"🚨 [MONITOR] Slip Detected! Theta Shift: {self.monitor_initial_theta:.1f} -> {theta_deg:.1f}")
                    self._trigger_slip_stop()


            # 1. Noise Filter (If force is too small, ignore norm)
            if max_norm < 0.5: 
                self.z_aligned = False
                self.contacted = False
            # 2. Heuristic Logic
            elif norm < 0.35:
                # self.get_logger().info(f" [Tactile] 0.35 이하 norm: {norm:.3f}, max_norm: {max_norm:.1f}, theta: {theta_deg:.1f}")
                self.z_aligned = False
                self.contacted = False
            else:
                # self.get_logger().info(f" [Tactile] 0.35 이상 norm: {norm:.3f}, max_norm: {max_norm:.1f}, theta: {theta_deg:.1f}")
                # if theta_deg < 70.0:
                if norm < 1.0: 
                    if theta_deg < 55.0:
                        # Vertical Force -> Z-Aligned
                        # if not self.z_aligned:
                        #     self.get_logger().info(f"\n⚡ [TACTILE] Z-Aligned! (θ={theta_deg:.1f}°)")
                        self.z_aligned = True
                        self.contacted = False
                    
                    # elif theta_deg <= 110.0:
                    elif theta_deg <= 80.0:
                        # Side Force -> Contact
                        # if not self.contacted:
                        #     self.get_logger().info(f"\n⚡ [TACTILE] Contact Detected! (θ={theta_deg:.1f}°)")
                        self.contacted = True
                        self.z_aligned = False
                    else:
                        self.z_aligned = False
                        self.contacted = False
                
                else: 
                    if theta_deg < 35.0:
                        # Vertical Force -> Z-Aligned
                        # if not self.z_aligned:
                        #     self.get_logger().info(f"\n⚡ [TACTILE] Z-Aligned! (θ={theta_deg:.1f}°)")
                        self.z_aligned = True
                        self.contacted = False
                    
                    # elif theta_deg <= 110.0:
                    elif theta_deg <= 80.0:
                        # Side Force -> Contact
                        # if not self.contacted:
                        #     self.get_logger().info(f"\n⚡ [TACTILE] Contact Detected! (θ={theta_deg:.1f}°)")
                        self.contacted = True
                        self.z_aligned = False
                    else:
                        self.z_aligned = False
                        self.contacted = False

            # self.norm = norm
            # self.theta = theta_deg

            # if norm < 0.55:
            #     self.z_aligned = False
            # else:
            #     if theta_deg < 70.0:
            #         self.z_aligned = True
                    
            #     elif theta_deg <= 110.0:
            #         self.contacted = True
            #         self.z_aligned = False
            #     else:
            #         self.z_aligned = False

                    


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

            # prev 갱신 (항상 마지막에)
            self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals
            
        except Exception as e:
            self.get_logger().error(f"Sensor callback error: {e}")


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

    def target_get_wm(self, mode="failure", first=True): ### basket, wm, mid
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

            # [Two-Stage Check] Sentinel Passthrough
            if result.target_pose.position.z == -99.0:
                self.get_logger().info("  >> [Sentinel] Empty Detected (-99.0). Returning immediately.")
                pose_stamped = PoseStamped()
                pose_stamped.pose = result.target_pose
                return pose_stamped

            R = np.array([
                [-0.07403140416428758,  0.9970253999896115, -0.02144068499412377],
                [-0.997213265121594,   -0.07381230005954229,  0.010837353249242954],
                [0.009222530183886607,  0.022183239967900933,  0.9997113827508088]
            ])

            t = np.array([[-0.04615831961252831],
                        [ 0.035761448900299364],
                        [ 0.06690399326489311]]) ######## 원래 07077399326489311 ############

            ee2cam = np.vstack((np.hstack((self.R_cal, self.t_cal)), [0, 0, 0, 1]))

            base2cam = self.ee_pose @ ee2cam ## O2C
            

            # 1. Transform to EE and Base frames
            p_cam = np.array([result.target_pose.position.x, result.target_pose.position.y, result.target_pose.position.z, 1.0])
            p_ee = ee2cam @ p_cam
            pose = self.transform_pose_with_offset(result.target_pose, base2cam) ## O2t

            # [DEBUG] Green Logs for coordinate verification
            self.get_logger().info(f"\033[92m[Debug] Camera frame (EA): [{p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f}]\033[0m")
            self.get_logger().info(f"\033[92m[Debug] EE frame (local): [{p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}]\033[0m")
            self.get_logger().info(f"\033[92m[Debug] Base frame (bA):  [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]\033[0m")
            self.get_logger().info(f"\033[94m변환한 bA: {pose}\033[0m") # Blue

            # [VISUALIZATION] Publish Camera Center Marker (Blue, ID=100)
            cam_center_pose = Pose()
            cam_center_pose.position.x = base2cam[0,3]
            cam_center_pose.position.y = base2cam[1,3]
            cam_center_pose.position.z = base2cam[2,3]
            self.publish_target_marker(cam_center_pose, id=100, r=0.0, g=0.5, b=1.0) # Light Blue

            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            
            # Use strict coordinate matching in case node2.py flag wasn't passed
            is_fixed_coord = (abs(p_cam[0] - 0.057) < 0.001 and 
                              abs(p_cam[1] - (-0.044)) < 0.001 and 
                              abs(p_cam[2] - 0.089) < 0.001)

            if result.target_pose.orientation.x > 90.0 or is_fixed_coord:
                pose_stamped.header.frame_id = "base_link|fixed"
            else:
                pose_stamped.header.frame_id = "base_link"
            
            # [USER REQUEST] Limit Minimum Z Height
            MIN_Z_LIMIT = 0.2 # 20cm
            MAX_Z_LIMIT = 0.265
            # MAX_Y_LIMIT = 0.33    # original
            MAX_Y_LIMIT = 0.24 
            if pose_stamped.pose.position.z < MIN_Z_LIMIT:
                self.get_logger().warn(f"  >> Target Z ({pose_stamped.pose.position.z:.3f}) is too low! Clamping to {MIN_Z_LIMIT}")
                pose_stamped.pose.position.z = MIN_Z_LIMIT
            if pose_stamped.pose.position.z > MAX_Z_LIMIT:
                self.get_logger().warn(f"  >> Target Z ({pose_stamped.pose.position.z:.3f}) is too high! Clamping to {MAX_Z_LIMIT}")
                pose_stamped.pose.position.z = MAX_Z_LIMIT

        
            pose_stamped.pose.position.x -= 0.03  # -1cm offset
            tx, ty = pose_stamped.pose.position.x, pose_stamped.pose.position.y
            self.get_logger().info(f"  [Wall Check] Target XY: ({tx:.3f}, {ty:.3f})")
            if pose_stamped.pose.position.y > MAX_Y_LIMIT:
                self.get_logger().warn(f" >> Target Y  ({pose_stamped.pose.position.y:.3f}) is too low! Clamping to {MAX_Y_LIMIT}")
                pose_stamped.pose.position.y = MAX_Y_LIMIT
            
            ## orientation 고정 ##
            pose_stamped.pose.orientation.x = 0.4635
            pose_stamped.pose.orientation.y = -0.5322
            pose_stamped.pose.orientation.z = -0.5512
            pose_stamped.pose.orientation.w = -0.4451

            ######################################################################## 
            # # [Adaptive Orientation] Calculate Yaw from Target Position
            # # Goal: Orient gripper radially to face the object (or align with arm extension)
            # # Roll=89, Pitch=80 (from original hardcoded), Yaw=Dynamic
            
            # target_yaw_rad = np.arctan2(pose.position.y, pose.position.x)
            # target_yaw_deg = np.degrees(target_yaw_rad)
            
            # # Using scipy Rotation to generate quaternion
            # # Original: [88.7, 79.9, -179.1] (For simple rear target)
            # # New: [89, 80, target_yaw_deg]
            # from scipy.spatial.transform import Rotation as R_scipy
            
            # # Note: You might need to adjust the yaw offset (e.g. +180) depending on gripper definition.
            # # Assuming '0' yaw points +X (Front). '180' points -X (Back).
            # # If target is at +Y (90), we want yaw=90.
            
            # r_adaptive = R_scipy.from_euler('xyz', [88.7, 80.0, target_yaw_deg], degrees=True)
            # qx, qy, qz, qw = r_adaptive.as_quat()
            
            # pose_stamped.pose.orientation.x = qx
            # pose_stamped.pose.orientation.y = qy
            # pose_stamped.pose.orientation.z = qz
            # pose_stamped.pose.orientation.w = qw
            
            # self.get_logger().info(f"  [Adaptive Yaw] Target (x={pose.position.x:.2f}, y={pose.position.y:.2f}) -> Yaw={target_yaw_deg:.1f} deg")
            ################################################################################################


            pose_stamped.header.frame_id = 'base_link'
            pose_stamped.header.stamp = Clock().now().to_msg()

            # [VISUALIZATION] Publish Marker (Blue)
            self.publish_target_marker(pose_stamped.pose, id=1, r=0.0, g=0.0, b=1.0)

            while not self.goal(self.ff, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")

            return pose_stamped
        else:

            return None

    def target_get_ba(self, mode="basket"): ### basket, wm, mid
        goal_msg = TargetPose.Goal()
        goal_msg.start = True  # 요청 트리거
        goal_msg.mode = mode

        # Goal 보내기
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

            # [Two-Stage Check] Sentinel Passthrough
            if result.target_pose.position.z == -99.0:
                self.get_logger().info("  >> [Sentinel] Empty Detected (-99.0). Returning immediately.")
                pose_stamped = PoseStamped()
                pose_stamped.pose = result.target_pose
                return pose_stamped


            ee2cam = np.vstack((np.hstack((self.R_cal, self.t_cal)), [0, 0, 0, 1]))

            base2cam = self.ee_pose @ ee2cam ## O2C
            
            # 1. Transform to EE and Base frames
            p_cam = np.array([result.target_pose.position.x, result.target_pose.position.y, result.target_pose.position.z, 1.0])
            p_ee = ee2cam @ p_cam
            pose = self.transform_pose_with_offset(result.target_pose, base2cam) ## O2t

            # [DEBUG] Green Logs for coordinate verification
            self.get_logger().info(f"\033[92m[Debug] Camera frame (EA): [{p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f}]\033[0m")
            self.get_logger().info(f"\033[92m[Debug] EE frame (local): [{p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}]\033[0m")
            self.get_logger().info(f"\033[92m[Debug] Base frame (bA):  [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]\033[0m")

            # [VISUALIZATION] Publish Camera Center Marker (Blue, ID=100)
            cam_center_pose = Pose()
            cam_center_pose.position.x = base2cam[0,3]
            cam_center_pose.position.y = base2cam[1,3]
            cam_center_pose.position.z = base2cam[2,3]
            self.publish_target_marker(cam_center_pose, id=100, r=0.0, g=0.5, b=1.0) # Light Blue for Camera
            
            # [USER REQUEST] Clamp Z to Basket Floor + Offset safe margin
            # Basket Floor (Surface) ~ 0.12m
            # DEEP (Grasp Point) Offset from Tip ~ 0.03m (Retracted)
            # Physical Tip Z = DEEP Z - 0.03
            # We want Physical Tip >= 0.12 + 0.005 (margin)
            # [USER REQUEST] Clamp Z to Basket Floor + Offset safe margin
            # Basket Floor (Surface) ~ 0.12m (World Frame)
            # Physical Tip Z >= 0.12 + 0.005 (margin) = 0.125 (World)
            # DEEP (Grasp Point) Z = Tip Z + 0.03 = 0.155 (World)
            # Converting to Base Frame: Base is at World Z + 0.445
            # So DEEP (Base) = DEEP (World) - 0.445
            # MIN_SAFE_Z_BASE = 0.155 - 0.445 = -0.290
            
            MIN_SAFE_Z_BASE = -0.310
            if pose.position.z < MIN_SAFE_Z_BASE:
                 self.get_logger().warn(f"Target Z ({pose.position.z:.3f}) too low! Clamping to {MIN_SAFE_Z_BASE} (Base Frame)")
                 pose.position.z = MIN_SAFE_Z_BASE
            
            # [USER REQUEST] Wall Distance Check & Conditional Orientation
            # Basket Inner Bounds (Approx from planning_scene.cpp)
            # Wall Thickness ~1.75cm. Center roughly (-0.06, -0.38)
            # Dimensions: 40x28cm Outer -> Inner ~36.5x24.5cm
            X_MIN_WALL = -0.06 - (0.40/2 - 0.0175) # -0.2425
            X_MAX_WALL = -0.06 + (0.40/2 - 0.0175) # +0.1225
            Y_MIN_WALL = -0.38 - (0.28/2 - 0.0175) # -0.5025
            Y_MAX_WALL = -0.38 + (0.28/2 - 0.0175) # -0.2575
            
            tx, ty = pose.position.x, pose.position.y
            dist_x = min(abs(tx - X_MIN_WALL), abs(tx - X_MAX_WALL))
            dist_y = min(abs(ty - Y_MIN_WALL), abs(ty - Y_MAX_WALL))
            min_dist = min(dist_x, dist_y)
            
            SAFE_DIST = 0.04 # 4cm
            
            # 벽 거리와 무관하게 항상 현재 EE orientation 사용
            # (벽 근처에서 identity orientation을 쓰면 IK 해가 없는 경우가 발생)
            from scipy.spatial.transform import Rotation as ScipyRotation
            current_rot_mat = self.ee_pose[:3, :3]
            r = ScipyRotation.from_matrix(current_rot_mat)
            qx, qy, qz, qw = r.as_quat()
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            if min_dist >= SAFE_DIST:
                self.get_logger().info(f"Target Safe (dist={min_dist*100:.1f}cm > 4cm). EE Orientation.")
            else:
                self.get_logger().info(f"Target Near Wall (dist={min_dist*100:.1f}cm < 4cm). EE Orientation (was identity).")

            self.get_logger().info(f"\033[94m변환한 bA: {pose}\033[0m") # Blue

            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.frame_id = 'base_link'
            pose_stamped.header.stamp = Clock().now().to_msg()

            self.get_logger().info(f"target_get_ba: (pose_stampled) {pose_stamped}")

            return pose_stamped
        else:
            self.get_logger().warn("결과 없음 -> 문 닫기 실행")
            self.close_door()
            while not self.goal(self.home, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
            rclpy.shutdown()
            sys.exit(0)
            return None

    def target_get_bg(self, mode = "failure_bg"):
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
            self.get_logger().info(f"success")
            return True
        else:
            self.get_logger().warn("None")
            return None
            return None
    
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

    def overcome2(self):
        while True:
            self.open_gripper()
            self.get_logger().info("Step 1: View Pose로 이동 (Empty Check)")
            while not self.goal(self.wm_view, 'keep'):
                self.get_logger().warn("  >> wm_view 이동 실패 → 재시도 중")
            
            chk_pose = self.target_get_wm(mode="failure_eh")

            if chk_pose is None:
                self.get_logger().warn("⚠️ [Overcome] 감지 결과 없음 (인식 실패). 루프 재시도.")
                continue

            if chk_pose.pose.position.z == -99.0:
                self.get_logger().info("🛑 [Overcome] 세탁물 없음 (RGB Gate). 작업 종료.")
                break

            self.get_logger().info("✅ [Overcome] 의류 감지됨. 파지 자세로 진입.")
            # [Stage 2] Move to Grasp Pose & Detect Target (TinyUNet)
            self.get_logger().info("Step 2: 파지 자세(wm_mid)로 이동 및 좌표 계산")
            
            while not self.goal(self.wm_mid, 'keep'):
                self.get_logger().warn("  >> wm_mid 이동 실패 → 재시도 중")
            import time
            time.sleep(1.0) # [CRITICAL] Let the controller settle before Vision blocks the thread

            # wm_mid 도달 후 EE orientation 캡처 (joint6 align 기준)
            rclpy.spin_once(self, timeout_sec=0.15)
            if self.ee_pose is not None:
                from scipy.spatial.transform import Rotation as R_scipy
                self.wm_init_ee_orientation = R_scipy.from_matrix(self.ee_pose[:3, :3])
                self.get_logger().info("[overcome2] wm_mid EE orientation 캡처 완료")
            
            # --- [USER SUGGESTED OPTIMIZATION] (주석 처리됨) ---
            # # wm_mid에서의 탐지("클러스터 너무 작음" 오류 빈발)를 생략하고
            # # 위쪽 wm_view에서 구한 목표 좌표(chk_pose)를 복사해서 바로 활용합니다.
            # self.get_logger().info("[OPTIMIZATION] Skipping failure_rgb_diff at wm_mid, using coordinate from wm_view.")
            # import copy
            # target_wm = copy.deepcopy(chk_pose)
            # target_wm.pose.position.y = 0.24
            
            # --- 기존 코드 복구 ---
            target_wm = self.target_get_wm(mode="failure_rgb_diff")  # 감지 및 잡기 시도 (RGB 배경오차)
            self.reset_sensor_data()

            if target_wm == None:
                self.get_logger().info("결과 없음")
                break

            # check if it's a fixed point via flag on frame_id
            self.get_logger().info(f"[DEBUG] target_wm frame_id is: '{target_wm.header.frame_id}'")
            is_fixed = (target_wm.header.frame_id == "base_link|fixed")
            target_wm.header.frame_id = "base_link"  # clean up for moveit

            # [VISUALIZATION] Overcome Target Marker (Fixed=Red, Default=Blue)
            t_r, t_g, t_b = (1.0, 0.0, 0.0) if is_fixed else (0.0, 0.0, 1.0)
            target_name = "Fixed WM Target" if is_fixed else "Target WM"
            self.publish_target_marker(target_wm.pose, id=2, r=t_r, g=t_g, b=t_b)
            self.get_logger().info(f"  [OVERCOME] {target_name}: \n{target_wm.pose}")
            
            # wm_view에서 가져온 좌표를 파란색 구 마커로 표시 (is_fixed 로직 건너뜀) (주석 처리)
            # is_fixed = False
            # self.publish_target_marker(target_wm.pose, id=2, r=0.0, g=0.0, b=1.0)
            # self.get_logger().info(f"  [OVERCOME] Target WM (from wm_view): \n{target_wm.pose}")

            # [Added] Pre-Grasp Approach (EE -5cm Retreat) to avoid collision checks (M0)
            import copy
            from scipy.spatial.transform import Rotation as R_scipy
            
            pre_grasp_pose = copy.deepcopy(target_wm)
            
            # Calculate Retreat Vector (-0.05 in Local Frame: -Z)
            # R * [0, 0, -0.05]
            q = [target_wm.pose.orientation.x, target_wm.pose.orientation.y, target_wm.pose.orientation.z, target_wm.pose.orientation.w]
            rot = R_scipy.from_quat(q)
            # retreat_vec = rot.apply([0, 0, -0.05]) # 5cm back
            retreat_vec = rot.apply([0, 0, -0.03]) # 3cm back

            pre_grasp_pose.pose.position.x += retreat_vec[0]
            pre_grasp_pose.pose.position.y += retreat_vec[1]
            pre_grasp_pose.pose.position.z += retreat_vec[2]

            # 1. Approach Pre-Grasp
            p_r, p_g, p_b = (1.0, 0.5, 0.0) if is_fixed else (0.0, 1.0, 1.0)
            self.get_logger().info(f" Pre-Grasp Approach ... ")
            self.publish_target_marker(pre_grasp_pose.pose, id=3, r=p_r, g=p_g, b=p_b)
            self.get_logger().info("  [OVERCOME] Valid Target found. Moving to Pre-Grasp (EE -3cm)...")
            res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=1, rpy_tol=[6.28, 6.28, 6.28])
            approach_success = (res == "success")
            if approach_success:
                self.get_logger().info(f"  [OVERCOME] Approach Success!")

                # joint6 align (수평 정렬)
                self.get_logger().info("[overcome2] joint6 align 중...")
                for _aj6_try in range(3):
                    if self.align_joint6():
                        break
                    self.get_logger().warn(f"  >> joint6 정렬 실패 → 재시도 ({_aj6_try+1}/3)")

                # align 후: 위치는 원래 target_wm 유지, orientation만 현재 EE 기준 갱신
                # [수정] 이전 코드는 EE+3cm_in_local_z로 위치를 재계산했으나,
                # joint6 회전으로 EE가 드럼 밖(y≈0.09)으로 밀려나고 local_z가 드럼 반대를 향해
                # 그리퍼가 y=0.09→0.09+작은값 (뒤쪽 허공)으로 가서 닫히는 문제가 있었음.
                # 원본 target_wm 위치(y≈0.23, 드럼 내부)를 유지하고 Mode 1으로 재진입
                rclpy.spin_once(self, timeout_sec=0.15)
                if self.ee_pose is not None:
                    from scipy.spatial.transform import Rotation as R_scipy
                    curr_rot = self.ee_pose[:3, :3]
                    q_cur = R_scipy.from_matrix(curr_rot).as_quat()
                    # 위치 변경 없이 orientation만 업데이트
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
                self.get_logger().warn("  >> [OVERCOME] Pre-Grasp failed (All Angles). Attempting Joint-based Fallback...")
                fallback_joints = [1.274389, 0.598975, -0.289762, -0.500294, 0.076265, 0.44763, 0.03, 0.03]
                res_joint = self.goal(fallback_joints, gripper_action='open')
                
                if res_joint:
                    self.get_logger().info("  >> Success moving to Fallback Joints. Calculating 1cm Blind Push...")
                    time.sleep(0.5) # Wait for TF update

                    # TF로 DEEP 프레임의 실제 위치를 가져옴
                    # /end_pose(하드웨어 FK)는 link6 위치이므로 사용 불가 - DEEP과 약 13cm 차이
                    curr_pose_mat = self.get_deep_pose()
                    if curr_pose_mat is None:
                        self.get_logger().error("  >> DEEP TF lookup 실패. 이 target 건너뜀.")
                        continue
                    curr_rot = curr_pose_mat[:3, :3]
                    curr_pos = curr_pose_mat[:3, 3]
                    self.get_logger().info(f"  >> DEEP pos(TF): [{curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}]")

                    # 1cm forward along DEEP Z-axis
                    push_vec = curr_rot @ np.array([0, 0, 0.01])

                    # Update target_wm to be this new relative target
                    target_wm.pose.position.x = float(curr_pos[0] + push_vec[0])
                    target_wm.pose.position.y = float(curr_pos[1] + push_vec[1])
                    target_wm.pose.position.z = float(curr_pos[2] + push_vec[2])

                    # Keep orientation from the DEEP frame
                    from scipy.spatial.transform import Rotation as ScipyRot
                    r = ScipyRot.from_matrix(curr_rot)
                    qx, qy, qz, qw = r.as_quat()
                    target_wm.pose.orientation.x = qx
                    target_wm.pose.orientation.y = qy
                    target_wm.pose.orientation.z = qz
                    target_wm.pose.orientation.w = qw

                    approach_success = True
                else:
                    self.get_logger().error("  >> Fallback Joint move failed. Skipping this target.")
                    continue
            # 2. Move to Grasp Target (Push)
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
                # pre_grasp에서 target_wm까지 3cm 직선 push.
                # Mode 9(Cartesian + collision ON)는 드럼 내부 collision으로 실패.
                # Mode 11(computeCartesianPath + avoid_collisions=false): orientation 유지 + 드럼 내부 push
                forward_state = self.move_forward(target_wm, fallback_to_ik=True, disable_torque=True, ik_mode=11, rpy_tol=[6.28, 6.28, 6.28])

                if forward_state == "success":
                    break  # 정상 성공

                if forward_state == "stop":
                    self._handle_stop_and_recover()
                    self.recover_robot() # [CRITICAL] Unlock safety latch immediately
                    break  # 접촉으로 판단 → retry 금지, 바로 다음 단계로

                self.get_logger().warn(f"forward 실패 ({i+1}/2) → 재시도 중")

            else:
                self.get_logger().warn("forward 2회 실패 → 새 target")

                # [진단] Mode 6 (IK only) 로 IK 존재 여부 확인
                # fallback_to_ik=True + ik_mode=6 → Mode 9 실패 후 Mode 6 IK 실제 시도
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
                    # [추가 fallback] Mode 0 (OMPL) — joint space 탐색으로 singularity 우회
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
                        continue     # 다음 target으로

            # 3. Close gripper & Retreat (pre_grasp 위치로 복귀, Mode 11)
            self.close_gripper()
            self.get_logger().info("  [OVERCOME] Retreating to pre-grasp...")
            # Mode 11: Cartesian 직선 (avoid_collisions=false) → push와 동일, 항상 부드러운 직선 경로
            ret = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=11, rpy_tol=[6.28, 6.28, 6.28])
            if ret != "success":
                self.get_logger().warn("  [OVERCOME] pre_grasp 복귀 실패 → Mode 6 재시도")
                ret = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=6, rpy_tol=[6.28, 6.28, 6.28])
            if ret != "success":
                self.get_logger().warn("  [OVERCOME] pre_grasp 복귀 실패 → Mode 0 (OMPL) 재시도")
                self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=0, rpy_tol=[6.28, 6.28, 6.28])

            # 4. 촉각 기반 파지 성공 판단
            rclpy.spin_once(self, timeout_sec=0.1)  # 센서 콜백 최신화
            is_grasp     = self.max_norm > 0.35
            is_valid_norm = self.norm > 0.35
            is_not_edge  = self.theta < 85.0
            grasp_success = is_grasp and is_valid_norm and is_not_edge
            if grasp_success:
                self.get_logger().info(
                    f"  [OVERCOME] 파지 성공 (norm={self.norm:.3f}, max_norm={self.max_norm:.3f}, theta={self.theta:.1f}°)"
                )
            else:
                self.get_logger().warn(
                    f"  [OVERCOME] 파지 실패 (norm={self.norm:.3f}, max_norm={self.max_norm:.3f}, theta={self.theta:.1f}°) → 그리퍼 오픈 후 재탐색"
                )
                self.open_gripper()
                continue  # while True 처음으로 → 새 target 탐색
            import copy
            self.get_logger().info("  >> [Overcome] Force Retreat 5cm (Blind)")
            
            # 1. Get Current Pose
            current_pose = self.ee_pose # 4x4 Matrix
            
            # 2. Calculate Back Vector (-Z local)
            # Z axis of EE is usually "Forward". Back is -Z.
            # R * [0, 0, -0.05]
            back_vec = current_pose[:3, :3] @ np.array([0, 0, -0.05])
            
            # 3. Create Target Matrix
            target_back_mat = current_pose.copy()
            target_back_mat[:3, 3] += back_vec
            
            # 4. Convert to PoseStamped
            from scipy.spatial.transform import Rotation as ScipyRotation
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
            
            # [Joint Space Retreat]
            # Cartesian Retreat (-Z) fails at low altitude due to IK limits.
            # Instead, we pull back by lifting the Elbow (Joint3) and Shoulder (Joint2) slightly.
            
            # [USER REQUEST] Skip Retreat, Just Move to wm_mid
            self.get_logger().info("  >> [Overcome] Skip Retreat -> Moving directly to ff")
            
            # Direct move attempt
            direct_success = self.goal(self.ff, 'keep')
            
            if not direct_success:
                self.get_logger().warn("  >> ff 직행 실패! (Deep Singularity?). Rescue Mode 진입")
                
                # [CRITICAL] Try to reset robot state if locked
                self.recover_robot()
                
                # Rescue Mode: Incrementally lift shoulder/elbow to escape
                # current_joint_state를 매 step 새로 읽으면 실제로 이동하지 않았을 때
                # 같은 목표를 5번 반복하게 됨. 대신 초기 pose에서 누적으로 목표를 증가시킴.
                import time
                for _ in range(10):
                    if self.current_joint_state is not None: break
                    time.sleep(0.05)

                if self.current_joint_state:
                    # 초기 joint 값을 한 번만 읽고, 누적 목표로 관리
                    rescue_base = list(self.current_joint_state)
                    for rescue_step in range(5):
                        self.get_logger().info(f"    >> [Rescue] Step {rescue_step+1}/5: Lifting arm slightly...")
                        target_rescue = list(rescue_base)
                        target_rescue[1] += 0.12 * (rescue_step + 1)  # J2 Shoulder lift 누적
                        target_rescue[2] -= 0.12 * (rescue_step + 1)  # J3 Elbow fold 누적
                        target_rescue[6] = rescue_base[6]
                        target_rescue[7] = rescue_base[7]

                        before = list(self.current_joint_state) if self.current_joint_state else rescue_base
                        if self.goal(target_rescue, 'keep'):
                            after = list(self.current_joint_state) if self.current_joint_state else before
                            actually_moved = abs(after[1] - before[1]) > 0.03 or abs(after[2] - before[2]) > 0.03
                            if actually_moved:
                                self.get_logger().info("    >> [Rescue] Small move success.")
                            else:
                                self.get_logger().warn("    >> [Rescue] goal() returned True but joint didn't move. Continuing...")
                        else:
                            self.get_logger().warn("    >> [Rescue] Small move failed.")

                        time.sleep(0.5)

                # Rescue 후 pre_grasp로 복귀 시도 (직접 JTC 대신 MoveIt Mode 6)
                self.get_logger().info("  >> [Rescue] Retrying pre_grasp via Mode 6 after rescue...")
                ret2 = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=6, rpy_tol=[6.28, 6.28, 6.28])
                if ret2 == "success":
                    self.get_logger().info("  >> [Rescue] pre_grasp 복귀 성공. ff로 이동...")
                    if not self.goal(self.ff, 'keep'):
                        self.get_logger().warn("  >> [Rescue] ff 이동 실패. 루프 계속.")
                else:
                    self.get_logger().error("  >> [Rescue] pre_grasp 복귀 실패. Critical Stuck.")

            retreat_done = True # Mark as done so loop continues correctly if needed

            #     self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")
            # else:  
            #     self.get_logger().warn("  >> back 5회 연속 실패 → 새 target 요청")
            #     continue  # while True 처음으로


            while not self.goal(self.ff, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
            self.get_logger().info("Z-축 정렬 감지는 생략합니다!!")
            #self.get_logger().info("Z-축 정렬 감지")
            # is_grasp = self.max_norm > 0.45
            # is_valid_norm = self.norm > 0.35
                
            # # Use Relative Logic INSTEAD of self.z_aligned (which uses absolute thresholds)
            # if is_valid_norm and is_grasp:
            if True:
                #self.get_logger().warn(f"실패 복구 잡기 성공 (Stable Grasp: norm={self.norm:.3f}, max_norm = {self.max_norm:.3f})")
                self.z_aligned = True
                
                # [USER REQUEST] Loop until Empty: Move to Drop -> Return to Scan
                self.get_logger().info("  >> [Loop] 세탁기(wm_end)로 이동하여 놓기")
                
                # 1. Move to Place (wm_end)
                while not self.goal(self.wm_end, 'keep'):
                    self.get_logger().warn("  >> wm_end 이동 실패 → 재시도 중")
                
                # 2. Release
                self.open_gripper()
                
                # 3. Return to Scan Pose (wm_mid)
                # self.get_logger().info("  >> [Loop] 다시 감지 위치(wm_mid)로 복귀")
                # while not self.goal(self.wm_mid, 'keep'):
                #     self.get_logger().warn("  >> wm_mid 이동 실패 → 재시도 중")
                
                # 4. Continue Loop (Do NOT Break)
                self.contacted = False # Reset flag if needed
                import time
                time.sleep(1.0) # [CRITICAL] Settle at wm_mid before next vision call
                continue 

            else:
                self.get_logger().warn(f" 실패 복구 잡기 실패 (Slip Detected: norm={self.norm:.3f,}, max_norm = {self.max_norm:.3f})")
                self.get_logger().warn(f"   (Criteria: norm>0.35, max norm > 0.5)")
                self.open_gripper()
                
                # Return to mid for retry?
                while not self.goal(self.wm_mid, 'keep'):
                     self.get_logger().warn("  >> wm_mid 복귀 실패 → 재시도")
                import time
                time.sleep(1.0) # [CRITICAL] Settle at wm_mid before next vision call
                continue


            # self.z_aligned = True ####
            # if self.z_aligned:
            #     self.get_logger().warn(" 잡기 성공")
            #     break
            # else:
            #     self.get_logger().warn(" 잡기 실패 →  재시도")
            #     while not self.open_gripper():
            #         self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중") 
            #     while not self.goal(self.wm_mid, 'keep'):
            #         self.get_logger().warn("  >> 실패 → 재시도 중") 

        # while not self.goal(self.ff, 'keep'):
        #     self.get_logger().warn("  >> 실패 → 재시도 중")


    def overcome(self):

        while True:

            # [Stage 1] Move to View Pose & Check Empty (RGB Gate)
            self.get_logger().info("Step 1: View Pose로 이동 (Empty Check)")
            while not self.goal(self.wm_view, 'keep'):
                self.get_logger().warn("  >> wm_view 이동 실패 → 재시도 중")
            
            # Check Emptiness
            chk_pose = self.target_get_wm(mode="failure_eh")
            
            # Sentinel Check from target_pose_server (-99.0 means RGB Gate said Empty)
            if chk_pose is not None and chk_pose.pose.position.z == -99.0:
                 self.get_logger().info("🛑 [Overcome] 빈 바구니 감지 (RGB Gate). 작업 종료.")
                 break
            
            self.get_logger().info("✅ [Overcome] 의류 감지됨. 파지 자세로 진입.")
            # [Stage 2] Move to Grasp Pose & Detect Target (TinyUNet)
            self.get_logger().info("Step 2: 파지 자세(wm_mid)로 이동 및 좌표 계산")

            while not self.goal(self.wm_mid, 'keep'):
                # self.get_logger().warn("  >> 실패 → 재시도 중")
                self.get_logger().warn("  >> wm_mid 이동 실패 → 재시도 중")

            # self.get_logger().info("Step 2: 잡기 시도 중...")
            self.get_logger().info("잡기 시도 중...")
            target_wm = self.target_get_wm(mode="failure_eh")  # 감지 및 잡기 시도
            self.reset_sensor_data()

            if target_wm == None:
                self.get_logger().info("결과 없음")
                break
            
            # [VISUALIZATION] Overcome Target Marker (Blue)
            self.publish_target_marker(target_wm.pose, id=2, r=0.0, g=0.0, b=1.0)
            self.get_logger().info(f"  [OVERCOME] Target WM: \n{target_wm.pose}")

            # [Added] Pre-Grasp Approach (EE -5cm Retreat) to avoid collision checks (M0)
            import copy
            from scipy.spatial.transform import Rotation as R_scipy
            
            pre_grasp_pose = copy.deepcopy(target_wm)
            
            # Calculate Retreat Vector (-0.05 in Local Frame: -Z)
            # R * [0, 0, -0.05]
            q = [target_wm.pose.orientation.x, target_wm.pose.orientation.y, target_wm.pose.orientation.z, target_wm.pose.orientation.w]
            rot = R_scipy.from_quat(q)
            retreat_vec = rot.apply([0, 0, -0.05]) # 5cm back
            
            pre_grasp_pose.pose.position.x += retreat_vec[0]
            pre_grasp_pose.pose.position.y += retreat_vec[1]
            pre_grasp_pose.pose.position.z += retreat_vec[2]
            
            # 1. Approach Pre-Grasp
            # 1. Approach Pre-Grasp (With Yaw Search)
            # [VISUALIZATION] Pre-Grasp Marker (Cyan)
            self.publish_target_marker(pre_grasp_pose.pose, id=3, r=0.0, g=1.0, b=1.0)
            
            self.get_logger().info("  [OVERCOME] Valid Target found. Moving to Pre-Grasp (EE -5cm)...")
            
            approach_success = False
            yaw_candidates = [0, 90, -90, 180, 45, -45, 135, -135] # 0 is default
            
            orig_q_pg = [pre_grasp_pose.pose.orientation.x, pre_grasp_pose.pose.orientation.y, 
                         pre_grasp_pose.pose.orientation.z, pre_grasp_pose.pose.orientation.w]
            
            for yaw in yaw_candidates:
                if yaw != 0:
                    self.get_logger().info(f"  [OVERCOME] Retrying with Yaw Rotation: {yaw} deg")
                    # Apply Yaw Rotation (Local Z)
                    rot_z = R_scipy.from_euler('z', yaw, degrees=True)
                    orig_rot = R_scipy.from_quat(orig_q_pg)
                    new_rot = orig_rot * rot_z
                    nx, ny, nz, nw = new_rot.as_quat()
                    pre_grasp_pose.pose.orientation.x = nx
                    pre_grasp_pose.pose.orientation.y = ny
                    pre_grasp_pose.pose.orientation.z = nz
                    pre_grasp_pose.pose.orientation.w = nw
                
                # Single attempt per angle
                res = self.move_forward(pre_grasp_pose, fallback_to_ik=True) 
                if res == "success":
                    approach_success = True
                    self.get_logger().info(f"  [OVERCOME] Approach Success with Yaw {yaw} deg!")
                     # Update final target orientation to match successful pre-grasp orientation
                    target_wm.pose.orientation = pre_grasp_pose.pose.orientation
                    break
                else:
                    self.get_logger().warn(f"  >> Pre-Grasp failed (Yaw {yaw}) result={res}")

            if not approach_success:
                 self.get_logger().warn("  >> Pre-Grasp failed (All Angles). Skipping this target.")
                 continue

            # 2. Move to Grasp Target (Push)
            self.get_logger().info("  [OVERCOME] Moving to Grasp Target...")
            forward_state = None
            for i in range(5):
                forward_state = self.move_forward(target_wm) # 6

                if forward_state == "success":
                    break  # 정상 성공

                if forward_state == "stop":
                    self._handle_stop_and_recover()
                    self.recover_robot() # [CRITICAL] Unlock safety latch immediately
                    break  # 접촉으로 판단 → retry 금지, 바로 다음 단계로

                self.get_logger().warn(f"forward 실패 ({i+1}/5) → 재시도 중")   

            else:
                self.get_logger().warn("forward 5회 실패 → 새 target")
                continue     # 다음 루프로

            self.close_gripper()

            # for i in range(5):
            #     if self.move_back(target_wm): # 7
            # [USER REQUEST] Force Back 5cm (Collision Ignored)
            import copy
            self.get_logger().info("  >> [Overcome] Force Retreat 5cm (Blind)")
            
            # 1. Get Current Pose
            current_pose = self.ee_pose # 4x4 Matrix
            
            # 2. Calculate Back Vector (-Z local)
            # Z axis of EE is usually "Forward". Back is -Z.
            # R * [0, 0, -0.05]
            back_vec = current_pose[:3, :3] @ np.array([0, 0, -0.05])
            
            # 3. Create Target Matrix
            target_back_mat = current_pose.copy()
            target_back_mat[:3, 3] += back_vec
            
            # 4. Convert to PoseStamped
            from scipy.spatial.transform import Rotation as ScipyRotation
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
            
            # [Joint Space Retreat]
            # Cartesian Retreat (-Z) fails at low altitude due to IK limits.
            # Instead, we pull back by lifting the Elbow (Joint3) and Shoulder (Joint2) slightly.
            
            # [USER REQUEST] Skip Retreat, Just Move to wm_mid
            self.get_logger().info("  >> [Overcome] Skip Retreat -> Moving directly to wm_mid")
            
            # Direct move attempt
            direct_success = self.goal(self.wm_mid_original, 'keep')
            
            if not direct_success:
                self.get_logger().warn("  >> wm_mid 직행 실패! (Deep Singularity?). Rescue Mode 진입")
                
                # [CRITICAL] Try to reset robot state if locked
                self.recover_robot()
                
                # Rescue Mode: Incrementally lift shoulder/elbow to escape
                for rescue_step in range(5):
                    self.get_logger().info(f"    >> [Rescue] Step {rescue_step+1}/5: Lifting arm slightly...")
                    
                    # Wait for fresh joint state
                    import time
                    for _ in range(10):
                        if self.current_joint_state is not None: break
                        time.sleep(0.05)
                        
                    if self.current_joint_state:
                         current = list(self.current_joint_state)
                         target_rescue = list(current)
                         # Lift Shoulder (J2) and Fold Elbow (J3) slightly
                        #  target_rescue[1] += 0.05 # ~3 deg
                         target_rescue[2] -= 0.01 # ~3 deg
                         
                         # Keep gripper
                         target_rescue[6] = current[6]
                         target_rescue[7] = current[7]
                         
                         if self.goal(target_rescue, 'keep'):
                             self.get_logger().info("    >> [Rescue] Small move success.")
                         else:
                             self.get_logger().warn("    >> [Rescue] Small move failed.")
                    
                    # Try wm_mid again? or just continue lifting?
                    # Let's try wm_mid every 2 steps or just finish lifting first.
                    time.sleep(0.5)
                
                # Final Attempt to wm_mid after rescue
                self.get_logger().info("  >> [Rescue] Retrying wm_mid after rescue moves...")
                if self.goal(self.wm_mid_original, 'keep'):
                     self.get_logger().info("  >> [Rescue] Success! Recovered to wm_mid.")
                else:
                     self.get_logger().error("  >> [Rescue] Final wm_mid attempt failed. Critical Stuck.")

            retreat_done = True # Mark as done so loop continues correctly if needed

            #     self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")
            # else:  
            #     self.get_logger().warn("  >> back 5회 연속 실패 → 새 target 요청")
            #     continue  # while True 처음으로


            while not self.goal(self.ff, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")

            self.get_logger().info("Z-축 정렬 감지")
            is_grasp = self.max_norm > 0.45
            is_valid_norm = self.norm > 0.35
                
            # Use Relative Logic INSTEAD of self.z_aligned (which uses absolute thresholds)
            if is_valid_norm and is_grasp:
                self.get_logger().warn(f"실패 복구 잡기 성공 (Stable Grasp: norm={self.norm:.3f}, max_norm = {self.max_norm:.3f})")
                self.z_aligned = True
                
                # [USER REQUEST] Loop until Empty: Move to Drop -> Return to Scan
                self.get_logger().info("  >> [Loop] 세탁기(wm_end)로 이동하여 놓기")
                
                # 1. Move to Place (wm_end)
                while not self.goal(self.wm_end, 'keep'):
                    self.get_logger().warn("  >> wm_end 이동 실패 → 재시도 중")
                
                # 2. Release
                self.open_gripper()
                
                # 3. Return to Scan Pose (wm_mid)
                self.get_logger().info("  >> [Loop] 다시 감지 위치(wm_mid)로 복귀")
                while not self.goal(self.wm_mid, 'keep'):
                    self.get_logger().warn("  >> wm_mid 이동 실패 → 재시도 중")
                
                # 4. Continue Loop (Do NOT Break)
                self.contacted = False # Reset flag if needed
                continue 

            else:
                self.get_logger().warn(f" 실패 복구 잡기 실패 (Slip Detected: norm={self.norm:.3f,}, max_norm = {self.max_norm:.3f})")
                self.get_logger().warn(f"   (Criteria: norm>0.35, max norm > 0.5)")
                self.open_gripper()
                
                # Return to mid for retry?
                while not self.goal(self.wm_mid, 'keep'):
                     self.get_logger().warn("  >> wm_mid 복귀 실패 → 재시도")
                continue


            # self.z_aligned = True ####
            # if self.z_aligned:
            #     self.get_logger().warn(" 잡기 성공")
            #     break
            # else:
            #     self.get_logger().warn(" 잡기 실패 →  재시도")
            #     while not self.open_gripper():
            #         self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중") 
            #     while not self.goal(self.wm_mid, 'keep'):
            #         self.get_logger().warn("  >> 실패 → 재시도 중") 

        while not self.goal(self.ff, 'keep'):
            self.get_logger().warn("  >> 실패 → 재시도 중")


    def execute_sequence(self):
        # self.get_logger().info("Step 0: 문 여는 중")
        # self.open_door()
        
        self.get_logger().info("Step 0.5: 배경 이미지(RGB/Depth) 자동 초기 캡처 진행")
        while not self.goal(self.wm_mid, 'keep'):
            self.get_logger().warn("  >> wm_mid(배경 촬영용) 이동 실패 → 재시도 중")
        import time
        time.sleep(1.0) # 로봇 떨림 안정화 대기
        self.get_logger().info("배경 촬영 중...")
        self.target_get_wm(mode="failure_bg_eh")
        self.get_logger().info("✅ 배경 이미지 캡처 완료.")

        # self.overcome2()

        while rclpy.ok():

            while True:
                self.open_gripper()
                self.z_aligned = False
                self.get_logger().info("Step 1: 이니셜 포즈(ba_init)로 이동 중")
                while not self.goal(self.ba_init, 'keep'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                self.reset_sensor_data()
                import time
                self.get_logger().info("Step 2: 잡기 시도 중...")
                target_ba = self.target_get_ba(mode="basket")  # 감지
                if target_ba.pose.position.z == -99.0:
                    self.get_logger().info("🛑 [Overcome] 빈 바구니 감지 (RGB Gate). overcome2 호출.")
                    self.overcome2()
                    break
            
                # --- Three-Stage Approach: 1. Approach Hover 2. Pre-Grasp 3. Push ---
                import copy
                
                # 1. Hover Pose (Safe Altitude above Target)
                hover_pose = copy.deepcopy(target_ba)
                # Keep original Target X, Y, Orientation, but set safe Z (e.g., 0.15m in Base Frame)
                hover_pose.pose.position.z = 0.0 
                
                # 2. Pre-Grasp Pose (+5cm above Target)
                pre_grasp_pose = copy.deepcopy(target_ba)
                pre_grasp_pose.pose.position.z += 0.05  # 5cm higher than target

                # [VISUALIZATION] Publish Markers IMMEDIATELY (Before Move)
                self.get_logger().info("\n==================================================")
                self.get_logger().info(f"  [TARGET] BA (Grasp): \n{target_ba.pose}")
                # Convert to World Frame
                target_ba_world = copy.deepcopy(target_ba.pose)
                target_ba_world.position.z += 0.445
                self.get_logger().info(f"  [WORLD] BA (Grasp) (Base+0.445): \n{target_ba_world}")
                self.get_logger().info("==================================================")
                self.publish_target_marker(target_ba.pose, id=0, r=1.0, g=0.0, b=0.0) # Red
                
                self.get_logger().info("\n==================================================")
                self.get_logger().info(f"  [PRE-TARGET] Hover Pose: \n{hover_pose.pose}")
                # Convert to World Frame
                hover_world = copy.deepcopy(hover_pose.pose)
                hover_world.position.z += 0.445
                self.get_logger().info(f"  [WORLD] Hover (Base+0.445): \n{hover_world}")
                self.get_logger().info("==================================================")
                self.publish_target_marker(hover_pose.pose, id=4, r=1.0, g=1.0, b=0.0) # Yellow
                
                self.get_logger().info("\n==================================================")
                self.get_logger().info(f"  [PRE-TARGET] Pre-Grasp Pose (+5cm): \n{pre_grasp_pose.pose}")
                # Convert to World Frame
                pre_grasp_world = copy.deepcopy(pre_grasp_pose.pose)
                pre_grasp_world.position.z += 0.445
                self.get_logger().info(f"  [WORLD] Pre-Grasp (Base+0.445): \n{pre_grasp_world}")
                self.get_logger().info("==================================================")
                self.publish_target_marker(pre_grasp_pose.pose, id=1, r=0.0, g=1.0, b=0.0) # Green

                # 1. Approach Hover
                self.get_logger().info("Step 2-1: 안전 위치(Hover) 이동 중... (Collision Check OFF)")
                hover_success = False
                for i in range(3):
                    # [USER REQUEST] Use Mode 10 (Env-aware) for Hover
                    res = self.move_forward(hover_pose, fallback_to_ik=True, disable_torque=True, ik_mode=10)
                    if res == "success":
                        hover_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Hover 도달!\033[0m")
                        break
                    self.get_logger().warn(f"  >> Hover 실패 ({i+1}/3) result={res} → 재시도")
                
                if not hover_success:
                    self.get_logger().warn("  >> Hover 이동 실패 → 새 target 요청")
                    continue

                # 2. Approach Pre-Grasp
                self.get_logger().info("Step 2-2: 접근(Pre-Grasp) 하강 중... (Collision Check ON)")
                approach_success = False

                for i in range(3):
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=False, disable_torque=True)
                    if res == "success":
                        approach_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Linear)\033[0m")
                        break
                    self.get_logger().info(f"\033[94m  >> [Stage 1] 직선 하강 실패 ({i+1}/3) result={res} → 재시도\033[0m")

                # Stage 1 실패 시 Stage 2 시도: 엄격한 수직 IK (Mode 10, Env-aware)
                if not approach_success:
                    self.get_logger().info("\033[94m [Stage 2] 엄격한 수직 IK(Mode 10) 시도\033[0m")
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, ik_mode=10, disable_torque=True)
                    if res == "success":
                        approach_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Strict IK)\033[0m")

                # Stage 2 실패 시 Stage 3 시도: 수직 유지 완화 계획 (Mode 1, ±15도 허용)
                # Mode 1: OMPL Planner를 사용하여 목표 위치(XYZ)에 도달하되, 
                # 수직 방향에서 상하좌우 ±30도 이내의 기울기를 허용하여 충돌을 회피하는 경로를 찾습니다.
                # if not approach_success:
                #     self.get_logger().info("\033[94m [Stage 3] 수직 유지 완화(Mode 1) 계획 시도\033[0m")
                #     # roll_tol, pitch_tol = 0.52 rad (~30 deg), yaw_tol = 6.28 (자유)
                #     res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, ik_mode=1, rpy_tol=[0.52, 0.52, 6.28], disable_torque=True)
                #     if res == "success":
                #         approach_success = True
                #         self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Relaxed Orientation)\033[0m")

                # Stage 4: orientation 완전 자유 (path constraint 없음)
                # d435i가 벽을 피하는 joint6 회전을 OMPL이 자유롭게 탐색하도록 허용
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
                    
                    # [IMPORTANT] Pre-Grasp에 도달한 자세(기울기 포함)를 최종 Push 타겟에도 적용
                    # 이렇게 해야 Push 시 수직으로 억지로 돌아오다가 벽에 부딪히는 것을 방지함
                    target_ba.pose.orientation = self.ee_pose_st.pose.orientation

                # if not approach_success:
                #     # IK solver가 task-space Yaw를 손목(joint6) 회전으로 해석하지 않고
                #     # 어깨/팔꿈치를 바꿔버리는 문제가 있음.
                #     # → 관절 공간에서 joint6을 직접 돌리고, 그 상태에서 Cartesian 하강.
                #     self.get_logger().warn("  >> 직선 하강 실패 → joint6 직접 회전 후 재하강 시도")
                #     import time

                #     # joint6 회전 후보 (라디안)
                #     j6_candidates_deg = [-135,-180]

                #     if self.current_joint_state is None:
                #         self.get_logger().warn("  >> joint_state 없음 → 폴백 불가, 새 target 요청")
                #     else:
                #         base_j6 = self.current_joint_state[5]  # 현재 joint6 (0-indexed)

                #         for deg in j6_candidates_deg:
                #             rad = deg * 3.14159265 / 180.0
                #             target_j6 = base_j6 + rad

                #             # joint6 범위 클램프 (-180도 ~ +180도)
                #             target_j6 = max(-3.14159, min(3.14159, target_j6))

                #             self.get_logger().info(f"  >> [joint6 폴백] joint6 {base_j6:.2f} → {target_j6:.2f} rad ({deg}°)")

                #             # 1) joint6만 돌리기 (현재 arm 자세에서 joint6만 변경)
                #             wrist_cmd = list(self.current_joint_state[:6])
                #             wrist_cmd[5] = target_j6
                #             wrist_ok = self._send_goal(
                #                 self._arm_client,
                #                 self.arm_joint_names,
                #                 wrist_cmd,
                #                 wait_for_result=True
                #             )
                #             if not wrist_ok:
                #                 self.get_logger().warn(f"  >> joint6 {deg}° 회전 실패 → 다음 각도")
                #                 continue

                #             time.sleep(0.3)  # 안정화 대기

                #             # 2) hover 재진행 (EE target은 그대로, 손목만 미리 돌린 상태)
                #             hover_ok = False
                #             for _ in range(2):
                #                 r = self.move_forward(hover_pose, fallback_to_ik=True, disable_torque=True, ik_mode=10)
                #                 if r == "success":
                #                     hover_ok = True
                #                     break
                #             if not hover_ok:
                #                 self.get_logger().warn(f"  >> joint6 {deg}° 후 hover 실패 → 다음 각도")
                #                 continue

                #             # 3) pre-grasp 재진행 (Mode 9 → 10)
                #             r = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=10)
                #             if r == "success":
                #                 approach_success = True
                #                 self.get_logger().info(f"\n\033[92m[SUCCESS] Pre-Grasp 도달! (joint6 {deg}°)\033[0m")
                #                 break
                #             self.get_logger().warn(f"  >> joint6 {deg}° 후 pre-grasp 실패 (result={r})")

                if not approach_success:
                    self.get_logger().warn("  >> Pre-Grasp 전부 실패 → 새 target 요청")
                    continue


                # 3. Push (Actual Grasp Move)
                self.get_logger().info("Step 2-3: 잡기(Push) 이동 중...")
                forward_state = None
                for i in range(3):
                    forward_state = self.move_forward(target_ba, ik_mode=11) # Cartesian, avoid_collisions=false
                    if forward_state == "success":
                        self.get_logger().info("\n\033[92m[SUCCESS] Target BA 도달!\033[0m")
                        # Calc Tip Pose
                        tip_pos_ba = self.ee_pose[:3, 3] + (self.ee_pose[:3, :3] @ np.array([0, 0, 0.11]))
                        self.get_logger().info(f"  >> Current Wrist : {self.ee_pose[:3,3]}")
                        self.get_logger().info(f"\033[92m  >> Computed Tip  : {tip_pos_ba}\033[0m") 
                        break

                    if forward_state == "success":
                        break  # 정상 성공

                    if forward_state == "stop":
                        self._handle_stop_and_recover()
                        break  # 접촉으로 판단 → retry 금지, 바로 다음 단계로

                    self.get_logger().warn(f"forward 실패 ({i+1}/5) result={forward_state} → 재시도 중")   

                else:
                    self.get_logger().warn("forward 5회 실패 → 새 target")
                    continue     # 다음 루프로

                self.close_gripper()

                # [DATA COLLECTION] 녹화 시작 (collect / inference 모드)
                if self.run_mode in ('collect', 'inference'):
                    self.grasp_buffer = []
                    self.grasp_raw_buffer = []
                    self.realtime_buffer.clear()
                    self.inference_frame_cnt  = 0
                    self.infer_consec_normal  = 0
                    self.infer_latest_score   = None
                    self.is_recording = True
                    self.get_logger().info(f"[DATA] 녹화 시작 (mode={self.run_mode})")

                # --- Retreat Added ---
                for i in range(5):
                    if self.move_backward(target_ba): # Using new hybrid move_backward
                        break
                    self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")
                # 실패해도 일단 진행 (or loop?)

                while not self.goal(self.ba_init, 'keep'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")

                # [DATA COLLECTION] 녹화 종료
                if self.run_mode in ('collect', 'inference'):
                    self.is_recording = False
                    self.get_logger().info(f"[DATA] 녹화 종료: {len(self.grasp_buffer)} 프레임 수집")

                self.get_logger().info(" Z-축 정렬 (Slip Check) 감지")
                #self.get_logger().info(" Z-축 정렬 (Slip Check) 감지")
                # [USER REQUEST] Relative Slip Detection
                # Criteria:
                # 1. Norm should be reasonable (> 0.35)
                # 2. Theta Deviation should be small (< 15.0 deg)
                # 3. Absolute Theta shouldn't be extreme (> 85.0 deg)
                
                grab_norm = self.norm
                grab_theta = self.theta
                current_theta = self.theta
                theta_diff = abs(current_theta - grab_theta)
                
                is_grasp = self.max_norm > 0.45
                is_stable = theta_diff < 6.0
                is_valid_norm = self.norm > 0.35
                is_not_edge = current_theta < 85.0
                
                # Use Relative Logic INSTEAD of self.z_aligned (which uses absolute thresholds)
                rule_success = is_stable and is_valid_norm and is_not_edge and is_grasp

                if self.run_mode == 'collect':
                    # 데이터 수집 모드: 사람 입력으로 로봇 동작 결정
                    label = self._request_human_label()
                    if label == 'success':
                        self.z_aligned = True
                        break
                    elif label in ('failure', 'perturbation'):
                        while not self.open_gripper():
                            self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
                    else:
                        # x (skip): rule-based 결과로 fallback
                        self.get_logger().info("[DATA] skip → rule-based fallback")
                        if rule_success:
                            self.z_aligned = True
                            break
                        else:
                            while not self.open_gripper():
                                self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")

                elif self.run_mode == 'inference':
                    # 슬라이딩 윈도우 실시간 판단 결과 사용
                    score = self.infer_latest_score
                    score_str = f"{score:.4f}" if score is not None else "None"
                    self.get_logger().info(
                        f"[INFERENCE] 연속 정상 {self.infer_consec_normal}회 "
                        f"(needed={self.infer_consec_needed}), 최종 score={score_str}")
                    if self.infer_consec_normal >= self.infer_consec_needed:
                        self.get_logger().info("[INFERENCE] 성공 판정")
                        self.z_aligned = True
                        break
                    else:
                        self.get_logger().warn("[INFERENCE] 실패 판정 → 재시도")
                        while not self.open_gripper():
                            self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")

                else:
                    # normal 모드: 기존 rule-based 동작
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
            
            # Start Monitoring
            self.monitor_initial_norm = self.norm
            self.monitor_initial_theta = self.theta
            self.monitor_slip = True
            self.get_logger().info(f"👀 Slip Monitoring START (Ref: norm={self.norm:.2f}, theta={self.theta:.1f})")
            self.get_logger().info(f"👀 Slip Monitoring START (Ref: norm={self.norm:.2f}, theta={self.theta:.1f})")
            
            while not self.goal(self.wm_init, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
            
            while not self.goal(self.wm_end, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
            
            # Stop Monitoring
            self.monitor_slip = False
            self.get_logger().info("👀 Slip Monitoring END")

            self.get_logger().info("Step 4: 접촉 감지 결과 확인 중")
            # If monitoring triggered 'contacted=True', overcome will be called below
            current_theta = self.theta
            theta_diff = abs(current_theta - grab_theta)
                
            is_stable = theta_diff < 6.0
            is_valid_norm = self.norm > 0.35
            is_not_edge = current_theta < 85.0
            
            if self.contacted:
                self.get_logger().warn(f" Slip Recorded during Motion! (Overcome Trigger)")
            elif is_stable and is_valid_norm and is_not_edge:
                self.get_logger().warn(f" No slip: Δθ={theta_diff:.1f}°, θ={current_theta:.1f}°), norm={self.norm:.3f}")
            # else:
            #     self.get_logger().warn(f" Slip Detected: Δθ={theta_diff:.1f}°, θ={current_theta:.1f}°, norm={self.norm:.3f})")
            #     self.get_logger().warn(f"   (Criteria: Δ<6.0, θ<85.0, norm>0.35)")
            #     self.contacted = True

            # self.contacted = True
            self.open_gripper()

            if self.contacted:
                self.get_logger().info("  >> 접촉 감지됨 → 실패 극복")
                self.overcome2() 

            else:
                self.get_logger().info("  >> 접촉 없음")
            
            while not self.goal(self.wm_init, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
            continue



# ================================================================= #
# ===================== 핵심 변경 부분: main 함수 =================== #
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
        marker.scale.x = 0.04  # sphere diameter
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.lifetime = Duration(sec=0, nanosec=0) # Infinite lifetime
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published Marker ID {id} to {marker.header.frame_id}")

    def publish_camera_center_callback(self):
        """
        주기적으로 현재 ee_pose와 ee2cam 행렬을 사용하여 카메라 중심 위치를 Rviz에 표시
        """
        if self.ee_pose is None:
            return

        ee2cam = np.vstack((np.hstack((self.R_cal, self.t_cal)), [0, 0, 0, 1]))
        
        # 2. Base 기준 카메라 위치 계산
        base2cam = self.ee_pose @ ee2cam
        
        cam_center_pose = Pose()
        cam_center_pose.position.x = base2cam[0,3]
        cam_center_pose.position.y = base2cam[1,3]
        cam_center_pose.position.z = base2cam[2,3]
        
        # 3. 마커 발행 (ID=100, Light Blue)
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
        marker.lifetime = Duration(sec=1, nanosec=0)
        
        self.marker_pub.publish(marker)

def main(args=None):
    parser = argparse.ArgumentParser(description='ControlTower 실행 모드 설정')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'collect', 'inference'],
                        help='normal: rule-based | collect: 데이터 수집 | inference: GRU-AE 판단')
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
        node.get_logger().info('노드 종료 중...')
        # Executor와 노드를 깔끔하게 종료합니다.
        executor.shutdown()
        # 노드가 이미 파괴되었을 수 있으므로 확인 후 파괴
        if rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == '__main__':
    main()

