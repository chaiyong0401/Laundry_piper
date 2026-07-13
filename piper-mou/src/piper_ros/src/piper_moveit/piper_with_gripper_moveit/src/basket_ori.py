
import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.clock import Clock
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
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
import queue
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

    def __init__(self, run_mode: str = 'normal', use_tactile_api: bool = False):
        super().__init__('control_tower_node')
        # run_mode: 'normal'   → 기존 동작 (rule-based 판단, 데이터 수집 없음)
        #           'collect'  → 데이터 수집 모드 (human labeling)
        #           'inference'→ 학습된 GRU-AE로 판단
        self.run_mode = run_mode
        self.use_tactile_api = use_tactile_api
        self.get_logger().info(f"[MODE] 실행 모드: {self.run_mode}, use_tactile_api={self.use_tactile_api}")

        self._timing_log = []  # [(label, elapsed_sec), ...]

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
        self.data_save_dir = "/home/dyros/mcy_ws/grasp_new"
        os.makedirs(self.data_save_dir, exist_ok=True)
        self.door_bag_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/door_open_bag"
        self.door_bag_hz   = 20.0
        self.door_bag_speed = 1.0

        self.is_insert_recording = False
        self.insert_buffer = []         # Model 2: 6D feature 시계열
        self.insert_raw_buffer = []     # Model 2: raw taxel 48D 시계열
        self.insert_save_dir = "/home/dyros/mcy_ws/transport_new"
        os.makedirs(self.insert_save_dir, exist_ok=True)

        # [INFERENCE] Model 1 (grasp) GRU-AE + 채널 B
        self.gru_ae_model      = None
        self.gru_ae_scaler     = None
        self.gru_ae_threshold  = 0.02   # 채널 A threshold
        self.gru_ae_threshold_b = 0.5   # 채널 B threshold
        self.gru_ae_infer_cfg  = {"relative": "none", "weight_alpha": 0.0}
        self.grasp_channel_b   = None   # FramewiseDeltaDetector

        # [INFERENCE] Model 2 (transport) GRU-AE + 채널 B
        self.insert_gru_ae_model     = None
        self.insert_gru_ae_scaler    = None
        self.insert_gru_ae_threshold = 0.02   # 채널 A threshold
        self.insert_gru_ae_threshold_b = 0.5  # 채널 B threshold
        self.insert_gru_ae_infer_cfg = {"relative": "none", "weight_alpha": 0.0}
        self.transport_channel_b     = None   # FramewiseDeltaDetector
        self.insert_max_mrl_seen     = 0.0

        self.grasp_api_detector = None  # DualChannelDetector (use_tactile_api=True 시 사용)
        self.insert_api_detector = None # DualChannelDetector (use_tactile_api=True 시 사용)

        if self.run_mode == 'inference':
            self._load_gru_ae()
            self._load_insert_gru_ae()
            if self.use_tactile_api:
                self._load_grasp_api_detector()
                self._load_insert_api_detector()

        # [SLIDING WINDOW] Model 1 (grasp) 실시간 inference 상태
        self.realtime_buffer     = deque(maxlen=50)   # 6D feature 실시간 버퍼
        self.realtime_raw_buffer = deque(maxlen=50)   # 48D raw taxel 실시간 버퍼
        self.inference_stride    = 10
        self.inference_frame_cnt = 0
        self.infer_consec_normal         = 0
        self.infer_consec_needed         = 3
        self.infer_consec_anomaly        = 0   # 채널 A 연속 이상 카운터
        self.infer_consec_anomaly_needed = 2   # 이상 확정에 필요한 연속 횟수
        self.infer_latest_score          = None
        self.infer_grasp_anomaly_detected = False  # recording 중 연속 이상 달성 여부 (eval 기준)
        self.grasp_detect_channel        = 'none' # 이상 확정 시 감지 채널 ('A','B','AB')
        self.grasp_channel_b_flag        = False  # 채널 B 이상 확정 플래그
        self.grasp_channel_b_consec      = 0      # 채널 B 연속 이상 프레임 수
        self.grasp_channel_b_consec_needed = 5    # 채널 B 이상 확정 프레임 수
        self.grasp_delta_b_latest        = 0.0
        self.grasp_channel_b_warmup_needed = 30  # 로봇 이동 초기 모션 아티팩트 무시 프레임 수

        # [SLIDING WINDOW] Model 2 (transport) 실시간 inference 상태
        self.insert_realtime_buffer           = deque(maxlen=50)
        self.insert_realtime_raw_buffer       = deque(maxlen=50)
        self.insert_infer_frame_cnt           = 0
        self.insert_infer_consec_normal       = 0
        self.insert_infer_consec_needed       = 3
        self.insert_infer_consec_anomaly      = 0
        self.insert_infer_consec_anomaly_needed = 2
        self.insert_infer_latest_score        = None
        self.insert_anomaly_ever_detected     = False
        self.transport_detect_channel         = 'none' # 이송 이상 확정 시 감지 채널 ('A','B','AB')
        self._transport_abort                 = threading.Event()
        self.transport_channel_b_flag         = False  # 채널 B 이상 확정 플래그
        self.transport_channel_b_consec       = 0
        self.transport_channel_b_consec_needed = 5
        self.transport_channel_b_warmup_needed = 30  # 이송 초기 모션 아티팩트 무시 프레임 수
        self.transport_delta_b_latest         = 0.0

        # [SENSOR PROCESSING THREAD] sensor_callback은 raw 데이터만 큐에 넣고 즉시 리턴
        # 모든 numpy 연산과 상태 업데이트는 이 스레드에서 수행 → DDS 버퍼 블로킹 방지
        self._raw_sensor_queue = queue.Queue(maxsize=200)
        self._sensor_proc_thread = threading.Thread(
            target=self._sensor_processing_worker, daemon=True)
        self._sensor_proc_thread.start()

        # [INFERENCE THREAD] callback 블로킹 방지: inference는 별도 스레드에서 수행
        self._infer_lock = threading.Lock()
        self._infer_thread_running = True
        self._infer_trigger    = threading.Event()  # grasp inference 요청
        self._insert_trigger   = threading.Event()  # insert inference 요청
        self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._infer_thread.start()

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

        # ── Callback Group 분리 ──────────────────────────────────────────────────────
        # tick() 타이머(100Hz, pin.rnea 포함)와 센서/조인트 구독이 같은 기본
        # MutuallyExclusiveCallbackGroup을 공유하면, rnea 실행 중 sensor_callback이
        # 대기(blocked)됨 → 0.5s 이상 데이터 없으면 "WebSocket 끊김" 오탐 발생.
        # 별도 그룹을 할당하면 MultiThreadedExecutor가 이들을 병렬 처리.
        self._sensor_cb_group = MutuallyExclusiveCallbackGroup()

        self.subscription = self.create_subscription(
            SensStream,
            '/xServTopic',
            self.sensor_callback,
            10,
            callback_group=self._sensor_cb_group
        )
        self.create_subscription(Pose, '/end_pose', self.ee_pose_callback, 10,
                                 callback_group=self._sensor_cb_group)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10,
                                 callback_group=self._sensor_cb_group)
        
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
        # self.wm_view = [1.567378, 0.584147, -0.976707, -0.132923, 1.276936, -1.211015, 0.0, 0.0]
        self.wm_view = [1.562546, 0.641172, -1.202764, -0.089156, 1.286792, -1.478571, 0.0, 0.0]

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
    def _send_goal(self, client, joint_names, joint_values, wait_for_result=True, timeout_sec=30.0, fast=False):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in joint_values]

        # Dynamic Duration Calculation to prevent "Status 6" (Too Slow/Fast)
        # fast=True (물체 미파지): 1.5 rad/s, fast=False (물체 파지 중): 0.5 rad/s
        rad_per_sec = 1.5 if fast else 0.5
        duration_sec = 3.0 # Default fallback

        try:
            if hasattr(self, 'current_joint_state') and self.current_joint_state is not None:
                n = len(joint_values)
                if len(self.current_joint_state) >= n:
                    cur_slice = self.current_joint_state[:n]
                    diffs = [abs(t - c) for t, c in zip(joint_values, cur_slice)]
                    max_diff = max(diffs)

                    req_duration = (max_diff / rad_per_sec) + 0.2
                    min_dur = 0.3 if fast else 0.8
                    duration_sec = max(min_dur, req_duration)

                    if max_diff < 0.05:
                        duration_sec = 0.2 if fast else 0.4
                    elif max_diff > 4.0:
                        duration_sec = 2.0 if fast else 5.0

                    # self.get_logger().info(f"  [Move] Max Diff={max_diff:.3f} rad, Duration={duration_sec:.2f}s (fast={fast})")
        except Exception as e:
            self.get_logger().warn(f"Duration calc failed: {e}")
            duration_sec = 1.5 if fast else 3.0

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
        self.get_logger().info(f"[Goal 명령] Pose: {joint_values_8dof}, Gripper: {gripper_action}, fast={fast}")
        arm_target = joint_values_8dof[:6]

        success = False
        # [Added] Check if already at goal to prevent MoveIt Status 6 Abort
        if hasattr(self, 'current_joint_state') and self.current_joint_state is not None and len(self.current_joint_state) >= 6:
            diffs = [abs(a - c) for a, c in zip(arm_target, self.current_joint_state[:6])]
            if all(d < 0.05 for d in diffs): # ~2.8 degrees tolerance
                self.get_logger().info("  >> 이미 목표 관절 각도에 도달해 있습니다. (명령 스킵)")
                success = True
            else:
                success = self._send_goal(self._arm_client, self.arm_joint_names, arm_target, wait_for_result=True, fast=fast)
        else:
            success = self._send_goal(self._arm_client, self.arm_joint_names, arm_target, wait_for_result=True, fast=fast)

        # MoveIt planning 실패(status 6) 시 direct JTC로 fallback
        if not success:
            self.get_logger().warn("  >> MoveIt planning 실패 → direct JTC fallback 시도")
            self._timing_log.append(("  [GAP] MoveIt fallback 발생", 0.0))
            success = self._send_goal(self._arm_direct_client, self.arm_joint_names, arm_target, wait_for_result=True, fast=fast)

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

    def send_pose_wm(self, pose: PoseStamped):
        goal_msg = PoseGoal.Goal()
        goal_msg.target_pose = pose

        goal_msg.mode = 0 # pose

        send_goal_future = self._pose_client.send_goal_async(goal_msg)
        self._wait_future(send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("pose_goal이 거부되었습니다.")
            return False
        self.get_logger().info("pose_goal 명령 전송 완료.")

        result_future = goal_handle.get_result_async()
        self._wait_future(result_future)
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
        self._wait_future(send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("pose_goal이 거부되었습니다.")
            return False
        self.get_logger().info("pose_goal 명령 전송 완료.")

        result_future = goal_handle.get_result_async()
        self._wait_future(result_future)
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
            self._wait_future(future)
            
            gh = future.result()
            if not gh or not gh.accepted:
                continue
                
            res_future = gh.get_result_async()
            self._wait_future(res_future)
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
        self._wait_future(send_goal_future)
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
        self._wait_future(result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f"✅ pose 명령 성공: {result.message}")
            return True
        else:
            self.get_logger().warn(f"⚠️ pose 명령 실패: {result.message}")
            return False
        

    def move_forward(self, pose_st: PoseStamped, fallback_to_ik=True, disable_torque=False, ik_mode=6, rpy_tol=[0.0, 0.0, 0.0], skip_mode9=False, vel_scale=0.6):
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
            goal_msg.velocity_scaling = vel_scale
            goal_msg.acceleration_scaling = vel_scale
             
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
        self._wait_future(send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("move_back goal rejected")
            return False

        # 2) 결과 대기
        result_future = goal_handle.get_result_async()
        self._wait_future(result_future)
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

    def reset_sensor_data(self):
        self.prev_x, self.prev_y, self.prev_z = None, None, None
        self.cum_x, self.cum_y, self.cum_z = [0.0] * 16, [0.0] * 16, [0.0] * 16
        self.z_aligned, self.contacted = False, False
        self.get_logger().info("센서 기준값 및 상태 초기화 완료.")

    def _spin_input(self, prompt: str) -> str:
        """input()을 별도 스레드에서 실행하며 대기.
        MultiThreadedExecutor가 이미 다른 스레드에서 spin() 중이므로
        rclpy.spin_once를 중복 호출할 필요 없음 → 충돌 방지."""
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
        done.wait()  # executor가 별도 스레드에서 spin 중이므로 단순 대기
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
        self.get_logger().info(f"[DATA] 저장 완료: {fname} (shape={seq.shape}), raw: {raw.shape}")

    def _inference_worker(self):
        """별도 스레드에서 grasp/transport 이중 채널 inference를 처리.
        채널 A (GRU-AE window MSE): 이 스레드에서 실행
        채널 B (MRL delta): 센서 스레드에서 실행 후 플래그로 전달
        최종 판정: A OR B
        """
        while self._infer_thread_running:
            grasp_fired  = self._infer_trigger.wait(timeout=0.05)
            insert_fired = self._insert_trigger.is_set()

            if grasp_fired:
                self._infer_trigger.clear()
                score_a = self._run_windowed_inference()
                if score_a is not None:
                    with self._infer_lock:
                        flag_a = score_a > self.gru_ae_threshold
                        flag_b = self.grasp_channel_b_flag
                        self.grasp_channel_b_flag = False  # 읽은 후 초기화 (sticky 방지)
                        is_anomaly = flag_a or flag_b
                        channel = ('AB' if flag_a and flag_b else
                                   'A'  if flag_a else
                                   'B'  if flag_b else 'none')
                        self.infer_latest_score = score_a
                        if is_anomaly:
                            self.infer_consec_anomaly += 1
                            self.infer_consec_normal = 0
                            if self.infer_consec_anomaly >= self.infer_consec_anomaly_needed:
                                self.infer_grasp_anomaly_detected = True
                                self.grasp_detect_channel = channel
                        else:
                            self.infer_consec_normal += 1
                            self.infer_consec_anomaly = 0
                        self.get_logger().info(
                            f"[GRASP] score_A={score_a:.4f} "
                            f"thr_A={self.gru_ae_threshold:.4f} "
                            f"delta_B={self.grasp_delta_b_latest:.4f} "
                            f"thr_B={self.gru_ae_threshold_b:.4f} "
                            f"ch={channel} "
                            f"{'ANOMALY' if is_anomaly else 'normal'} "
                            f"consec_anom={self.infer_consec_anomaly} "
                            f"max_norm={self.max_norm:.3f} norm={self.norm:.3f}"
                        )

            if insert_fired:
                self._insert_trigger.clear()
                score_a = self._run_insert_windowed_inference()
                if score_a is not None:
                    cfg = getattr(self, "insert_gru_ae_infer_cfg", {})
                    mrl_thresh = float(cfg.get("mrl_thresh", 0.2))
                    cutoff_mult = float(cfg.get("cutoff_multiplier", 2.2))
                    
                    with self._infer_lock:
                        # Rule 1: Miss Guard Rail
                        flag_a_rule1 = (self.insert_max_mrl_seen < mrl_thresh)
                        # Rule 2: Super High Error Cutoff
                        flag_a_rule2 = (score_a > self.insert_gru_ae_threshold * cutoff_mult)
                        # Rule 3: Regular Threshold
                        flag_a_rule3 = (score_a > self.insert_gru_ae_threshold)
                        
                        flag_a = flag_a_rule1 or flag_a_rule2 or flag_a_rule3
                        flag_b = self.transport_channel_b_flag
                        is_anomaly = flag_a or flag_b
                        
                        channel = ('AB' if flag_a and flag_b else
                                   'A'  if flag_a else
                                   'B'  if flag_b else 'none')
                        self.insert_infer_latest_score = score_a
                        
                        if is_anomaly:
                            if flag_a_rule1 or flag_a_rule2 or flag_b:
                                self.insert_infer_consec_anomaly = self.insert_infer_consec_anomaly_needed
                            else:
                                self.insert_infer_consec_anomaly += 1
                            self.insert_infer_consec_normal = 0
                            
                            if (self.insert_infer_consec_anomaly >=
                                    self.insert_infer_consec_anomaly_needed):
                                if not self.insert_anomaly_ever_detected:
                                    self.insert_anomaly_ever_detected = True
                                    self.transport_detect_channel = channel
                                    self._transport_abort.set()
                                    self.get_logger().warn(
                                        f"🚨 [TRANSPORT] ANOMALY DETECTED! "
                                        f"score_A={score_a:.4f} (rule1={flag_a_rule1}, rule2={flag_a_rule2}) "
                                        f"delta_B={self.transport_delta_b_latest:.4f} ch={channel}"
                                    )
                        else:
                            self.insert_infer_consec_normal += 1
                            self.insert_infer_consec_anomaly = 0
                        self.get_logger().info(
                            f"[TRANSPORT] score_A={score_a:.4f} "
                            f"thr_A={self.insert_gru_ae_threshold:.4f} "
                            f"delta_B={self.transport_delta_b_latest:.4f} "
                            f"thr_B={self.insert_gru_ae_threshold_b:.4f} "
                            f"ch={channel} "
                            f"{'ANOMALY' if is_anomaly else 'normal'} "
                            f"consec_anom={self.insert_infer_consec_anomaly} "
                            f"ever={self.insert_anomaly_ever_detected} "
                            f"max_norm={self.max_norm:.3f} norm={self.norm:.3f}"
                        )


    def _load_gru_ae(self):
        """학습된 GRU-AE 모델 + scaler + threshold A/B + infer_cfg 로드 (Model 1).
        raw(_raw) 파일이 존재하면 48D raw 모델을 우선 사용.
        """
        import torch
        import joblib
        import pickle
        # Grasp 단계에서는 MRL 모델 사용을 강제하기 위해 use_raw = False 설정
        use_raw = False
        suffix = ""
        data_dir  = self.data_save_dir
        model_path    = os.path.join(data_dir, f"grasp_gru_ae{suffix}.pt")
        scaler_path   = os.path.join(data_dir, f"grasp_scaler{suffix}.pkl")
        thresh_path   = os.path.join(data_dir, f"grasp_threshold{suffix}.npy")
        thresh_b_path = os.path.join(data_dir, f"grasp_threshold_b{suffix}.npy")
        cfg_path      = os.path.join(data_dir, f"grasp_infer_cfg{suffix}.pkl")

        for p in [model_path, scaler_path, thresh_path]:
            if not os.path.exists(p):
                self.get_logger().warn(f"[INFERENCE] 파일 없음: {p}")
                return
        try:
            sys.path.insert(0, "/home/dyros/mcy_ws")
            from train_gru_ae import GRUAE
            from dual_channel_detector import FramewiseDeltaDetector
            checkpoint = torch.load(model_path, map_location="cpu")
            scaler = joblib.load(scaler_path)
            input_dim = scaler.n_features_in_
            hidden_dim = checkpoint['encoder.gru.weight_hh_l0'].shape[1]
            model = GRUAE(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=2, dropout=0.2, seq_len=50)
            model.load_state_dict(checkpoint)
            model.eval()
            self.gru_ae_model     = model
            self.gru_ae_scaler    = scaler
            self.gru_ae_threshold = float(np.load(thresh_path)[0])
            if os.path.exists(thresh_b_path):
                self.gru_ae_threshold_b = float(np.load(thresh_b_path)[0])
            if os.path.exists(cfg_path):
                with open(cfg_path, "rb") as f:
                    self.gru_ae_infer_cfg = pickle.load(f)
            self.gru_ae_infer_cfg["use_raw"] = use_raw
            self.grasp_channel_b = FramewiseDeltaDetector(
                delta_threshold=self.gru_ae_threshold_b, smooth_window=3)
            self.get_logger().info(
                f"[INFERENCE] Grasp GRU-AE 로드 완료 "
                f"(input_dim={input_dim}, use_raw={use_raw}, "
                f"thr_A={self.gru_ae_threshold:.6f}, "
                f"thr_B={self.gru_ae_threshold_b:.6f})"
            )
        except Exception as e:
            self.get_logger().error(f"[INFERENCE] Grasp 모델 로드 실패: {e}")

    def _load_grasp_api_detector(self):
        """tactile_anomaly.DualChannelDetector 로드 (use_tactile_api=True 시 사용).
        _load_gru_ae와 동일한 파일을 사용하므로 해당 메서드 호출 이후에 실행해야 함.
        """
        try:
            import sys
            sys.path.insert(0, "/home/dyros/mcy_ws")
            from tactile_anomaly.detector import DualChannelDetector
            data_dir  = self.data_save_dir
            # Grasp 단계에서는 MRL 모델 사용을 강제하기 위해 use_raw = False 설정
            use_raw   = False
            suffix    = ""
            model_path    = os.path.join(data_dir, f"grasp_gru_ae{suffix}.pt")
            scaler_path   = os.path.join(data_dir, f"grasp_scaler{suffix}.pkl")
            thresh_path   = os.path.join(data_dir, f"grasp_threshold{suffix}.npy")
            thresh_b_path = os.path.join(data_dir, f"grasp_threshold_b{suffix}.npy")
            cfg_path      = os.path.join(data_dir, f"grasp_infer_cfg{suffix}.pkl")
            for p in [model_path, scaler_path, thresh_path, thresh_b_path]:
                if not os.path.exists(p):
                    self.get_logger().warn(f"[API] 파일 없음, API detector 로드 불가: {p}")
                    return
            self.grasp_api_detector = DualChannelDetector.from_checkpoint(
                model_path       = model_path,
                scaler_path      = scaler_path,
                threshold_a_path = thresh_path,
                threshold_b_path = thresh_b_path,
                infer_cfg_path   = cfg_path if os.path.exists(cfg_path) else None,
            )
            self.get_logger().info(
                f"[API] DualChannelDetector 로드 완료 (use_raw={use_raw})"
            )
        except Exception as e:
            self.get_logger().error(f"[API] DualChannelDetector 로드 실패: {e}")

    def _run_inference(self):
        """수집된 grasp_buffer로 GRU-AE anomaly score 계산.
        Returns: True(정상), False(이상), None(판단 불가)
        """
        if self.gru_ae_model is None:
            self.get_logger().warn("[INFERENCE] 모델 미로드, rule-based로 fallback")
            return None
        use_raw = self.gru_ae_infer_cfg.get("use_raw", False)
        buf = self.grasp_raw_buffer if use_raw else self.grasp_buffer
        if len(buf) < 10:
            self.get_logger().warn(f"[INFERENCE] 프레임 수 부족 ({len(buf)})")
            return None
        try:
            import torch
            from scipy.signal import resample
            from train_gru_ae import apply_relative_transform, build_hybrid_window
            relative_mode = self.gru_ae_infer_cfg.get("relative", "none")
            weight_alpha  = self.gru_ae_infer_cfg.get("weight_alpha", 0.0)
            hybrid        = self.gru_ae_infer_cfg.get("hybrid", False)
            n_rel_dims    = self.gru_ae_infer_cfg.get("n_rel_dims", None)
            seq = np.array(buf, dtype=np.float32)                          # [T, 6 or 48]
            seq_fixed = resample(seq, 50, axis=0).astype(np.float32)       # [50, D]
            if hybrid:
                seq_fixed = build_hybrid_window(seq_fixed)             # [50, 7]
            seq_norm  = self.gru_ae_scaler.transform(seq_fixed)        # 정규화
            seq_norm  = np.clip(seq_norm, -3.0, 3.0)                   # 극단 taxel 억제
            seq_rel   = apply_relative_transform(seq_norm, relative_mode,
                                                 n_rel_dims=n_rel_dims)
            x = torch.tensor(seq_rel, dtype=torch.float32).unsqueeze(0)  # [1, 50, D]
            with torch.no_grad():
                score = float(self.gru_ae_model.reconstruction_error(
                    x, weight_alpha=weight_alpha).item())
            is_normal = score <= self.gru_ae_threshold
            self.get_logger().info(
                f"[INFERENCE] score={score:.6f}, "
                f"threshold={self.gru_ae_threshold:.6f}, "
                f"판정={'정상' if is_normal else '이상'}"
            )
            return is_normal
        except Exception as e:
            self.get_logger().error(f"[INFERENCE] 추론 실패: {e}")
            return None

    def _run_windowed_inference(self) -> float | None:
        """realtime_buffer 의 현재 50프레임 window 로 즉시 anomaly score 계산.
        Returns: reconstruction error (float), 판단 불가 시 None
        """
        if self.gru_ae_model is None or self.gru_ae_scaler is None:
            return None
        if len(self.realtime_buffer) < 50:
            return None
        try:
            import torch
            from train_gru_ae import apply_relative_transform, build_hybrid_window
            relative_mode = self.gru_ae_infer_cfg.get("relative", "none")
            weight_alpha  = self.gru_ae_infer_cfg.get("weight_alpha", 0.0)
            hybrid        = self.gru_ae_infer_cfg.get("hybrid", False)
            n_rel_dims    = self.gru_ae_infer_cfg.get("n_rel_dims", None)
            use_raw       = self.gru_ae_infer_cfg.get("use_raw", False)
            # use_raw이면 48D raw 버퍼, 아니면 6D 버퍼 사용
            buf = self.realtime_raw_buffer if use_raw else self.realtime_buffer
            seq = np.array(list(buf), dtype=np.float32)                    # [50, D]
            if hybrid:
                seq = build_hybrid_window(seq)
            seq_s = self.gru_ae_scaler.transform(seq).astype(np.float32)
            # 특정 taxel의 극단값이 재구성 오차를 독점하는 것을 방지 (±3σ 클리핑)
            seq_s = np.clip(seq_s, -3.0, 3.0)
            seq_r = apply_relative_transform(seq_s, relative_mode,
                                             n_rel_dims=n_rel_dims)
            x = torch.tensor(seq_r).unsqueeze(0)                           # [1, 50, D]
            with torch.no_grad():
                score = float(self.gru_ae_model.reconstruction_error(
                    x, weight_alpha=weight_alpha).item())
            return score
        except Exception as e:
            self.get_logger().error(f"[SLIDING] inference error: {e}")
            return None

    def _load_insert_gru_ae(self):
        """학습된 Transport GRU-AE 모델 + scaler + threshold A/B + infer_cfg 로드 (Model 2)."""
        import torch
        import joblib
        import pickle
        data_dir    = self.insert_save_dir
        use_raw     = os.path.exists(os.path.join(data_dir, "transport_gru_ae_raw.pt"))
        suffix      = "_raw" if use_raw else ""
        model_path    = os.path.join(data_dir, f"transport_gru_ae{suffix}.pt")
        scaler_path   = os.path.join(data_dir, f"transport_scaler{suffix}.pkl")
        thresh_path   = os.path.join(data_dir, f"transport_threshold{suffix}.npy")
        thresh_b_path = os.path.join(data_dir, f"transport_threshold_b{suffix}.npy")
        cfg_path      = os.path.join(data_dir, f"transport_infer_cfg{suffix}.pkl")

        for p in [model_path, scaler_path, thresh_path]:
            if not os.path.exists(p):
                self.get_logger().warn(f"[INFERENCE] Transport 파일 없음: {p}")
                return
        try:
            sys.path.insert(0, "/home/dyros/mcy_ws")
            from train_gru_ae import GRUAE
            from dual_channel_detector import FramewiseDeltaDetector
            checkpoint = torch.load(model_path, map_location="cpu")
            scaler = joblib.load(scaler_path)
            input_dim = scaler.n_features_in_
            hidden_dim = checkpoint['encoder.gru.weight_hh_l0'].shape[1]
            model = GRUAE(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=2, dropout=0.2, seq_len=50)
            model.load_state_dict(checkpoint)
            model.eval()
            self.insert_gru_ae_model     = model
            self.insert_gru_ae_scaler    = scaler
            self.insert_gru_ae_threshold = float(np.load(thresh_path)[0])
            if os.path.exists(thresh_b_path):
                self.insert_gru_ae_threshold_b = float(np.load(thresh_b_path)[0])
            self.insert_gru_ae_mrl_thresh = 0.2
            self.insert_gru_ae_cutoff_multiplier = 2.2
            if os.path.exists(cfg_path):
                with open(cfg_path, "rb") as f:
                    self.insert_gru_ae_infer_cfg = pickle.load(f)
                self.insert_gru_ae_mrl_thresh = float(self.insert_gru_ae_infer_cfg.get("mrl_thresh", 0.2))
                self.insert_gru_ae_cutoff_multiplier = float(self.insert_gru_ae_infer_cfg.get("cutoff_multiplier", 2.2))
            self.transport_channel_b = FramewiseDeltaDetector(
                delta_threshold=self.insert_gru_ae_threshold_b, smooth_window=3)
            self.insert_gru_ae_infer_cfg["use_raw"] = use_raw
            self.get_logger().info(
                f"[INFERENCE] Transport GRU-AE 로드 완료 "
                f"(input_dim={input_dim}, use_raw={use_raw}, "
                f"thr_A={self.insert_gru_ae_threshold:.6f}, "
                f"thr_B={self.insert_gru_ae_threshold_b:.6f}, "
                f"mrl_thresh={self.insert_gru_ae_mrl_thresh:.2f}, "
                f"cutoff_mult={self.insert_gru_ae_cutoff_multiplier:.2f})"
            )
        except Exception as e:
            self.get_logger().error(f"[INFERENCE] Transport 모델 로드 실패: {e}")

    def _load_insert_api_detector(self):
        """tactile_anomaly.DualChannelDetector 로드 (Model 2, use_tactile_api=True 시 사용)."""
        try:
            import sys
            sys.path.insert(0, "/home/dyros/mcy_ws")
            from tactile_anomaly.detector import DualChannelDetector
            data_dir = self.insert_save_dir
            use_raw = os.path.exists(os.path.join(data_dir, "transport_gru_ae_raw.pt"))
            suffix = "_raw" if use_raw else ""
            model_path    = os.path.join(data_dir, f"transport_gru_ae{suffix}.pt")
            scaler_path   = os.path.join(data_dir, f"transport_scaler{suffix}.pkl")
            thresh_path   = os.path.join(data_dir, f"transport_threshold{suffix}.npy")
            thresh_b_path = os.path.join(data_dir, f"transport_threshold_b{suffix}.npy")
            cfg_path      = os.path.join(data_dir, f"transport_infer_cfg{suffix}.pkl")
            for p in [model_path, scaler_path, thresh_path, thresh_b_path]:
                if not os.path.exists(p):
                    self.get_logger().warn(f"[API] 파일 없음, Transport API detector 로드 불가: {p}")
                    return
            self.insert_api_detector = DualChannelDetector.from_checkpoint(
                model_path       = model_path,
                scaler_path      = scaler_path,
                threshold_a_path = thresh_path,
                threshold_b_path = thresh_b_path,
                infer_cfg_path   = cfg_path if os.path.exists(cfg_path) else None,
            )
            self.get_logger().info(
                f"[API] Transport DualChannelDetector 로드 완료 (use_raw={use_raw})"
            )
        except Exception as e:
            self.get_logger().error(f"[API] Transport DualChannelDetector 로드 실패: {e}")

    def _run_insert_windowed_inference(self) -> float | None:
        """insert_realtime_buffer 의 현재 50프레임 window 로 anomaly score 계산.
        Returns: reconstruction error (float), 판단 불가 시 None
        """
        if self.insert_gru_ae_model is None or self.insert_gru_ae_scaler is None:
            return None
        try:
            import torch
            from train_gru_ae import apply_relative_transform, build_hybrid_window
            relative_mode = self.insert_gru_ae_infer_cfg.get("relative", "none")
            weight_alpha  = self.insert_gru_ae_infer_cfg.get("weight_alpha", 0.0)
            hybrid        = self.insert_gru_ae_infer_cfg.get("hybrid", False)
            n_rel_dims    = self.insert_gru_ae_infer_cfg.get("n_rel_dims", None)
            use_raw       = self.insert_gru_ae_infer_cfg.get("use_raw", False)
            buf = self.insert_realtime_raw_buffer if use_raw else self.insert_realtime_buffer
            if len(buf) < 50:
                return None
            seq = np.array(list(buf), dtype=np.float32)  # [50, 6] or [50, 48]
            if hybrid:
                seq = build_hybrid_window(seq)                                   # [50, 7]
            seq_s = self.insert_gru_ae_scaler.transform(seq).astype(np.float32)  # 정규화
            seq_r = apply_relative_transform(seq_s, relative_mode,
                                             n_rel_dims=n_rel_dims)              # relative
            x = torch.tensor(seq_r).unsqueeze(0)                                 # [1, 50, D]
            with torch.no_grad():
                score = float(self.insert_gru_ae_model.reconstruction_error(
                    x, weight_alpha=weight_alpha).item())
            return score
        except Exception as e:
            self.get_logger().error(f"[SLIDING-INSERT] inference error: {e}")
            return None

    def _save_insert_data(self, label: str):
        """수집된 이송 구간(ba_init → open_gripper) 시계열을 레이블과 함께 저장."""
        if len(self.insert_buffer) < 10:
            self.get_logger().warn(f"[DATA] 이송 프레임 수 부족 ({len(self.insert_buffer)}), 저장 스킵")
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seq = np.array(self.insert_buffer, dtype=np.float32)       # [T, 6]
        fname = os.path.join(self.insert_save_dir, f"transport_{ts}_{label}.npy")
        np.save(fname, seq)
        raw = np.array(self.insert_raw_buffer, dtype=np.float32)   # [T, 48]
        fname_raw = os.path.join(self.insert_save_dir, f"transport_{ts}_{label}_raw.npy")
        np.save(fname_raw, raw)
        self.get_logger().info(f"[DATA] 이송 저장 완료: {fname} (shape={seq.shape}), raw: {raw.shape}")

    def _ask_insertion_result(self):
        """open_gripper 직전: 이송 구간 파지 유지 성공 여부를 사람에게 확인.
        성공 = ba_init ~ open_gripper 동안 세탁물을 놓치지 않음.
        실패 = 이송 중 낙하 (벽 접촉 여부는 관계 없음).
        Returns: True=성공 / False=실패 / None=skip
        """
        print("\n" + "="*50)
        print("[DATA COLLECTION] 이송 구간 파지 유지 결과 확인")
        print("  y = 성공 (세탁물을 놓치지 않고 세탁기에 넣음)")
        print("  n = 실패 (이송 중 낙하)")
        print("  x = skip")
        print("="*50)
        while True:
            key = self._spin_input("입력 (y/n/x): ")
            if key in ('y', 'n', 'x'):
                break
            print("  y / n / x 중 하나를 입력하세요")
        if key == 'y':
            return True
        elif key == 'n':
            return False
        else:
            return None

    def _ask_gripper_held(self):
        """그리퍼에 물체가 잡혀있는지 사람에게 확인.
        Returns: True=잡혀있음 / False=비어있음 / None=skip
        """
        print("\n" + "="*50)
        print("[DATA COLLECTION] 그리퍼 상태 확인")
        print("  y = 물체가 잡혀있음  → 이송 시도 후 결과로 레이블 결정")
        print("  n = 그리퍼 비어있음  → failure 저장, 재시도")
        print("  x = skip             → rule-based fallback")
        print(f"  [참고] norm={self.norm:.3f}, theta={self.theta:.1f}, max_norm={self.max_norm:.3f}")
        print("="*50)
        while True:
            key = self._spin_input("입력 (y/n/x): ")
            if key in ('y', 'n', 'x'):
                break
            print("  y / n / x 중 하나를 입력하세요")
        if key == 'y':
            return True
        elif key == 'n':
            return False
        else:
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
        """DDS 블로킹 방지: raw 데이터만 큐에 넣고 즉시 리턴."""
        try:
            x_vals = [t.x for t in msg.sensors[0].taxels]
            y_vals = [t.y for t in msg.sensors[0].taxels]
            z_vals = [t.z for t in msg.sensors[0].taxels]
            try:
                self._raw_sensor_queue.put_nowait((x_vals, y_vals, z_vals))
            except queue.Full:
                pass  # 처리 스레드가 밀리면 최신 데이터를 위해 드롭
        except Exception:
            pass

    def _sensor_processing_worker(self):
        """raw 센서 데이터를 큐에서 꺼내 feature 계산 + 상태 업데이트.
        - 중복 제거: 센서 publish 속도(~500Hz) > 실제 업데이트(~15Hz)이므로 값이 바뀐 프레임만 처리
        - stale 감지: WebSocket 끊김 후 재연결 시 누적값 자동 리셋
        """
        _prev_raw_x = None  # 중복 제거용 직전 raw 값 (정규화 전)
        _prev_raw_y = None
        _prev_raw_z = None
        _last_data_time = time.time()
        _STALE_TIMEOUT = 0.5   # 0.5초 이상 데이터 없으면 WebSocket 끊김으로 판단

        while True:
            try:
                x_vals, y_vals, z_vals = self._raw_sensor_queue.get(timeout=0.1)
            except queue.Empty:
                # ── stale 감지: 마지막 데이터로부터 일정 시간 경과 시 누적값 리셋
                if (self.prev_x is not None and
                        time.time() - _last_data_time > _STALE_TIMEOUT):
                    self.get_logger().warn(
                        "[SENSOR] WebSocket 끊김 감지 → 누적값 리셋"
                    )
                    self.cum_x = [0.0] * 16
                    self.cum_y = [0.0] * 16
                    self.cum_z = [0.0] * 16
                    self.prev_x = None
                    _prev_raw_x = _prev_raw_y = _prev_raw_z = None
                    _last_data_time = time.time()  # 재진입 방지
                continue

            _last_data_time = time.time()

            # ── 중복 제거: raw 값이 직전과 동일하면 skip (센서 재발행 프레임)
            if x_vals == _prev_raw_x and y_vals == _prev_raw_y and z_vals == _prev_raw_z:
                continue
            _prev_raw_x, _prev_raw_y, _prev_raw_z = x_vals, y_vals, z_vals

            try:
                x_vals, y_vals, z_vals = self._normalize_taxels(x_vals, y_vals, z_vals)

                if self.prev_x is None:
                    self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals
                    self.z_aligned = self.contacted = False
                    self.z_aligned_k = self.contacted_k = False
                    self.z_aligned_f = self.contacted_f = False
                    self.contacted_ff = self.contacted_kk = False
                    continue

                dx = [c - p for c, p in zip(x_vals, self.prev_x)]
                dy = [c - p for c, p in zip(y_vals, self.prev_y)]
                dz = [c - p for c, p in zip(z_vals, self.prev_z)]

                # ── 점프 감지: 재연결이 stale timeout(0.5s)보다 빠른 경우,
                #   prev_x가 유지된 채로 센서가 다른 값으로 갑자기 돌아올 수 있음.
                #   정규화 후 [0,1] 범위에서 프레임 간 diff > 0.5 는 정상 동작 불가능 → 리셋
                _max_jump = max(
                    max(abs(d) for d in dx),
                    max(abs(d) for d in dy),
                    max(abs(d) for d in dz),
                )
                if _max_jump > 0.5:
                    self.cum_x = [0.0] * 16
                    self.cum_y = [0.0] * 16
                    self.cum_z = [0.0] * 16
                    self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals
                    _prev_raw_x = _prev_raw_y = _prev_raw_z = None
                    if self.is_insert_recording:
                        # 이송 중 점프 = 낙하로 인한 센서 급감일 수 있음 → 버퍼/카운터 유지
                        self.get_logger().warn(
                            f"[SENSOR] 점프 감지 (max_diff={_max_jump:.3f}) → 누적값 리셋 (이송 중: 버퍼 유지)"
                        )
                    else:
                        # 이송 외 구간: WebSocket 재연결 아티팩트 → 버퍼 flush
                        self.insert_realtime_buffer.clear()
                        self.insert_realtime_raw_buffer.clear()
                        with self._infer_lock:
                            self.insert_infer_consec_anomaly = 0
                            self.insert_infer_consec_normal  = 0
                        self.get_logger().warn(
                            f"[SENSOR] 점프 감지 (max_diff={_max_jump:.3f}) → 누적값 리셋 + inference 버퍼 flush"
                        )
                    continue

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
                weights = norms.copy()
                weights[weights == 0] = 1e-6

                mean_vec_biased = np.sum(vecs, axis=0) / np.sum(weights)

                norm = np.linalg.norm(mean_vec_biased)
                cos_theta = np.dot(mean_vec_biased, np.array([0, 0, 1])) / (norm + 1e-8)
                theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

                self.norm = norm
                self.theta = theta_deg
                self.max_norm = max_norm

                # MRL 계산
                valid_mask = norms > 0.1
                if valid_mask.sum() >= 2:
                    unit_vecs = vecs[valid_mask] / norms[valid_mask, np.newaxis]
                    mean_unit = np.mean(unit_vecs, axis=0)
                    mrl = float(np.linalg.norm(mean_unit))
                else:
                    mean_unit = np.zeros(3)
                    mrl = 0.0
                self.mrl = mrl

                # 6D feature
                frame = [
                    mrl,
                    float(norm),
                    float(max_norm),
                    float(mean_unit[0]),
                    float(mean_unit[1]),
                    float(mean_unit[2]),
                ]
                raw_frame = vecs.flatten().tolist()

                if self.is_recording:
                    self.grasp_buffer.append(frame)
                    self.grasp_raw_buffer.append(raw_frame)
                    if self.run_mode == 'inference':
                        if self.use_tactile_api and self.grasp_api_detector is not None:
                            # ── API 경로: DualChannelDetector가 채널 A+B를 동시에 처리
                            frame_arr = np.array(frame,     dtype=np.float32)
                            raw_arr   = np.array(raw_frame, dtype=np.float32)
                            self.inference_frame_cnt += 1
                            res      = self.grasp_api_detector.update(frame_arr, raw_frame=raw_arr)
                            is_anom  = res['anomaly']
                            score_a  = res['score_a']
                            delta_b  = res['delta_b']
                            channel  = res['channel']
                            with self._infer_lock:
                                self.infer_latest_score   = score_a
                                self.grasp_delta_b_latest = delta_b
                                if is_anom:
                                    self.infer_consec_anomaly += 1
                                    self.infer_consec_normal   = 0
                                    if self.infer_consec_anomaly >= self.infer_consec_anomaly_needed:
                                        self.infer_grasp_anomaly_detected = True
                                else:
                                    self.infer_consec_normal   += 1
                                    self.infer_consec_anomaly   = 0
                            if score_a > 0:
                                self.get_logger().info(
                                    f"[API-GRASP] score_A={score_a:.4f} "
                                    f"delta_B={delta_b:.4f} ch={channel} "
                                    f"{'ANOMALY' if is_anom else 'normal'} "
                                    f"consec={self.infer_consec_anomaly}"
                                )
                        else:
                            # ── 기존 경로: 채널 A는 별도 worker thread, 채널 B는 여기서 처리
                            self.realtime_buffer.append(frame)
                            self.realtime_raw_buffer.append(raw_frame)
                            self.inference_frame_cnt += 1
                            if (len(self.realtime_buffer) >= 50 and
                                    self.inference_frame_cnt % self.inference_stride == 0):
                                self._infer_trigger.set()
                            # 채널 B: 퍼프레임 MRL delta → 즉시 consec_normal 리셋 (50프레임 대기 없음)
                            # 초기 로봇 이동 모션 아티팩트 방지용 워밍업 기간 적용
                            # 최소 힘 게이트: max_norm이 너무 작으면 MRL이 불안정 → 체크 생략
                            if (self.grasp_channel_b is not None and
                                    self.inference_frame_cnt > self.grasp_channel_b_warmup_needed and
                                    self.max_norm > 0.3):
                                frame_arr = np.array(frame, dtype=np.float32)
                                delta_b, flag_b = self.grasp_channel_b.update(frame_arr)
                                if flag_b:
                                    self.grasp_channel_b_consec += 1
                                    if (self.grasp_channel_b_consec >=
                                            self.grasp_channel_b_consec_needed):
                                        with self._infer_lock:
                                            self.grasp_channel_b_flag = True
                                            self.grasp_delta_b_latest = delta_b
                                            self.infer_consec_normal = 0
                                        self.get_logger().error(
                                            f"[CH-B] Grasp 이상 즉시 감지! "
                                            f"delta={delta_b:.4f} thr={self.gru_ae_threshold_b:.4f}"
                                        )
                                else:
                                    self.grasp_channel_b_consec = 0

                if self.is_insert_recording:
                    self.insert_buffer.append(frame)
                    self.insert_raw_buffer.append(raw_frame)
                    self.insert_max_mrl_seen = max(self.insert_max_mrl_seen, mrl)
                    if self.run_mode == 'inference':
                        if self.use_tactile_api and self.insert_api_detector is not None:
                            frame_arr = np.array(frame,     dtype=np.float32)
                            raw_arr   = np.array(raw_frame, dtype=np.float32)
                            self.insert_infer_frame_cnt += 1
                            res      = self.insert_api_detector.update(frame_arr, raw_frame=raw_arr)
                            is_anom  = res['anomaly']
                            score_a  = res['score_a']
                            delta_b  = res['delta_b']
                            channel  = res['channel']
                            with self._infer_lock:
                                self.insert_infer_latest_score = score_a
                                self.transport_delta_b_latest  = delta_b
                                
                                # Warmup check for Channel B
                                flag_b = (channel in ('B', 'AB'))
                                if self.insert_infer_frame_cnt <= self.transport_channel_b_warmup_needed:
                                    flag_b = False
                                
                                if flag_b:
                                    self.transport_channel_b_consec += 1
                                    if self.transport_channel_b_consec >= self.transport_channel_b_consec_needed:
                                        self.transport_channel_b_flag = True
                                else:
                                    self.transport_channel_b_consec = 0

                                # Check Channel A
                                flag_a = (channel in ('A', 'AB'))
                                if flag_a:
                                    self.insert_infer_consec_anomaly += 1
                                    self.insert_infer_consec_normal   = 0
                                else:
                                    self.insert_infer_consec_normal   += 1
                                    self.insert_infer_consec_anomaly   = 0
                                    
                                is_final_anom = (self.insert_infer_consec_anomaly >= self.insert_infer_consec_anomaly_needed) or self.transport_channel_b_flag
                                if is_final_anom:
                                    if not self.insert_anomaly_ever_detected:
                                        self.insert_anomaly_ever_detected = True
                                        self._transport_abort.set()
                                        self.get_logger().warn(
                                            f"🚨 [API-TRANSPORT] ANOMALY DETECTED! "
                                            f"score_A={score_a:.4f} delta_B={delta_b:.4f} ch={channel} "
                                            f"consec_anom={self.insert_infer_consec_anomaly} "
                                            f"consec_B={self.transport_channel_b_consec}"
                                        )
                        else:
                            # ── 기존 경로: 채널 A는 별도 worker thread, 채널 B는 여기서 처리
                            self.insert_realtime_buffer.append(frame)
                            self.insert_realtime_raw_buffer.append(raw_frame)
                            self.insert_infer_frame_cnt += 1
                            if (len(self.insert_realtime_buffer) >= 50 and
                                    self.insert_infer_frame_cnt % self.inference_stride == 0):
                                self._insert_trigger.set()
                            # 채널 B: 퍼프레임 MRL delta → 즉시 anomaly 확정 (50프레임 대기 없음)
                            # 이송 초기 모션 아티팩트 방지용 워밍업 기간 적용
                            # 최소 힘 게이트: max_norm이 너무 작으면 MRL이 불안정 → 체크 생략
                            if (self.transport_channel_b is not None and
                                    self.insert_infer_frame_cnt > self.transport_channel_b_warmup_needed and
                                    self.max_norm > 0.3):
                                frame_arr = np.array(frame, dtype=np.float32)
                                delta_b, flag_b = self.transport_channel_b.update(frame_arr)
                                if flag_b:
                                    self.transport_channel_b_consec += 1
                                    if (self.transport_channel_b_consec >=
                                            self.transport_channel_b_consec_needed):
                                        with self._infer_lock:
                                            self.transport_channel_b_flag = True
                                            self.transport_delta_b_latest = delta_b
                                            if not self.insert_anomaly_ever_detected:
                                                self.insert_anomaly_ever_detected = True  # 즉시 확정
                                                self.transport_detect_channel = 'B'
                                                self._transport_abort.set()
                                                self.get_logger().warn(
                                                    "🚨 Slip Recorded! Will recover after motion.")
                                        self.get_logger().error(
                                            f"[CH-B] Transport 낙하 즉시 감지! "
                                            f"delta={delta_b:.4f} thr={self.insert_gru_ae_threshold_b:.4f}"
                                        )
                                else:
                                    self.transport_channel_b_consec = 0

                # Slip Monitoring
                if getattr(self, "monitor_slip", False):
                    if self.monitor_initial_norm > 0 and norm < (self.monitor_initial_norm * 0.5):
                        self.get_logger().error(
                            f"🚨 [MONITOR] Slip Detected! Norm Dropped: "
                            f"{self.monitor_initial_norm:.3f} -> {norm:.3f}")
                        self._trigger_slip_stop()
                    elif abs(theta_deg - self.monitor_initial_theta) > 20.0:
                        self.get_logger().error(
                            f"🚨 [MONITOR] Slip Detected! Theta Shift: "
                            f"{self.monitor_initial_theta:.1f} -> {theta_deg:.1f}")
                        self._trigger_slip_stop()

                # Heuristic z_aligned / contacted 판단
                if max_norm < 0.5:
                    self.z_aligned = False
                    self.contacted = False
                elif norm < 0.35:
                    self.z_aligned = False
                    self.contacted = False
                else:
                    if norm < 1.0:
                        if theta_deg < 55.0:
                            self.z_aligned = True
                            self.contacted = False
                        elif theta_deg <= 80.0:
                            self.contacted = True
                            self.z_aligned = False
                        else:
                            self.z_aligned = False
                            self.contacted = False
                    else:
                        if theta_deg < 35.0:
                            self.z_aligned = True
                            self.contacted = False
                        elif theta_deg <= 80.0:
                            self.contacted = True
                            self.z_aligned = False
                        else:
                            self.z_aligned = False
                            self.contacted = False

                # prev 갱신
                self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals

            except Exception as e:
                self.get_logger().error(f"[SENSOR] 처리 오류: {e}")

    def close_door(self):
        pass

    def open_door(self):
        import sqlite3
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        import glob

        bag_dir = self.door_bag_path
        hz      = self.door_bag_hz
        speed   = self.door_bag_speed

        # --- bag 파싱 ---
        db_files = sorted(glob.glob(os.path.join(bag_dir, '*.db3')))
        if not db_files:
            self.get_logger().error(f"[open_door] bag 파일 없음: {bag_dir}")
            return
        JointStateMsg = get_message('sensor_msgs/msg/JointState')
        interval = 1.0 / hz
        records  = []
        last_t   = -1e9
        for db_path in db_files:
            con = sqlite3.connect(db_path)
            for ts_ns, data in con.execute(
                "SELECT timestamp, data FROM messages ORDER BY timestamp"
            ):
                t = ts_ns * 1e-9
                if t - last_t < interval:
                    continue
                msg = deserialize_message(data, JointStateMsg)
                records.append((t, list(msg.position[:6])))
                last_t = t
            con.close()

        if not records:
            self.get_logger().error("[open_door] bag에 데이터 없음")
            return
        self.get_logger().info(f"[open_door] {len(records)}개 포인트 로드 완료")

        # --- JointTrajectory 전송 ---
        t0   = records[0][0]
        traj = JointTrajectory()
        traj.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        for t, pos in records:
            elapsed = (t - t0) / speed
            sec     = int(elapsed)
            nsec    = int((elapsed - sec) * 1e9)
            pt = JointTrajectoryPoint()
            pt.positions       = pos
            pt.velocities      = [0.0] * 6
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            traj.points.append(pt)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        future = self._arm_direct_client.send_goal_async(goal_msg)
        self._wait_future(future, timeout_sec=10.0)
        if not future.done() or not future.result().accepted:
            self.get_logger().error("[open_door] goal 거절됨")
            return

        duration_sec = (records[-1][0] - records[0][0]) / speed + 2.0
        result_future = future.result().get_result_async()
        self._wait_future(result_future, timeout_sec=duration_sec)
        self.get_logger().info("[open_door] 완료")

    def _floor_recovery_grasp(self) -> bool:
        """이송 중 낙하한 세탁물을 바닥에서 찾아 재파지. True=성공."""
        import copy

        if self.current_joint_state is None:
            self.get_logger().error("[RECOVERY] joint_state 미수신")
            return False

        j1_now = float(self.current_joint_state[0])
        recovery_pose = [j1_now] + list(self.ba_init[1:6]) + [0.0, 0.0]
        self.get_logger().info(f"[RECOVERY] 복구 자세 이동 (j1={j1_now:.4f}, joints2-6=ba_init)")

        # 복구 자세 이동 (그리퍼 열린 상태 → 빠른 이동)
        for _ in range(3):
            if self.goal(recovery_pose, 'keep', fast=True):
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

        # Hover → Pre-Grasp → Push
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
        time.sleep(0.3)

        # Retreat
        for _ in range(3):
            if self.move_backward(target):
                break
            self.get_logger().warn("[RECOVERY] Retreat 실패 → 재시도")

        # 복구 자세 복귀 (물체 파지 중 → 기본 속도)
        while not self.goal(recovery_pose, 'keep'):
            self.get_logger().warn("[RECOVERY] 복구 자세 복귀 실패 → 재시도")
            time.sleep(0.2)

        # 파지 성공 여부 판정 (rule-based)
        is_grasp = self.max_norm > 0.45
        self.get_logger().info(
            f"[RECOVERY] norm={self.norm:.3f} max_norm={self.max_norm:.3f} "
            f"→ {'재파지 성공' if is_grasp else '재파지 실패'}"
        )

        if not is_grasp:
            self.open_gripper()
            return False

        return True

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

        if not self._target_client.server_is_ready():
            self.get_logger().error('target_pose 액션 서버 미연결 → node2.py 실행 중인지 확인')
            return None
        self.get_logger().info('target_pose 요청 전송')
        _tg0 = time.time()
        future = self._target_client.send_goal_async(goal_msg)
        self._wait_future(future, timeout_sec=30.0)
        _gap_accept = time.time() - _tg0
        self.get_logger().info(f"[GAP_TIMER] goal_accept: {_gap_accept:.2f}s")
        if _gap_accept > 2.0:
            self._timing_log.append((f"  [GAP] goal_accept 지연 ({mode})", _gap_accept))

        if not future.done():
            self.get_logger().error('Goal 요청 실패 (timeout)')
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal이 액션 서버에서 거부됨')
            return None

        # 결과 대기 (vision inference 최대 60s)
        _tg1 = time.time()
        result_future = goal_handle.get_result_async()
        self._wait_future(result_future, timeout_sec=60.0)
        _gap_result = time.time() - _tg1
        self.get_logger().info(f"[GAP_TIMER] result_wait: {_gap_result:.2f}s")
        if _gap_result > 20.0:
            self._timing_log.append((f"  [GAP] result_wait 이상 ({mode})", _gap_result))
        if not result_future.done():
            self.get_logger().error('Vision 결과 대기 timeout (60s) → goal 취소')
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
            return None
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
            MAX_Z_LIMIT = 0.25
            MAX_Y_LIMIT = 0.365   # original
            # MAX_Y_LIMIT = 0.24 
            if pose_stamped.pose.position.z < MIN_Z_LIMIT:
                self.get_logger().warn(f"  >> Target Z ({pose_stamped.pose.position.z:.3f}) is too low! Clamping to {MIN_Z_LIMIT}")
                pose_stamped.pose.position.z = MIN_Z_LIMIT
            if pose_stamped.pose.position.z > MAX_Z_LIMIT:
                self.get_logger().warn(f"  >> Target Z ({pose_stamped.pose.position.z:.3f}) is too high! Clamping to {MAX_Z_LIMIT}")
                pose_stamped.pose.position.z = MAX_Z_LIMIT

        
            pose_stamped.pose.position.x -= 0.01  # -1cm offset (wall margin)
            tx, ty, tz = pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z
            self.get_logger().info(f"  [Wall Check] Target XYZ: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

            if tz == MAX_Z_LIMIT:
                HIGH_Z_MAX_Y = MAX_Y_LIMIT
                if pose_stamped.pose.position.y > HIGH_Z_MAX_Y:
                    self.get_logger().warn(f"  >> Target Y ({pose_stamped.pose.position.y:.3f}) > {HIGH_Z_MAX_Y} → Clamping")
                    pose_stamped.pose.position.y = HIGH_Z_MAX_Y
                raw_y = pose_stamped.pose.position.y
                grasp_y = raw_y + (HIGH_Z_MAX_Y - raw_y) * (1.0 / 3.0)
                self.get_logger().info(f"  [Grasp Y] z={tz:.3f} > 0.265 → raw={raw_y:.3f} → grasp={grasp_y:.3f} (1/3 to limit={HIGH_Z_MAX_Y})")
                pose_stamped.pose.position.y = grasp_y
            else:
                if pose_stamped.pose.position.y > MAX_Y_LIMIT:
                    self.get_logger().warn(f" >> Target Y  ({pose_stamped.pose.position.y:.3f}) is too low! Clamping to {MAX_Y_LIMIT}")
                    pose_stamped.pose.position.y = MAX_Y_LIMIT
                # grasp y = target_y와 MAX_Y_LIMIT 사이 2/3 지점
                raw_y = pose_stamped.pose.position.y
                grasp_y = raw_y + (MAX_Y_LIMIT - raw_y) * (2.0 / 3.0)
                self.get_logger().info(f"  [Grasp Y] raw={raw_y:.3f} → grasp={grasp_y:.3f} (2/3 to limit={MAX_Y_LIMIT})")
                pose_stamped.pose.position.y = grasp_y

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
                time.sleep(0.2)

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
        self._wait_future(future)

        if not future.done():
            self.get_logger().error('Goal 요청 실패')
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal이 액션 서버에서 거부됨')
            return None

        # 결과 대기
        result_future = goal_handle.get_result_async()
        self._wait_future(result_future)
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
            
            MIN_SAFE_Z_BASE = -0.300
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
                time.sleep(0.2)
            rclpy.shutdown()
            sys.exit(0)
            return None

    def target_get_bg(self, mode = "failure_bg"):
        goal_msg = TargetPose.Goal()
        goal_msg.start = True  # 요청 트리거
        goal_msg.mode = mode

        self.get_logger().info('target_pose 요청 전송')
        future = self._target_client.send_goal_async(goal_msg)
        self._wait_future(future)

        if not future.done():
            self.get_logger().error('Goal 요청 실패')
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal이 액션 서버에서 거부됨')
            return None

        # 결과 대기
        result_future = goal_handle.get_result_async()
        self._wait_future(result_future)
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
        import time
        while True:
            _t_overcome = time.time()
            self.open_gripper()
            self.get_logger().info("Step 1: View Pose로 이동 (Empty Check)")
            _t = time.time()
            while not self.goal(self.wm_view, 'keep', fast=True):  # 물체 없음 → 빠른 이동
                self.get_logger().warn("  >> wm_view 이동 실패 → 재시도 중")
                time.sleep(0.2)
            self._step_time("overcome2: wm_view 이동", _t)

            _t = time.time()
            chk_pose = self.target_get_wm(mode="failure_eh")
            self._step_time("overcome2: target_get_wm (Empty Check)", _t)

            if chk_pose is None:
                self.get_logger().warn("⚠️ [Overcome] 감지 결과 없음 (인식 실패). 루프 재시도.")
                continue

            if chk_pose.pose.position.z == -99.0:
                self.get_logger().info("🛑 [Overcome] 세탁물 없음 (RGB Gate). 작업 종료.")
                break

            self.get_logger().info("✅ [Overcome] 의류 감지됨. 파지 자세로 진입.")
            # [Stage 2] Move to Grasp Pose & Detect Target (TinyUNet)
            self.get_logger().info("Step 2: 파지 자세(wm_mid)로 이동 및 좌표 계산")

            _t = time.time()
            while not self.goal(self.wm_mid, 'keep', fast=True):  # 물체 없음 → 빠른 이동
                self.get_logger().warn("  >> wm_mid 이동 실패 → 재시도 중")
                time.sleep(0.2)
            self._step_time("overcome2: wm_mid 이동", _t)
            import time
            time.sleep(0.5) # controller settle before Vision

            # wm_mid 도달 후 EE orientation 캡처 (joint6 align 기준)
            # rclpy.spin_once은 main executor와 충돌 → time.sleep으로 대체 (main executor가 처리)
            time.sleep(0.05)
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
            _t = time.time()
            target_wm = self.target_get_wm(mode="failure_eh_strict")  # 정밀 좌표용: 클러스터 작으면 미감지
            self._step_time("overcome2: target_get_wm (정밀 좌표)", _t)
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
            # Mode 9(collision ON)는 드럼 내부 collision으로 실패 → Mode 11(collision OFF)로 직행
            _t = time.time()
            res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=11, rpy_tol=[6.28, 6.28, 6.28])
            if res != "success":
                self.get_logger().warn("  [OVERCOME] Pre-Grasp Mode 11 실패 → Mode 6 재시도")
                res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=6, rpy_tol=[6.28, 6.28, 6.28])
            if res != "success":
                self.get_logger().warn("  [OVERCOME] Pre-Grasp Mode 6 실패 → Mode 0 (OMPL) 재시도")
                res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=0, rpy_tol=[6.28, 6.28, 6.28])
            approach_success = (res == "success")
            self._step_time("overcome2: Pre-Grasp 접근", _t)
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
                time.sleep(0.05)
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
                self.get_logger().warn("  >> [OVERCOME] Pre-Grasp failed (All Angles). Skipping this target.")
                continue
                # fallback_joints = [1.274389, 0.598975, -0.289762, -0.500294, 0.076265, 0.44763, 0.03, 0.03]
                # res_joint = self.goal(fallback_joints, gripper_action='open')
                #
                # if res_joint:
                #     self.get_logger().info("  >> Success moving to Fallback Joints. Calculating 1cm Blind Push...")
                #     time.sleep(0.2) # Wait for TF update
                #
                #     curr_pose_mat = self.get_deep_pose()
                #     if curr_pose_mat is None:
                #         self.get_logger().error("  >> DEEP TF lookup 실패. 이 target 건너뜀.")
                #         continue
                #     curr_rot = curr_pose_mat[:3, :3]
                #     curr_pos = curr_pose_mat[:3, 3]
                #     self.get_logger().info(f"  >> DEEP pos(TF): [{curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}]")
                #
                #     push_vec = curr_rot @ np.array([0, 0, 0.01])
                #     target_wm.pose.position.x = float(curr_pos[0] + push_vec[0])
                #     target_wm.pose.position.y = float(curr_pos[1] + push_vec[1])
                #     target_wm.pose.position.z = float(curr_pos[2] + push_vec[2])
                #
                #     from scipy.spatial.transform import Rotation as ScipyRot
                #     r = ScipyRot.from_matrix(curr_rot)
                #     qx, qy, qz, qw = r.as_quat()
                #     target_wm.pose.orientation.x = qx
                #     target_wm.pose.orientation.y = qy
                #     target_wm.pose.orientation.z = qz
                #     target_wm.pose.orientation.w = qw
                #
                #     approach_success = True
                # else:
                #     self.get_logger().error("  >> Fallback Joint move failed. Skipping this target.")
                #     continue
            # 2. Move to Grasp Target (Push)
            self.get_logger().info("  [OVERCOME] Moving to Grasp Target...")
            p = target_wm.pose.position
            o = target_wm.pose.orientation
            self.get_logger().info(
                f"  [OVERCOME] target_wm pose:\n"
                f"    position   : x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}\n"
                f"    orientation: x={o.x:.4f}  y={o.y:.4f}  z={o.z:.4f}  w={o.w:.4f}"
            )
            _t = time.time()
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
            self._step_time("overcome2: Push 이동", _t)
            self.grasp_buffer = []
            self.grasp_raw_buffer = []
            self.is_recording = True
            _t = time.time()
            self.close_gripper()
            self._step_time("overcome2: 그리퍼 닫기", _t)
            self.get_logger().info("  [OVERCOME] Retreating to pre-grasp...")
            _t = time.time()
            # Mode 11: Cartesian 직선 (avoid_collisions=false) → push와 동일, 항상 부드러운 직선 경로
            ret = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=11, rpy_tol=[6.28, 6.28, 6.28])
            if ret != "success":
                self.get_logger().warn("  [OVERCOME] pre_grasp 복귀 실패 → Mode 6 재시도")
                ret = self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=6, rpy_tol=[6.28, 6.28, 6.28])
            if ret != "success":
                self.get_logger().warn("  [OVERCOME] pre_grasp 복귀 실패 → Mode 0 (OMPL) 재시도")
                self.move_forward(pre_grasp_pose, fallback_to_ik=True, disable_torque=True, ik_mode=0, rpy_tol=[6.28, 6.28, 6.28])
            self._step_time("overcome2: Pre-Grasp 복귀", _t)

            # 4. 촉각 기반 파지 성공 판단 (overcome2: rule-based only)
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
            
            # pre_grasp → ff → wm_end (물체 파지 중 → 기본 속도)
            _t = time.time()
            while not self.goal(self.ff, 'keep'):
                self.get_logger().warn("  >> ff 이동 실패 → 재시도 중")
                time.sleep(0.2)
            self._step_time("overcome2: ff 이동", _t)
            _t = time.time()
            while not self.goal(self.wm_end, 'keep'):
                self.get_logger().warn("  >> wm_end 이동 실패 → 재시도 중")
                time.sleep(0.2)
            self._step_time("overcome2: wm_end 이동", _t)
            self._step_time("overcome2: 루프 1회 완료", _t_overcome)
            self.get_logger().info("Z-축 정렬 감지는 생략합니다!!")
            if True:
                self.z_aligned = True

                # [USER REQUEST] Loop until Empty: Move to Drop -> Return to Scan
                self.get_logger().info("  >> [Loop] 세탁기(wm_end)로 이동하여 놓기")

                # 1. Move to Place (wm_end, 물체 파지 중 → 기본 속도)
                while not self.goal(self.wm_end, 'keep'):
                    self.get_logger().warn("  >> wm_end 이동 실패 → 재시도 중")
                    time.sleep(0.2)

                # 2. Release
                self.open_gripper()

                # 4. Continue Loop (Do NOT Break)
                self.contacted = False
                import time
                time.sleep(0.5)
                continue

            else:
                self.get_logger().warn(f" 실패 복구 잡기 실패 (Slip Detected: norm={self.norm:.3f,}, max_norm = {self.max_norm:.3f})")
                self.get_logger().warn(f"   (Criteria: norm>0.35, max norm > 0.5)")
                self.open_gripper()

                while not self.goal(self.wm_mid, 'keep', fast=True):  # 물체 없음 → 빠른 복귀
                     self.get_logger().warn("  >> wm_mid 복귀 실패 → 재시도")
                     time.sleep(0.2)
                import time
                time.sleep(0.5)
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
                time.sleep(0.2)

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
                time.sleep(0.2)

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
                    time.sleep(0.2)
                
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
                time.sleep(0.2)

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
                    time.sleep(0.2)

                # 2. Release
                self.open_gripper()

                # 3. Return to Scan Pose (wm_mid)
                self.get_logger().info("  >> [Loop] 다시 감지 위치(wm_mid)로 복귀")
                while not self.goal(self.wm_mid, 'keep'):
                    self.get_logger().warn("  >> wm_mid 이동 실패 → 재시도 중")
                    time.sleep(0.2)
                
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
                     time.sleep(0.2)
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
            time.sleep(0.2)


    def _step_time(self, label: str, t0: float) -> float:
        dt = time.time() - t0
        self.get_logger().info(f"[TIMER] {label:<45s} {dt:6.2f}s")
        self._timing_log.append((label, dt))
        return dt

    def _print_timing_summary(self):
        if not self._timing_log:
            return
        sep = "=" * 64
        lines = [sep, f"{'[ TIMING SUMMARY ]':^64}"]
        total = 0.0
        for label, dt in self._timing_log:
            lines.append(f"  {label:<48s} {dt:6.2f}s")
            if not label.startswith("===") and not label.startswith("overcome2: 루프"):
                total += dt
        lines.append("-" * 64)
        lines.append(f"  {'합계 (사이클 내 측정 구간)':<48s} {total:6.2f}s")
        lines.append(sep)
        self.get_logger().info("\n" + "\n".join(lines))
        self._timing_log = []

    def execute_sequence(self):
        import time
        # self.get_logger().info("Step 0: 문 여는 중")
        # self.open_door()
        
        
        # self.get_logger().info("Step 0.5: 배경 이미지(RGB/Depth) 자동 초기 캡처 진행")
        # while not self.goal(self.wm_mid, 'keep'):
        #     self.get_logger().warn("  >> wm_mid(배경 촬영용) 이동 실패 → 재시도 중")
        # import time
        # time.sleep(1.0) # 로봇 떨림 안정화 대기
        # self.get_logger().info("배경 촬영 중...")
        # self.target_get_wm(mode="failure_bg_eh")
        # self.get_logger().info("✅ 배경 이미지 캡처 완료.")

        # self.overcome2()

        while rclpy.ok():
            _t_cycle = time.time()

            while True:
                self.open_gripper()
                self.z_aligned = False
                self.get_logger().info("Step 1: 이니셜 포즈(ba_init)로 이동 중")
                _t = time.time()
                while not self.goal(self.ba_init, 'keep', fast=True):  # 물체 없음 → 빠른 이동
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                    time.sleep(0.2)
                self._step_time("ba_init 이동", _t)
                self.reset_sensor_data()
                import time
                self.get_logger().info("Step 2: 잡기 시도 중...")
                _t = time.time()
                target_ba = self.target_get_ba(mode="basket")  # 감지
                self._step_time("target_get_ba 비전 처리", _t)
                if target_ba.pose.position.z == -99.0:
                    self.get_logger().info("🛑 [Overcome] 빈 바구니 감지 (RGB Gate). overcome2 호출.")
                    self.overcome2()
                    break
            
                # --- Three-Stage Approach: 1. Approach Hover 2. Pre-Grasp 3. Push ---
                import copy
                
                # 1. Hover Pose (Safe Altitude above Target)
                # Pre-Grasp Pose (+5cm above Target)
                pre_grasp_pose = copy.deepcopy(target_ba)
                pre_grasp_pose.pose.position.z += 0.05  # 5cm higher than target

                # [VISUALIZATION] Publish Markers IMMEDIATELY (Before Move)
                self.get_logger().info("\n==================================================")
                self.get_logger().info(f"  [TARGET] BA (Grasp): \n{target_ba.pose}")
                target_ba_world = copy.deepcopy(target_ba.pose)
                target_ba_world.position.z += 0.445
                self.get_logger().info(f"  [WORLD] BA (Grasp) (Base+0.445): \n{target_ba_world}")
                self.get_logger().info("==================================================")
                self.publish_target_marker(target_ba.pose, id=0, r=1.0, g=0.0, b=0.0) # Red

                self.get_logger().info("\n==================================================")
                self.get_logger().info(f"  [PRE-TARGET] Pre-Grasp Pose (+5cm): \n{pre_grasp_pose.pose}")
                pre_grasp_world = copy.deepcopy(pre_grasp_pose.pose)
                pre_grasp_world.position.z += 0.445
                self.get_logger().info(f"  [WORLD] Pre-Grasp (Base+0.445): \n{pre_grasp_world}")
                self.get_logger().info("==================================================")
                self.publish_target_marker(pre_grasp_pose.pose, id=1, r=0.0, g=1.0, b=0.0) # Green

                # 1. Approach Pre-Grasp (Hover 생략)
                _t_pregrasp = time.time()
                self.get_logger().info("Step 2-1: 접근(Pre-Grasp) 하강 중... (Collision Check ON)")
                approach_success = False

                for i in range(3):
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=False, disable_torque=True, vel_scale=1.0)
                    if res == "success":
                        approach_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Linear)\033[0m")
                        break
                    self.get_logger().info(f"\033[94m  >> [Stage 1] 직선 하강 실패 ({i+1}/3) result={res} → 재시도\033[0m")

                # Stage 1 실패 시 Stage 2 시도: 엄격한 수직 IK (Mode 10, Env-aware)
                if not approach_success:
                    self.get_logger().info("\033[94m [Stage 2] 엄격한 수직 IK(Mode 10) 시도\033[0m")
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, ik_mode=10, disable_torque=True, vel_scale=1.0)
                    if res == "success":
                        approach_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Strict IK)\033[0m")

                # Stage 2 실패 시 Stage 3 시도: 수직 유지 완화 계획 (Mode 1, ±15도 허용)
                # Mode 1: OMPL Planner를 사용하여 목표 위치(XYZ)에 도달하되,
                # 수직 방향에서 상하좌우 ±30도 이내의 기울기를 허용하여 충돌을 회피하는 경로를 찾습니다.
                # if not approach_success:
                #     self.get_logger().info("\033[94m [Stage 3] 수직 유지 완화(Mode 1) 계획 시도\033[0m")
                #     # roll_tol, pitch_tol = 0.52 rad (~30 deg), yaw_tol = 6.28 (자유)
                #     res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, ik_mode=1, rpy_tol=[0.52, 0.52, 6.28], disable_torque=True, vel_scale=1.0)
                #     if res == "success":
                #         approach_success = True
                #         self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달! (Relaxed Orientation)\033[0m")

                # Stage 4: orientation 완전 자유 (path constraint 없음)
                # d435i가 벽을 피하는 joint6 회전을 OMPL이 자유롭게 탐색하도록 허용
                if not approach_success:
                    self.get_logger().info("\033[94m [Stage 4] Orientation 완전 자유(Mode 1, no path constraint) 시도\033[0m")
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=True, ik_mode=1, rpy_tol=[6.28, 6.28, 6.28], disable_torque=True, vel_scale=1.0)
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
                    self._step_time("Pre-Grasp 접근 (실패)", _t_pregrasp)
                    continue

                self._step_time("Pre-Grasp 접근 (성공)", _t_pregrasp)

                # 3. Push (Actual Grasp Move)
                self.get_logger().info("Step 2-3: 잡기(Push) 이동 중...")
                _t = time.time()
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
                    self._step_time("Push 이동 (실패)", _t)
                    continue     # 다음 루프로

                self._step_time("Push 이동 (성공)", _t)
                _t = time.time()
                self.close_gripper()
                self._step_time("그리퍼 닫기", _t)

                # [DATA COLLECTION] 녹화 시작 (collect / inference 모드)
                if self.run_mode in ('collect', 'inference'):
                    # cum_x/y/z 리셋: 접근 경로 누적값을 제거하고
                    # 순수한 접촉+retreat 동역학만 포착
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
                    self.grasp_detect_channel        = 'none'
                    self.grasp_channel_b_flag        = False
                    self.grasp_channel_b_consec      = 0
                    self.grasp_delta_b_latest        = 0.0
                    if self.grasp_channel_b is not None:
                        self.grasp_channel_b.reset()
                    if self.grasp_api_detector is not None:
                        self.grasp_api_detector.reset()
                    self.is_recording = True
                    self.get_logger().info(f"[DATA] 녹화 시작 (mode={self.run_mode})")

                # --- Retreat (바구니 없는 환경에서는 생략 가능) ---
                # _t = time.time()
                # for i in range(5):
                #     if self.move_backward(target_ba): # Using new hybrid move_backward
                #         break
                #     self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")
                # self._step_time("후퇴 (move_backward)", _t)

                _t = time.time()
                while not self.goal(self.ba_init, 'keep'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                    time.sleep(0.2)
                self._step_time("ba_init 복귀", _t)

                # [DATA COLLECTION] 녹화 종료
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

                    # [DEBUG] inference 모드에서 raw 버퍼 저장 (학습 데이터 비교용, 나중에 제거)
                    if self.run_mode == 'inference' and len(self.grasp_raw_buffer) > 0:
                        import datetime
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        dbg_path = f"/home/dyros/mcy_ws/grasp_05/infer_debug_{ts}_raw.npy"
                        np.save(dbg_path, np.array(self.grasp_raw_buffer, dtype=np.float32))
                        self.get_logger().info(f"[DEBUG] inference raw 저장: {dbg_path} shape={np.array(self.grasp_raw_buffer).shape}")

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
                    # ba_init에서 파지 성공 여부 즉시 판정 → Model 1 레이블 확정
                    held = self._ask_gripper_held()
                    if held:
                        # 파지 성공 → Model 1 success 즉시 저장 후 이송 진행
                        self._save_grasp_data(label='success')
                        self.z_aligned = True
                        break
                    elif held is False:
                        # 파지 실패 → Model 1 failure 저장 후 재시도
                        self._save_grasp_data(label='failure')
                        while not self.open_gripper():
                            self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
                    else:
                        # skip → rule-based fallback
                        if rule_success:
                            self.z_aligned = True
                            break
                        else:
                            while not self.open_gripper():
                                self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")

                elif self.run_mode == 'inference':
                    # 슬라이딩 윈도우 실시간 판단 결과 사용 (lock으로 스레드 안전 읽기)
                    # eval 기준과 동일: recording 중 연속 이상 consec_anomaly_needed 회 미달성 → 성공
                    with self._infer_lock:
                        anom_detected = self.infer_grasp_anomaly_detected
                        score         = self.infer_latest_score
                        detect_ch     = self.grasp_detect_channel
                    # 슬라이딩 윈도우가 한 번도 실행되지 않은 경우 → rule-based fallback
                    if score is None:
                        self.get_logger().warn(
                            f"[INFERENCE] 시퀀스 짧음 ({len(self.grasp_buffer)}프레임 < 50) "
                            f"→ rule-based fallback (rule_success={rule_success})")
                        if rule_success:
                            self.get_logger().info("[INFERENCE] ✅ 파지 성공 판정 (rule-based fallback)")
                            self.z_aligned = True
                            break
                        else:
                            self.get_logger().warn("[INFERENCE] ❌ 파지 실패 판정 (rule-based fallback) → 재시도")
                            while not self.open_gripper():
                                self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
                            continue

                    score_str = f"{score:.4f}" if score is not None else "None"
                    self.get_logger().info(
                        f"[INFERENCE] anomaly_detected={anom_detected} "
                        f"(consec_needed={self.infer_consec_anomaly_needed}), "
                        f"최종 score={score_str}")
                    if not anom_detected:
                        self.get_logger().info("[INFERENCE] ✅ 파지 성공 판정")
                        self.z_aligned = True
                        break
                    else:
                        ch_desc = {'A': 'Channel A (GRU-AE)',
                                   'B': 'Channel B (MRL delta)',
                                   'AB': 'Channel A + B (둘 다)'}.get(detect_ch, detect_ch)
                        self.get_logger().warn(
                            f"[INFERENCE] ❌ 파지 실패 판정 → 재시도  |  감지 채널: {ch_desc}  |  score_A={score_str}")
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
                self._print_timing_summary()
                break
            
            self.get_logger().info("Step 3: 엔드 포즈(wm_end)로 이동 중")
            self.contacted = False
            
            # Start Monitoring (inference 모드에서는 GRU-AE가 담당하므로 rule-based 비활성화)
            self.monitor_initial_norm = self.norm
            self.monitor_initial_theta = self.theta
            self.monitor_slip = (self.run_mode != 'inference')
            self.get_logger().info(f"👀 Slip Monitoring START (Ref: norm={self.norm:.2f}, theta={self.theta:.1f})")
            self.get_logger().info(f"👀 Slip Monitoring START (Ref: norm={self.norm:.2f}, theta={self.theta:.1f})")
            
            # [DATA] Model 2 녹화 시작: ba_init 이후 이송 전 구간 (ba_init → open_gripper)
            if self.run_mode in ('collect', 'inference'):
                # cum_x/y/z 리셋: 그리퍼 접촉+retreat 누적값 제거,
                # 이송 구간의 순수한 접촉 동역학만 포착
                self.reset_sensor_data()
                self.insert_buffer = []
                self.insert_raw_buffer = []
                self.insert_realtime_buffer.clear()
                self.insert_max_mrl_seen              = 0.0
                self.insert_infer_frame_cnt           = 0
                self.insert_infer_consec_normal       = 0
                self.insert_infer_consec_anomaly      = 0
                self.insert_infer_latest_score        = None
                self.insert_anomaly_ever_detected     = False
                self.transport_detect_channel         = 'none'
                self.transport_channel_b_flag         = False
                self.transport_channel_b_consec       = 0
                self.transport_delta_b_latest         = 0.0
                if self.transport_channel_b is not None:
                    self.transport_channel_b.reset()
                if self.use_tactile_api and self.insert_api_detector is not None:
                    self.insert_api_detector.reset()
                self.is_insert_recording = True
                self.get_logger().info("[DATA] 이송 구간 녹화 시작 (ba_init → open_gripper)")

            self._transport_abort.clear()

            _t = time.time()
            while not self.goal(self.wm_init, 'keep'):
                if self._transport_abort.is_set():
                    break
                self.get_logger().warn("  >> 실패 → 재시도 중")
                time.sleep(0.2)
            self._step_time("wm_init 이동", _t)

            # wm_init 도달 시점에 이상 감지 종료 (wm_end 이동은 감지 없이 진행)
            abort_at_wm_init = self._transport_abort.is_set()
            if self.run_mode == 'inference':
                self.is_insert_recording = False
                self.get_logger().info(f"[DATA] 이송 구간 녹화 종료 (wm_init 도달): {len(self.insert_buffer)} 프레임")

            # 낙하 감지로 중단된 경우: joint1 조건 확인 후 재파지 또는 바구니 복귀
            if abort_at_wm_init and self.run_mode == 'inference':
                self.get_logger().warn("[INFERENCE] 낙하 감지 → 재파지 조건 확인")
                time.sleep(0.5)  # MoveIt/JTC 완료 대기 (status 6 방지)
                self._transport_abort.clear()  # _send_goal abort 체크 해제 후 그리퍼 명령
                # 그리퍼 오픈 retry (arm 컨트롤러 상태 불안정 시 첫 시도 묻힐 수 있음)
                for _retry in range(3):
                    if self.open_gripper():
                        break
                    self.get_logger().warn(f"[RECOVERY] 그리퍼 오픈 실패 → 재시도 ({_retry+1}/3)")
                    time.sleep(0.3)

                j1_now = self.current_joint_state[0] if self.current_joint_state else 999.0
                if j1_now <= 0.5:
                    self.get_logger().warn(f"[RECOVERY] joint1={j1_now:.3f} ≤ 0.5 → 바닥 재파지 시도")
                    recovered = self._floor_recovery_grasp()
                    if recovered:
                        self.get_logger().info("[RECOVERY] 재파지 성공 → 이송 재개")
                        self._transport_abort.clear()
                        while not self.goal(self.wm_init, 'keep'):  # 물체 파지 중 → 기본 속도
                            if self._transport_abort.is_set():
                                break
                            self.get_logger().warn("  >> [RECOVERY] wm_init 이동 실패 → 재시도")
                            time.sleep(0.2)
                        if not self._transport_abort.is_set():
                            while not self.goal(self.wm_end, 'keep'):  # 물체 파지 중 → 기본 속도
                                if self._transport_abort.is_set():
                                    break
                                self.get_logger().warn("  >> [RECOVERY] wm_end 이동 실패 → 재시도")
                                time.sleep(0.2)
                        if not self._transport_abort.is_set():
                            self.open_gripper()
                            time.sleep(0.3)
                        # 정상 경로와 동일하게 wm_init 경유 후 복귀
                        self.get_logger().info("[RECOVERY] wm_init 경유 후 ba_init 복귀")
                        while not self.goal(self.wm_init, 'keep', fast=True):  # 물체 없음 → 빠른 복귀
                            self.get_logger().warn("  >> [RECOVERY] wm_init 복귀 실패 → 재시도")
                            time.sleep(0.2)
                    else:
                        self.get_logger().warn("[RECOVERY] 재파지 실패 → 바구니로 복귀")
                        while not self.goal(self.ba_init, 'keep', fast=True):  # 물체 없음 → 빠른 복귀
                            self.get_logger().warn("  >> ba_init 복귀 실패 → 재시도 중")
                            time.sleep(0.2)
                else:
                    self.get_logger().warn(f"[RECOVERY] joint1={j1_now:.3f} > 0.5 → 재파지 생략, 바구니로 복귀")
                    while not self.goal(self.ba_init, 'keep', fast=True):  # 물체 없음 → 빠른 복귀
                        self.get_logger().warn("  >> ba_init 복귀 실패 → 재시도 중")
                        time.sleep(0.2)
                self._step_time("=== 낙하 복구 사이클 완료", _t_cycle)
                # summary는 overcome2 종료 시(3742)에 모든 cycle 누적 후 한번에 출력
                continue

            # wm_init 이후: abort 클리어 후 wm_end로 이동 (이상 감지 없음)
            self._transport_abort.clear()
            _t = time.time()
            while not self.goal(self.wm_end, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
                time.sleep(0.2)
            self._step_time("wm_end 이동", _t)

            # [DATA] collect 모드 녹화 종료 (wm_end 도달 후, open_gripper 직전)
            if self.run_mode == 'collect':
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
                    detect_ch      = self.transport_detect_channel
                insert_ok = (not anomaly_ever) and (insert_consec >= insert_needed)
                score_str = (f"{insert_score:.6f}"
                             if insert_score is not None else "N/A")
                self.get_logger().info(
                    f"[INFERENCE] 이송 판정: {'정상' if insert_ok else '이상'} "
                    f"(anomaly_ever={anomaly_ever}, consec={insert_consec}/{insert_needed}, "
                    f"latest_score={score_str})"
                )
                if not insert_ok:
                    ch_desc = {'A': 'Channel A (GRU-AE)',
                               'B': 'Channel B (MRL delta)',
                               'AB': 'Channel A + B (둘 다)'}.get(detect_ch, detect_ch)
                    self.get_logger().warn(
                        f"[INFERENCE] ❌ 이송 실패 판정 → 극복 동작 트리거  |  감지 채널: {ch_desc}  |  score_A={score_str}")
                    self.contacted = True

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

            _t = time.time()
            while not self.goal(self.wm_init, 'keep', fast=True):  # 물체 내려놓은 후 → 빠른 복귀
                self.get_logger().warn("  >> 실패 → 재시도 중")
                time.sleep(0.2)
            self._step_time("wm_init 최종 복귀", _t)
            self._step_time("=== 전체 사이클 완료", _t_cycle)
            # summary는 overcome2 종료 시(3731)에 모든 cycle 누적 후 한번에 출력
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
    parser.add_argument('--use_tactile_api', action='store_true',
                        help='inference 모드에서 tactile_anomaly.DualChannelDetector API 사용')
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args if ros_args else args)

    try:
        node = ControlTower(run_mode=parsed.mode, use_tactile_api=parsed.use_tactile_api)
    except RuntimeError as e:
        print(f"노드 초기화 중 오류 발생: {e}")
        return

    # MultiThreadedExecutor: 메인 스레드에서 지속적으로 spin → 항상 콜백 처리
    # (sensor_callback, joint_state_callback 등이 spin_until_future_complete 유무와 무관하게 동작)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    # 메인 로직을 실행할 작업 스레드
    sequence_thread = threading.Thread(target=node.execute_sequence, daemon=True)
    sequence_thread.start()

    try:
        # 메인 스레드에서 executor를 지속 spin → 콜백 항상 처리
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                node.get_logger().warn(f"Spin 오류 (계속 진행): {e}")
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 중단 요청됨.')
    finally:
        node.get_logger().info('노드 종료 중...')
        executor.shutdown()
        if rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == '__main__':
    main()

