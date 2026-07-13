
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
from std_msgs.msg import Bool, Float32MultiArray
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

from basket_components.constants import (
    URDF_OR_XACRO, URDF_JOINT_NAMES, CTRL_JOINT_NAMES,
    DISPLAY_IDX, GRAVITY, PRED_RATE_HZ, SIGMA, FOUR_SIGMA, A, B,
    build_pin_model
)
from basket_components.motion_mixin import MotionMixin
from basket_components.sensor_mixin import SensorMixin
from basket_components.inference_mixin import InferenceMixin
from basket_components.data_mixin import DataMixin
from basket_components.vision_mixin import VisionMixin
from basket_components.recovery_mixin import RecoveryMixin
from basket_components.task_mixin import TaskMixin


class ControlTower(
    TaskMixin, RecoveryMixin, VisionMixin, DataMixin,
    InferenceMixin, SensorMixin, MotionMixin, Node
):
    def __init__(self, run_mode: str = 'normal', use_tactile_api: bool = False):
        super().__init__('control_tower_node')
        self.run_mode = run_mode
        self.use_tactile_api = use_tactile_api
        self.get_logger().info(f"[MODE] 실행 모드: {self.run_mode}, use_tactile_api={self.use_tactile_api}")

        self.ee_pose = None
        self.wm_init_ee_orientation = None
        self.ee_pose_st = PoseStamped()
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

        self.norm = 0.0
        self.theta = 0.0
        self.max_norm = 0.0

        self.minmax_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/taxel_minmax_norm.npy"
        self.gmin, self.grng = self._load_minmax(self.minmax_path)
        self.get_logger().info(f"MinMax loaded: {self.minmax_path}")

        self.monitor_slip = False
        self.monitor_initial_norm = 0.0
        self.monitor_initial_theta = 0.0
        self.current_goal_handle = None

        self.is_recording = False
        self.grasp_buffer = []
        self.grasp_raw_buffer = []
        self.mrl = 0.0
        self.data_save_dir = "/home/dyros/mcy_ws/grasp_05"
        os.makedirs(self.data_save_dir, exist_ok=True)
        self.door_bag_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/door_open_bag"
        self.door_bag_hz   = 20.0
        self.door_bag_speed = 1.0

        self.is_insert_recording = False
        self.insert_buffer = []
        self.insert_raw_buffer = []
        self.insert_save_dir = "/home/dyros/mcy_ws/transport_05"
        os.makedirs(self.insert_save_dir, exist_ok=True)

        # Model 1 (grasp) GRU-AE + 채널 B
        self.gru_ae_model      = None
        self.gru_ae_scaler     = None
        self.gru_ae_threshold  = 0.02
        self.gru_ae_threshold_b = 0.5
        self.gru_ae_infer_cfg  = {"relative": "none", "weight_alpha": 0.0}
        self.grasp_channel_b   = None

        # Model 2 (transport) GRU-AE + 채널 B
        self.insert_gru_ae_model     = None
        self.insert_gru_ae_scaler    = None
        self.insert_gru_ae_threshold = 0.02
        self.insert_gru_ae_threshold_b = 0.5
        self.insert_gru_ae_infer_cfg = {"relative": "none", "weight_alpha": 0.0}
        self.transport_channel_b     = None

        self.grasp_api_detector = None

        if self.run_mode == 'inference':
            self._load_gru_ae()
            self._load_insert_gru_ae()
            if self.use_tactile_api:
                self._load_grasp_api_detector()

        # Sliding window Model 1 (grasp)
        self.realtime_buffer     = deque(maxlen=50)
        self.realtime_raw_buffer = deque(maxlen=50)
        self.inference_stride    = 5
        self.inference_frame_cnt = 0
        self.infer_consec_normal         = 0
        self.infer_consec_needed         = 3
        self.infer_consec_anomaly        = 0
        self.infer_consec_anomaly_needed = 2
        self.infer_latest_score          = None
        self.infer_grasp_anomaly_detected = False
        self.grasp_channel_b_flag        = False
        self.grasp_channel_b_consec      = 0
        self.grasp_channel_b_consec_needed = 5
        self.grasp_delta_b_latest        = 0.0
        self.grasp_channel_b_warmup_needed = 30

        # Sliding window Model 2 (transport)
        self.insert_realtime_buffer           = deque(maxlen=50)
        self.insert_realtime_raw_buffer       = deque(maxlen=50)
        self.insert_infer_frame_cnt           = 0
        self.insert_infer_consec_normal       = 0
        self.insert_infer_consec_needed       = 3
        self.insert_infer_consec_anomaly      = 0
        self.insert_infer_consec_anomaly_needed = 3
        self.insert_infer_latest_score        = None
        self.insert_anomaly_ever_detected     = False
        self._transport_abort                 = threading.Event()
        self.transport_channel_b_flag         = False
        self.transport_channel_b_consec       = 0
        self.transport_channel_b_consec_needed = 4
        self.transport_channel_b_warmup_needed = 0
        self.transport_delta_b_latest         = 0.0

        self._raw_sensor_queue = queue.Queue(maxsize=200)
        self._sensor_proc_thread = threading.Thread(
            target=self._sensor_processing_worker, daemon=True)
        self._sensor_proc_thread.start()

        self._infer_lock = threading.Lock()
        self._infer_thread_running = True
        self._infer_trigger    = threading.Event()
        self._insert_trigger   = threading.Event()
        self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._infer_thread.start()

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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        from piper_msgs.srv import Enable
        self._enable_client = self.create_client(Enable, '/enable_srv')

        self.marker_pub = self.create_publisher(Marker, 'grasp_target_marker', 10)

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

        self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.gripper_joint_names = ['joint7', 'joint8']

        # [D405] Camera Extrinsics
        self.R_cal = np.array([
            [-0.084547, 0.992441, -0.088956],
            [-0.995873, -0.081206, 0.040533],
            [0.033003, 0.092016, 0.995210]
        ])
        self.t_cal = np.array([[-0.050437],
                               [0.007361],
                               [0.061211]])

        self.ba_init = [-1.6379, 1.5724, -1.2721, -0.0566, 1.2838, 0.2908, 0.0, 0.0]
        self.wm_init = [1.5640, 0.4538, -1.5319, -0.0488, 1.2797, 0.3169, 0.0, 0.0]
        self.ba_end = [-1.5527, 2.6613, -1.5133, -0.2642, 0.6119, 0.3951, 0.0, 0.0]
        self.wm_end = [1.571, 2.146, -2.268, 0.000, 0.594, 0.000, 0.0, 0.0]
        self.wm_mid_original = [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -1.357858404, 0.0, 0.0]
        self.wm_mid = [1.569035, 0.043279, 0.026288, -0.075341, 0.159264, 0.256235, 0.0, 0.0]
        self.wm_view = [1.562546, 0.641172, -1.202764, -0.089156, 1.286792, -1.478571, 0.0, 0.0]
        self.home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.f = [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -0.1, 0.0, 0.0]
        self.ff = [1.5533, 0.1396, -0.4014, 0.0349, 0.3665, 0.1745, 0.0, 0.0]

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

        self.create_timer(1.0 / PRED_RATE_HZ, self.tick)
        self.camera_timer = self.create_timer(0.5, self.publish_camera_center_callback)

        self._lock = threading.Lock()
        self.paused = False

        self.stop_requested = False
        self.stop_reason = ""
        self.last_forward_pose = None
        self.recover_pending = False
        self.monitor_torque = False
        self.torque_stop = False


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

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    sequence_thread = threading.Thread(target=node.execute_sequence, daemon=True)
    sequence_thread.start()

    try:
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
