import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.clock import Clock
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
import sys, termios, tty, select
from scipy.spatial.transform import Rotation as R
from rclpy.clock import Clock
from collections import deque
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState

import math
from pathlib import Path
import pinocchio as pin


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

SIGMA = 0.0856
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

    def __init__(self):
        super().__init__('control_tower_node')

        self.current_label = "NC"
        self._label_lock = threading.Lock()

        self.recording = False
        self.xserv_log = {"frames": []}
       
        self.scenario = ""
        self.tag = ""
       
        self._key_thread_alive = True
        threading.Thread(target=self._keyboard_loop, daemon=True).start()
       
        self.ee_pose = None
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
        self.theta = None

        self.minmax_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/taxel_minmax_norm.npy"
        self.minmax_path = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/taxel_minmax_norm.npy"
        self.gmin, self.grng = self._load_minmax(self.minmax_path)
        self.get_logger().info(f"MinMax loaded: {self.minmax_path}")
       
        # [USER REQUEST] Continuous Monitoring Variables
        self.monitor_slip = False
        self.monitor_initial_norm = 0.0
        self.monitor_initial_theta = 0.0
        self.current_goal_handle = None
        self._n = 0
        self._t0 = self.get_clock().now().nanoseconds * 1e-9

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
        self._gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        self._pose_client = ActionClient(self, PoseGoal, 'pose_goal')
       
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

        self.ba_init = [-1.6379, 1.5724, -1.2721, -0.0566, 1.2838, 0.2908, 0.0, 0.0]
        self.wm_init = [1.5640, 0.4538, -1.5319, -0.0488, 1.2797, 0.3169, 0.0, 0.0]
        self.ba_end = [-1.5527, 2.6613, -1.5133, -0.2642, 0.6119, 0.3951, 0.0, 0.0]
        self.wm_end = [1.571, 2.146, -2.268, 0.000, 0.594, 0.000, 0.0, 0.0]
        self.wm_mid_original = [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -1.357858404, 0.0, 0.0]
        self.wm_mid= [1.5058533, 0.30410125200000004, -0.15608891200000002, -0.15819963600000003, 0.0684677, -1.267376376,0.0, 0.0]
        self.wm_view = [1.535473212,1.2058165,-1.5244486040000003,-0.045633504000000005, 1.271702488,-1.3131843200000002,0.0,0.0]
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

        # paused 플래그/락
        self._lock = threading.Lock()
        self.paused = False

        self.stop_requested = False
        self.stop_reason = ""
        self.last_forward_pose = None   # move_forward에서 저장할 pose
        self.recover_pending = False
        self.monitor_torque = False   # ✅ forward 구간에서만 True
        self.torque_stop = False


    def ee_pose_callback(self, msg):
        pos = msg.position
        ori = msg.orientation
        r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = [pos.x, pos.y, pos.z]
        self.ee_pose = T

    def _keyboard_loop(self):
        '''
        터미널에서 키를 상시 감지해서 current_label을 바꿈
        g -> G, n -> NC, c -> C, w -> WG
        '''
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd) #현재 터미널 설정을 백업. 함수 종료 시 원래 상태로 백업하지 않으면 터미널 망가짐
        try:
            tty.setcbreak(fd) # 터미널을 cbreak 모드로 전환해서 키를 엔터 없이 즉시 읽을 수 있음
            while self._key_thread_alive and rclpy.ok(): # 키 스레드가 살아있고 rclpy 가 잘 작동 중일 때
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                k = ch.lower(); new_label = None
                if k == 'g':
                    new_label = 'G'
                elif k == 'n':
                    new_label = 'NC'
                elif k == 'c':
                    new_label = 'C'
                elif k == 'w':
                    new_label = 'WG'
                if new_label is not None:
                    with self._label_lock:
                        self.current_label = new_label
                    self.get_logger().warn(f"[LABEL] -> {new_label} (key='{k}')")
        except Exception as e:
            self.get_logger().error(f"keyboard loop error: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old) #터미널 설정 복원
               
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
        duration_sec = 5.0 # Default fallback
           
        try:
            # Calculate max joint difference if current state is available
            if hasattr(self, 'current_joint_state') and self.current_joint_state is not None:
                # joint_values is target
                # self.current_joint_state is current list
                # Order is assumed same if length matches
                if len(joint_values) == len(self.current_joint_state):
                    diffs = [abs(t - c) for t, c in zip(joint_values, self.current_joint_state)]
                    max_diff = max(diffs)
                       
                    # Speed Policy:
                    # Small move (rescue): 0.05 rad -> want ~0.5s (0.1 rad/s)
                    # Large move (home): 3.0 rad -> want ~5.0s (0.6 rad/s)
                   
                    # Set min duration 0.5s, target speed ~0.6 rad/s
                    req_duration = max_diff / 0.6
                    duration_sec = max(0.8, req_duration)
                   
                    # Cap max duration to avoid boredom? No, safety first.
                    # Check specific small move case
                    if max_diff < 0.1:
                        duration_sec = 0.8
                    elif max_diff > 4.0: # Very large
                        duration_sec = 8.0
                       
                    # self.get_logger().info(f"  [Move] Max Diff={max_diff:.3f} rad, Duration set to {duration_sec:.2f}s")
        except Exception as e:
            self.get_logger().warn(f"Duration calc failed: {e}")
            duration_sec = 5.0

        if set(joint_names) == set(self.gripper_joint_names):
            point.time_from_start = Duration(sec=2, nanosec=0)
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
        else:
            self.get_logger().warn(f"Action '{client._action_name}' failed with status: {result.status}")
            # return False

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
       

    def move_forward(self, pose_st: PoseStamped, fallback_to_ik=True):
        """
        Hybrid Move Forward (Mode 9 -> Mode 6)
        with 4-sigma Torque Stop Protection.
        Returns: "success", "stop", "fail"
        """
        self.get_logger().info("[forward] Starting Hybrid Move (Mode 9 -> Mode 6)")
       
        # --- Internal Helper to run a specific mode ---
        def run_mode(mode_val):
            # 1. Reset Stop Flags & Enable Torque Monitor
            with self._lock:
                self.stop_requested = False
                self.stop_reason = ""
                self.torque_stop = False
                self.monitor_torque = True
             
            goal_msg = PoseGoal.Goal()
            goal_msg.target_pose = pose_st
            goal_msg.mode = mode_val
             
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
             
            # 3. Wait Result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=20.0) # Enough time
           
            if not result_future.done():
                self.get_logger().error(f"[forward] mode={mode_val} result wait timeout")
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
            self.get_logger().warn(f"Mode 9 result='{res9}'. Retrying with Mode 6 (Joint IK)...")
            # --- Step 2. Try Mode 6 (Joint) ---
            res6 = run_mode(6)
            return res6
        else:
            self.get_logger().error(f"Mode 9 result='{res9}'. Fallback disabled -> Abort.")
            return "fail"



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
       
        # move_forward 재활용 (Mode 9 -> Mode 6 Fallback 포함됨)
        res = self.move_forward(retreat_pose)
       
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

    def save_xserv_data(self):
        '''
        xserv_log 구조 :
        {
            "frames": [ (timestamp, label, feat_3x4x4), ... ],
            "scenario": integer,
            "tag": string
        }
        '''
        import json
        import pickle
        from datetime import datetime
        from pathlib import Path
        import numpy as np
        from collections import Counter
       
        self.save_dir = (
            Path.home()
            / "mcy_ws"
            / "piper-mou"
            / "src"
            / "piper_ros"
            / "src"
            / "piper_moveit"
            / "piper_with_gripper_moveit"
            / "src"
            / "data"
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_dir = self.save_dir
       
        frames = self.xserv_log.get("frames", None)
        if frames is None:
            self.get_logger().error("[SAVE] xserv_log['frames'] is missing. Abort.")
            return False
        if len(frames) == 0:
            self.get_logger().error("[SAVE] frames is empty.")
            return False
       
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = self.xserv_log.get("scenario", -1)
        tag = self.xserv_log.get("tag", "")
        # 파일명에 들어갈 tag sanitize
        safe_tag = "".join([c if c.isalnum() or c in "_-" else "_" for c in str(tag)])
        if safe_tag:
            base_name = f"{now}_S{scenario}_{safe_tag}"
        else:
            base_name = f"{now}_S{scenario}"
        base_path = save_dir / base_name
        meta = dict(self.xserv_log)
        meta.pop("frames", None)
        meta["n_frames"] = len(frames)
        meta["savd_at"] = now
        t_arr = np.array([float(f[0]) for f in frames], dtype=np.float64)
        lab_arr = np.array([f[1] for f in frames], dtype=object)
        feat_list = []
        for i, f in enumerate(frames):
            feat = np.asarray(f[2], dtype=np.float32)
            if feat.shape != (3,4,4):
                raise ValueError(f"Frame[{i}] feat shape {feat.shape} != (3,4,4)")
            feat_list.append(feat)
        x_arr = np.stack(feat_list, axis=0)
        np.savez_compressed(
            str(base_path) + ".npz",
            t=t_arr,
            label=lab_arr,
            x = x_arr,
            meta = json.dumps(meta, ensure_ascii=False)
        )
        with open(str(base_path) + ".json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
           
        # LABEL COUNT CHECK
        label_counts = Counter(lab_arr)
        self.get_logger().info("[LABEL CHECK]")
        for lbl, count in label_counts.items():
            self.get_logger().info(f"{{ {lbl} : {count} }}")

        self.get_logger().warn(
            f"[SAVE] OK -> {base_path}.npz  (frames={len(frames)}, scenario={scenario}, tag='{tag}')"
        )

        return True

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
                   
            if getattr(self, "recording", False):
                feat_3x4x4 = np.zeros((3,4,4), dtype=np.float32)
                for i in range(16):
                    r = i // 4
                    c = i % 4
                    feat_3x4x4[0, r, c] = x_vals[i]
                    feat_3x4x4[1, r, c] = y_vals[i]
                    feat_3x4x4[2, r, c] = z_vals[i]
                t = self.get_clock().now().nanoseconds * 1e-9
                with self._label_lock:
                    lab = self.current_label
                t = self.get_clock().now().nanoseconds * 1e-9
                self.xserv_log["frames"].append((t, lab, feat_3x4x4))
            # t = self.get_clock().now().nanoseconds * 1e-9
            # self._n += 1
            # dt = t - self._t0
            # if dt >= 1.0:
            #     hz = self._n / dt
            #     self.get_logger().info(f"callback rate: {hz:.1f} Hz over {dt:.2f}s")
            #     self._n = 0
            #     self._t0 = t

            # 1. Noise Filter (If force is too small, ignore norm)
            # if max_norm < 0.5:
            #     self.z_aligned = False
            #     self.contacted = False
            # # 2. Heuristic Logic
            # elif norm < 0.35:
            #     self.get_logger().info(f" [Tactile] 0.35 이하 norm: {norm:.3f}, max_norm: {max_norm:.1f}, theta: {theta_deg:.1f}")
            #     self.z_aligned = False
            #     self.contacted = False
            # else:
                # self.get_logger().info(f" [Tactile] 0.35 이상 norm: {norm:.3f}, max_norm: {max_norm:.1f}, theta: {theta_deg:.1f}")
                # if theta_deg < 70.0:
                # if norm < 1.0:
                #     if theta_deg < 55.0:
                #         # Vertical Force -> Z-Aligned
                #         if not self.z_aligned:
                #             self.get_logger().info(f"\n⚡ [TACTILE] Z-Aligned! (θ={theta_deg:.1f}°)")
                #         self.z_aligned = True
                #         self.contacted = False
                   
                #     # elif theta_deg <= 110.0:
                #     elif theta_deg <= 80.0:
                #         # Side Force -> Contact
                #         if not self.contacted:
                #             self.get_logger().info(f"\n⚡ [TACTILE] Contact Detected! (θ={theta_deg:.1f}°)")
                #         self.contacted = True
                #         self.z_aligned = False
                #     else:
                #         self.z_aligned = False
                #         self.contacted = False
               
                # else:
                #     if theta_deg < 35.0:
                #         # Vertical Force -> Z-Aligned
                #         if not self.z_aligned:
                #             self.get_logger().info(f"\n⚡ [TACTILE] Z-Aligned! (θ={theta_deg:.1f}°)")
                #         self.z_aligned = True
                #         self.contacted = False
                   
                #     # elif theta_deg <= 110.0:
                #     elif theta_deg <= 80.0:
                #         # Side Force -> Contact
                #         if not self.contacted:
                #             self.get_logger().info(f"\n⚡ [TACTILE] Contact Detected! (θ={theta_deg:.1f}°)")
                #         self.contacted = True
                #         self.z_aligned = False
                #     else:
                #         self.z_aligned = False
                #         self.contacted = False

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
    
    def target_get_wm_eh(self, mode="failure_eh", first=True): ### basket, wm, mid
        # 액션 클라이언트가 처음이면 생성
        if not hasattr(self, 'action_client'):
            self.action_client = ActionClient(self, TargetPose, 'target_pose')

        # 서버 대기
        if not self.action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('target_pose 액션 서버를 찾을 수 없습니다.')
            return None

        # Goal 메시지 생성
        goal_msg = TargetPose.Goal()
        goal_msg.start = True  # 요청 트리거
        goal_msg.mode = mode

        # Goal 보내기
        self.get_logger().info('target_pose 요청 전송')
        future = self.action_client.send_goal_async(goal_msg)
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

            ee2cam = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

            base2cam = self.ee_pose @ ee2cam ## O2C
           

            pose = self.transform_pose_with_offset(result.target_pose, base2cam) ## O2t
            self.get_logger().info(f"변환한 bA: {pose}") # bA

            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
           
            # [USER REQUEST] Limit Minimum Z Height
            MIN_Z_LIMIT = 0.2 # 20cm
            MAX_Y_LIMIT = 0.33
            if pose_stamped.pose.position.z < MIN_Z_LIMIT:
                self.get_logger().warn(f"  >> Target Z ({pose_stamped.pose.position.z:.3f}) is too low! Clamping to {MIN_Z_LIMIT}")
                pose_stamped.pose.position.z = MIN_Z_LIMIT

       
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

            # [VISUALIZATION] Overcome Target Marker (Blue)
            # self.publish_target_marker(pose_stamped.pose, id=2, r=0.0, g=0.0, b=1.0)

            while not self.goal(self.ff, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")

            return pose_stamped
        else:

            return None


    def target_get_wm(self, mode="failure", first=True): ### basket, wm, mid
        # 액션 클라이언트가 처음이면 생성
        if not hasattr(self, 'action_client'):
            self.action_client = ActionClient(self, TargetPose, 'target_pose')

        # 서버 대기
        if not self.action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('target_pose 액션 서버를 찾을 수 없습니다.')
            return None

        # Goal 메시지 생성
        goal_msg = TargetPose.Goal()
        goal_msg.start = True  # 요청 트리거
        goal_msg.mode = mode

        # Goal 보내기
        self.get_logger().info('target_pose 요청 전송')
        future = self.action_client.send_goal_async(goal_msg)
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

            ee2cam = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

            base2cam = self.ee_pose @ ee2cam ## O2C
           

            pose = self.transform_pose_with_offset(result.target_pose, base2cam) ## O2t
            self.get_logger().info(f"변환한 bA: {pose}") # bA

            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
           
            # [USER REQUEST] Limit Minimum Z Height
            MIN_Z_LIMIT = 0.2 # 20cm
            MAX_Y_LIMIT = 0.33
            if pose_stamped.pose.position.z < MIN_Z_LIMIT:
                self.get_logger().warn(f"  >> Target Z ({pose_stamped.pose.position.z:.3f}) is too low! Clamping to {MIN_Z_LIMIT}")
                pose_stamped.pose.position.z = MIN_Z_LIMIT

       
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

            # [VISUALIZATION] Overcome Target Marker (Blue)
            # self.publish_target_marker(pose_stamped.pose, id=2, r=0.0, g=0.0, b=1.0)

            while not self.goal(self.ff, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")

            return pose_stamped
        else:
            return None

    def target_get_ba(self, mode="basket"): ### basket, wm, mid
        # 액션 클라이언트가 처음이면 생성
        if not hasattr(self, 'action_client'):
            self.action_client = ActionClient(self, TargetPose, 'target_pose')

        # 서버 대기
        if not self.action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('target_pose 액션 서버를 찾을 수 없습니다.')
            return None

        # Goal 메시지 생성
        goal_msg = TargetPose.Goal()
        goal_msg.start = True  # 요청 트리거
        goal_msg.mode = mode

        # Goal 보내기
        self.get_logger().info('target_pose 요청 전송')
        future = self.action_client.send_goal_async(goal_msg)
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

       
            R = np.array([
                [-0.07403140416428758,  0.9970253999896115, -0.02144068499412377],
                [-0.997213265121594,   -0.07381230005954229,  0.010837353249242954],
                [0.009222530183886607,  0.022183239967900933,  0.9997113827508088]
            ])

            t = np.array([[-0.04615831961252831],
                        [ 0.035761448900299364],
                        [ 0.06590399326489311]]) ######## 원래 07077399326489311 ############

            ee2cam = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

            base2cam = self.ee_pose @ ee2cam ## O2C
           

            pose = self.transform_pose_with_offset(result.target_pose, base2cam) ## O2t
           
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
           
            MIN_SAFE_Z_BASE = -0.290
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
           
            if min_dist >= SAFE_DIST: # Safe from wall -> Maintain Vertical
                 # --- Fix Orientation: Use current EE orientation ---
                from scipy.spatial.transform import Rotation as ScipyRotation
                current_rot_mat = self.ee_pose[:3, :3]
                r = ScipyRotation.from_matrix(current_rot_mat)
                qx, qy, qz, qw = r.as_quat()
                pose.orientation.x = qx
                pose.orientation.y = qy
                pose.orientation.z = qz
                pose.orientation.w = qw
                self.get_logger().info(f"Target Safe (dist={min_dist*100:.1f}cm > 4cm). Force Vertical Orientation.")
            else:
                self.get_logger().info(f"Target Near Wall (dist={min_dist*100:.1f}cm < 4cm). Use Vision Orientation.")

            self.get_logger().info(f"변환한 bA: {pose}") # bA

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
        
    def target_get_bg_eh(self, mode = "failure_bg_eh"):
        # 액션 클라이언트가 처음이면 생성
        if not hasattr(self, 'action_client'):
            self.action_client = ActionClient(self, TargetPose, 'target_pose')

        # 서버 대기
        if not self.action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('target_pose 액션 서버를 찾을 수 없습니다.')
            return None

        # Goal 메시지 생성
        goal_msg = TargetPose.Goal()
        goal_msg.start = True  # 요청 트리거
        goal_msg.mode = mode

        # Goal 보내기
        self.get_logger().info('target_pose 요청 전송')
        future = self.action_client.send_goal_async(goal_msg)
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

    def target_get_bg(self, mode = "failure_bg"):
        # 액션 클라이언트가 처음이면 생성
        if not hasattr(self, 'action_client'):
            self.action_client = ActionClient(self, TargetPose, 'target_pose')

        # 서버 대기
        if not self.action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('target_pose 액션 서버를 찾을 수 없습니다.')
            return None

        # Goal 메시지 생성
        goal_msg = TargetPose.Goal()
        goal_msg.start = True  # 요청 트리거
        goal_msg.mode = mode

        # Goal 보내기
        self.get_logger().info('target_pose 요청 전송')
        future = self.action_client.send_goal_async(goal_msg)
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

    def overcome(self):

        while True:

            # [Stage 1] Move to View Pose & Check Empty (RGB Gate)
            self.get_logger().info("Step 1: View Pose로 이동 (Empty Check)")
            while not self.goal(self.wm_view, 'keep'):
                self.get_logger().warn("  >> wm_view 이동 실패 → 재시도 중")
           
            # Check Emptiness
            chk_pose = self.target_get_wm(mode="failure")
           
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
            target_wm = self.target_get_wm()  # 감지 및 잡기 시도
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
   
    def end_sequence(self):
        self.scenario = int(input("시퀀스가 종료되었습니다. 시나리오를 골라주세요 [1,2,3,4,5]: "))
        if self.scenario in[1,2,3,4,5]:
            pass
        else:
            print("다시 선택하세요")
        if self.scenario == 3 or self.scenario ==5:
            self.tag = "abort_timeout"
        else:
            self.tag = input("시나리오에 해당되는 tag를 골라주세요 [u, s]: ")
            if self.tag in ["u","s"]:
                pass
            else:
                print("다시 선택하세요")

    def eunho_overcome_sequence(self):
        import time
        # Step 0: 비어 있는 세탁기 외벽 사진 촬영
        self.open_gripper()
        while not self.goal(self.wm_mid, 'keep'):
            self.get_logger().warn(" >> 실패 → 재시도 중")
        self.target_get_bg()
        while True:
            self.recording = False
            self.xserv_log = {"frames": []}
            self.open_gripper()
       
            # Step 1: wm_mid로 이동 후 time sleep 4초 대기 (그 사이 사람이 세탁기 외벽에 옷 걸치게 걸어 놓음)
            self.z_aligned = False
            self.get_logger().warn("Step 1: 5초 대기 후 view pose로 이동 (세탁기 외벽에 옷을 지금 걸쳐 놓으세요.)")
            while not self.goal(self.wm_view, 'keep'):
                self.get_logger().warn(" >> wm_view 실패 → 재시도 중")
            import time
            time.sleep(5)
       
            # Step 2: 파지 자세로 이동 후 파지점 계산 및 이동
            chk_pose = self.target_get_wm(mode="failure_eh")
            if chk_pose is not None and chk_pose.pose.position.z == -99.0:
                self.get_logger().warn("옷이 제 때 제대로 걸쳐 놓이지 않았습니다. 데이터를 뽑을 것도 없습니다.")
                return
            self.get_logger().info("Step 2: 파지 자세(wm_mid)로 이동 및 좌표 계산")
            while not self.goal(self.wm_mid, 'keep'):
                self.get_logger().warn("  >> wm_mid 이동 실패 → 재시도 중")
            self.get_logger().info("잡기 시도 중...")
            target_wm = self.target_get_wm(mode="failure_eh")  # 감지 및 잡기 시도
            if target_wm == None:
                self.get_logger().info("target_wm 결과가 없습니다. 시퀀스를 종료할테니, 다시 시작하십시오.")
                self.recording=False
                while not self.goal(self.wm_mid, 'keep'):
                    self.get_logger().warn(" >> 실패 → 재시도 중")
                return
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
            # 1. Approach Pre-Grasp (With Yaw Search)
            # [VISUALIZATION] Pre-Grasp Marker (Cyan)
            for i in range(20):
                self.get_logger().info(f" Pre-Grasp 접근 시도 {i+1} 번째 ... ")
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
                if approach_success:
                    break
            if not approach_success:
                self.get_logger().warn("  >> Pre-Grasp 에 실패하였습니다. (All Angles). 시퀀스를 종료할테니, 다시 시작하십시오.")
                self.recording=False
                while not self.goal(self.wm_mid, 'keep'):
                    self.get_logger().warn(" >> 실패 → 재시도 중")
                return
            # 2. Move to Grasp Target (Push)
            # 이때부터 데이터 받으면 될듯
            self.reset_sensor_data()
            self.recording=True
            with self._label_lock:
                self.current_label="NC"
            self.get_logger().warn("[LABEL] init -> NC (recording start)")
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
                self.get_logger().warn("forward 5회 실패 → 재시도 가치 없음. 시퀀스를 종료할테니, 다시 시작하십시오.")
                self.recording=False
                while not self.goal(self.wm_mid, 'keep'):
                    self.get_logger().warn(" >> 실패 → 재시도 중")
                return
            time.sleep(2)
            self.close_gripper()
            time.sleep(0.5)
            # 휴리스틱으로 잡기 성공 판단
            if True:
                self.get_logger().warn(f" 잡기 성공, G키를 눌러 상태를 업데이트하세요")
                self.z_aligned = True
                break
            else:
                self.open_gripper()
                self.get_logger().warn(f" firm grasp 실패, 처음부터 다시 시도합니다.")
                for i in range(5):
                    if self.move_backward(target_wm):
                        break
                    self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")
                continue
                self.get_logger().warn(f" firm grasp 실패 ")
                self.get_logger().warn(f"   S5 시나리오 데이터 수집 완료됨.   ")
                while not self.open_gripper():
                    self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
                self.recording=False
                self.end_sequence()
                self.xserv_log["scenario"] = self.scenario
                self.xserv_log["tag"] = self.tag
                while not self.goal(self.wm_mid, 'keep'):
                    self.get_logger().warn(" >> 실패 → 재시도 중")
                return
        self.get_logger().info("   [Overcome] Skip Retreat -> Moving directly to wm_mid")
        direct_success = self.goal(self.ff, 'keep')
        if not direct_success:
            self.get_logger().warn("  >> wm_mid 직행 실패! (Deep Singularity?). Rescue Mode 진입")
            self.recover_robot()
            for rescue_step in range(5):
                self.get_logger().info("    >> [Rescue] Step {rescue_step+1}/5: Lifting arm slightly...")
                import time
                for _ in range(10):
                    if self.current_joint_state is not None: break
                    time.sleep(0.05)
                if self.current_joint_state:
                    current = list(self.current_joint_state)
                    target_rescue = list(current)
                    target_rescue[2] -= 0.01 # ~3 deg
                    target_rescue[6] = current[6]
                    target_rescue[7] = current[7]
                    if self.goal(target_rescue, 'keep'):
                        self.get_logger().info("    >> [Rescue] Small move success.")
                    else:
                        self.get_logger().warn("    >> [Rescue] Small move failed.")
                time.sleep(0.5)
            self.get_logger().info("  >> [Rescue] Retrying wm_mid after rescue moves...")
            if self.goal(self.wm_mid_original, 'keep'):
                self.get_logger().info("  >> [Rescue] Success! Recovered to wm_mid.")
            else:
                self.get_logger().error("  >> [Rescue] Final wm_mid attempt failed. Critical Stuck.")
        while not self.goal(self.ff, 'keep'):
            self.get_logger().warn("  >> 실패 → 재시도 중")
        # 휴리스틱으로 잡기 성공 판단
        # self.get_logger().info("Z-축 정렬 감지")
        # grab_theta = self.theta
        # current_theta = self.theta; theta_diff = abs(current_theta - grab_theta)
        # is_grasp = self.max_norm > 0.45; is_stable = theta_diff < 5.0; is_valid_norm = self.norm > 0.35; is_not_edge = current_theta < 85.0
        # heuristic_1 = is_grasp and is_stable and is_valid_norm and is_not_edge
        # heuristic_2 = is_grasp and is_stable and is_valid_norm
        # if heuristic_2:
        #     self.get_logger().warn(f" 잡기 성공, G키를 눌러 상태를 업데이트하세요")
        #     self.z_aligned = True
        # else:
        #     self.get_logger().warn(f" firm grasp 실패 ")
        #     self.get_logger().warn(f"   S5 시나리오 데이터 수집 완료됨.   ")
        #     while not self.open_gripper():
        #         self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
        #     self.recording=False
        #     self.end_sequence()
        #     self.xserv_log["scenario"] = self.scenario
        #     self.xserv_log["tag"] = self.tag
        #     while not self.goal(self.wm_mid, 'keep'):
        #         self.get_logger().warn(" >> 실패 → 재시도 중")
        #     return
        self.get_logger().info("  >> [Loop] 세탁기(wm_end)로 이동하여 놓기")
        while not self.goal(self.wm_end, 'keep'):
            self.get_logger().warn("  >> wm_end 이동 실패 → 재시도 중")
        while not self.open_gripper():
            self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
        self.get_logger().warn(f"   그리퍼를 열었습니다. N킬을 눌러 상태를 업데이트하세요.")
        while not self.goal(self.wm_mid, 'keep'):
            self.get_logger().warn("  >> wm_mid 복귀 실패 → 재시도")
        self.get_logger().info("라벨링 기록을 종료하고 시퀀스를 종료합니다.")
        time.sleep(1)
        self.recording=False
        self.end_sequence()
        self.xserv_log["scenario"] = self.scenario
        self.xserv_log["tag"] = self.tag
        self.save_xserv_data()
        return
    
    def eunho_basket_sequence(self):
        self.open_gripper()
        self.z_aligned = False

        self.get_logger().info("Step 1: 이니셜 포즈(ba_init)로 이동 중")
        while not self.goal(self.ba_init, 'keep'):
            self.get_logger().warn(" >> 실패 → 재시도 중")
        self.reset_sensor_data()
        self.get_logger().info("Step 2-0: 잡기 시도 중...")
        target_ba = self.target_get_ba(mode="basket")
        
        # --- Two-Stage Approach: 1. Approach (+5cm) 2. Push ---
        import copy
        pre_grasp_pose = copy.deepcopy(target_ba)
        pre_grasp_pose.pose.position.z += 0.05  # 10cm higher

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
        self.get_logger().info(f"  [PRE-TARGET] Pre-Grasp Pose (+5cm): \n{pre_grasp_pose.pose}")
        # Convert to World Frame
        pre_grasp_world = copy.deepcopy(pre_grasp_pose.pose)
        pre_grasp_world.position.z += 0.445
        self.get_logger().info(f"  [WORLD] Pre-Grasp (Base+0.445): \n{pre_grasp_world}")
        self.get_logger().info("==================================================")
        self.publish_target_marker(pre_grasp_pose.pose, id=1, r=0.0, g=1.0, b=0.0) # Green

        # 1. Approach
        self.get_logger().info("Step 2-1: 접근(Approach) 이동 중... (Collision Check ON)")
        approach_success = False
        for i in range(3):
            # [USER REQUEST] Disable Fallback to Mode 6 for Pre-Grasp
            res = self.move_forward(pre_grasp_pose, fallback_to_ik=False)
            if res == "success":
                approach_success = True
                self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달!\033[0m")
                # Calc Tip Pose
                tip_pose = np.eye(4)
                tip_pose[:3, :3] = self.ee_pose[:3, :3]
                # Tip = Wrist + R * [0, 0, 0.11]
                offset_vec = self.ee_pose[:3, :3] @ np.array([0, 0, 0.11])
                tip_pose[:3, 3] = self.ee_pose[:3, 3] + offset_vec
                
                self.get_logger().info(f"  >> Current Wrist : {self.ee_pose[:3,3]}")
                self.get_logger().info(f"\033[92m  >> Computed Tip  : {tip_pose[:3,3]}\033[0m") # Should match pre_grasp
                break
            self.get_logger().warn(f"  >> 접근 실패 ({i+1}/3) result={res} → 재시도")
            # if res == "stop": # 접근 중 충돌?
            #      self._handle_stop_and_recover()
            #      break # 일단 멈춤
        
        if not approach_success:
            self.get_logger().warn("  >> 접근 이동 실패 → 시퀀스 중단")
            return

        # 2. Push (Actual Grasp Move)
        self.get_logger().info("Step 2-2: 잡기(Push) 이동 중...")
        forward_state = None
        for i in range(5):
            forward_state = self.move_forward(target_ba) # Hybrid
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
            return

        self.close_gripper()
        # [USER REQUEST] Stabilize & Capture Initial Theta
        import time
        time.sleep(1.0)
        
        
        # --- Retreat Added ---
        for i in range(5):
            if self.move_backward(target_ba): # Using new hybrid move_backward
                break
            self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")
        # 실패해도 일단 진행 (or loop?)
        
        while not self.goal(self.ba_init, 'keep'):
            self.get_logger().warn("  >> 실패 → 재시도 중")

    def eunho_sequence(self):
            # 데이터 뽑는 용도
        
            self.open_gripper()
            self.z_aligned = False
            # self.get_logger().info("Step 0: wm_mid로 이동 중")
            # while not self.goal(self.wm_mid, 'keep'):
            #     self.get_logger().warn(" >> 실패 → 재시도 중")
            self.get_logger().info("Step 1: 이니셜 포즈(ba_init)로 이동 중")
            while not self.goal(self.ba_init, 'keep'):
                self.get_logger().warn(" >> 실패 → 재시도 중")
            self.reset_sensor_data()
        
            ####################################################################################################
            ######## [시각화] 실제 로봇을 움직이기 전에, 잡을 target_pose와 그 위의 pre-grasp 포즈를 검증·시각화하는 블록#########
            ####################################################################################################
            self.get_logger().info("Step 2-0: 잡기 시도 중...")
            target_ba = self.target_get_ba(mode="basket")
            import copy
            pre_grasp_pose = copy.deepcopy(target_ba)
            pre_grasp_pose.pose.position.z += 0.05
            self.get_logger().info("\n==================================================")
            self.get_logger().info(f"  [TARGET] BA (Grasp): \n{target_ba.pose}") # taget_ba.pose를 출력
            # Convert to World Frame
            target_ba_world = copy.deepcopy(target_ba.pose)
            target_ba_world.position.z += 0.445 # base 기준 포즈를 world 기준 pose로 보정
            self.get_logger().info(f"  [WORLD] BA (Grasp) (Base+0.445): \n{target_ba_world}")
            self.get_logger().info("==================================================")
            self.publish_target_marker(target_ba.pose, id=0, r=1.0, g=0.0, b=0.0) # grasp 포즈를 RViz marker(RED)로 즉시 publish
            self.get_logger().info("\n==================================================")
            self.get_logger().info(f"  [PRE-TARGET] Pre-Grasp Pose (+5cm): \n{pre_grasp_pose.pose}") # pre_grasp_pose.pose를 출력
            # Convert to World Frame
            pre_grasp_world = copy.deepcopy(pre_grasp_pose.pose)
            pre_grasp_world.position.z += 0.445 # base 기준 포즈를 world 기준 pose로 보정
            self.get_logger().info(f"  [WORLD] Pre-Grasp (Base+0.445): \n{pre_grasp_world}")
            self.get_logger().info("==================================================")
            self.publish_target_marker(pre_grasp_pose.pose, id=1, r=0.0, g=1.0, b=0.0) # pre grasp 포즈를 RViz marker(GREEN)로 즉시 publish
        
            ####################################################################################################
            ####### 1. Approach : pre-grasp 포즈로 충돌 검사를 켠 상태로 접근을 최대 3회 시도하고, 성공 시 실제 EE pose #######
            #######             로부터 tip 위치를 재계산해서 내가 진짜 의도한 위치에 왔는지를 검증하는 코드               #######
            ####################################################################################################
            # 이때부터 데이터 받으면 될듯 (NC 라벨링 시작)
            self.recording = True
            with self._label_lock:
                self.current_label = "NC"
            self.get_logger().warn("[LABEL] init -> NC (recording start)")
            
            self.get_logger().info("Step 2-1: 접근(Approach) 이동 중... (Collision Check ON)")
            approach_success = False
            for i in range(5):
                res = self.move_forward(pre_grasp_pose, fallback_to_ik=False) # pre_grasp_pose로 이동 시도, mode 6 폴백 금지 (정확히 갈 수 있을 때만 가라)
                if res == "success":
                    approach_success = True
                    self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달!\033[0m")
                    # Calc Tip Pose
                    tip_pose = np.eye(4)
                    tip_pose[:3, :3] = self.ee_pose[:3, :3] # tip pose의 회전 성분은 현재 EE pose와 동일
                    # Tip = Wrist + R * [0, 0, 0.11]
                    offset_vec = self.ee_pose[:3, :3] @ np.array([0, 0, 0.11])
                    tip_pose[:3, 3] = self.ee_pose[:3, 3] + offset_vec
                
                    self.get_logger().info(f"  >> Current Wrist : {self.ee_pose[:3,3]}")
                    self.get_logger().info(f"\033[92m  >> Computed Tip  : {tip_pose[:3,3]}\033[0m") # Should match pre_grasp
                    break
                self.get_logger().warn(f"  >> 접근 실패 ({i+1}/5) result={res} → 재시도")

            if not approach_success:
                self.get_logger().warn(" >> 접근 이동 실패 → 재시도 가치 없음. 본 데이터를 폐기합니다. 재시작하세요.")
                self.recording = False
                while not self.goal(self.wm_mid, 'keep'): # 사진
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                return # eunho_sequence 함수 종료
        
            ####################################################################################################
            ############# 2. Push : target_grasp 포즈까지 전진 이동을 시도하고 gripper를 닫는 실제 grasp 단계 #############
            ####################################################################################################
            self.get_logger().info("Step 2-2: 잡기(Push) 이동 중...")
            forward_state = None
            for i in range(5):
                forward_state = self.move_forward(target_ba)
                if forward_state =="success":
                    self.get_logger().info("\n\033[92m[SUCCESS] Target BA 도달!\033[0m")
                    # Calc Tip Pose
                    tip_pos_ba = self.ee_pose[:3, 3] + (self.ee_pose[:3, :3] @ np.array([0, 0, 0.11]))
                    self.get_logger().info(f"  >> Current Wrist : {self.ee_pose[:3,3]}")
                    self.get_logger().info(f"\033[92m  >> Computed Tip  : {tip_pos_ba}\033[0m")
                    break
                if forward_state == "stop": # 강한 충돌이므로 그리퍼를 닫으면 물체를 잡는다고 가정.
                    self._handle_stop_and_recover()
                    break
                else:
                    self.get_logger().warn(f"forward 실패 ({i+1}/5) result={forward_state} -> 재시도 중")
            else:
                self.get_logger().warn(" >> 물체 집으러 forward 이동 5회 연속 실패 -> 재시도 가치 없음. 본 데이터를 폐기합니다. 재시작하세요.")
                self.recording = False
                while not self.goal(self.wm_mid, 'keep'): # 사진
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                return # eunho sequence 종료

            self.get_logger().info("그리퍼 닫습니다 ... ")
            self.close_gripper()
            
            import time
            for i in range(5):
                if self.move_backward(target_ba):
                    break
                self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")

            self.get_logger().info("휴리스틱 기반 z축 정렬 감지 중 ... ")
            if True:
                self.get_logger().warn(f" 잡기 성공, G키를 눌러 상태를 업데이트하세요")
                self.z_aligned = True
            else:
                self.get_logger().warn(f" firm grasp 실패 ")
                self.get_logger().warn(f"   S3 시나리오 데이터 수집 완료됨. ")
                while not self.open_gripper():
                    self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")
                self.recording = False
                self.end_sequence()
                self.xserv_log["scenario"] = self.scenario
                self.xserv_log["tag"] = self.tag
                self.save_xserv_data()
                while not self.goal(self.wm_mid, 'keep'): # 사진
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                return # S3 시나리오 데이터 저장 후 eunho sequence 종료

            while not self.goal(self.ba_init, "keep"):
                self.get_logger().warn(" >> 초기 자세 복귀 실패 → 재시도 중")
                


            ####################################################################################################
            ####################################### 파지된 상태에서 로봇팔 이동 #######################################
            ####################################################################################################
            self.get_logger().info("Step 3: 파지된 사태에서 엔드포즈(wm_end)로 이동 중")
            self.get_logger().info("중간에 빨래가 세탁기 외벽에 걸리면 C키를 눌러 상태를 업데이트하세요")
            while not self.goal(self.wm_init, 'keep'):
                self.get_logger().warn(" >> wm_init 이동 실패 → 재시도 중")
            while not self.goal(self.wm_end, 'keep'):
                self.get_logger().warn(" >> wm_end 이동 실패 → 재시도 중")
        
            ####################################################################################################
            ####################################### 그리퍼 열고 시퀀스 종료하기 #######################################
            ####################################################################################################
            self.get_logger().info("Step 4: 그리퍼 열고 시퀀스 종료를 시도합니다.")
            self.open_gripper()
            self.get_logger().info("그리퍼를 열었습니다. N키를 눌러 상태를 업데이트하세요.")
            while not self.goal(self.wm_mid, 'keep'): # 사진
                self.get_logger().warn("  >> 실패 → 재시도 중")
            self.get_logger().info("라벨링 기록을 종료하고 시퀀스를 종료합니다 ...")
            self.recording = False
            self.end_sequence()
            self.xserv_log["scenario"] = self.scenario
            self.xserv_log["tag"] = self.tag
            self.save_xserv_data()
            return # S1 | S2 시나리오 데이터 저장 후 eunho sequence 종료
   
       
    def execute_sequence(self):
        self.get_logger().info("Step 0: 문 여는 중")
        # self.open_door()
        # self.overcome()

        while not self.goal(self.wm_mid, 'keep'): # 사진
            self.get_logger().warn("  >> 실패 → 재시도 중")
        self.target_get_bg()

        while rclpy.ok():
           
            self.open_gripper()

            while True:
                self.open_gripper()
                self.z_aligned = False
                self.get_logger().info("Step 1: 이니셜 포즈(ba_init)로 이동 중")
                while not self.goal(self.ba_init, 'keep'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")
                self.reset_sensor_data()

                self.get_logger().info("Step 2: 잡기 시도 중...")

                target_ba = self.target_get_ba(mode="basket")  # 감지
               

                # Removed direct move to target_ba to respect Pre-Grasp -> Push sequence
                # for i in range(5):
                #     if self.send_pose_ba(target_ba): #1
                #         break
                #     self.get_logger().warn(f"  >> pose 명령 실패 ({i+1}/5) → 재시도 중")
                # else:  
                #     self.get_logger().warn("  >> pose 명령 5회 연속 실패 → 새 target 요청")
                #     continue  # while True 처음으로

           
                # --- Two-Stage Approach: 1. Approach (+5cm) 2. Push ---
                import copy
                pre_grasp_pose = copy.deepcopy(target_ba)
                pre_grasp_pose.pose.position.z += 0.05  # 10cm higher

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
                self.get_logger().info(f"  [PRE-TARGET] Pre-Grasp Pose (+5cm): \n{pre_grasp_pose.pose}")
                # Convert to World Frame
                pre_grasp_world = copy.deepcopy(pre_grasp_pose.pose)
                pre_grasp_world.position.z += 0.445
                self.get_logger().info(f"  [WORLD] Pre-Grasp (Base+0.445): \n{pre_grasp_world}")
                self.get_logger().info("==================================================")
                self.publish_target_marker(pre_grasp_pose.pose, id=1, r=0.0, g=1.0, b=0.0) # Green

                # 1. Approach
                self.get_logger().info("Step 2-1: 접근(Approach) 이동 중... (Collision Check ON)")
                approach_success = False
                for i in range(3):
                    # [USER REQUEST] Disable Fallback to Mode 6 for Pre-Grasp
                    res = self.move_forward(pre_grasp_pose, fallback_to_ik=False)
                    if res == "success":
                        approach_success = True
                        self.get_logger().info("\n\033[92m[SUCCESS] Pre-Grasp 도달!\033[0m")
                        # Calc Tip Pose
                        tip_pose = np.eye(4)
                        tip_pose[:3, :3] = self.ee_pose[:3, :3]
                        # Tip = Wrist + R * [0, 0, 0.11]
                        offset_vec = self.ee_pose[:3, :3] @ np.array([0, 0, 0.11])
                        tip_pose[:3, 3] = self.ee_pose[:3, 3] + offset_vec
                       
                        self.get_logger().info(f"  >> Current Wrist : {self.ee_pose[:3,3]}")
                        self.get_logger().info(f"\033[92m  >> Computed Tip  : {tip_pose[:3,3]}\033[0m") # Should match pre_grasp
                        break
                    self.get_logger().warn(f"  >> 접근 실패 ({i+1}/3) result={res} → 재시도")
                    # if res == "stop": # 접근 중 충돌?
                    #      self._handle_stop_and_recover()
                    #      break # 일단 멈춤
               
                if not approach_success:
                    self.get_logger().warn("  >> 접근 이동 실패 → 새 target 요청")
                    continue


                # 2. Push (Actual Grasp Move)
                self.get_logger().info("Step 2-2: 잡기(Push) 이동 중...")
                forward_state = None
                for i in range(5):
                    forward_state = self.move_forward(target_ba) # Hybrid
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
                # [USER REQUEST] Stabilize & Capture Initial Theta
                import time
                time.sleep(1.0)
               
               
                # --- Retreat Added ---
                for i in range(5):
                    if self.move_backward(target_ba): # Using new hybrid move_backward
                        break
                    self.get_logger().warn(f"  >> back 실패 ({i+1}/5) → 재시도 중")
                # 실패해도 일단 진행 (or loop?)
               
                while not self.goal(self.ba_init, 'keep'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")


                self.get_logger().info(" Z-축 정렬 (Slip Check) 감지")
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
                if is_stable and is_valid_norm and is_not_edge and is_grasp:
                    self.get_logger().warn(f" 잡기 성공 (Stable Grasp: Δθ={theta_diff:.1f}°, θ={current_theta:.1f}°, norm={self.norm:.3f}, max_norm = {self.max_norm:.3f})")
                    self.z_aligned = True
                    break
                else:
                    self.get_logger().warn(f" 잡기 실패 (Unstable grasp): Δθ={theta_diff:.1f}°, θ={current_theta:.1f}°, norm={self.norm:.3f}, max_norm = {self.max_norm:.3f})")
                    self.get_logger().warn(f"   (Criteria: Δ<6.0, θ<85.0, norm>0.35, max norm > 0.5)")
                    while not self.open_gripper():
                        self.get_logger().warn("  >> 그리퍼 오픈 실패 → 재시도 중")


            self.get_logger().info("Step 3: 엔드 포즈(wm_end)로 이동 중")
            self.contacted = False
           
            # Start Monitoring
            self.monitor_initial_norm = self.norm
            self.monitor_initial_theta = self.theta
            self.monitor_slip = True
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
            else:
                self.get_logger().warn(f" Slip Detected: Δθ={theta_diff:.1f}°, θ={current_theta:.1f}°, norm={self.norm:.3f})")
                self.get_logger().warn(f"   (Criteria: Δ<6.0, θ<85.0, norm>0.35)")
                self.contacted = True


            self.contacted = True ####
            self.open_gripper()

            if self.contacted:
                self.get_logger().info("  >> 접촉 감지됨 → 실패 극복")
                self.overcome()
                # (overcome 기능은 필요시 추가하거나 현재는 주석)

                self.get_logger().info("  >> 다시 엔드 포즈(wm_end)로 이동 중")
                while not self.goal(self.wm_end, 'open'):
                    self.get_logger().warn("  >> 실패 → 재시도 중")
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
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.03 # arrow length (Longer)
        marker.scale.y = 0.03 # arrow width (Thicker)
        marker.scale.z = 0.03 # arrow height (Thicker)
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.lifetime = Duration(sec=0, nanosec=0) # Infinite lifetime
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published Marker ID {id} to {marker.header.frame_id}")

def main(args=None):
    rclpy.init(args=args)
   
    try:
        node = ControlTower()
    except RuntimeError as e:
        print(f"노드 초기화 중 오류 발생: {e}")
        return

    # MultiThreadedExecutor는 여전히 사용합니다.
    # 콜백들이 블로킹 함수(spin_until_future_complete)에 의해 중단되지 않도록 하기 위함입니다.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    # 메인 로직을 실행할 작업 스레드
    sequence_thread = threading.Thread(target=node.eunho_overcome_sequence, daemon=True)
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
