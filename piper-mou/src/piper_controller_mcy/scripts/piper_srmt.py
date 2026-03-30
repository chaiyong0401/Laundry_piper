#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from threading import Lock, Thread, Condition
import time

from dyros_robot_controller.robot_data.manipulator.base import ManipulatorBase
from dyros_robot_controller.robot_controller.manipulator.base import ManipulatorControllerBase
from srmt2.planning_scene import PlanningScene
from srmt2.planner.rrt_connect import SuhanRRTConnect
from srmt2.kinematics import TRACIK


class PiperSRMT(Node):
    def __init__(self):
        super().__init__('piper_srmt')

        # === ROS 2 PUB/SUB ===
        self.joint_ctrl_pub = self.create_publisher(JointState, 'joint_ctrl_single_clik', 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, 'ee_pose', 10)
        self.create_subscription(JointState, 'joint_states', self.joint_states_cb, 10)
        self.create_subscription(PoseStamped, 'desired_ee_pose', self.desired_ee_pose_cb, 10)
        self.create_subscription(Bool, 'mode_', self.mode_change_cb, 1)

        # === CONTROL SETTINGS ===
        self.control_dt = 0.01  # 100 Hz

        # === ROBOT SETUP ===
        robot_pkg_path = get_package_share_directory("piper_with_gripper_moveit")
        urdf_path = str(Path(get_package_share_directory("piper_description"), "urdf", "piper_description_wo_gripper.urdf"))
        srdf_path = str(Path(robot_pkg_path, "config", "piper_wo_gripper.srdf"))
        self.robot_data = ManipulatorBase(urdf_path=urdf_path, srdf_path=srdf_path, packages_path=robot_pkg_path)
        self.robot_controller = ManipulatorControllerBase(dt=self.control_dt, robot_data=self.robot_data)
        self.pc = PlanningScene(arm_names=["piper_arm"], arm_dofs=[6], base_link="base_link",
                                node_name="PiperPlanningScene", topic_name="/piper_planning_scene",
                                urdf_file_path=urdf_path, srdf_file_path=srdf_path)
        self.tracIK = TRACIK(base_link="base_link", tip_link="link7", urdf_file_path=urdf_path)
        self.joint_upper_limit = self.tracIK.get_upper_bound()
        self.joint_lower_limit = self.tracIK.get_lower_bound()
        self.rrt_planner = SuhanRRTConnect(state_dim=6, lb=self.joint_lower_limit, ub=self.joint_upper_limit,
                                           validity_fn=self.pc.is_valid)

        # === STATE VARIABLES (shared with control thread) ===
        self.q = np.zeros(6)
        self.qdot = np.zeros(6)
        self.q_desired = np.zeros(6)
        self.qdot_desired = np.zeros(6)

        self.x = np.eye(4)
        self.x_init = np.eye(4)
        self.x_target = np.eye(4)
        self.xdot = np.zeros(6)
        self.xdot_init = np.zeros(6)

        self.is_initialized = False
        self.mode = False  # False = MoveIt passthrough, True = CLIK trajectory following

        # === Locks & conditions ===
        self.state_lock = Lock()          # protects q, qdot, x, xdot
        self.desired_lock = Lock()        # protects x_target
        self.traj_lock = Lock()           # protects trajectory buffers
        self.plan_cv = Condition(Lock())  # coordinate plan requests with planner thread
        self.mode_lock = Lock()

        # === Trajectory buffers (shared) ===
        self.traj_q = None
        self.traj_qdot = None
        self.traj_time = None
        self._traj_ready = False
        self._traj_request_pending = False
        self._traj_request_stamp = 0.0    # to invalidate old requests if new target arrives

        self.control_start_time = 0.0
        self._last_print_time = 0.0  # throttled logging

        # === Start 100 Hz control timer ===
        self.create_timer(self.control_dt, self.compute)

        # === Start free-running planning thread ===
        self._planner_alive = True
        self._planner_thread = Thread(target=self._planning_loop, name="planner", daemon=True)
        self._planner_thread.start()

    # === CALLBACKS ===

    def mode_change_cb(self, msg: Bool):
        with self.mode_lock:
            self.mode = msg.data
        with self.state_lock:
            self.q_desired = self.q.copy()
            self.qdot_desired = self.qdot.copy()
            self.publish_joint_state()
        mode_str = "CLIK" if self.mode else "MoveIt"
        self.get_logger().info(f"Switched to mode: {mode_str}")

    def joint_states_cb(self, msg: JointState):
        with self.state_lock:
            print(msg.position)
            self.q = np.array(msg.position[:6])
            print(self.q)
            self.qdot = np.array(msg.velocity[:6])

            # update robot model
            if not self.robot_data.update_state(self.q, self.qdot):
                self.get_logger().error("Failed to update robot state.")
                return
            self.x = self.robot_data.get_pose("link7")
            self.xdot = self.robot_data.get_velocity("link7")

            if not self.is_initialized and np.linalg.norm(self.x[:3, 3]) > 1E-3:
                self.q_desired = self.q.copy()
                self.qdot_desired = self.qdot.copy()
                self.x_init = self.x.copy()
                self.x_target = self.x.copy()
                self.xdot_init = self.xdot.copy()
                self.control_start_time = self._now_sec()
                self.is_initialized = True
                self.get_logger().info("Initialized!")

        self.pc.display(self.q)
        self.publish_ee_pose()

    def desired_ee_pose_cb(self, msg: PoseStamped):
        quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                         msg.pose.orientation.z, msg.pose.orientation.w])
        norm = np.linalg.norm(quat)
        if norm == 0:
            self.get_logger().warn("Received zero quaternion; ignoring")
            return
        quat /= norm
        rot = R.from_quat(quat).as_matrix()

        with self.desired_lock:
            self.x_target[:3, :3] = rot
            self.x_target[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        # Signal planner thread to (re)plan
        with self.plan_cv:
            self._traj_request_pending = True
            self._traj_request_stamp = self._now_sec()
            self.plan_cv.notify()

        self.get_logger().info(f"New EE target received. Planner notified.")

    # === 100 Hz CONTROL LOOP ===

    def compute(self):
        if not self.is_initialized:
            return

        now = self._now_sec()
        if (now - self._last_print_time) >= 0.1:
            self._last_print_time = now
            self.get_logger().info(
                "\n[CTRL {:.3f}s]"
                "\n q         : {}"
                "\n q_desired : {}"
                "\n qdot         : {}"
                "\n qdot_desired : {}"
                "\n x (4x4) :\n{}"
                "\n xdot : {}"
                "\n x_target :\n{}".format(
                    now,
                    self._fmt(self.q),
                    self._fmt(self.q_desired),
                    self._fmt(self.qdot),
                    self._fmt(self.qdot_desired),
                    self._fmt(self.x),
                    self._fmt(self.xdot),
                    self._fmt(self.x_target)
                )
            )

        # If not in CLIK mode, just mirror the measured state
        with self.mode_lock:
            mode = self.mode
        if not mode:
            with self.state_lock:
                self.q_desired = self.q.copy()
                self.qdot_desired = self.qdot.copy()
                self.publish_joint_state()
            return

        # Follow trajectory if available
        with self.traj_lock:
            if self._traj_ready and self.traj_time is not None:
                t = now
                # find active segment
                idx = np.searchsorted(self.traj_time, t, side='right') - 1
                if 0 <= idx < len(self.traj_time) - 1:
                    i = idx
                    # cubic interpolation to next waypoint
                    self.q_desired = self.robot_controller.move_joint_position_cubic(
                        self.traj_q[i + 1], self.traj_qdot[i + 1],
                        self.traj_q[i], self.traj_qdot[i],
                        self.control_start_time, self.traj_time[i],
                        self.traj_time[i + 1] - self.traj_time[i])

                    self.qdot_desired = self.robot_controller.move_joint_velocity_cubic(
                        self.traj_q[i + 1], self.traj_qdot[i + 1],
                        self.traj_q[i], self.traj_qdot[i],
                        self.control_start_time, self.traj_time[i],
                        self.traj_time[i + 1] - self.traj_time[i])
                else:
                    # finished; hold last
                    self.q_desired = self.traj_q[-1].copy()
                    self.qdot_desired = np.zeros_like(self.qdot_desired)

        self.publish_joint_state()

    # === FREE-RUNNING PLANNER THREAD ===

    def _planning_loop(self):
        """Runs in a separate thread; blocks on condition and computes RRT trajectories."""
        self.get_logger().info("Planner thread started.")
        while self._planner_alive and rclpy.ok():
            # Wait for a plan request
            with self.plan_cv:
                while not self._traj_request_pending and self._planner_alive:
                    self.plan_cv.wait(timeout=0.2)
                if not self._planner_alive:
                    break
                request_stamp = self._traj_request_stamp
                self._traj_request_pending = False

            # Snapshot start & goal safely
            with self.state_lock:
                q_start = self.q.copy()
                x_target_copy = self.x_target.copy()

            # === IK ===
            ik_ok = False
            q_goal = None
            for _ in range(10):
                ok, qg = self.tracIK.solve(pos=x_target_copy[:3, 3],
                                           quat=R.from_matrix(x_target_copy[:3, :3]).as_quat(),
                                           q_init=q_start)
                if ok and self.pc.is_valid(qg):
                    ik_ok = True
                    q_goal = qg
                    break
            if not ik_ok:
                self.get_logger().warn("[Planner] Failed IK; skipping request")
                continue

            # === RRT PLAN ===
            self.rrt_planner.set_start(q_start)
            self.rrt_planner.set_goal(q_goal)
            self.rrt_planner.max_distance = 0.01

            found = False
            for _ in range(10):
                found, path = self.rrt_planner.solve()
                if found:
                    break
            if not found:
                self.get_logger().warn("[Planner] RRT failed; skipping request")
                continue

            # Time parameterization
            traj_q, traj_qdot, traj_qddot, traj_time = self.pc.time_parameterize(path, 1.0, 1.0)
            # Align to wall-clock
            now = self._now_sec()
            traj_time = now + traj_time - traj_time[0]

            # If a newer request arrived while planning, drop this result
            with self.plan_cv:
                if request_stamp != self._traj_request_stamp:
                    self.get_logger().info("[Planner] Newer request arrived; dropping old plan")
                    continue

            # Publish trajectory buffers
            with self.traj_lock:
                self.traj_q = traj_q
                self.traj_qdot = traj_qdot
                self.traj_time = traj_time
                self._traj_ready = True

            # Reset control segment time base
            self.control_start_time = now
            self.get_logger().info("[Planner] Trajectory ready ({} waypoints)".format(len(traj_time)))

        self.get_logger().info("Planner thread exiting.")

    # === UTILITIES ===

    def publish_ee_pose(self):
        trans = self.x[:3, 3]
        quat = R.from_matrix(self.x[:3, :3]).as_quat()

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = trans
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        self.ee_pose_pub.publish(msg)

    def publish_joint_state(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [f'joint{i+1}' for i in range(6)]
        js.position = self.q_desired.copy().tolist()
        js.velocity = self.qdot_desired.copy().tolist()
        js.effort = [0.0] * 6
        self.joint_ctrl_pub.publish(js)

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _fmt(self, arr) -> str:
        return np.array2string(np.asarray(arr), precision=4, suppress_small=True, max_line_width=120)

    def destroy_node(self):
        # stop planner thread cleanly
        with self.plan_cv:
            self._planner_alive = False
            self.plan_cv.notify_all()
        if hasattr(self, '_planner_thread') and self._planner_thread.is_alive():
            self._planner_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiperSRMT()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
