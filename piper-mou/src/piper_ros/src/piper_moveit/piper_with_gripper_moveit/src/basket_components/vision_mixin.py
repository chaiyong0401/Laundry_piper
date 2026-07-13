import time
import numpy as np
from rclpy.clock import Clock
from rclpy.duration import Duration
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker
from piper_with_gripper_moveit.action import TargetPose


class VisionMixin:
    """카메라 기반 타겟 감지 및 마커 시각화."""

    def target_get_wm(self, mode="failure", first=True):
        goal_msg = TargetPose.Goal()
        goal_msg.start = True
        goal_msg.mode = mode

        if not self._target_client.server_is_ready():
            self.get_logger().error('target_pose 액션 서버 미연결 → node2.py 실행 중인지 확인')
            return None
        self.get_logger().info('target_pose 요청 전송')
        future = self._target_client.send_goal_async(goal_msg)
        self._wait_future(future, timeout_sec=30.0)

        if not future.done():
            self.get_logger().error('Goal 요청 실패 (timeout)')
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal이 액션 서버에서 거부됨')
            return None

        result_future = goal_handle.get_result_async()
        self._wait_future(result_future, timeout_sec=60.0)
        if not result_future.done():
            self.get_logger().error('Vision 결과 대기 timeout (60s) → goal 취소')
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
            return None
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f"받은 EA: {result.target_pose}")
            self.get_logger().info(f"현재 Pose: {self.ee_pose}")

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
                        [ 0.06690399326489311]])

            ee2cam = np.vstack((np.hstack((self.R_cal, self.t_cal)), [0, 0, 0, 1]))
            base2cam = self.ee_pose @ ee2cam

            p_cam = np.array([result.target_pose.position.x, result.target_pose.position.y, result.target_pose.position.z, 1.0])
            p_ee = ee2cam @ p_cam
            pose = self.transform_pose_with_offset(result.target_pose, base2cam)

            self.get_logger().info(f"\033[92m[Debug] Camera frame (EA): [{p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f}]\033[0m")
            self.get_logger().info(f"\033[92m[Debug] EE frame (local): [{p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}]\033[0m")
            self.get_logger().info(f"\033[92m[Debug] Base frame (bA):  [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]\033[0m")
            self.get_logger().info(f"\033[94m변환한 bA: {pose}\033[0m")

            cam_center_pose = Pose()
            cam_center_pose.position.x = base2cam[0,3]
            cam_center_pose.position.y = base2cam[1,3]
            cam_center_pose.position.z = base2cam[2,3]
            self.publish_target_marker(cam_center_pose, id=100, r=0.0, g=0.5, b=1.0)

            pose_stamped = PoseStamped()
            pose_stamped.pose = pose

            is_fixed_coord = (abs(p_cam[0] - 0.057) < 0.001 and
                              abs(p_cam[1] - (-0.044)) < 0.001 and
                              abs(p_cam[2] - 0.089) < 0.001)

            if result.target_pose.orientation.x > 90.0 or is_fixed_coord:
                pose_stamped.header.frame_id = "base_link|fixed"
            else:
                pose_stamped.header.frame_id = "base_link"

            MIN_Z_LIMIT = 0.2
            MAX_Z_LIMIT = 0.265
            MAX_Y_LIMIT = 0.365
            if pose_stamped.pose.position.z < MIN_Z_LIMIT:
                self.get_logger().warn(f"  >> Target Z ({pose_stamped.pose.position.z:.3f}) is too low! Clamping to {MIN_Z_LIMIT}")
                pose_stamped.pose.position.z = MIN_Z_LIMIT
            if pose_stamped.pose.position.z > MAX_Z_LIMIT:
                self.get_logger().warn(f"  >> Target Z ({pose_stamped.pose.position.z:.3f}) is too high! Clamping to {MAX_Z_LIMIT}")
                pose_stamped.pose.position.z = MAX_Z_LIMIT

            pose_stamped.pose.position.x -= 0.01
            tx, ty = pose_stamped.pose.position.x, pose_stamped.pose.position.y
            self.get_logger().info(f"  [Wall Check] Target XY: ({tx:.3f}, {ty:.3f})")
            if pose_stamped.pose.position.y > MAX_Y_LIMIT:
                self.get_logger().warn(f" >> Target Y  ({pose_stamped.pose.position.y:.3f}) is too low! Clamping to {MAX_Y_LIMIT}")
                pose_stamped.pose.position.y = MAX_Y_LIMIT

            raw_y = pose_stamped.pose.position.y
            grasp_y = raw_y + (MAX_Y_LIMIT - raw_y) * (2.0 / 3.0)
            self.get_logger().info(f"  [Grasp Y] raw={raw_y:.3f} → grasp={grasp_y:.3f} (2/3 to limit={MAX_Y_LIMIT})")
            pose_stamped.pose.position.y = grasp_y

            pose_stamped.pose.orientation.x = 0.4635
            pose_stamped.pose.orientation.y = -0.5322
            pose_stamped.pose.orientation.z = -0.5512
            pose_stamped.pose.orientation.w = -0.4451

            pose_stamped.header.frame_id = 'base_link'
            pose_stamped.header.stamp = Clock().now().to_msg()

            self.publish_target_marker(pose_stamped.pose, id=1, r=0.0, g=0.0, b=1.0)

            while not self.goal(self.ff, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
                time.sleep(0.5)

            return pose_stamped
        else:
            return None

    def target_get_ba(self, mode="basket"):
        goal_msg = TargetPose.Goal()
        goal_msg.start = True
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

        result_future = goal_handle.get_result_async()
        self._wait_future(result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f"받은 EA: {result.target_pose}")
            self.get_logger().info(f"현재 Pose: {self.ee_pose}")

            if result.target_pose.position.z == -99.0:
                self.get_logger().info("  >> [Sentinel] Empty Detected (-99.0). Returning immediately.")
                pose_stamped = PoseStamped()
                pose_stamped.pose = result.target_pose
                return pose_stamped

            ee2cam = np.vstack((np.hstack((self.R_cal, self.t_cal)), [0, 0, 0, 1]))
            base2cam = self.ee_pose @ ee2cam

            p_cam = np.array([result.target_pose.position.x, result.target_pose.position.y, result.target_pose.position.z, 1.0])
            p_ee = ee2cam @ p_cam
            pose = self.transform_pose_with_offset(result.target_pose, base2cam)

            self.get_logger().info(f"\033[92m[Debug] Camera frame (EA): [{p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f}]\033[0m")
            self.get_logger().info(f"\033[92m[Debug] EE frame (local): [{p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}]\033[0m")
            self.get_logger().info(f"\033[92m[Debug] Base frame (bA):  [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]\033[0m")

            cam_center_pose = Pose()
            cam_center_pose.position.x = base2cam[0,3]
            cam_center_pose.position.y = base2cam[1,3]
            cam_center_pose.position.z = base2cam[2,3]
            self.publish_target_marker(cam_center_pose, id=100, r=0.0, g=0.5, b=1.0)

            MIN_SAFE_Z_BASE = -0.310
            if pose.position.z < MIN_SAFE_Z_BASE:
                self.get_logger().warn(f"Target Z ({pose.position.z:.3f}) too low! Clamping to {MIN_SAFE_Z_BASE} (Base Frame)")
                pose.position.z = MIN_SAFE_Z_BASE

            X_MIN_WALL = -0.06 - (0.40/2 - 0.0175)
            X_MAX_WALL = -0.06 + (0.40/2 - 0.0175)
            Y_MIN_WALL = -0.38 - (0.28/2 - 0.0175)
            Y_MAX_WALL = -0.38 + (0.28/2 - 0.0175)

            tx, ty = pose.position.x, pose.position.y
            dist_x = min(abs(tx - X_MIN_WALL), abs(tx - X_MAX_WALL))
            dist_y = min(abs(ty - Y_MIN_WALL), abs(ty - Y_MAX_WALL))
            min_dist = min(dist_x, dist_y)

            SAFE_DIST = 0.04

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

            self.get_logger().info(f"\033[94m변환한 bA: {pose}\033[0m")

            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.frame_id = 'base_link'
            pose_stamped.header.stamp = Clock().now().to_msg()

            self.get_logger().info(f"target_get_ba: (pose_stampled) {pose_stamped}")

            return pose_stamped
        else:
            self.get_logger().warn("결과 없음 -> 문 닫기 실행")
            import rclpy
            import sys
            while not self.goal(self.home, 'keep'):
                self.get_logger().warn("  >> 실패 → 재시도 중")
                time.sleep(0.5)
            rclpy.shutdown()
            sys.exit(0)
            return None

    def publish_target_marker(self, pose, id=0, r=1.0, g=0.0, b=0.0):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "grasp_target"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.lifetime = Duration(sec=0, nanosec=0)
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published Marker ID {id} to {marker.header.frame_id}")

    def publish_camera_center_callback(self):
        """주기적으로 현재 ee_pose와 ee2cam 행렬을 사용하여 카메라 중심 위치를 Rviz에 표시."""
        if self.ee_pose is None:
            return

        ee2cam = np.vstack((np.hstack((self.R_cal, self.t_cal)), [0, 0, 0, 1]))
        base2cam = self.ee_pose @ ee2cam

        cam_center_pose = Pose()
        cam_center_pose.position.x = base2cam[0,3]
        cam_center_pose.position.y = base2cam[1,3]
        cam_center_pose.position.z = base2cam[2,3]

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
