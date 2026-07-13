"""
지정한 자세로 로봇 이동
사용법:
  python go_to_pose.py wm_mid
  python go_to_pose.py wm_view
  python go_to_pose.py ba_init
"""
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

POSES = {
    'wm_mid':  [1.569035,  0.043279,  0.026288, -0.075341,  0.159264,  0.256235],
    'wm_view': [1.562546,  0.641172, -1.202764, -0.089156,  1.286792, -1.478571],
    'wm_init': [1.5640,    0.4538,   -1.5319,   -0.0488,    1.2797,    0.3169  ],
    'wm_end':  [1.571,     2.146,    -2.268,     0.000,     0.594,     0.000   ],
    'ff': [1.5533, 0.1396, -0.4014, 0.0349, 0.3665, 0.1745, 0.0, 0.0],
    'ba_init': None,  # basket_ori.py에서 확인 필요
}

ARM_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
# MoveIt 플래너 경유 (충돌 체크 O): /moveit_action/arm_controller/follow_joint_trajectory
# 직접 컨트롤러   (충돌 체크 X): /arm_controller/follow_joint_trajectory
ACTION_NAME = '/arm_controller/follow_joint_trajectory'


class GoToPose(Node):
    def __init__(self, pose_name):
        super().__init__('go_to_pose')
        self._client = ActionClient(self, FollowJointTrajectory, ACTION_NAME)
        self._pose_name = pose_name
        self._joint_values = POSES[pose_name]

    def send(self):
        self.get_logger().info(f'액션 서버 대기 중...')
        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('액션 서버 연결 실패')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in self._joint_values]
        point.time_from_start = Duration(sec=5, nanosec=0)
        goal.trajectory.points = [point]

        self.get_logger().info(f'{self._pose_name} 로 이동 중: {self._joint_values}')
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal 거부됨')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info(f'완료: {self._pose_name}')
        return True


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in POSES:
        print(f"사용법: python go_to_pose.py <pose_name>")
        print(f"사용 가능: {list(POSES.keys())}")
        sys.exit(1)

    pose_name = sys.argv[1]
    if POSES[pose_name] is None:
        print(f"{pose_name} 값이 정의되지 않았습니다. basket_ori.py에서 확인 후 추가하세요.")
        sys.exit(1)

    rclpy.init()
    node = GoToPose(pose_name)
    node.send()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
