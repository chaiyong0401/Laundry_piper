#!/usr/bin/env python3
"""
rosbag /joint_states 기록을 arm controller로 재생.

사용법:
  python3 replay_bag.py --bag door_open_bag
  python3 replay_bag.py --bag door_open_bag --hz 20 --speed 1.0
"""

import argparse
import sqlite3
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import numpy as np


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
ARM_ACTION  = '/arm_controller/follow_joint_trajectory'


def load_bag(bag_dir: str, hz: float):
    """bag에서 /joint_states 읽어 (timestamp_sec, positions) 리스트 반환 (hz로 다운샘플)."""
    import glob, os
    db_files = sorted(glob.glob(os.path.join(bag_dir, '*.db3')))
    if not db_files:
        raise FileNotFoundError(f"db3 파일 없음: {bag_dir}")

    JointState = get_message('sensor_msgs/msg/JointState')
    interval   = 1.0 / hz
    records    = []
    last_t     = -1e9

    for db_path in db_files:
        con = sqlite3.connect(db_path)
        for ts_ns, data in con.execute(
            "SELECT timestamp, data FROM messages ORDER BY timestamp"
        ):
            t = ts_ns * 1e-9
            if t - last_t < interval:
                continue
            msg = deserialize_message(data, JointState)
            pos = list(msg.position[:6])
            records.append((t, pos))
            last_t = t
        con.close()

    return records


class BagReplayer(Node):
    def __init__(self):
        super().__init__('bag_replayer')
        self._client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)

    def replay(self, records, speed: float):
        self.get_logger().info(f"arm controller 대기 중...")
        self._client.wait_for_server()
        self.get_logger().info(f"연결 완료. {len(records)}개 포인트 전송 준비.")

        t0      = records[0][0]
        goal    = FollowJointTrajectory.Goal()
        traj    = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        for t, pos in records:
            elapsed = (t - t0) / speed
            sec     = int(elapsed)
            nanosec = int((elapsed - sec) * 1e9)
            pt = JointTrajectoryPoint()
            pt.positions       = pos
            pt.velocities      = [0.0] * 6
            pt.time_from_start = Duration(sec=sec, nanosec=nanosec)
            traj.points.append(pt)

        goal.trajectory = traj

        self.get_logger().info("전송 중...")
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("goal 거절됨")
            return

        self.get_logger().info("실행 중... (완료 대기)")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("재생 완료.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',   required=True, help='bag 디렉토리 경로')
    parser.add_argument('--hz',    type=float, default=20.0, help='재생 샘플링 Hz (default: 20)')
    parser.add_argument('--speed', type=float, default=1.0,  help='재생 속도 배율 (default: 1.0, 느리게: 0.5)')
    args = parser.parse_args()

    records = load_bag(args.bag, args.hz)
    duration = records[-1][0] - records[0][0]
    print(f"[INFO] {len(records)}개 포인트 로드 ({duration:.1f}초 / {args.hz}Hz / speed={args.speed}x)")

    rclpy.init()
    node = BagReplayer()
    node.replay(records, args.speed)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
