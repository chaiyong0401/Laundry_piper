#!/usr/bin/env python3
"""
Teaching mode waypoint recorder
  - 팔을 원하는 자세에 놓고 엔터 → 기록
  - 'u' + 엔터 → 마지막 waypoint 취소
  - 'q' + 엔터 → 종료 및 결과 출력

실행:
  python3 record_waypoints.py
  python3 record_waypoints.py --name door_open   # waypoint 이름 prefix 지정
"""

import sys
import argparse
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')
        self.latest = None
        self.lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._cb, 10)

    def _cb(self, msg):
        with self.lock:
            self.latest = list(msg.position[:6])

    def get_current(self):
        with self.lock:
            return list(self.latest) if self.latest else None


def spin_thread(node):
    rclpy.spin(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='waypoint', help='waypoint 변수명 prefix')
    args = parser.parse_args()

    rclpy.init()
    node = WaypointRecorder()

    t = threading.Thread(target=spin_thread, args=(node,), daemon=True)
    t.start()

    poses = []
    print("=== Waypoint Recorder ===")
    print("  엔터     : 현재 자세 기록")
    print("  u + 엔터 : 마지막 기록 취소")
    print("  q + 엔터 : 종료 및 결과 출력")
    print("=" * 26)

    while True:
        try:
            key = input(f"[{len(poses)}개 기록됨] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key == 'q':
            break

        elif key == 'u':
            if poses:
                removed = poses.pop()
                print(f"  취소: {[round(v, 4) for v in removed]}")
            else:
                print("  취소할 waypoint 없음")

        else:
            pos = node.get_current()
            if pos is None:
                print("  /joint_states 수신 대기 중...")
                continue
            poses.append(pos)
            print(f"  [{len(poses)}] {[round(v, 4) for v in pos]}")

    node.destroy_node()
    rclpy.shutdown()

    if not poses:
        print("\n기록된 waypoint 없음.")
        return

    print("\n" + "=" * 50)
    print("=== basket_ori.py 에 붙여넣기 ===")
    print("=" * 50)
    print(f"self.{args.name}_poses = [")
    for p in poses:
        formatted = [round(v, 4) for v in p]
        print(f"    {formatted + [0.0, 0.0]},")
    print("]")

    print("\n=== 실행 코드 ===")
    print(f"for pose in self.{args.name}_poses:")
    print(f"    while not self.goal(pose, 'keep'):")
    print(f"        self.get_logger().warn('재시도 중')")


if __name__ == '__main__':
    main()
