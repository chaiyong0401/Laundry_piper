#!/usr/bin/env python3
"""
return_init.py
==============
작업 완료 후 로봇을 안전하게 init pose로 복귀시키는 스크립트.

동작 순서:
  1. 현재 로봇 자세와 가장 가까운 학습 에피소드 마지막 프레임 탐색
  2. 해당 에피소드를 역순(reverse)으로 재생 → 세탁기 밖으로 안전하게 탈출
  3. init pose (첫 에피소드 첫 프레임)로 이동

Usage:
  python3 -m act_learning.return_init \
    --data_path /home/dyros/mcy_ws/act_learning/data \
    --vel 10.0 \
    --thr 0.08
"""

import os
import glob
import argparse
import time
import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ReturnInitNode(Node):
    def __init__(self, data_path, vel, thr, step_dt):
        super().__init__('return_init')
        self.data_path = data_path
        self.vel       = vel
        self.thr       = thr
        self.step_dt   = step_dt

        self.curr_qpos   = None
        self.joint_names = None

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 1)
        self.action_pub = self.create_publisher(
            JointState, '/joint_ctrl_single', 10)

    def _joint_cb(self, msg):
        self.joint_names = list(msg.name)
        self.curr_qpos   = np.array(msg.position)

    def _publish(self, qpos, vel=None):
        if vel is None:
            vel = self.vel
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = self.joint_names if self.joint_names else \
                       ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","joint8"]
        msg.position = qpos.tolist()
        msg.velocity = [vel] * len(qpos)
        self.action_pub.publish(msg)

    def _wait_for_qpos(self, timeout=10.0):
        """joint_states 수신 대기."""
        deadline = time.time() + timeout
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.curr_qpos is not None:
                return True
        return False

    def _move_to(self, target, label=''):
        """target에 도달할 때까지 반복 퍼블리시."""
        self.get_logger().info(f"[{label}] 이동 → {target.round(3)}")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.curr_qpos is None:
                continue
            self._publish(target)
            err = np.max(np.abs(target - self.curr_qpos))
            if err < self.thr:
                self.get_logger().info(f"[{label}] 도달 (err={err:.4f})")
                break
            time.sleep(self.step_dt)

    def run(self):
        # 1. 현재 qpos 수신 대기
        self.get_logger().info("[Return] joint_states 대기 중...")
        if not self._wait_for_qpos():
            self.get_logger().error("[Return] joint_states 수신 실패. 로봇 드라이버 확인 필요.")
            return

        self.get_logger().info(f"[Return] 현재 자세: {self.curr_qpos.round(3)}")

        # 2. 가장 가까운 에피소드 탐색
        files = sorted(glob.glob(os.path.join(self.data_path, "*.h5")))
        if not files:
            self.get_logger().error(f"[Return] h5 파일 없음: {self.data_path}")
            return

        curr = self.curr_qpos.copy()
        best_file, best_err = None, 1e9
        for f_path in files:
            with h5py.File(f_path, 'r') as f:
                last = np.array(f['observations/qpos'][-1])
            err = np.max(np.abs(last - curr))
            if err < best_err:
                best_err, best_file = err, f_path

        self.get_logger().info(
            f"[Return] 선택 에피소드: {os.path.basename(best_file)}  "
            f"마지막프레임 오차={best_err:.3f} rad")

        # 에피소드 qpos 로드
        with h5py.File(best_file, 'r') as f:
            traj     = np.array(f['observations/qpos'])   # [T, 8]
            init_pos = np.array(f['observations/qpos'][0])

        traj_rev = traj[::-1]  # 역순
        n = len(traj_rev)
        self.get_logger().info(f"[Return] {n} 프레임 역재생 시작 (속도 {self.vel}%)")

        # 3. 역순 재생
        for i, waypoint in enumerate(traj_rev):
            if not rclpy.ok():
                break

            self._publish(waypoint)

            # 각 waypoint에 thr 이내로 도달하면 다음으로
            deadline = time.time() + 3.0
            while time.time() < deadline and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
                if self.curr_qpos is not None and \
                   np.max(np.abs(waypoint - self.curr_qpos)) < self.thr:
                    break
                time.sleep(self.step_dt)

            if i % 100 == 0 or i == n - 1:
                self.get_logger().info(f"[Return] {i+1}/{n} 완료")

        # 4. Init pose로 최종 이동
        self.get_logger().info("[Return] 역재생 완료. Init pose로 최종 이동...")
        self._move_to(init_pos, label='InitPose')
        self.get_logger().info("[Return] 복귀 완료!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/home/dyros/mcy_ws/act_learning/data',
                        help='h5 에피소드 폴더')
    parser.add_argument('--vel',     type=float, default=10.0,
                        help='이동 속도 %% (default: 10)')
    parser.add_argument('--thr',     type=float, default=0.08,
                        help='waypoint 도달 판정 오차 rad (default: 0.08)')
    parser.add_argument('--step_dt', type=float, default=0.02,
                        help='waypoint 간 대기 시간 s (default: 0.02)')
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ReturnInitNode(
        data_path=parsed.data_path,
        vel=parsed.vel,
        thr=parsed.thr,
        step_dt=parsed.step_dt)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn("[Return] 사용자 중단.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
