#!/usr/bin/env python3
import sys

_ros_paths = [(i, p) for i, p in enumerate(sys.path) if 'opt/ros' in p]
for _, p in reversed(_ros_paths):
    sys.path.remove(p)

import torch
import numpy as np
import cv2

for i, p in _ros_paths:
    sys.path.insert(i, p)

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from act_learning.policy import ACTPolicy
import threading
import time

class ACTInferenceNode(Node):
    def __init__(self, ckpt_path='/home/dyros/mcy_ws/act_learning/ckpts/policy_last.ckpt',
                 init_pose=False, init_data_path='', init_vel=10.0, init_thr=0.05):
        super().__init__('act_inference')
        self.init_pose      = init_pose
        self.init_data_path = init_data_path
        self.init_vel       = init_vel
        self.init_thr       = init_thr
        self.ready          = not init_pose

        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Loading model on {self.device}...")

        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            self.get_logger().error(f"Checkpoint not found: {ckpt_path}")
            raise SystemExit

        self.config = checkpoint['config']
        self.stats = checkpoint['stats']

        self.policy = ACTPolicy(self.config).to(self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()
        self.get_logger().info("Model loaded successfully.")

        # Subs
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/camera/color/image_raw/compressed',
            self.image_callback, qos_profile_sensor_data)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 1)

        # Pub
        self.action_pub = self.create_publisher(
            JointState, '/joint_ctrl_single', 10)

        # Buffers
        self.curr_image = None
        self.curr_qpos  = None
        self.joint_names = None
        self.lock = threading.Lock()

        # Init pose thread
        if self.init_pose:
            init_thread = threading.Thread(target=self.init_pose_loop)
            init_thread.daemon = True
            init_thread.start()

        # Control Loop (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

    def image_callback(self, msg):
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if cv_img is None:
                return
            cv_img = cv2.resize(cv_img, (320, 240))
            with self.lock:
                self.curr_image = cv_img
        except Exception as e:
            self.get_logger().error(f"Img error: {e}")

    def joint_callback(self, msg):
        with self.lock:
            self.joint_names = list(msg.name)
            self.curr_qpos   = np.array(msg.position)

    def normalize(self, data, stats_key):
        mean = self.stats[f'{stats_key}_mean']
        std  = self.stats[f'{stats_key}_std']
        return (data - mean) / std

    def unnormalize(self, data, stats_key):
        mean = self.stats[f'{stats_key}_mean']
        std  = self.stats[f'{stats_key}_std']
        return data * std + mean

    def init_pose_loop(self):
        import h5py, glob
        files = sorted(glob.glob(os.path.join(self.init_data_path, "*.h5")))
        if not files:
            self.get_logger().error(f"[Init] h5 파일 없음: {self.init_data_path}")
            self.ready = True
            return

        with h5py.File(files[0], 'r') as f:
            target = np.array(f['observations/qpos'][0])
        self.get_logger().info(f"[Init] 에피소드: {os.path.basename(files[0])}")
        self.get_logger().info(f"[Init] 목표 자세: {target.round(3)}")
        self.get_logger().info(f"[Init] 속도 {self.init_vel}% 로 이동 시작...")

        rate_s = 0.02
        while rclpy.ok():
            with self.lock:
                qpos  = self.curr_qpos
                names = self.joint_names
            if qpos is None:
                time.sleep(rate_s)
                continue

            err = np.abs(target - qpos)
            self.get_logger().info(
                f"[Init] max_err={err.max():.4f}  thr={self.init_thr}",
                throttle_duration_sec=1.0)

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name     = names if names else \
                           ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","joint8"]
            msg.position = target.tolist()
            msg.velocity = [self.init_vel] * len(target)
            self.action_pub.publish(msg)

            if err.max() < self.init_thr:
                self.get_logger().info("[Init] 목표 자세 도달! 추론 시작.")
                self.ready = True
                break
            time.sleep(rate_s)

    def control_loop(self):
        if not self.ready:
            return

        with self.lock:
            img   = self.curr_image
            qpos  = self.curr_qpos
            names = self.joint_names

        if img is None or qpos is None:
            self.get_logger().warn("Waiting for data...", throttle_duration_sec=1)
            return

        start_time = time.time()

        with torch.no_grad():
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            qpos_norm   = self.normalize(qpos, 'qpos')
            qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)

            action_preds, _, _ = self.policy(qpos_tensor, img_tensor, actions=None)

            next_action_norm = action_preds[0, 0].cpu().numpy()
            next_action      = self.unnormalize(next_action_norm, 'qpos')

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name     = names if names else \
                           ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","joint8"]
            msg.position = next_action.tolist()
            msg.velocity = [20.0] * len(next_action)
            self.action_pub.publish(msg)

            process_dur = time.time() - start_time
            self.get_logger().info(
                f"Inference {1.0/max(process_dur,0.001):.1f} Hz  "
                f"current: {qpos.round(3)}  action[0]: {next_action.round(3)}",
                throttle_duration_sec=1.0)

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='/home/dyros/mcy_ws/act_learning/ckpts/policy_last.ckpt')
    parser.add_argument('--init_pose', action='store_true')
    parser.add_argument('--init_data_path', type=str,
                        default='/home/dyros/mcy_ws/act_learning/data')
    parser.add_argument('--init_vel',  type=float, default=10.0)
    parser.add_argument('--init_thr',  type=float, default=0.05)
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ACTInferenceNode(
        ckpt_path=parsed.ckpt_path,
        init_pose=parsed.init_pose,
        init_data_path=parsed.init_data_path,
        init_vel=parsed.init_vel,
        init_thr=parsed.init_thr)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
