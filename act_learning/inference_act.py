#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from act_learning.policy import ACTPolicy
from act_learning.utils import PiperDataset # For normalization utils
import threading
import time

class ACTInferenceNode(Node):
    def __init__(self):
        super().__init__('act_inference')
        
        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Loading model on {self.device}...")
        
        ckpt_path = '/home/dyros/mcy_ws/act_learning/ckpts/policy_last.ckpt'
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.stats = checkpoint['stats']
        
        self.policy = ACTPolicy(self.config).to(self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()
        self.get_logger().info("Model loaded successfully.")
        
        # ROS
        self.cv_bridge = CvBridge()
        
        # Subs
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 1)
        self.joint_sub = self.create_subscription(
            JointState, '/follower/joint_states', self.joint_callback, 1)
            
        # Pub
        self.action_pub = self.create_publisher(
            JointState, '/follower/joint_ctrl_single', 10)
            
        # Buffers
        self.curr_image = None
        self.curr_qpos = None
        
        # Control Loop
        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz
        
    def image_callback(self, msg):
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Preprocess: Resize to 320x240 matched training
            cv_img = cv2.resize(cv_img, (320, 240))
            self.curr_image = cv_img
        except Exception as e:
            self.get_logger().error(f"Img error: {e}")

    def joint_callback(self, msg):
        # assumes joint names order matches training (usually yes/hardware fixed)
        # Taking position
        self.curr_qpos = np.array(msg.position)

    def normalize(self, data, stats_key):
        mean = self.stats[f'{stats_key}_mean']
        std = self.stats[f'{stats_key}_std']
        return (data - mean) / std

    def unnormalize(self, data, stats_key):
        mean = self.stats[f'{stats_key}_mean']
        std = self.stats[f'{stats_key}_std']
        return data * std + mean

    def control_loop(self):
        start_time = time.time()
        
        if self.curr_image is None or self.curr_qpos is None:
            self.get_logger().warn("Waiting for data...", throttle_duration_sec=1)
            return
            
        # Prepare Input
        with torch.no_grad():
            # 1. Image
            img_tensor = torch.from_numpy(self.curr_image).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device) # [1, C, H, W]
            
            # 2. Qpos (Normalize)
            qpos_norm = self.normalize(self.curr_qpos, 'qpos')
            qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)
            
            # 3. Inference
            # For inference, actions=None (CVAE samples prior)
            action_preds, _, _ = self.policy(qpos_tensor, img_tensor, actions=None)
            
            # 4. Process Output (Receding Horizon: Take 1st step)
            # action_preds: [1, Chunk, Dim]
            next_action_norm = action_preds[0, 0].cpu().numpy()
            
            # Unnormalize
            next_action = self.unnormalize(next_action_norm, 'qpos') # action mean/std same as qpos
            
            # 5. Publish
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"] # Dummy names or fixed
            msg.position = next_action.tolist()
            msg.velocity = [20.0] * len(next_action) # Speed limit
            
            self.action_pub.publish(msg)
            
            # Logging FPS
            process_dur = time.time() - start_time
            if hasattr(self, 'last_log_time'):
                if time.time() - self.last_log_time > 1.0:
                    self.get_logger().info(f"Inference Speed: {1.0/max(process_dur, 0.001):.1f} Hz (Process: {process_dur*1000:.1f}ms)")
                    self.last_log_time = time.time()
            else:
                self.last_log_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = ACTInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
