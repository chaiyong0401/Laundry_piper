import sys
import os

# HACK: Remove ROS python path to prefer Conda's OpenCV/NumPy
# ROS Humble usually adds /opt/ros/humble/lib/python3.10/site-packages
to_remove = [p for p in sys.path if 'opt/ros' in p]
for p in to_remove:
    sys.path.remove(p)

import torch
import numpy as np
import cv2

# Restore ROS path for rclpy
for p in to_remove:
    sys.path.append(p)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
import threading
import time

# Import the Full Model (specifically the class user renamed)
from act_learning.policy_full import ACTPolicyBaseline
# Re-use utils from existing setup
from act_learning.utils import PiperDataset 

class ACTFullInferenceNode(Node):
    def __init__(self):
        super().__init__('act_full_inference')
        
        # Device Check
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Loading FULL model on {self.device}...")
        if self.device.type == 'cpu':
            self.get_logger().warn("Running FULL model on CPU. Latency might be high.")
        
        # Load Checkpoint
        # Assuming user copies the file to this location
        ckpt_path = '/home/dyros/mcy_ws/act_learning/ckpts/policy_full_last.ckpt'
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            self.get_logger().error(f"Checkpoint not found at {ckpt_path}. Please copy it from your training PC.")
            raise SystemExit
            
        self.config = checkpoint['config']
        self.stats = checkpoint['stats']
        
        # Instantiate Full Policy
        self.policy = ACTPolicyBaseline(self.config).to(self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()
        self.get_logger().info("Model loaded successfully.")
        
        # ROS Setup
        # self.cv_bridge = CvBridge()
        
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
        self.lock = threading.Lock()
        
        # Action Buffer for Interpolation
        # Stores (time, action) pairs
        self.action_chunk = None 
        self.chunk_start_time = 0.0
        self.dt = 0.02 # Assuming model was trained with 50Hz data (0.02s)
                       # If training was 30Hz, this should be 0.033. 
                       # Config usually has this, but defaulting to 50Hz (standard ACT).
        
        # Recording for Debug
        self.history_actual = []
        self.history_target = []
        
        # 1. Inference Thread (As fast as possible)
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        # 2. Control Timer (High Frequency 100Hz)
        self.control_timer = self.create_timer(0.01, self.control_loop) 
        
    def image_callback(self, msg):
        try:
            # Manual bridge
            dtype = np.uint8
            n_channels = 3
            if msg.encoding in ['bgr8', 'rgb8']:
                img_arr = np.array(msg.data, dtype=dtype).reshape(msg.height, msg.width, n_channels)
                img_arr = np.ascontiguousarray(img_arr)
                if msg.encoding == 'rgb8':
                     img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                cv_img = img_arr
            else:
                 return

            cv_img = cv2.resize(cv_img, (320, 240))
            with self.lock:
                self.curr_image = cv_img
        except Exception as e:
            self.get_logger().error(f"Img error: {e}")

    def joint_callback(self, msg):
        with self.lock:
            self.curr_qpos = np.array(msg.position)

    def normalize(self, data, stats_key):
        mean = self.stats[f'{stats_key}_mean']
        std = self.stats[f'{stats_key}_std']
        return (data - mean) / std

    def unnormalize(self, data, stats_key):
        mean = self.stats[f'{stats_key}_mean']
        std = self.stats[f'{stats_key}_std']
        return data * std + mean

    def inference_loop(self):
        # Runs in background, updates self.action_queue
        while rclpy.ok():
            with self.lock:
                img = self.curr_image
                qpos = self.curr_qpos
            
            if img is None or qpos is None:
                time.sleep(0.1)
                continue
                
            # Prepare Input
            with torch.no_grad():
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                qpos_norm = self.normalize(qpos, 'qpos')
                qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)
                
                # Inference
                t0 = time.time()
                action_preds, _, _ = self.policy(qpos_tensor, img_tensor, actions=None)
                
                # Process Output: Get entire chunk
                # [1, Chunk, Dim] -> [Chunk, Dim]
                chunk_norm = action_preds[0].cpu().numpy()
                chunk = self.unnormalize(chunk_norm, 'qpos')
                
                # Add to Temporal Ensemble Buffer
                with self.lock:
                    if self.action_chunk is None:
                        self.action_chunk = []
                    # Append (start_time, trajectory)
                    self.action_chunk.append((time.time(), chunk))
                    
                    # Clean up old chunks that are fully past
                    # Keep last 2 seconds (chunk duration usually)
                    # Simple cleanup: remove chunks older than 2s
                    now = time.time()
                    self.action_chunk = [ct for ct in self.action_chunk if (now - ct[0]) < 2.0]
            
            time.sleep(0.01)

    def control_loop(self):
        # Temporal Ensembling
        with self.lock:
            chunk_list = self.action_chunk
            curr_qpos = self.curr_qpos
            
        if chunk_list is None or not chunk_list or curr_qpos is None:
            return
            
        now = time.time()
        
        # Aggregate overlapping actions
        actions = []
        weights = []
        k = 0.01 # weight exponent factor
        
        for start_t, chunk in chunk_list:
            time_since_start = now - start_t
            
            # Index calculation
            idx_float = time_since_start / self.dt
            idx = int(idx_float)
            
            if 0 <= idx < len(chunk) - 1:
                # Linear Interpolation
                alpha = idx_float - idx
                val = chunk[idx] * (1 - alpha) + chunk[idx+1] * alpha
                
                # Exponential Weighting: exp(-k * t)
                # Gives more weight to newer predictions (t small) 
                # OR ACT standard: w = exp(-k * index)? 
                # Actually standard ACT uses: weight = exp(-k * time_since_start)
                w = np.exp(-k * (idx_float))
                
                actions.append(val)
                weights.append(w)
                
        if not actions:
            # Fallback if no valid chunk overlaps (should rarely happen if inference is fast enough)
            # Use the last frame of the latest chunk
            latest_chunk = chunk_list[-1][1]
            target_action = latest_chunk[-1] 
        else:
            # Weighted Average
            actions = np.array(actions)
            weights = np.array(weights)
            target_action = np.sum(actions * weights[:, None], axis=0) / np.sum(weights)
            
        # Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
        msg.position = target_action.tolist()
        msg.velocity = [0.0] * len(target_action)
        
        self.action_pub.publish(msg)
        
        # Record
        if len(self.history_actual) < 1000:
            self.history_actual.append(curr_qpos)
            self.history_target.append(target_action)

    def save_plot(self):
        if len(self.history_actual) == 0:
            print("No data to plot.")
            return
            
        print("Saving analysis plot...")
        import matplotlib.pyplot as plt
        
        actual = np.array(self.history_actual)
        target = np.array(self.history_target)
        
        dim = actual.shape[1]
        fig, axes = plt.subplots(dim, 1, figsize=(10, 2*dim))
        if dim == 1: axes = [axes]

        for i in range(dim):
            ax = axes[i]
            ax.plot(actual[:, i], label='Actual (Robot)', color='black')
            ax.plot(target[:, i], label='Predicted (Model)', color='blue', linestyle='--')
            ax.set_title(f'Joint {i+1}')
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        plt.savefig('inference_analysis.png')
        print("Saved inference_analysis.png")

def main(args=None):
    rclpy.init(args=args)
    node = ACTFullInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_plot() # Plot on exit
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
