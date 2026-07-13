import sys
import os

# HACK: Remove ROS python path to prefer Conda's OpenCV/NumPy
# ROS Humble usually adds /opt/ros/humble/lib/python3.10/site-packages
# Save original (index, path) pairs to restore at exact positions
_ros_paths = [(i, p) for i, p in enumerate(sys.path) if 'opt/ros' in p]
for _, p in reversed(_ros_paths):
    sys.path.remove(p)

import torch
import numpy as np
import cv2

# Restore ROS paths at their original positions (not appended to end)
for i, p in _ros_paths:
    sys.path.insert(i, p)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
import threading
import time

# Import the Full Model (specifically the class user renamed)
from act_learning.policy_full import ACTPolicyBaseline
# Re-use utils from existing setup
from act_learning.utils import PiperDataset 

class ACTFullInferenceNode(Node):
    def __init__(self, ckpt_path='/home/dyros/mcy_ws/act_learning/ckpts/policy_full_last.ckpt',
                 init_pose=False, init_data_path='', init_vel=10.0, init_thr=0.05):
        super().__init__('act_full_inference')
        self.init_pose      = init_pose
        self.init_data_path = init_data_path
        self.init_vel       = init_vel
        self.init_thr       = init_thr
        self.ready          = not init_pose

        # Device Check
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Loading FULL model on {self.device}...")
        if self.device.type == 'cpu':
            self.get_logger().warn("Running FULL model on CPU. Latency might be high.")

        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            self.get_logger().error(f"Checkpoint not found: {ckpt_path}")
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
        from rclpy.qos import qos_profile_sensor_data
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 1)

        # 카메라: compressed topic (~50KB JPEG) — raw(2.7MB)보다 DDS 전달 안정적
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/camera/color/image_raw/compressed',
            self.image_callback, qos_profile_sensor_data)

        # Pub
        self.action_pub = self.create_publisher(
            JointState, '/joint_ctrl_single', 10)
            
        # Buffers
        self.curr_image = None
        self.curr_qpos = None
        self.joint_names = None
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

        # 1-b. Init pose thread (init_pose=True 일 때만 동작)
        if self.init_pose:
            init_thread = threading.Thread(target=self.init_pose_loop)
            init_thread.daemon = True
            init_thread.start()

        # 2. Control Timer (High Frequency 100Hz)
        self.control_timer = self.create_timer(0.01, self.control_loop)
        
    def image_callback(self, msg):
        self.get_logger().info(
            f"[IMG CB] format={msg.format} data_len={len(msg.data)}",
            throttle_duration_sec=2.0)
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # -> BGR
            if cv_img is None:
                self.get_logger().warn("[IMG CB] imdecode 실패")
                return
            cv_img = cv2.resize(cv_img, (320, 240))
            with self.lock:
                self.curr_image = cv_img
        except Exception as e:
            self.get_logger().error(f"Img error: {e}")

    def joint_callback(self, msg):
        with self.lock:
            self.joint_names = list(msg.name)
            self.curr_qpos = np.array(msg.position)

    def normalize(self, data, stats_key):
        mean = self.stats[f'{stats_key}_mean']
        std = self.stats[f'{stats_key}_std']
        return (data - mean) / std

    def unnormalize(self, data, stats_key):
        mean = self.stats[f'{stats_key}_mean']
        std = self.stats[f'{stats_key}_std']
        return data * std + mean

    def init_pose_loop(self):
        """학습 데이터 첫 에피소드 첫 프레임 qpos로 로봇을 이동."""
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

        rate_s = 0.02  # 50Hz
        while rclpy.ok():
            with self.lock:
                qpos = self.curr_qpos
                names = self.joint_names
            if qpos is None:
                time.sleep(rate_s)
                continue

            err = np.abs(target - qpos)
            self.get_logger().info(
                f"[Init] max_err={err.max():.4f}  thr={self.init_thr}",
                throttle_duration_sec=1.0)

            # 목표 자세 퍼블리시
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = names if names else \
                       ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","joint8"]
            msg.position = target.tolist()
            msg.velocity = [self.init_vel] * len(target)
            self.action_pub.publish(msg)

            if err.max() < self.init_thr:
                self.get_logger().info("[Init] 목표 자세 도달! 추론 시작.")
                self.ready = True
                break
            time.sleep(rate_s)

    def inference_loop(self):
        # Runs in background, updates self.action_queue
        while rclpy.ok():
            with self.lock:
                img = self.curr_image
                qpos = self.curr_qpos
            
            if img is None or qpos is None:
                self.get_logger().warn(
                    f"Waiting for data... image={'OK' if img is not None else 'MISSING'}  "
                    f"qpos={'OK' if qpos is not None else 'MISSING'}",
                    throttle_duration_sec=2.0)
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
                
                t_infer = time.time() - t0
                self.get_logger().info(
                    f"Inference OK  {1.0/max(t_infer,1e-4):.1f} Hz\n"
                    f"  current : {qpos.round(3)}\n"
                    f"  action[0]: {chunk[0].round(3)}",
                    throttle_duration_sec=1.0)

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
        if not self.ready:
            return

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
        with self.lock:
            names = self.joint_names
        msg.name = names if names else \
                   ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
        msg.position = target_action.tolist()
        msg.velocity = [20.0] * len(target_action)
        
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='/home/dyros/mcy_ws/act_learning/ckpts/policy_full_last.ckpt')
    parser.add_argument('--init_pose', action='store_true',
                        help='추론 전 학습 데이터 첫 프레임 자세로 로봇 이동')
    parser.add_argument('--init_data_path', type=str,
                        default='/home/dyros/mcy_ws/act_learning/data',
                        help='초기 자세를 읽을 h5 데이터 폴더')
    parser.add_argument('--init_vel',  type=float, default=10.0,
                        help='초기 이동 속도 %% (default: 10)')
    parser.add_argument('--init_thr',  type=float, default=0.05,
                        help='도달 판정 오차 rad (default: 0.05)')
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ACTFullInferenceNode(ckpt_path=parsed.ckpt_path,
                                init_pose=parsed.init_pose,
                                init_data_path=parsed.init_data_path,
                                init_vel=parsed.init_vel,
                                init_thr=parsed.init_thr)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_plot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
