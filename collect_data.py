#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import h5py
import numpy as np
import cv2
import time
from cv_bridge import CvBridge

# ROS Messages
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose
import message_filters

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # Parameters
        # Parameters
        # Auto-generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_file = f"data/episode_{timestamp}.h5"
        self.declare_parameter('output', self.output_file)
        self.output_file = self.get_parameter('output').value
        
        # Init buffers
        self.images = []
        self.qpos = [] # joint_states
        self.actions = [] # next joint_states ? just recording current for now.
        self.ee_pose = [] # x,y,z,qx,qy,qz,qw
        
        self.last_pose = None
        
        self.cv_bridge = CvBridge()
        
        # Subscribers
        
        # 1. Unstamped Pose (Manual Callback)
        self.pose_sub = self.create_subscription(
            Pose, 
            '/leader/end_pose', 
            self.pose_callback, 
            10
        )
        
        # 2. Stamped Topics (Sync)
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.joint_sub = message_filters.Subscriber(self, JointState, '/leader/joint_states')
        
        # Sync policy: Allow some slop (e.g. 0.05s)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.joint_sub], 
            queue_size=10, 
            slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.recording = False
        self.get_logger().info("Data Collector Initialized.")
        self.get_logger().info(f"Saving to {self.output_file}")
        self.get_logger().info("Press 'Ctrl+C' to stop and save.")
        self.get_logger().info("Waiting for data stream...")

    def pose_callback(self, msg):
        # Update latest pose
        # Pose: position(x,y,z), orientation(x,y,z,w)
        self.last_pose = [
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ]

    def sync_callback(self, img_msg, joint_msg):
        # Callback only triggered when Image and JointState are synced
        print(".", end="", flush=True)
        
        if self.last_pose is None:
            # We haven't received a pose yet, skip this frame or use zeros?
            # Better to skip to ensure data quality.
            return

        try:
            # 1. Image Processing
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            # Optimization: Resize to 320x240 immediately to save space
            cv_image = cv2.resize(cv_image, (320, 240))
            
            # 2. Joint Processing
            # JointState: position, velocity, effort. We take position.
            # Assuming 6 joints + 2 gripper?
            # Filter out gripper if needed, or keep all.
            # joint_msg.position is a tuple/list
            current_qpos = list(joint_msg.position)
            
            # 3. Store Data
            self.images.append(cv_image)
            self.qpos.append(current_qpos)
            self.ee_pose.append(self.last_pose)
            
        except Exception as e:
            self.get_logger().error(f"Error in processing: {e}")

    def save_data(self):
        self.get_logger().info("Saving data to HDF5...")
        
        if len(self.images) == 0:
            self.get_logger().warn("No data collected!")
            return

        with h5py.File(self.output_file, 'w') as f:
            # Create datasets
            # Images: (N, H, W, C)
            f.create_dataset('observations/images', data=np.array(self.images), compression="gzip")
            
            # Qpos: (N, D)
            f.create_dataset('observations/qpos', data=np.array(self.qpos))
            
            # EE Pose: (N, 7)
            f.create_dataset('observations/ee_pose', data=np.array(self.ee_pose))
            
            # Metadata
            f.attrs['num_samples'] = len(self.images)
            
        self.get_logger().info(f"Saved {len(self.images)} samples to {self.output_file}")

def main(args=None):
    rclpy.init(args=args)
    collector = DataCollector()
    
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        pass
    finally:
        collector.save_data()
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
