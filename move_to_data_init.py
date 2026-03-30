#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import h5py
import numpy as np
import os
import glob
import time

class MoveToDataInit(Node):
    def __init__(self):
        super().__init__('move_to_data_init')
        
        # 1. Find latest data file
        data_dir = '/home/dyros/mcy_ws/data'
        files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        if not files:
            self.get_logger().error(f"No .h5 files found in {data_dir}")
            raise SystemExit
            
        latest_file = files[-1]
        self.get_logger().info(f"Loading initial pose from: {latest_file}")
        
        # 2. Read first frame qpos
        with h5py.File(latest_file, 'r') as f:
            # qpos shape: [N, Dim]
            init_qpos = f['observations/qpos'][0]
            
        self.target_qpos = np.array(init_qpos)
        self.get_logger().info(f"Target Init Pose: {self.target_qpos}")
        
        # 3. Setup ROS
        # Need to know START pose to interpolate safely
        self.current_qpos = None
        self.joint_sub = self.create_subscription(JointState, '/follower/joint_states', self.joint_callback, 1)
        
        self.pub = self.create_publisher(JointState, '/follower/joint_ctrl_single', 10)
        self.timer = self.create_timer(0.02, self.timer_callback) # 50Hz
        
        # Interpolation State
        self.start_qpos = None
        self.start_time = None
        self.duration = 5.0 # 5 Seconds for safety
        self.moving = False

    def joint_callback(self, msg):
        # Only update if not moving (lock start pose)
        if not self.moving:
            self.current_qpos = np.array(msg.position)

    def timer_callback(self):
        if self.current_qpos is None:
            self.get_logger().info("Waiting for current joint state...", throttle_duration_sec=1.0)
            return

        if not self.moving:
            # Start Movement
            self.start_qpos = self.current_qpos
            
            # Check Dimension Mismatch
            if len(self.start_qpos) != len(self.target_qpos):
                 self.get_logger().warn(f"Size Mismatch! Curr:{len(self.start_qpos)} vs Target:{len(self.target_qpos)}.")
                 min_len = min(len(self.start_qpos), len(self.target_qpos))
                 self.start_qpos = self.start_qpos[:min_len]
                 self.target_qpos = self.target_qpos[:min_len]
            
            self.start_time = time.time()
            self.moving = True
            self.phase = 1 # Start Phase 1
            self.get_logger().info(f"PHASE 1 START: Moving Joints 2-8 (Duration: {self.duration}s)...")
        
        # Interpolate
        now = time.time()
        elapsed = now - self.start_time
        
        cmd_qpos = self.start_qpos.copy()
        
        if self.phase == 1:
            # Phase 1: Move J2,J3,J5,J6,J7,J8. 
            # Keep J1 (Index 0) and J4 (Index 3) at Start.
            if elapsed > self.duration:
                # Phase 1 Done
                self.phase = 2
                self.start_time = time.time() # Reset timer for Phase 2
                self.get_logger().info(f"PHASE 1 DONE. PHASE 2 START: Moving Joint 1 & 4 (Duration: {self.duration}s)...")
                # Set cmd to target for others just in case
                cmd_qpos[:] = self.target_qpos[:]
                # Reset J1, J4 to start for now (will be handled in phase 2 logic if elapsed=0)
                cmd_qpos[0] = self.start_qpos[0]
                cmd_qpos[4] = self.start_qpos[4]
            else:
                alpha = elapsed / self.duration
                
                # All interpolate first
                cmd_qpos = (1 - alpha) * self.start_qpos + alpha * self.target_qpos
                
                # Overwrite J1 and J4 with START position (Freeze)
                cmd_qpos[0] = self.start_qpos[0]
                cmd_qpos[4] = self.start_qpos[4]
                
        elif self.phase == 2:
            # Phase 2: Move J1 (0) and J4 (3). Keep others at Target.
            
            # Base is Target for everyone
            cmd_qpos[:] = self.target_qpos[:]
            
            if elapsed > self.duration:
                # Phase 2 Done
                if elapsed > self.duration + 1.0:
                     self.get_logger().info("Moved to Init Pose Complete.")
                     raise SystemExit
            else:
                alpha = elapsed / self.duration
                # J1 and J4 interpolate from START to TARGET
                cmd_qpos[0] = (1 - alpha) * self.start_qpos[0] + alpha * self.target_qpos[0]
                cmd_qpos[4] = (1 - alpha) * self.start_qpos[4] + alpha * self.target_qpos[4]
            
        # Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
        
        # Assign to 8 dim
        final_pos = [0.0] * 8
        for i in range(min(len(cmd_qpos), 8)):
            final_pos[i] = float(cmd_qpos[i])
            
        msg.position = final_pos
        msg.velocity = [0.0] * 8 
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MoveToDataInit()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
