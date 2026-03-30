#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class PiperMSBridge(Node):
    def __init__(self):
        super().__init__('piper_ms_bridge')
        
        # Parameters for namespaces
        self.declare_parameter('leader_ns', '/leader')
        self.declare_parameter('follower_ns', '/follower')
        
        self.leader_ns = self.get_parameter('leader_ns').get_parameter_value().string_value
        self.follower_ns = self.get_parameter('follower_ns').get_parameter_value().string_value
        
        self.get_logger().info(f"Starting Multi-Topic Bridge: {self.leader_ns} -> {self.follower_ns}")

        # Subscribe to Leader's joint states (Feedback mode)
        self.create_subscription(
            JointState,
            f'{self.leader_ns}/joint_states',
            self.joint_callback,
            10
        )
        
        # Subscribe to Leader's joint ctrl (Master/Teach mode)
        # In Master mode, the robot sends control frames (0x15x) instead of feedback (0x2Ax)
        # However, testing shows this sends 0 values while Drag Teaching, interfering with valid joint_states.
        # self.create_subscription(
        #     JointState,
        #     f'{self.leader_ns}/joint_ctrl',
        #     self.joint_callback,
        #     10
        # )
        
        # Publish to Follower's control topic
        self.follower_pub = self.create_publisher(
            JointState,
            f'{self.follower_ns}/joint_ctrl_single',
            10
        )
        
        self.last_sync_time = 0

    def joint_callback(self, msg):
        # Create a new command message for the follower
        follow_cmd = JointState()
        follow_cmd.header.stamp = self.get_clock().now().to_msg()
        follow_cmd.name = msg.name
        follow_cmd.position = msg.position
        
        # Set a safe constant velocity for tracking
        follow_cmd.velocity = [30.0] * len(msg.position)
        
        # Copy efforts
        follow_cmd.effort = msg.effort
        
        self.follower_pub.publish(follow_cmd)
        
        # Debug logging every 2 seconds to confirm data flow
        now = time.time()
        if now - self.last_sync_time > 2.0:
            gripper_val = msg.position[6] if len(msg.position) > 6 else 0.0
            self.get_logger().info(f"Syncing Leader -> Follower (Joints: {len(msg.position)}, Gripper: {gripper_val:.4f})")
            self.last_sync_time = now

def main(args=None):
    rclpy.init(args=args)
    node = PiperMSBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
