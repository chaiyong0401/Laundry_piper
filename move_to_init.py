#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class MoveToInit(Node):
    def __init__(self):
        super().__init__('move_to_init')
        self.pub = self.create_publisher(JointState, '/joint_ctrl_single', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.count = 0
        self.max_count = 20 # Send for 2 seconds

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # 7 joints + gripper? Or 6+gripper? 
        # Based on piper_ctrl_single_node.py it has 8 elements (joint1..joint7, joint8/gripper)
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
        
        # Init Pose (All Zeros)
        # You can adjust this if INIT means something else
        msg.position = [1.5058533, 0.30410125200000004, -0.15608891200000002, -0.15819963600000003, 0.0684677, -1.267376376,0.0, 0.0]
        msg.position = [1.535473212,1.2058165,-1.5244486040000003,-0.045633504000000005, 1.271702488,-1.3131843200000002,0.0,0.0]

        # msg.position= [1.574251224, 0.336651756, -1.159398016, -0.185900708, 1.287576528, -1.357858404, 0.0, 0.0]
        # msg.position  = [1.715346952, 0.323793323, -1.007969640, -0.335967800, 1.178093273, -1.223699806, 0.0, 0.0]
        msg.velocity = [20.0] * 8 # Speed limit
        
        self.pub.publish(msg)
        self.get_logger().info(f"Publishing Init Pose... {self.count}")
        
        self.count += 1
        if self.count >= self.max_count:
            self.get_logger().info("Done.")
            raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = MoveToInit()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()