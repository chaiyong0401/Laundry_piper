import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import sys

class GripperDebug(Node):
    def __init__(self):
        super().__init__('gripper_debug_node')
        self.gripper_joint_names = ['joint7', 'joint8']
        
        # Connect to Gripper Controller directly (same as basket.py)
        # Note: If running with /follower namespace, this might need /follower prefix.
        # But user said they reverted to "Original Version" (Root namespace).
        self.topic = '/gripper_controller/follow_joint_trajectory'
        self._gripper_client = ActionClient(self, FollowJointTrajectory, self.topic)
        
        self.get_logger().info(f"Waiting for action server: {self.topic}")
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action server {self.topic} not available! Check namespace.")
            sys.exit(1)
        self.get_logger().info("Server connected.")

    def send_gripper(self, j7, j8):
        self.get_logger().info(f"Sending Gripper: joint7={j7}, joint8={j8}")
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.gripper_joint_names
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()
        
        point = JointTrajectoryPoint()
        point.positions = [float(j7), float(j8)]
        point.time_from_start = Duration(sec=2, nanosec=0)
        
        goal_msg.trajectory.points.append(point)
        
        send_goal_future = self._gripper_client.send_goal_async(goal_msg)
        self.get_logger().info("Sending goal...")
        rclpy.spin_until_future_complete(self, send_goal_future)
        
        goal_handle = send_goal_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("Goal rejected.")
            return False
            
        self.get_logger().info("Goal accepted. Waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        if result:
            self.get_logger().info(f"Result Status: {result.status}") # 4 = SUCCEEDED
            if result.status == 4:
                self.get_logger().info("SUCCESS!")
                return True
        
        self.get_logger().error("Failed or Aborted.")
        return False

def main():
    rclpy.init()
    node = GripperDebug()
    
    print("Test 1: Open Gripper (0.03, 0.03)")
    node.send_gripper(0.03, 0.03)
    
    # print("Test 2: Close Gripper (0.0, 0.0)")
    # node.send_gripper(0.0, 0.0)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
