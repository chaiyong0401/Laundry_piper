import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import sys

class GripperTester(Node):
    def __init__(self):
        super().__init__('gripper_tester_node')
        self._client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        self.get_logger().info("Waiting for gripper action server...")
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available!")
            sys.exit(1)
        self.get_logger().info("Gripper action server found.")

    def send_goal(self, j7, j8):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint7', 'joint8']
        
        point = JointTrajectoryPoint()
        point.positions = [float(j7), float(j8)]
        point.time_from_start.sec = 2 # 2 seconds to move
        
        goal_msg.trajectory.points.append(point)
        
        self.get_logger().info(f"Sending goal: joint7={j7}, joint8={j8}")
        self._client.wait_for_server()
        
        future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return
            
        self.get_logger().info("Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Done.")

def main():
    rclpy.init()
    node = GripperTester()
    
    print("\n--- Gripper Test Tool ---")
    print("Commands:")
    print("  open  : Move to [0.03, 0.03]")
    print("  close : Move to [0.00, 0.00]")
    print("  set J7 J8 : Custom values (e.g. 'set 0.01 -0.01')")
    print("  q     : Quit")
    
    try:
        while rclpy.ok():
            cmd = input("\nEnter command: ").strip().lower()
            if cmd == 'q':
                break
            elif cmd == 'open':
                node.send_goal(0.03, 0.03)
            elif cmd == 'close':
                node.send_goal(0.0, 0.0)
            elif cmd.startswith('set'):
                try:
                    parts = cmd.split()
                    j7 = float(parts[1])
                    j8 = float(parts[2])
                    node.send_goal(j7, j8)
                except:
                    print("Usage: set [joint7_val] [joint8_val]")
            else:
                print("Unknown command")
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
