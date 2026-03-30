#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointSubscriber(Node):
    def __init__(self):
        super().__init__('joint_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        print("현재 관절 값을 읽어오는 중입니다... (Ctrl+C로 종료)")

    def listener_callback(self, msg):
        # joint1~8 순서대로 정렬 (드라이버마다 순서가 다를 수 있음)
        joint_names = [f"joint{i+1}" for i in range(8)]
        joint_map = dict(zip(msg.name, msg.position))
        
        try:
            current_joints = [joint_map[name] for name in joint_names]
            # 출력 포맷 (Logic_inte_wm.py에 바로 복사 가능)
            formatted_joints = [round(j, 6) for j in current_joints]
            print(f"\n[현재 관절 값 (joint 1~8)]")
            print(f"{formatted_joints}")
            print("\n위 값을 복사해서 사용하세요.")
            
            # 한 번만 출력하고 종료하려면 아래 주석 해제
            # rclpy.shutdown()
        except KeyError as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    joint_subscriber = JointSubscriber()
    try:
        rclpy.spin(joint_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        joint_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
