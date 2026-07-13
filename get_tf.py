import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import sys

class FrameListener(Node):
    def __init__(self):
        super().__init__('tf_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.on_timer)
        self.count = 0

    def on_timer(self):
        try:
            t1 = self.tf_buffer.lookup_transform('tcp', 'EEE', rclpy.time.Time())
            t2 = self.tf_buffer.lookup_transform('tcp', 'DEEP', rclpy.time.Time())
            print(f"tcp -> EEE: {t1.transform.translation.x}, {t1.transform.translation.y}, {t1.transform.translation.z}")
            print(f"tcp -> DEEP: {t2.transform.translation.x}, {t2.transform.translation.y}, {t2.transform.translation.z}")
            sys.exit(0)
        except TransformException as ex:
            self.count += 1
            if self.count > 3:
                print(f"Could not transform: {ex}")
                sys.exit(1)

def main():
    rclpy.init()
    node = FrameListener()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
