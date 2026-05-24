import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os

class Saver(Node):
    def __init__(self):
        super().__init__('image_saver')

        self.count = 0
        os.makedirs("calib_images", exist_ok=True)

        self.sub = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.callback,
            10
        )

    def callback(self, msg):
        if self.count % 20 != 0:
            self.count += 1
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        filename = f"calib_images/img_{self.count:04d}.png"
        cv2.imwrite(filename, frame)

        print("saved", filename)

        self.count += 1

def main():
    rclpy.init()
    node = Saver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()