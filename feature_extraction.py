import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class ArucoFeatureExtractor:
    def __init__(self, dictionary_name=cv2.aruco.DICT_4X4_50):
        self.dictionary = cv2.aruco.Dictionary_get(dictionary_name)
        self.parameters = cv2.aruco.DetectorParameters_create()

    def extract(self, frame):
        features = []

        if frame is None:
            return features

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.parameters
        )

        if ids is None:
            return features

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            pts = marker_corners[0]
            center = np.mean(pts, axis=0)

            features.append({
                "id": int(marker_id),
                "type": "aruco",
                "corners": pts,
                "center_px": center
            })

            center_int = tuple(center.astype(int))
            cv2.circle(frame, center_int, 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"ID {int(marker_id)} ({center[0]:.1f},{center[1]:.1f})",
                center_int,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return features


class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_feature_extractor")

        self.bridge = CvBridge()
        self.extractor = ArucoFeatureExtractor()

        cv2.namedWindow("Aruco Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Aruco Detection", 1280, 800)

        self.sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10
        )

        self.get_logger().info("Aruco node started, waiting for /image_raw...")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        features = self.extractor.extract(frame)

        for f in features:
            self.get_logger().info(
                f"ID={f['id']} center={f['center_px']}"
            )

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ArucoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()