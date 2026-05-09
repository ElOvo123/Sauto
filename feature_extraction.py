import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import math


class ArucoFeatureExtractor:
    def __init__(self, dictionary_name=cv2.aruco.DICT_4X4_50):
        self.dictionary = cv2.aruco.Dictionary_get(dictionary_name)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Marker real size in meters
        self.marker_size = 0.10  # 10 cm

        # Approximate camera calibration for 1280x800 image
        # Replace these with your real camera calibration if available
        self.camera_matrix = np.array([
            [640.0,   0.0, 640.0],
            [  0.0, 640.0, 400.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

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

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_size,
            self.camera_matrix,
            self.dist_coeffs
        )

        for marker_corners, marker_id, rvec, tvec in zip(
            corners,
            ids.flatten(),
            rvecs,
            tvecs
        ):
            pts = marker_corners[0]
            center = np.mean(pts, axis=0)
            center_int = tuple(center.astype(int))

            x = float(tvec[0][0])
            y = float(tvec[0][1])
            z = float(tvec[0][2])

            range_m = math.sqrt(x**2 + y**2 + z**2)
            bearing_rad = math.atan2(x, z)
            bearing_deg = math.degrees(bearing_rad)

            features.append({
                "id": int(marker_id),
                "type": "aruco",
                "corners": pts,
                "center_px": center,
                "range_m": range_m,
                "bearing_rad": bearing_rad,
                "bearing_deg": bearing_deg,
                "tvec": tvec,
                "rvec": rvec
            })

            cv2.circle(frame, center_int, 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"ID {int(marker_id)} R={range_m:.2f}m B={bearing_deg:.1f}deg",
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
                f"ID={f['id']} range={f['range_m']:.2f} m "
                f"bearing={f['bearing_deg']:.1f} deg"
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