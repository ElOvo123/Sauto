import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import cv2
import numpy as np
import math


class ArucoFeatureExtractor:
    def __init__(self, dictionary_name=cv2.aruco.DICT_4X4_50):
        self.dictionary = cv2.aruco.Dictionary_get(dictionary_name)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.marker_size = 0.16872

        self.camera_matrix = np.array([
            [261.00813352, 0.0, 172.1808022],
            [0.0, 262.1472986, 120.76379966],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # Stores all positions for each ArUco ID
        self.aruco_positions = {}

    def extract(self, frame, robot_pose=None):
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

        for marker_corners, aruco_id, rvec, tvec in zip(
            corners,
            ids.flatten(),
            rvecs,
            tvecs
        ):
            aruco_id = int(aruco_id)

            pts = marker_corners[0]
            center = np.mean(pts, axis=0)

            x = float(tvec[0][0])
            y = float(tvec[0][1])
            z = float(tvec[0][2])

            range_m = math.sqrt(x**2 + y**2 + z**2)
            bearing_rad = math.atan2(x, z)

            if robot_pose is not None:
                rx, ry, rtheta = robot_pose

                landmark_x = rx + range_m * math.cos(rtheta + bearing_rad)
                landmark_y = ry + range_m * math.sin(rtheta + bearing_rad)
            else:
                landmark_x = x
                landmark_y = z

            # Store positions
            if aruco_id not in self.aruco_positions:
                self.aruco_positions[aruco_id] = []

            self.aruco_positions[aruco_id].append([
                float(landmark_x),
                float(landmark_y)
            ])

            features.append({
                "aruco_id": aruco_id,
                "landmark_x": float(landmark_x),
                "landmark_y": float(landmark_y),
            })

            center_int = tuple(center.astype(int))

            cv2.circle(frame, center_int, 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"A{aruco_id}",
                center_int,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return features

    def get_aruco_positions(self):
        return self.aruco_positions

    def get_mean_positions(self):
        mean_positions = {}

        for aruco_id, positions in self.aruco_positions.items():
            positions_np = np.array(positions)

            mean_x = np.mean(positions_np[:, 0])
            mean_y = np.mean(positions_np[:, 1])

            mean_positions[aruco_id] = [mean_x, mean_y]

        return mean_positions


class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_feature_extractor")

        self.bridge = CvBridge()
        self.extractor = ArucoFeatureExtractor()

        self.current_pose = None

        cv2.namedWindow("Aruco Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Aruco Detection", 1280, 800)

        self.image_sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.get_logger().info(
            "Aruco node started, waiting for /image_raw and /odom..."
        )

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        theta = math.atan2(siny_cosp, cosy_cosp)

        self.current_pose = [x, y, theta]

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        features = self.extractor.extract(
            frame,
            self.current_pose
        )

        printed_ids = set()

        for f in features:
            aruco_id = f["aruco_id"]

            if aruco_id not in printed_ids:
                printed_ids.add(aruco_id)

                self.get_logger().info(
                    f"Aruco ID={aruco_id} "
                    f"pos=({f['landmark_x']:.2f}, "
                    f"{f['landmark_y']:.2f})"
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

    print("\n===== MEAN ARUCO POSITIONS =====")

    mean_positions = node.extractor.get_mean_positions()

    for aruco_id, mean_pos in mean_positions.items():
        print(
            f"Aruco ID {aruco_id}: "
            f"mean_x={mean_pos[0]:.3f}, "
            f"mean_y={mean_pos[1]:.3f}"
        )

    print("================================\n")

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()