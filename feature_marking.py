import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import cv2
import numpy as np
import math


class Feature:
    def __init__(self, feature_id, data):
        self.id = feature_id
        self.data = np.array(data, dtype=float)
        self.seen_count = 1


class FeatureMarker:
    def __init__(self, threshold=1.0, distance_fn=None):
        self.features = []
        self.next_id = 0
        self.threshold = threshold
        self.distance_fn = distance_fn or self.euclidean_distance

    def euclidean_distance(self, a, b):
        return np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float))

    def mark_features(self, detected_features):
        marked_features = []

        for detected in detected_features:
            detected = np.array(detected, dtype=float)

            best_feature = None
            best_distance = float("inf")

            for feature in self.features:
                distance = self.distance_fn(detected, feature.data)

                if distance < best_distance:
                    best_distance = distance
                    best_feature = feature

            if best_feature is not None and best_distance < self.threshold:
                best_feature.data = detected
                best_feature.seen_count += 1
                marked_features.append((detected, best_feature.id, False))
            else:
                new_feature = Feature(self.next_id, detected)
                self.features.append(new_feature)
                marked_features.append((detected, self.next_id, True))
                self.next_id += 1

        return marked_features

    def get_features(self):
        return self.features


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

        self.feature_marker = FeatureMarker(threshold=0.75)

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

        detected_world_positions = []
        marker_info = []

        for marker_corners, aruco_id, rvec, tvec in zip(
            corners,
            ids.flatten(),
            rvecs,
            tvecs
        ):
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

                detected_world_positions.append([landmark_x, landmark_y])
            else:
                detected_world_positions.append([x, z])

            marker_info.append({
                "aruco_id": int(aruco_id),
                "corners": pts,
                "center_px": center,
                "range_m": range_m,
                "bearing_rad": bearing_rad,
                "bearing_deg": math.degrees(bearing_rad),
                "tvec": tvec,
                "rvec": rvec
            })

        marked = self.feature_marker.mark_features(detected_world_positions)

        for info, marked_data in zip(marker_info, marked):
            detected_position, landmark_id, is_new = marked_data

            landmark_x = float(detected_position[0])
            landmark_y = float(detected_position[1])

            features.append({
                "id": int(landmark_id),
                "aruco_id": info["aruco_id"],
                "type": "aruco",
                "corners": info["corners"],
                "center_px": info["center_px"],
                "range_m": info["range_m"],
                "bearing_rad": info["bearing_rad"],
                "bearing_deg": info["bearing_deg"],
                "landmark_x": landmark_x,
                "landmark_y": landmark_y,
                "is_new": is_new,
                "tvec": info["tvec"],
                "rvec": info["rvec"]
            })

            center_int = tuple(info["center_px"].astype(int))

            cv2.circle(frame, center_int, 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"L{landmark_id} ({landmark_x:.2f}, {landmark_y:.2f})",
                center_int,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return features