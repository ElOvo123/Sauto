import math
import numpy as np
from feature_marking import FeatureMarker


class LandmarkLapDetector:
    def __init__(self, cooldown_frames, threshold=0.5):
        self.lap_count = 0

        self.cooldown = 0
        self.cooldown_frames = cooldown_frames

        self.seen_ids = set()
        self.absent_ids = set()

        self.frame_count = 0

        # Pose-based fallback lap detection.
        self.in_start_area = False
        self.has_left_start_area = False

        # Prevent early false laps
        self.min_frames_before_lap = 300

        # Only use bottom/start landmark
        self.start_landmark_ids = {1}

    def reset(self):
        self.lap_count = 0

        self.cooldown = 0

        self.seen_ids = set()
        self.absent_ids = set()

        self.frame_count = 0
        self.in_start_area = False
        self.has_left_start_area = False

    def update(self, sensor_data, robot_pose=None):
        self.frame_count += 1

        rx, ry, _ = robot_pose

        # Only allow lap detection near bottom/start corridor
        near_start_area = (
            rx < 5.0
            and ry > 4.0
            and ry < 8.0
        )

        visible_ids = set(
            m[0] for m in sensor_data
        ).intersection(self.start_landmark_ids)

        # Track whether the robot has left and re-entered the start area.
        entered_start_area = near_start_area and not self.in_start_area
        if not near_start_area and self.in_start_area:
            self.has_left_start_area = True

        self.in_start_area = near_start_area

        # Ignore landmark detections outside start area
        if not near_start_area:
            visible_ids = set()

        if self.cooldown > 0:
            self.cooldown -= 1

        # Mark old seen landmarks as absent
        for landmark_id in self.seen_ids:
            if landmark_id not in visible_ids:
                self.absent_ids.add(landmark_id)

        if (
            self.frame_count > self.min_frames_before_lap
            and self.cooldown == 0
        ):
            returning_ids = visible_ids.intersection(self.absent_ids)

            landmark_based_lap = False
            if len(returning_ids) > 0:
                lap_id = list(returning_ids)[0]

                self.lap_count += 1

                self.cooldown = self.cooldown_frames

                self.absent_ids.clear()

                print(
                    f"Lap completed by returning "
                    f"landmark {lap_id}: "
                    f"{self.lap_count}"
                )

                landmark_based_lap = True

            pose_based_lap = (
                self.has_left_start_area
                and entered_start_area
            )

            if pose_based_lap and not landmark_based_lap:
                self.lap_count += 1
                self.cooldown = self.cooldown_frames
                self.has_left_start_area = False

                print(
                    f"Lap completed by start-area re-entry: "
                    f"{self.lap_count}"
                )

                return True

            if landmark_based_lap:
                self.has_left_start_area = False
                return True

        self.seen_ids.update(visible_ids)

        return False

    def get_lap_feature_position(self):
        return None