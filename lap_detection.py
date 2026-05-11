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

        # Ignore landmark detections outside start area
        if not near_start_area:
            visible_ids = set()

        if self.cooldown > 0:
            self.cooldown -= 1

        # Mark old seen landmarks as absent
        for landmark_id in self.seen_ids:
            if landmark_id not in visible_ids:
                self.absent_ids.add(landmark_id)

        # Detect returning landmark
        if (
            self.frame_count > self.min_frames_before_lap
            and self.cooldown == 0
        ):
            returning_ids = visible_ids.intersection(
                self.absent_ids
            )

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

                return True

        self.seen_ids.update(visible_ids)

        return False

    def get_lap_feature_position(self):
        return None