import math
import numpy as np
from feature_marking import FeatureMarker

class LandmarkLapDetector:
    def __init__(self, cooldown_frames, threshold=0.5):
        self.threshold = threshold
        self.feature_marker = FeatureMarker(threshold=threshold)

        self.marker_id = None
        self.lap_count = 0
        self.seen_once = False
        self.armed = False
        self.cooldown = 0
        self.cooldown_frames = cooldown_frames

    def reset(self):
        self.feature_marker = FeatureMarker(threshold=self.threshold)

        self.marker_id = None
        self.lap_count = 0
        self.seen_once = False
        self.armed = False
        self.cooldown = 0

    def update(self, sensor_data, robot_pose):
        rx, ry, rtheta = robot_pose

        detected_features = []

        for _, range_m, bearing in sensor_data:
            global_x = rx + range_m * math.cos(rtheta + bearing)
            global_y = ry + range_m * math.sin(rtheta + bearing)
            detected_features.append([global_x, global_y])

        marked_features = self.feature_marker.mark_features(detected_features)

        visible_feature_ids = [
            feature_id for _, feature_id, _ in marked_features
        ]

        # Select first detected feature automatically
        if self.marker_id is None and len(visible_feature_ids) > 0:
            self.marker_id = visible_feature_ids[0]
            print(f"Lap feature selected: {self.marker_id}")

        if self.cooldown > 0:
            self.cooldown -= 1

        # First time selected feature is seen
        if (
            self.marker_id is not None
            and self.marker_id in visible_feature_ids
            and not self.seen_once
        ):
            self.seen_once = True
            self.armed = False

        # Selected feature disappeared -> arm lap detection
        if (
            self.marker_id is not None
            and self.seen_once
            and self.marker_id not in visible_feature_ids
        ):
            self.armed = True

        # Selected feature seen again -> lap completed
        if (
            self.marker_id is not None
            and self.seen_once
            and self.armed
            and self.marker_id in visible_feature_ids
            and self.cooldown == 0
        ):
            self.lap_count += 1
            self.armed = False
            self.cooldown = self.cooldown_frames

            print(f"Lap completed by feature {self.marker_id}: {self.lap_count}")
            return True

        return False

    def get_lap_feature_position(self):
        if self.marker_id is None:
            return None

        for feature in self.feature_marker.get_features():
            if feature.id == self.marker_id:
                return feature.data

        return None