import numpy as np
import unittest

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

        if distance_fn is None:
            self.distance_fn = self.euclidean_distance
        else:
            self.distance_fn = distance_fn

    def euclidean_distance(self, a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        return np.linalg.norm(a - b)

    def mark_features(self, detected_features):
        marked_features = []

        for detected in detected_features:
            detected = np.array(detected, dtype=float)

            best_id = None
            best_distance = float("inf")
            best_feature = None

            for feature in self.features:
                distance = self.distance_fn(detected, feature.data)

                if distance < best_distance:
                    best_distance = distance
                    best_id = feature.id
                    best_feature = feature

            if best_distance < self.threshold:
                best_feature.data = detected
                best_feature.seen_count += 1

                marked_features.append((detected, best_id, False))

            else:
                new_feature = Feature(self.next_id, detected)
                self.features.append(new_feature)

                marked_features.append((detected, self.next_id, True))

                self.next_id += 1

        return marked_features

    def get_features(self):
        return self.features

    def print_features(self):
        for feature in self.features:
            print(
                f"Feature ID: {feature.id}, "
                f"Data: {feature.data}, "
                f"Seen: {feature.seen_count}"
            )