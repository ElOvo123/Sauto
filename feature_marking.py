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

def test_feature_marker():

    print("=== Testing FeatureMarker ===")

    marker = FeatureMarker(threshold=0.8)

    detections_1 = [
        [1.0, 2.0],
        [5.0, 3.0],
        [10.0, 8.0]
    ]

    result_1 = marker.mark_features(detections_1)

    print("\nTest 1: First detections")
    for f, fid, is_new in result_1:
        print(f, "-> ID:", fid, "New:", is_new)

    assert all(is_new for _, _, is_new in result_1), "Test 1 failed"

    detections_2 = [
        [1.1, 2.1],
        [5.2, 3.1]
    ]

    result_2 = marker.mark_features(detections_2)

    print("\nTest 2: Matching detections")
    for f, fid, is_new in result_2:
        print(f, "-> ID:", fid, "New:", is_new)

    assert not any(is_new for _, _, is_new in result_2), "Test 2 failed"

    detections_3 = [
        [20.0, 20.0]
    ]

    result_3 = marker.mark_features(detections_3)

    print("\nTest 3: New far detection")
    for f, fid, is_new in result_3:
        print(f, "-> ID:", fid, "New:", is_new)

    assert result_3[0][2] is True, "Test 3 failed"

    marker.mark_features([[1.0, 2.0]])
    marker.mark_features([[1.05, 2.05]])

    feature = marker.get_features()[0]

    print("\nTest 4: Seen count =", feature.seen_count)

    assert feature.seen_count >= 3, "Test 4 failed"

    marker.mark_features([[2.0, 3.0]])  # far → new
    marker.mark_features([[2.1, 3.1]])  # update

    updated_feature = marker.get_features()[-1]

    print("\nTest 5: Updated feature =", updated_feature.data)

    assert np.allclose(updated_feature.data, [2.1, 3.1]), "Test 5 failed"

    print("\n All tests passed!")

if __name__ == "__main__":
    test_feature_marker()