#!/usr/bin/env python3

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG_PATH = "camera_calibration"   # folder with metadata.yaml and .db3
IMAGE_TOPIC = "/image_raw"

CHECKERBOARD = (9, 6)             # inner corners
SQUARE_SIZE = 0.025               # meters
FRAME_SKIP = 1
MIN_DETECTIONS = 4

objpoints = []
imgpoints = []

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

bridge = CvBridge()
frame_count = 0
used_count = 0
gray_shape = None
video_writer = None

storage_options = rosbag2_py.StorageOptions(
    uri=BAG_PATH,
    storage_id="sqlite3"
)

converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {topic.name: topic.type for topic in topic_types}

print("Topics found:")
for name, typ in type_map.items():
    print(f"  {name} -> {typ}")

if IMAGE_TOPIC not in type_map:
    raise RuntimeError(f"Topic {IMAGE_TOPIC} not found in bag.")

msg_type = get_message(type_map[IMAGE_TOPIC])

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic != IMAGE_TOPIC:
        continue

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    msg = deserialize_message(data, msg_type)

    if type_map[IMAGE_TOPIC] == "sensor_msgs/msg/CompressedImage":
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)

    if found:
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used_count += 1

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, found)
        print(f"Found checkerboard: frame {frame_count}, total {used_count}")

    if video_writer is None:
        h, w = img.shape[:2]
        video_writer = cv2.VideoWriter(
            "checkerboard_debug.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            20,
            (w, h)
        )

    video_writer.write(img)

    cv2.imshow("checkerboard detection", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if video_writer:
    video_writer.release()

cv2.destroyAllWindows()

print(f"\nFrames read: {frame_count}")
print(f"Valid checkerboard frames: {used_count}")

if used_count < MIN_DETECTIONS:
    raise RuntimeError("Not enough valid checkerboard detections.")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray_shape,
    None,
    None
)

print("\nReprojection error:")
print(ret)

print("\ncamera_matrix:")
print(camera_matrix)

print("\ndist_coeffs:")
print(dist_coeffs)

np.savez(
    "camera_calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs,
    reprojection_error=ret
)

print("\nSaved calibration to camera_calibration.npz")
print("Saved debug video to checkerboard_debug.avi")