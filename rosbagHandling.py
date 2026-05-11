from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores

import pandas as pd

bagpath = Path('/home/goncalo/Downloads/SAUtobags/rosbag_good_Arucos_2_laps')

# Choose your ROS2 distro typestore
typestore = get_typestore(Stores.ROS2_HUMBLE)
# Examples:
# Stores.ROS2_FOXY
# Stores.ROS2_GALACTIC
# Stores.ROS2_HUMBLE
# Stores.ROS2_IRON

data = []

with AnyReader([bagpath], default_typestore=typestore) as reader:

    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/cmd_vel':
            msg = reader.deserialize(rawdata, connection.msgtype)

            data.append({
                'time': timestamp,
                'linear_x': msg.linear.x,
                'angular_z': msg.angular.z
            })

df = pd.DataFrame(data)
df.to_csv('velocities.csv', index=False)

print(df.T)