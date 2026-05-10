import numpy as np
import math

def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Math helper to find distance from robot to a wall."""
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0: return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return math.hypot(px - proj_x, py - proj_y)

class SimulatedTurtlebot:
    def __init__(self, start_x, start_y, start_theta):
        # 1. The TRUE state of the robot (used for physics and camera)
        self.x = start_x
        self.y = start_y
        self.theta = start_theta
        
        # 2. The ODOMETRY state (what the robot's encoders "think" happened)
        self.odom_x = start_x
        self.odom_y = start_y
        self.odom_theta = start_theta
        
        # Physical attributes
        self.radius = 0.15          
        self.body_offset = 0.08     
        self.camera_offset = 0.03   
        
        # Camera specs
        self.max_range = 5.0        
        self.fov = math.radians(60) 

    def get_body_center(self):
        bx = self.x - self.body_offset * math.cos(self.theta)
        by = self.y - self.body_offset * math.sin(self.theta)
        return bx, by

    def get_camera_position(self):
        cx = self.x + self.camera_offset * math.cos(self.theta)
        cy = self.y + self.camera_offset * math.sin(self.theta)
        return cx, cy

    def move(self, v, w, dt, walls):
        """Updates both the true physics and the noisy odometry."""
        
        # --- 1. UPDATE ODOMETRY (What the encoders read) ---
        # Encoders have slight reading errors (slip, resolution limits)
        # We simulate this by adding noise to the velocity before integrating
        noisy_v = v + np.random.normal(0, 0.02) if v != 0 else 0.0
        noisy_w = w + np.random.normal(0, 0.01) if w != 0 else 0.0
        
        self.odom_theta += noisy_w * dt
        self.odom_theta = (self.odom_theta + math.pi) % (2 * math.pi) - math.pi
        self.odom_x += noisy_v * math.cos(self.odom_theta) * dt
        self.odom_y += noisy_v * math.sin(self.odom_theta) * dt

        # --- 2. UPDATE TRUE PHYSICS ---
        # The true robot moves perfectly according to physics (unless it hits a wall)
        next_theta = self.theta + w * dt
        next_theta = (next_theta + math.pi) % (2 * math.pi) - math.pi
        next_x = self.x + v * math.cos(self.theta) * dt
        next_y = self.y + v * math.sin(self.theta) * dt

        next_bx = next_x - self.body_offset * math.cos(next_theta)
        next_by = next_y - self.body_offset * math.sin(next_theta)

        collision = False
        for (p1, p2) in walls:
            dist = distance_point_to_segment(next_bx, next_by, p1[0], p1[1], p2[0], p2[1])
            if dist < self.radius:
                collision = True
                break
        
        if not collision:
            self.x = next_x
            self.y = next_y
            self.theta = next_theta

    def get_odometry(self):
        """Returns the robot's internal, drifting belief of its pose."""
        return [self.odom_x, self.odom_y, self.odom_theta]

    def get_camera_measurements(self, true_landmarks):
        measurements = []
        cx, cy = self.get_camera_position()
        
        for marker_id, (lx, ly) in true_landmarks.items():
            dx = lx - cx
            dy = ly - cy
            
            true_range = math.hypot(dx, dy)
            true_bearing = math.atan2(dy, dx) - self.theta
            true_bearing = (true_bearing + math.pi) % (2 * math.pi) - math.pi
            
            if true_range < self.max_range and abs(true_bearing) < (self.fov / 2):
                noise_range = np.random.normal(0, 0.05)   
                noise_bearing = np.random.normal(0, 0.02) 
                
                noisy_range = max(0.01, true_range + noise_range)
                noisy_bearing = true_bearing + noise_bearing
                noisy_bearing = (noisy_bearing + math.pi) % (2 * math.pi) - math.pi
                measurements.append([marker_id, noisy_range, noisy_bearing])
                                
        return measurements