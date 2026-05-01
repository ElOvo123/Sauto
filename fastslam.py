import math
import numpy as np

# Your friend will import their particle filter and EKF here
# from particle_filter import ParticleFilter
# from ekf import ExtendedKalmanFilter

class FastSLAM:
    def __init__(self, initial_pose, num_particles=100):
        # Initialize particles
        self.particles = [] 
        
        # The estimated state to return to the simulator
        self.estimated_pose = initial_pose # [x, y, theta]
        
        # Track the previous odometry to calculate movement deltas!
        self.prev_odom = initial_pose

    def step(self, current_odom, measurements, dt):
        """
        The main SLAM loop called every frame by the Pygame simulator.
        current_odom: [x, y, theta] from the robot's noisy encoders
        measurements: list of [[id, range, bearing], ...] from the camera
        dt: time step
        """
        
        # --- CALCULATE ODOMETRY DELTA ---
        # The standard Probabilistic Robotics motion model uses:
        # delta_rot1, delta_trans, delta_rot2
        dx = current_odom[0] - self.prev_odom[0]
        dy = current_odom[1] - self.prev_odom[1]
        
        delta_trans = math.hypot(dx, dy)
        delta_rot1 = math.atan2(dy, dx) - self.prev_odom[2]
        
        # Normalize rot1
        delta_rot1 = (delta_rot1 + math.pi) % (2 * math.pi) - math.pi 
        
        delta_rot2 = current_odom[2] - self.prev_odom[2] - delta_rot1
        
        # Normalize rot2
        delta_rot2 = (delta_rot2 + math.pi) % (2 * math.pi) - math.pi

        # Save current odometry for the next frame
        self.prev_odom = current_odom

        # --- 1. PREDICT STEP ---
        # Eduardo uses delta_rot1, delta_trans, delta_rot2 to move particles.
        # They MUST add artificial motion noise here, otherwise particles don't spread.
        
        # --- 2. UPDATE STEP ---
        # Eduardo loops through measurements to update EKFs and particle weights.
        
        # --- 3. RESAMPLE STEP ---
        # Eduardo resamples particles based on weights.
        
        # --- 4. ESTIMATE EXTRACTION ---
        # Eduardo calculates the best estimated robot pose (e.g., highest weight particle)
        self.estimated_pose = current_odom # Placeholder: replace with actual estimate
        
        # Eduardo calculates the estimated map (x, y of landmarks from best particle)
        estimated_landmarks = {10: [0.0, 0.0]} # Placeholder
        
        # Return data so the Pygame simulator can draw it
        return self.particles, self.estimated_pose, estimated_landmarks