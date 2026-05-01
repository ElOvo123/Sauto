import pygame
import sys
import math
from turtlebot import SimulatedTurtlebot
from environment import Environment
from fastslam import FastSLAM

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000
SCALE = 45.0  # Pixels per meter
FPS = 30
DT = 1.0 / FPS

def to_screen(x, y):
    """Maps mathematical coordinates (meters) to Pygame pixels."""
    screen_x = int(x * SCALE)
    screen_y = int(HEIGHT - (y * SCALE))
    return (screen_x, screen_y)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("FastSLAM Micro-Simulator")
    clock = pygame.time.Clock()

    env = Environment()
    
    true_start_pose = [3.0, 3.0, 0.0] 
    robot = SimulatedTurtlebot(*true_start_pose)
    slam = FastSLAM(initial_pose=true_start_pose)

    running = True
    while running:
        # 1. PROCESS KEYBOARD INPUTS
        v, w = 0.0, 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:    v = 0.5
        if keys[pygame.K_DOWN]:  v = -0.5
        if keys[pygame.K_LEFT]:  w = 1.0
        if keys[pygame.K_RIGHT]: w = -1.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. RUN REAL-WORLD PHYSICS 
        # Notice we pass env.walls to the move function now!
        robot.move(v, w, DT, env.outer_walls)
        sensor_data = robot.get_camera_measurements(env.landmarks)
        odom_data = robot.get_odometry() # Returns the noisy [x, y, theta]

        #print (sensor_data)
        print (odom_data)

        # 3. RUN SLAM ALGORITHM
        particles, est_pose, est_map = slam.step(odom_data, sensor_data, DT)

        # 4. RENDER GRAPHICS
        screen.fill((240, 240, 240)) 
        
        # Draw Floor Plan
        env.draw(screen, to_screen)

        # Draw SLAM Estimated Map (Blue squares)
        for mark_id, (lx, ly) in est_map.items():
            pos = to_screen(lx, ly)
            pygame.draw.rect(screen, (0, 0, 255), (pos[0]-4, pos[1]-4, 8, 8))

        # Draw True Robot (Body Center, not wheel axis)
        bx, by = robot.get_body_center()
        rob_pos = to_screen(bx, by)
        pixel_radius = int(robot.radius * SCALE)
        pygame.draw.circle(screen, (0, 0, 0), rob_pos, pixel_radius)
        
        # Draw heading line from body center pointing forward
        end_x = bx + (robot.radius + 0.1) * math.cos(robot.theta)
        end_y = by + (robot.radius + 0.1) * math.sin(robot.theta)
        pygame.draw.line(screen, (255, 0, 0), rob_pos, to_screen(end_x, end_y), 2)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()