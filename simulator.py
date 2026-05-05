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
    true_trajectory = []  # To store the true trajectory for visualization
    
    true_start_pose = [3.0, 3.0, 0.0] 
    robot = SimulatedTurtlebot(*true_start_pose)
    slam = FastSLAM(initial_pose=true_start_pose)

    pygame.font.init()
    # Cria uma fonte Arial, tamanho 24
    my_font = pygame.font.SysFont('Arial', 24)

    running = True
    v, w = 0.0, 0.0
    while running:
        # 1. PROCESS KEYBOARD INPUTS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # KEYDOWN só dispara UMA vez quando carregas na tecla
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    v += 0.02
                elif event.key == pygame.K_DOWN:
                    v -= 0.02
                elif event.key == pygame.K_LEFT:
                    w += 0.02
                elif event.key == pygame.K_RIGHT:
                    w -= 0.02

        # 2. RUN REAL-WORLD PHYSICS 
        # Notice we pass env.walls to the move function now!
        robot.move(v, w, DT, env.outer_walls)
        
        # Save true robot trajectory
        bx, by = robot.get_body_center()
        true_trajectory.append((bx, by))

        sensor_data = robot.get_camera_measurements(env.landmarks)
        odom_data = robot.get_odometry() # Returns the noisy [x, y, theta]

        #print (sensor_data)
        print (odom_data)

        # 3. RUN SLAM ALGORITHM
        particles, est_pose, est_map = slam.step(odom_data, sensor_data, DT)

        # 4. RENDER GRAPHICS
        screen.fill((240, 240, 240)) 
        
        # Draw Floor Plan
        # Draw Floor Plan
        env.draw(screen, to_screen)

        # Draw true robot trajectory
        if len(true_trajectory) > 1:
            trajectory_points = [to_screen(x, y) for x, y in true_trajectory]
            pygame.draw.lines(screen, (0, 200, 0), False, trajectory_points, 2)

        # Draw the particles
        for px, py, ptheta in particles:
            pos = to_screen(px, py)
            pygame.draw.circle(screen, (255, 100, 100), pos, 2)

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

        # Draw velocity
        text_v = my_font.render(f"Velocidade Linear (v): {v:.2f} m/s", True, (0, 0, 0))
        text_w = my_font.render(f"Velocidade Angular (w): {w:.2f} rad/s", True, (0, 0, 0))
        screen.blit(text_v, (10, 10))
        screen.blit(text_w, (10, 40))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()