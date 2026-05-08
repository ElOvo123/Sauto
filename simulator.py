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
    true_trajectory = []
    current_lap_best_trajectory = []
    displayed_best_trajectory = []
        
    true_start_pose = [3.0, 3.0, 0.0] 
    robot = SimulatedTurtlebot(*true_start_pose)
    slam = FastSLAM(initial_pose=true_start_pose)

    pygame.font.init()
    # Cria uma fonte Arial, tamanho 24
    my_font = pygame.font.SysFont('Arial', 24)

    true_pose_history = []
    estimated_pose_history = []
    position_error_history = []
    landmark_error_history = []

    running = True
    v, w = 0.0, 0.0

    auto_drive = False
    auto_speed = 3.0

    # Corridor center lines
    left_x = 3.125
    right_x = 17.375
    bottom_y = 3.125
    top_y = 17.375

    auto_state = "BOTTOM"
    lap_count = 0
    previous_auto_state = auto_state
    was_in_left_state = False

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
                elif event.key == pygame.K_t:
                    auto_drive = True

                    robot.x = left_x
                    robot.y = bottom_y
                    robot.theta = 0.0

                    robot.odom_x = robot.x
                    robot.odom_y = robot.y
                    robot.odom_theta = robot.theta

                    auto_state = "BOTTOM"
        if auto_drive:
            corner_margin = 0.6
            k_heading = 3.0
            k_center = 0.8
            max_w = 1.2

            if auto_state == "BOTTOM" and robot.x >= right_x - corner_margin:
                auto_state = "RIGHT"
            elif auto_state == "RIGHT" and robot.y >= top_y - corner_margin:
                auto_state = "TOP"
            elif auto_state == "TOP" and robot.x <= left_x + corner_margin:
                auto_state = "LEFT"
            elif auto_state == "LEFT" and robot.y <= bottom_y + corner_margin:
                auto_state = "BOTTOM"

            # Detect completed lap
            if auto_state == "LEFT":
                was_in_left_state = True

            if was_in_left_state and auto_state == "BOTTOM":
                lap_count += 1
                was_in_left_state = False

                print(f"Lap completed: {lap_count}")
                print(f"Saved trajectory points: {len(current_lap_best_trajectory)}")

                displayed_best_trajectory = current_lap_best_trajectory.copy()
                current_lap_best_trajectory = []

            # Then compute target heading and center error for CURRENT state
            if auto_state == "BOTTOM":
                target_theta = 0.0
                cross_track_error = robot.y - bottom_y

            elif auto_state == "RIGHT":
                target_theta = math.pi / 2
                cross_track_error = right_x - robot.x

            elif auto_state == "TOP":
                target_theta = math.pi
                cross_track_error = top_y - robot.y

            elif auto_state == "LEFT":
                target_theta = -math.pi / 2
                cross_track_error = robot.x - left_x

            angle_error = (target_theta - robot.theta + math.pi) % (2 * math.pi) - math.pi

            w = k_heading * angle_error - k_center * cross_track_error

            # Limit angular speed to avoid huge arcs
            w = max(-max_w, min(max_w, w))

            # Slow down only while turning
            if abs(angle_error) > 0.25:
                v = 1.0
            else:
                v = auto_speed

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

        # Record first-lap best particle trajectory
        # Record current lap best-particle trajectory
        current_lap_best_trajectory.append((est_pose[0], est_pose[1]))
        
        # --- STORE ERRORS FOR EVALUATION ---
        true_pose = [robot.x, robot.y, robot.theta]

        true_pose_history.append(true_pose)
        estimated_pose_history.append(est_pose)

        # Robot position error
        pos_error = math.hypot(
            robot.x - est_pose[0],
            robot.y - est_pose[1]
        )
        position_error_history.append(pos_error)

        # Landmark mapping error
        lm_errors = []

        for lm_id, est_lm in est_map.items():
            if lm_id in env.landmarks:
                true_lm = env.landmarks[lm_id]

                lm_error = math.hypot(
                    true_lm[0] - est_lm[0],
                    true_lm[1] - est_lm[1]
                )

                lm_errors.append(lm_error)

        if len(lm_errors) > 0:
            landmark_error_history.append(sum(lm_errors) / len(lm_errors))

        # 4. RENDER GRAPHICS
        screen.fill((240, 240, 240)) 

        # Draw Floor Plan
        # Draw Floor Plan
        env.draw(screen, to_screen)

        # Draw true robot trajectory
        if len(true_trajectory) > 1:
            trajectory_points = [to_screen(x, y) for x, y in true_trajectory]
            pygame.draw.lines(screen, (0, 200, 0), False, trajectory_points, 2)

        # Draw best particle trajectory after full lap
        if len(displayed_best_trajectory) > 1:
            best_particle_points = [to_screen(x, y) for x, y in displayed_best_trajectory]
            pygame.draw.lines(screen, (0, 0, 255), False, best_particle_points, 3)

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

    # --- FINAL EVALUATION ---

    if len(position_error_history) > 0:
        mean_position_error = sum(position_error_history) / len(position_error_history)
        final_position_error = position_error_history[-1]

        print("\n===== FASTSLAM EVALUATION =====")
        print(f"Mean robot position error: {mean_position_error:.3f} m")
        print(f"Final robot position error: {final_position_error:.3f} m")

    if len(landmark_error_history) > 0:
        mean_landmark_error = sum(landmark_error_history) / len(landmark_error_history)
        final_landmark_error = landmark_error_history[-1]

        print(f"Mean landmark error: {mean_landmark_error:.3f} m")
        print(f"Final landmark error: {final_landmark_error:.3f} m")

    print("================================\n")

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()