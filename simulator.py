import pygame
import sys
import math

from turtlebot import SimulatedTurtlebot
from environment import Environment
from fastslam2 import FastSLAM2
from fastslam1 import FastSLAM1
from lap_detection import LandmarkLapDetector

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000
SCALE = 45.0
FPS = 30
DT = 1.0 / FPS


def to_screen(x, y):
    screen_x = int(x * SCALE)
    screen_y = int(HEIGHT - (y * SCALE))
    return (screen_x, screen_y)


class EstimationAlignment:
    def __init__(self, rotation_offset_deg=0.0, center_x=0.0, center_y=0.0):
        self.rotation_offset_deg = rotation_offset_deg
        self.rotation_offset_rad = math.radians(rotation_offset_deg)
        self.center_x = center_x
        self.center_y = center_y

    def rotate_point(self, x, y):
        dx = x - self.center_x
        dy = y - self.center_y

        rotated_x = (
            self.center_x
            + dx * math.cos(self.rotation_offset_rad)
            - dy * math.sin(self.rotation_offset_rad)
        )

        rotated_y = (
            self.center_y
            + dx * math.sin(self.rotation_offset_rad)
            + dy * math.cos(self.rotation_offset_rad)
        )

        return rotated_x, rotated_y

    def align_pose(self, x, y, theta=None):
        aligned_x, aligned_y = self.rotate_point(x, y)

        if theta is not None:
            aligned_theta = theta + self.rotation_offset_rad
            aligned_theta = (
                aligned_theta + math.pi
            ) % (2 * math.pi) - math.pi

            return aligned_x, aligned_y, aligned_theta

        return aligned_x, aligned_y


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("FastSLAM Micro-Simulator")
    clock = pygame.time.Clock()

    env = Environment()

    true_trajectory = []
    current_lap_particle_paths = {}
    displayed_best_trajectory = []

    left_x = 2.835
    right_x = 16.865
    bottom_y = 2.835
    top_y = 16.865

    true_start_pose = [left_x, bottom_y, math.pi / 2]
    robot = SimulatedTurtlebot(*true_start_pose)
    slam = FastSLAM1(initial_pose=true_start_pose)

    alignment = EstimationAlignment(
        rotation_offset_deg=-1.3,
        center_x=9.85,
        center_y=9.85
    )

    pygame.font.init()
    my_font = pygame.font.SysFont("Arial", 24)

    true_pose_history = []
    estimated_pose_history = []
    position_error_history = []
    landmark_error_history = []

    running = True
    v, w = 0.0, 0.0

    auto_drive = False
    waypoint_index = 0

    # Go UP first, then right, then down, then left
    waypoints = [
        (left_x, top_y),
        (right_x, top_y),
        (right_x, bottom_y),
        (left_x, bottom_y),
    ]

    lap_detector = LandmarkLapDetector(
        cooldown_frames=FPS * 2,
        threshold=3.0
    )

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
                    robot.theta = math.pi / 2

                    robot.odom_x = robot.x
                    robot.odom_y = robot.y
                    robot.odom_theta = robot.theta

                    waypoint_index = 0

                    true_trajectory = []
                    current_lap_particle_paths = {}
                    displayed_best_trajectory = []

                    lap_detector.reset()

        if auto_drive:
            k_heading = 2.0
            max_w = 0.8
            waypoint_radius = 0.35

            target_x, target_y = waypoints[waypoint_index]

            dx = target_x - robot.x
            dy = target_y - robot.y

            distance_to_target = math.hypot(dx, dy)

            if distance_to_target < waypoint_radius:
                waypoint_index = (waypoint_index + 1) % len(waypoints)

                target_x, target_y = waypoints[waypoint_index]

                dx = target_x - robot.x
                dy = target_y - robot.y

            target_theta = math.atan2(dy, dx)

            angle_error = (
                target_theta - robot.theta + math.pi
            ) % (2 * math.pi) - math.pi

            w = k_heading * angle_error
            w = max(-max_w, min(max_w, w))

            if abs(angle_error) > 0.2:
                v = 0.4
            else:
                v = 1.2

        all_walls = env.outer_walls + env.inner_walls
        robot.move(v, w, DT, all_walls)

        bx, by = robot.get_body_center()
        true_trajectory.append((bx, by))

        sensor_data = robot.get_camera_measurements(env.landmarks)
        odom_data = robot.get_odometry()

        lap_completed = lap_detector.update(
            sensor_data,
            [robot.x, robot.y, robot.theta]
        )

        if lap_completed:
            best_particle_index = None
            best_final_weight = -1.0

            for particle_index, path in current_lap_particle_paths.items():
                if len(path) == 0:
                    continue

                final_weight = path[-1][2]

                if final_weight > best_final_weight:
                    best_final_weight = final_weight
                    best_particle_index = particle_index

            if best_particle_index is not None:
                displayed_best_trajectory = [
                    (x, y)
                    for x, y, weight in current_lap_particle_paths[best_particle_index]
                ]

                print(
                    f"Lap best particle: {best_particle_index}, "
                    f"weight={best_final_weight:.6f}, "
                    f"points={len(displayed_best_trajectory)}"
                )

            current_lap_particle_paths = {}

        particles, est_pose, est_map = slam.step(odom_data, sensor_data, DT)

        for particle_index, particle in enumerate(particles):
            px, py, ptheta, pweight = particle

            if particle_index not in current_lap_particle_paths:
                current_lap_particle_paths[particle_index] = []

            body_offset = robot.body_offset
            corrected_x = px - body_offset * math.cos(ptheta)
            corrected_y = py - body_offset * math.sin(ptheta)

            aligned_x, aligned_y = alignment.align_pose(
                corrected_x,
                corrected_y
            )

            current_lap_particle_paths[particle_index].append(
                (aligned_x, aligned_y, pweight)
            )

        true_pose = [robot.x, robot.y, robot.theta]
        true_pose_history.append(true_pose)
        estimated_pose_history.append(est_pose)

        aligned_est_x, aligned_est_y = alignment.align_pose(
            est_pose[0],
            est_pose[1]
        )

        pos_error = math.hypot(
            robot.x - aligned_est_x,
            robot.y - aligned_est_y
        )

        position_error_history.append(pos_error)

        lm_errors = []

        for lm_id, est_lm in est_map.items():
            if lm_id in env.landmarks:
                true_lm = env.landmarks[lm_id]

                aligned_lm_x, aligned_lm_y = alignment.align_pose(
                    est_lm[0],
                    est_lm[1]
                )

                lm_error = math.hypot(
                    true_lm[0] - aligned_lm_x,
                    true_lm[1] - aligned_lm_y
                )

                lm_errors.append(lm_error)

        if len(lm_errors) > 0:
            landmark_error_history.append(sum(lm_errors) / len(lm_errors))

        screen.fill((240, 240, 240))

        env.draw(screen, to_screen)

        if len(true_trajectory) > 1:
            trajectory_points = [to_screen(x, y) for x, y in true_trajectory]
            pygame.draw.lines(screen, (0, 200, 0), False, trajectory_points, 2)

        if len(displayed_best_trajectory) > 1:
            best_particle_points = [
                to_screen(x, y) for x, y in displayed_best_trajectory
            ]
            pygame.draw.lines(screen, (0, 0, 255), False, best_particle_points, 3)

        for particle in particles:
            px, py, ptheta = particle[:3]

            pos = to_screen(px, py)
            pygame.draw.circle(screen, (255, 100, 100), pos, 2)

        for mark_id, (lx, ly) in est_map.items():
            aligned_lx, aligned_ly = alignment.align_pose(lx, ly)
            pos = to_screen(aligned_lx, aligned_ly)

            pygame.draw.rect(
                screen,
                (0, 0, 255),
                (pos[0] - 4, pos[1] - 4, 8, 8)
            )

        bx, by = robot.get_body_center()
        rob_pos = to_screen(bx, by)
        pixel_radius = int(robot.radius * SCALE)

        pygame.draw.circle(screen, (0, 0, 0), rob_pos, pixel_radius)

        end_x = bx + (robot.radius + 0.1) * math.cos(robot.theta)
        end_y = by + (robot.radius + 0.1) * math.sin(robot.theta)

        pygame.draw.line(
            screen,
            (255, 0, 0),
            rob_pos,
            to_screen(end_x, end_y),
            2
        )

        text_v = my_font.render(
            f"Velocidade Linear (v): {v:.2f} m/s",
            True,
            (0, 0, 0)
        )
        text_w = my_font.render(
            f"Velocidade Angular (w): {w:.2f} rad/s",
            True,
            (0, 0, 0)
        )
        text_align = my_font.render(
            f"Alignment rot: {alignment.rotation_offset_deg:.1f} deg",
            True,
            (0, 0, 0)
        )

        screen.blit(text_v, (10, 10))
        screen.blit(text_w, (10, 40))
        screen.blit(text_align, (10, 70))

        pygame.display.flip()
        clock.tick(FPS)

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