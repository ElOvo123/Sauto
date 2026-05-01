import pygame

class Environment:
    def __init__(self):
        # True map landmarks {id: [x, y]}
        # Landmarks: 4 per corridor (bottom, right, top, left).
        # For each corridor there are 2 landmarks 0.1 m from the outer wall
        # and 2 landmarks 0.1 m from the inner wall. IDs 1..16.
        self.landmarks = {
            # Bottom corridor (y near outer_min_y=2.0 and inner_min_y=4.25)
            1: [4.5, 2.1],    # outer-close, left
            2: [14.5, 2.1],   # outer-close, right
            3: [7.5, 4.15],   # inner-close, left
            4: [11.5, 4.15],  # inner-close, right
            # Right corridor (x near outer_max_x=18.5 and inner_max_x=16.25)
            5: [18.4, 4.5],   # outer-close, bottom
            6: [18.4, 14.5],  # outer-close, top
            7: [16.35, 7.5],  # inner-close, bottom
            8: [16.35, 11.5], # inner-close, top
            # Top corridor (y near outer_max_y=18.5 and inner_max_y=16.25)
            9: [4.5, 18.4],   # outer-close, left
            10: [14.5, 18.4], # outer-close, right
            11: [7.5, 16.35], # inner-close, left
            12: [11.5, 16.35],# inner-close, right
            # Left corridor (x near outer_min_x=2.0 and inner_min_x=4.25)
            13: [2.1, 4.5],   # outer-close, bottom
            14: [2.1, 14.5],  # outer-close, top
            15: [4.15, 7.5],  # inner-close, bottom
            16: [4.15, 11.5], # inner-close, top
        }
        # Walls defined as line segments: ((x1, y1), (x2, y2)) in meters
        # Define outer bounds explicitly then compute inner bounds using a margin
        outer_min_x = 2.0
        outer_min_y = 2.0
        outer_max_x = 18.5
        outer_max_y = 18.5

        self.outer_walls = [
            ((outer_min_x, outer_min_y), (outer_max_x, outer_min_y)),   # Bottom
            ((outer_max_x, outer_min_y), (outer_max_x, outer_max_y)),   # Right
            ((outer_max_x, outer_max_y), (outer_min_x, outer_max_y)),   # Top
            ((outer_min_x, outer_max_y), (outer_min_x, outer_min_y)),   # Left
        ]

        # Desired uniform gap from outer walls to inner walls (meters)
        margin = 2.25

        inner_min_x = outer_min_x + margin
        inner_min_y = outer_min_y + margin
        inner_max_x = outer_max_x - margin
        inner_max_y = outer_max_y - margin

        self.inner_walls = [
            ((inner_min_x, inner_min_y), (inner_max_x, inner_min_y)),   # Bottom
            ((inner_max_x, inner_min_y), (inner_max_x, inner_max_y)),   # Right
            ((inner_max_x, inner_max_y), (inner_min_x, inner_max_y)),   # Top
            ((inner_min_x, inner_max_y), (inner_min_x, inner_min_y)),   # Left
        ]

    def draw(self, screen, to_screen_fn):
        # Draw Walls
        for wall in self.outer_walls:
            start_px = to_screen_fn(wall[0][0], wall[0][1])
            end_px = to_screen_fn(wall[1][0], wall[1][1])
            pygame.draw.line(screen, (50, 50, 50), start_px, end_px, 4)

        for wall in self.inner_walls:
            start_px = to_screen_fn(wall[0][0], wall[0][1])
            end_px = to_screen_fn(wall[1][0], wall[1][1])
            pygame.draw.line(screen, (50, 50, 50), start_px, end_px, 4)
            
        # Draw True Landmarks (Green squares)
        for mark_id, (lx, ly) in self.landmarks.items():
            pos = to_screen_fn(lx, ly)
            pygame.draw.rect(screen, (0, 200, 0), (pos[0]-5, pos[1]-5, 10, 10))