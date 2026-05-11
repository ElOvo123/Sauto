import pygame

class Environment:
    def __init__(self):
        # True map landmarks {id: [x, y]}
        # Landmarks: 4 per corridor (bottom, right, top, left).
        # For each corridor there are 2 landmarks 0.1 m from the outer wall
        # and 2 landmarks 0.1 m from the inner wall. IDs 1..16.
        self.landmarks = {
            # Left corridor
            1: [2.07, 6.34],    
            2: [3.67, 10.78],   
            3: [2.07, 15.06],   
            4: [3.67, 17.72],  
            # Top corridor
            5: [4.66, 16.05],   
            6: [8.23, 17.72],  
            7: [13.28, 16.49],  
            8: [16.11, 17.72], 
            # Right corridor
            9: [17.66, 11.84],   
            10: [17.59, 2.01], 
            #11: [7.5, 16.35], 
            #12: [11.5, 16.35],
            # bottom corridor 
            13: [5.7, 3.6],   
            14: [10.5, 3.6],  
            #15: [4.15, 7.5], 
            #16: [4.15, 11.5],
        }
        # Walls defined as line segments: ((x1, y1), (x2, y2)) in meters
        # Define outer bounds explicitly then compute inner bounds using a margin
        outer_min_x = 2.0
        outer_min_y = 2.0
        outer_max_x = 17.7
        outer_max_y = 17.7

        self.outer_walls = [
            ((outer_min_x, outer_min_y), (outer_max_x, outer_min_y)),   # Bottom
            ((outer_max_x, outer_min_y), (outer_max_x, outer_max_y)),   # Right
            ((outer_max_x, outer_max_y), (outer_min_x, outer_max_y)),   # Top
            ((outer_min_x, outer_max_y), (outer_min_x, outer_min_y)),   # Left
        ]

        # Desired uniform gap from outer walls to inner walls (meters)
        margin = 1.67

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