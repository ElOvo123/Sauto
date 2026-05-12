import random

import pygame

class Environment:
    def __init__(self, fake_environment=True, landmarks_per_corridor=10, seed=None):
        # True map landmarks {id: [x, y]}
        # fake_environment=True generates random landmark positions inside each corridor.
        # landmarks_per_corridor can be an int (same count for all corridors) or a dict.
        self.random = random.Random(seed)

        self.doors = {
            #Left corridor
            #left
            ((2.0, 5.6), (2.0, 6.4)),
            ((2.0, 8.6), (2.0, 9.4)),
            ((2.0, 11.55), (2.0, 12.35)),
            ((2.0, 14.55), (2.0, 15.35)),
            #right
            ((3.67, 9.9), (3.67, 10.8)),

            #Top corridor
            #left
            ((2.15, 17.75), (3.3, 17.75)),
            ((7.35, 17.75), (8.15, 17.75)),
            ((14.15, 17.75), (15.3, 17.75)),
            ((16.45, 17.75), (17.6, 17.75)),          
            
            #Right side
            #left
            ((17.75, 12.9), (17.75, 11.75)),
            ((17.75, 9.65), (17.75, 8.85)),
            ((17.75, 5.55), (17.75, 4.4)),
            #right
            ((16.08, 14.20), (16.08, 13.40)),  
            ((16.08, 11.10), (16.08, 10.30)), 
            ((16.08, 9.95), (16.08, 8.85)),   
             
            
            #Bottom side
            #left
            ((2.15, 2.0), (3.3, 2.0)),
            ((5.7, 2.0), (7.2, 2.0)),
            ((11.6, 2.0), (12.75, 2.0)),
            ((16.45, 2.0), (17.6, 2.0)),
            #right
            ((5.85, 3.67), (7.95, 3.67)),

            
        }

        if fake_environment == True:
            self.landmarks = self._generate_fake_landmarks(landmarks_per_corridor)
        else:
            self.landmarks = {
                # Left corridor
                1: [2.07, 6.34],    
                2: [3.67, 10.75],   
                3: [2.07, 15.15],   
                4: [3.67, 17.72],  
                # Top corridor
                5: [4.66, 16.05],   
                6: [7.85, 17.72],  
                7: [13.28, 16.49],  
                8: [15.0, 17.72], 
                # Right corridor
                9: [17.66, 11.8],   
                10: [17.59, 2.01], 
                #11: [7.5, 16.35], 
                #12: [11.5, 16.35],
                # bottom corridor 
                13: [5.85, 3.6],   
                14: [10.20, 3.6],  
                #15: [4.15, 7.5], 
                #16: [4.15, 11.5],
            }
        # Walls defined as line segments: ((x1, y1), (x2, y2)) in meters
        # Define outer bounds explicitly then compute inner bounds using a margin
        outer_min_x = 2.0
        outer_min_y = 2.0
        outer_max_x = 17.75
        outer_max_y = 17.75

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

    def _generate_fake_landmarks(self, landmarks_per_corridor):
        # Corridor boundaries used by the fake environment.
        outer_min_x = 2.0
        outer_min_y = 2.0
        outer_max_x = 17.75
        outer_max_y = 17.75

        inner_min_x = 3.67
        inner_max_x = 16.08
        inner_min_y = 3.67
        inner_max_y = 16.08

        def count_for(corridor_name):
            if isinstance(landmarks_per_corridor, dict):
                return int(landmarks_per_corridor.get(corridor_name, 0))
            return int(landmarks_per_corridor)

        def sorted_positions(count, low, high):
            if count <= 0:
                return []
            if count == 1:
                return [(low + high) / 2.0]
            values = [self.random.uniform(low, high) for _ in range(count)]
            values.sort()
            return values

        landmarks = {}
        landmark_id = 1

        # Keep landmark 1 as a stable start landmark so lap detection can trigger.
        landmarks[landmark_id] = [2.07, self.random.uniform(4.5, 7.5)]
        landmark_id += 1

        def add_landmarks_for_corridor(corridor_name, wall_placements, fixed_axis_is_x, count_offset=0):
            nonlocal landmark_id

            count = max(0, count_for(corridor_name) - count_offset)
            if count <= 0:
                return

            for _ in range(count):
                fixed_value, span_low, span_high = self.random.choice(wall_placements)
                variable_value = self.random.uniform(span_low, span_high)
                if fixed_axis_is_x:
                    landmarks[landmark_id] = [fixed_value, variable_value]
                else:
                    landmarks[landmark_id] = [variable_value, fixed_value]
                landmark_id += 1

        # Left corridor: x is fixed on either wall, y uses the corresponding wall span.
        add_landmarks_for_corridor(
            "left",
            [
                (outer_min_x, outer_min_y, outer_max_y),
                (inner_min_x, inner_min_y, inner_max_y),
            ],
            True,
            count_offset=1,
        )

        # Top corridor: y is fixed on either wall, x uses the corresponding wall span.
        add_landmarks_for_corridor(
            "top",
            [
                (outer_max_y, outer_min_x, outer_max_x),
                (inner_max_y, inner_min_x, inner_max_x),
            ],
            False,
        )

        # Right corridor: x is fixed on either wall, y uses the corresponding wall span.
        add_landmarks_for_corridor(
            "right",
            [
                (outer_max_x, outer_min_y, outer_max_y),
                (inner_max_x, inner_min_y, inner_max_y),
            ],
            True,
        )

        # Bottom corridor: y is fixed on either wall, x uses the corresponding wall span.
        add_landmarks_for_corridor(
            "bottom",
            [
                (outer_min_y, outer_min_x, outer_max_x),
                (inner_min_y, inner_min_x, inner_max_x),
            ],
            False,
        )

        return landmarks

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

        # Draw Doors
        for door in self.doors:
            start_px = to_screen_fn(door[0][0], door[0][1])
            end_px = to_screen_fn(door[1][0], door[1][1])
            pygame.draw.line(screen, (200, 200, 200), start_px, end_px, 4)
            
        # Draw True Landmarks (Green squares)
        for mark_id, (lx, ly) in self.landmarks.items():
            pos = to_screen_fn(lx, ly)
            pygame.draw.rect(screen, (0, 200, 0), (pos[0]-5, pos[1]-5, 10, 10))