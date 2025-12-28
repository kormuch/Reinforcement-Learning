# env_physics_engine.py
"""
Physics Engine for Atari-Authentic Pong
Handles ball movement, wall collisions, and speed management.

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""
import numpy as np


class PhysicsEngine:
    """Handles all physics calculations: ball movement, collisions, and speed management"""
    
    def __init__(self, width, height, ball_size, ball_speed_x, ball_speed_y, 
                 ball_speed_constant=True, normalize_after_edge=True):
        """
        Initialize physics engine.
        
        Args:
            width, height: Court dimensions
            ball_size: Ball dimensions
            ball_speed_x, ball_speed_y: Initial speed components
            ball_speed_constant: If True, maintain constant speed (Atari authentic)
            normalize_after_edge: If True, renormalize speed after collisions
        """
        self.width = width
        self.height = height
        self.ball_size = ball_size
        self.ball_speed_constant = ball_speed_constant
        self.normalize_after_edge = normalize_after_edge
        
        # Calculate target speed for normalization
        self.target_ball_speed = np.sqrt(ball_speed_x**2 + ball_speed_y**2)
        
        # Ball state
        self.ball_x = float(width // 2)
        self.ball_y = float(height // 2)
        self.ball_vx = float(ball_speed_x)
        self.ball_vy = float(ball_speed_y)
    
    def reset_ball(self, vx_direction='random', vy_direction='random'):
        """
        Reset ball to center with initial velocity.
        
        Args:
            vx_direction: 'left', 'right', or 'random'
            vy_direction: 'up', 'down', or 'random'
        """
        self.ball_x = float(self.width // 2)
        self.ball_y = float(self.height // 2)
        
        base_speed_x = self.target_ball_speed * 0.7  # Approximate
        base_speed_y = self.target_ball_speed * 0.7
        
        # Determine horizontal direction
        if vx_direction == 'left':
            self.ball_vx = -base_speed_x
        elif vx_direction == 'right':
            self.ball_vx = base_speed_x
        else:  # random
            self.ball_vx = float(np.random.choice([-base_speed_x, base_speed_x]))
        
        # Determine vertical direction
        if vy_direction == 'up':
            self.ball_vy = -base_speed_y
        elif vy_direction == 'down':
            self.ball_vy = base_speed_y
        else:  # random
            self.ball_vy = float(np.random.choice([-base_speed_y, base_speed_y]))
    
    def update(self):
        """Update ball position based on current velocity"""
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
    
    def check_wall_collisions(self):
        """
        Check and handle collisions with top/bottom walls.
        
        Returns:
            bool: True if wall collision occurred
        """
        collision = False
        
        # Top wall
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy = abs(self.ball_vy)
            collision = True
        
        # Bottom wall
        elif self.ball_y >= self.height - self.ball_size:
            self.ball_y = self.height - self.ball_size
            self.ball_vy = -abs(self.ball_vy)
            collision = True
        
        return collision
    
    def normalize_speed(self):
        """Normalize ball velocity to maintain constant speed (Atari authentic)"""
        current_speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
        if current_speed > 0:
            scale = self.target_ball_speed / current_speed
            self.ball_vx *= scale
            self.ball_vy *= scale
    
    def get_position(self):
        """Get current ball position"""
        return (self.ball_x, self.ball_y)
    
    def get_velocity(self):
        """Get current ball velocity"""
        return (self.ball_vx, self.ball_vy)
    
    def get_speed(self):
        """Get current ball speed magnitude"""
        return np.sqrt(self.ball_vx**2 + self.ball_vy**2)
    
    def apply_velocity_change(self, dvx=0.0, dvy=0.0):
        """Apply velocity change (used for paddle angle effects)"""
        self.ball_vx += dvx
        self.ball_vy += dvy
    
    def bounce_horizontal(self):
        """Reverse horizontal velocity (paddle bounce)"""
        self.ball_vx = -self.ball_vx
    
    def bounce_vertical(self):
        """Reverse vertical velocity (wall bounce)"""
        self.ball_vy = -self.ball_vy