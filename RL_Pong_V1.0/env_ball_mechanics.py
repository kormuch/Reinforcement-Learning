# env_ball_mechanics.py
"""
Ball Mechanics for Atari-Authentic Pong
Implements paddle angle effects and progressive angle difficulty.

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""
import numpy as np


class BallMechanics:
    """Handles ball angle effects and progressive difficulty (authentic 1972 Pong)"""
    
    def __init__(self, angle_variation, max_ball_angle, use_progressive_angles=True,
                 angle_increase_volleys=None, angle_multiplier_per_threshold=1.15,
                 max_angle_multiplier=2.0):
        """
        Initialize ball mechanics.
        
        Args:
            angle_variation: Base angle change from paddle hit
            max_ball_angle: Maximum allowed vertical velocity
            use_progressive_angles: If True, angles increase with rally volleys
            angle_increase_volleys: List of volley counts where angle increases
            angle_multiplier_per_threshold: Multiplier applied at each threshold
            max_angle_multiplier: Maximum angle multiplier cap
        """
        self.angle_variation = angle_variation
        self.max_ball_angle = max_ball_angle
        self.use_progressive_angles = use_progressive_angles
        self.angle_increase_volleys = angle_increase_volleys or [4, 12, 20, 28]
        self.angle_multiplier_per_threshold = angle_multiplier_per_threshold
        self.max_angle_multiplier = max_angle_multiplier
        
        # Rally state
        self.volley_count = 0
        self.current_angle_multiplier = 1.0
    
    def reset_rally(self):
        """Reset rally tracking for new point"""
        self.volley_count = 0
        self.current_angle_multiplier = 1.0
    
    def on_paddle_hit(self):
        """Called when ball hits a paddle - increments volley count and applies progressive angles"""
        self.volley_count += 1
        
        if self.use_progressive_angles:
            self._apply_progressive_angles()
    
    def _apply_progressive_angles(self):
        """
        Apply authentic 1972 Atari Pong angle progression.
        
        AUTHENTIC ORIGINAL PONG MECHANIC:
        - Ball SPEED stays constant (never increases)
        - Ball ANGLES get steeper at volley thresholds
        - This creates difficulty without changing speed
        """
        # Check if we've reached an angle increase threshold
        if self.volley_count in self.angle_increase_volleys:
            # Increase angle multiplier
            self.current_angle_multiplier *= self.angle_multiplier_per_threshold
            
            # Cap at maximum multiplier
            self.current_angle_multiplier = min(self.current_angle_multiplier,
                                                self.max_angle_multiplier)
    
    def apply_paddle_angle_effect(self, ball_vy, paddle_hit_offset):
        """
        Modify ball vertical velocity based on where it hits the paddle.
        
        AUTHENTIC 1972 ATARI PONG:
        - Edge hits create steeper angles
        - Center hits create shallower angles
        - Angle steepness increases during rallies (not speed!)
        
        Args:
            ball_vy: Current ball vertical velocity
            paddle_hit_offset: Normalized hit position (-1.0 to +1.0)
        
        Returns:
            float: New ball vertical velocity
        """
        # Apply angle variation (scaled by current angle multiplier)
        # As volleys increase, angles get steeper
        angle_effect = paddle_hit_offset * self.angle_variation * self.current_angle_multiplier
        ball_vy += angle_effect
        
        # Clamp to maximum angle (also scaled by multiplier)
        max_angle = self.max_ball_angle * self.current_angle_multiplier
        ball_vy = np.clip(ball_vy, -max_angle, max_angle)
        
        return ball_vy
    
    def get_game_info(self):
        """Get current ball mechanics state for debugging"""
        return {
            'volley_count': self.volley_count,
            'angle_multiplier': self.current_angle_multiplier
        }