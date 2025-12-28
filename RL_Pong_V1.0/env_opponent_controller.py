# env_opponent_controller.py
"""
CPU Opponent Controller for Atari-Authentic Pong
Implements realistic, beatable AI opponent behavior.

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""


class CPUOpponentController:
    """Controls CPU opponent paddle behavior with realistic, beatable AI"""
    
    def __init__(self, court_width, court_height, paddle_height, cpu_speed, 
                 reaction_enabled=True):
        """
        Initialize CPU opponent controller.
        
        Args:
            court_width, court_height: Court dimensions
            paddle_height: Height of CPU paddle
            cpu_speed: Pixels per step the CPU can move
            reaction_enabled: If True, CPU only reacts when ball crosses midline
        """
        self.court_width = court_width
        self.court_height = court_height
        self.paddle_height = paddle_height
        self.cpu_speed = cpu_speed
        self.reaction_enabled = reaction_enabled
        
        # Tracking state
        self.ball_crossed_midline = False
    
    def update(self, cpu_y, ball_x, ball_y, ball_vx, ball_size):
        """
        Update CPU paddle position based on ball position.
        
        CPU follows realistic rules:
        1. Limited paddle speed (beatable)
        2. Only reacts when ball crosses midline (realistic delay)
        3. Only tracks when ball is moving toward it
        
        Args:
            cpu_y: Current CPU paddle Y position
            ball_x, ball_y: Ball position
            ball_vx: Ball horizontal velocity
            ball_size: Size of ball
        
        Returns:
            float: New CPU paddle Y position
        """
        midline_x = self.court_width / 2
        
        # Track if ball has crossed midline
        if ball_vx < 0:  # Ball moving left (toward CPU)
            if ball_x < midline_x:
                self.ball_crossed_midline = True
        else:
            # Ball moving away, reset tracking
            self.ball_crossed_midline = False
        
        # CPU can only react if ball is moving toward it
        can_react = ball_vx < 0
        
        # Apply midline rule if enabled
        if self.reaction_enabled:
            can_react = can_react and self.ball_crossed_midline
        
        if can_react:
            # Calculate centers
            ball_center = ball_y + ball_size / 2
            cpu_center = cpu_y + self.paddle_height / 2
            
            # Move CPU paddle toward ball center (with limited speed)
            # Use tolerance of 2 pixels instead of 1 for more movement
            if ball_center < cpu_center - 2:
                # Ball is above CPU center - move up
                cpu_y = max(0, cpu_y - self.cpu_speed)
            elif ball_center > cpu_center + 2:
                # Ball is below CPU center - move down
                max_y = self.court_height - self.paddle_height
                cpu_y = min(max_y, cpu_y + self.cpu_speed)
        
        # If ball moving away or hasn't crossed midline, CPU doesn't move
        # This creates realistic, beatable behavior
        
        return cpu_y
    
    def reset(self):
        """Reset CPU state tracking"""
        self.ball_crossed_midline = False