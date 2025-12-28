# env_state_representation.py
"""
State Representation for Atari-Authentic Pong
Handles state arrays, frame stacking, and distinguishable values.

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""
import numpy as np


class StateRepresentation:
    """Manages game state representation for RL agents"""
    
    def __init__(self, width, height, use_distinct_values=True,
                 state_value_player=1.0, state_value_cpu=0.5, state_value_ball=0.75,
                 use_frame_stacking=False, frame_stack_size=2):
        """
        Initialize state representation.
        
        Args:
            width, height: Court dimensions
            use_distinct_values: If True, use different values for player/CPU/ball
            state_value_player: Value for player paddle pixels
            state_value_cpu: Value for CPU paddle pixels
            state_value_ball: Value for ball pixels
            use_frame_stacking: If True, stack frames for velocity inference
            frame_stack_size: Number of frames to stack
        """
        self.width = width
        self.height = height
        self.use_distinct_values = use_distinct_values
        self.state_value_player = state_value_player
        self.state_value_cpu = state_value_cpu
        self.state_value_ball = state_value_ball
        self.use_frame_stacking = use_frame_stacking
        self.frame_stack_size = frame_stack_size
        self.frame_stack = []
    
    def get_state(self, player_y, cpu_y, ball_x, ball_y, 
                  player_paddle_x, cpu_paddle_x, paddle_width, paddle_height, ball_size):
        """
        Get current game state as 2D array.
        
        Args:
            player_y, cpu_y: Paddle Y positions
            ball_x, ball_y: Ball position
            player_paddle_x, cpu_paddle_x: Paddle X positions
            paddle_width, paddle_height: Paddle dimensions
            ball_size: Ball dimensions
        
        Returns:
            np.ndarray: State array of shape (height, width)
        """
        state = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Determine values for different objects
        if self.use_distinct_values:
            player_val = self.state_value_player
            cpu_val = self.state_value_cpu
            ball_val = self.state_value_ball
        else:
            player_val = cpu_val = ball_val = 1.0
        
        # Draw player paddle (right side)
        for i in range(paddle_height):
            y_pos = int(player_y) + i
            if 0 <= y_pos < self.height:
                x_start = player_paddle_x
                x_end = min(x_start + paddle_width, self.width)
                state[y_pos, x_start:x_end] = player_val
        
        # Draw CPU paddle (left side)
        for i in range(paddle_height):
            y_pos = int(cpu_y) + i
            if 0 <= y_pos < self.height:
                x_start = cpu_paddle_x
                x_end = min(x_start + paddle_width, self.width)
                state[y_pos, x_start:x_end] = cpu_val
        
        # Draw ball
        for i in range(ball_size):
            for j in range(ball_size):
                ball_y_pos = int(ball_y) + i
                ball_x_pos = int(ball_x) + j
                if 0 <= ball_y_pos < self.height and 0 <= ball_x_pos < self.width:
                    state[ball_y_pos, ball_x_pos] = ball_val
        
        return state
    
    def initialize_frame_stack(self):
        """Initialize frame stack with a zero-filled frame"""
        if self.use_frame_stacking:
            zero_frame = np.zeros((self.height, self.width), dtype=np.float32)
            self.frame_stack = [zero_frame.copy() for _ in range(self.frame_stack_size)]
    
    def update_frame_stack(self, new_frame):
        """
        Update frame stack with new frame.
        
        Args:
            new_frame: New state array to add to stack
        
        Returns:
            np.ndarray: Stacked frames of shape (height, width, frame_stack_size)
        """
        if self.use_frame_stacking:
            self.frame_stack.pop(0)
            self.frame_stack.append(new_frame.copy())
            return self.get_stacked_state()
        return new_frame
    
    def get_stacked_state(self):
        """
        Get stacked frames for velocity inference.
        
        Returns:
            np.ndarray: Stacked frames of shape (height, width, frame_stack_size)
        """
        if self.use_frame_stacking:
            return np.stack(self.frame_stack, axis=-1)
        return None
    
    def reset(self):
        """Reset frame stack"""
        self.initialize_frame_stack()