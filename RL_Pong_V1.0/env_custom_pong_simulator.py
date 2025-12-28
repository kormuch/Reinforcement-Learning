# env_custom_pong_simulator.py
"""
Custom Pong Simulator - Refactored Main Orchestrator
Coordinates physics, collisions, opponent AI, and state management.

This is the primary environment class that implements the gym-like API.
It delegates to specialized modules for physics, collision detection, etc.

Enhanced with collision tracking for analysis.

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import EnvConfig, VisualConfig, TrainingConfig

# Import refactored modules
from env_physics_engine import PhysicsEngine
from env_collision_detector import CollisionDetector
from env_opponent_controller import CPUOpponentController
from env_ball_mechanics import BallMechanics
from env_state_representation import StateRepresentation

# Global collision analyzer (set by orchestrator)
COLLISION_ANALYZER = None


class CustomPongSimulator:
    """
    Atari-like Pong environment with configurable parameters.
    
    Main orchestrator that coordinates:
    - PhysicsEngine: Ball movement and speed
    - CollisionDetector: Paddle and scoring collisions
    - CPUOpponentController: Opponent AI behavior
    - BallMechanics: Angle effects and rally system
    - StateRepresentation: State arrays and frame stacking
    
    Implements gym-like API: reset(), step(action)
    """
    
    def __init__(self, width=None, height=None, max_score=None, max_steps=None, cpu_difficulty=None):
        """
        Initialize Custom Pong Simulator.
        
        Args:
            width, height: Court dimensions (from config if None)
            max_score: Points to win (from config if None)
            max_steps: Max steps per episode (from config if None)
            cpu_difficulty: CPU difficulty level (from config if None)
        """
        # Load parameters from config
        self.width = width if width is not None else EnvConfig.WIDTH
        self.height = height if height is not None else EnvConfig.HEIGHT
        self.max_score = max_score if max_score is not None else EnvConfig.MAX_SCORE
        self.max_steps = max_steps if max_steps is not None else EnvConfig.MAX_STEPS_PER_EPISODE
        
        # Paddle configuration
        self.paddle_height = EnvConfig.PADDLE_HEIGHT
        self.paddle_width = EnvConfig.PADDLE_WIDTH
        self.paddle_speed_player = EnvConfig.PADDLE_SPEED_PLAYER
        self.player_paddle_x = self.width - EnvConfig.PADDLE_OFFSET_FROM_EDGE
        self.cpu_paddle_x = EnvConfig.PADDLE_OFFSET_FROM_EDGE
        
        # Initialize specialized modules
        self.physics = PhysicsEngine(
            self.width, self.height,
            EnvConfig.BALL_SIZE,
            EnvConfig.BALL_SPEED_X,
            EnvConfig.BALL_SPEED_Y,
            EnvConfig.BALL_SPEED_CONSTANT,
            EnvConfig.BALL_SPEED_NORMALIZE_AFTER_EDGE
        )
        
        self.collision_detector = CollisionDetector(
            self.width, self.height,
            EnvConfig.BALL_SIZE,
            self.paddle_width,
            self.paddle_height
        )
        
        cpu_speed = EnvConfig.OPPONENT_SPEEDS.get(cpu_difficulty or EnvConfig.OPPONENT_DIFFICULTY,
                                                    EnvConfig.OPPONENT_SPEEDS['medium'])
        self.cpu_opponent = CPUOpponentController(
            self.width, self.height,
            self.paddle_height,
            cpu_speed,
            EnvConfig.OPPONENT_REACTION_DELAY
        )
        
        self.ball_mechanics = BallMechanics(
            EnvConfig.ANGLE_VARIATION,
            EnvConfig.MAX_BALL_ANGLE,
            EnvConfig.USE_PROGRESSIVE_ANGLES,
            EnvConfig.ANGLE_INCREASE_VOLLEYS,
            EnvConfig.ANGLE_MULTIPLIER_PER_THRESHOLD,
            EnvConfig.MAX_ANGLE_MULTIPLIER
        )
        
        self.state_repr = StateRepresentation(
            self.width, self.height,
            EnvConfig.USE_DISTINCT_VALUES,
            EnvConfig.STATE_VALUE_PLAYER,
            EnvConfig.STATE_VALUE_OPPONENT,
            EnvConfig.STATE_VALUE_BALL,
            EnvConfig.USE_FRAME_STACKING,
            EnvConfig.FRAME_STACK_SIZE
        )
        
        # Reward configuration
        self.reward_score = TrainingConfig.REWARD_SCORE
        self.reward_opponent_score = TrainingConfig.REWARD_OPPONENT_SCORE
        self.reward_ball_hit = TrainingConfig.REWARD_BALL_HIT
        
        # Game state
        self.player_score = 0
        self.cpu_score = 0
        self.step_count = 0
        self.last_scorer = None
        
        # Paddle positions
        self.player_y = 0.0
        self.cpu_y = 0.0
        
        # Serve rule
        self.loser_serves = EnvConfig.LOSER_SERVES
        
        # For compatibility with old code
        self.cpu_speed = cpu_speed
        self.cpu_reaction_enabled = EnvConfig.OPPONENT_REACTION_DELAY
        self.OPPONENT_reaction_enabled = EnvConfig.OPPONENT_REACTION_DELAY
        
        # Episode tracking for collision analyzer
        self._current_episode = 0
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset scores and timing
        self.player_score = 0
        self.cpu_score = 0
        self.step_count = 0
        self.last_scorer = None
        
        # Reset paddle positions (centered)
        self.player_y = self.height // 2 - self.paddle_height // 2
        self.cpu_y = self.height // 2 - self.paddle_height // 2
        
        # Reset physics and mechanics
        self.physics.reset_ball()
        self.ball_mechanics.reset_rally()
        self.cpu_opponent.reset()
        self.state_repr.reset()
        
        return self._build_state()
    
    def step(self, action):
        """
        Execute one timestep.
        
        Args:
            action: 0=stay, 1=up, 2=down
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.step_count += 1
        
        # Update player paddle
        self._update_player_paddle(action)
        
        # Update CPU opponent
        self.cpu_y = self.cpu_opponent.update(
            self.cpu_y,
            self.physics.ball_x,
            self.physics.ball_y,
            self.physics.ball_vx,
            EnvConfig.BALL_SIZE
        )
        
        # Update ball physics
        self.physics.update()
        self.physics.check_wall_collisions()
        
        # Check collisions and handle scoring
        reward, point_scored = self._check_collisions()
        
        # Reset ball if point was scored
        if point_scored:
            self._reset_ball_for_new_point()
        
        # Check termination
        done, termination_reason = self._check_termination()
        
        # Build state and info
        new_state = self._build_state()
        info = {
            'player_score': self.player_score,
            'cpu_score': self.cpu_score,
            'step_count': self.step_count,
            'done_reason': termination_reason if done else None
        }
        
        return new_state, reward, done, info
    
    def _update_player_paddle(self, action):
        """Update player paddle position based on action"""
        if action == 1:  # Move up
            self.player_y = max(0, self.player_y - self.paddle_speed_player)
        elif action == 2:  # Move down
            max_y = self.height - self.paddle_height
            self.player_y = min(max_y, self.player_y + self.paddle_speed_player)
        
        # Prevent paddle from overlapping field boundaries
        BOUNDARY_MARGIN = 2
        min_y = BOUNDARY_MARGIN
        max_y = self.height - self.paddle_height - BOUNDARY_MARGIN
        self.player_y = np.clip(self.player_y, min_y, max_y)
    
    def _check_collisions(self):
        """Check for paddle collisions and scoring"""
        global COLLISION_ANALYZER
        
        reward = 0.0
        point_scored = False
        
        # Check all collision states
        player_collision = self.collision_detector.check_player_paddle_collision(
            self.physics.ball_x, self.physics.ball_y,
            self.player_paddle_x, self.player_y)
        
        cpu_collision = self.collision_detector.check_cpu_paddle_collision(
            self.physics.ball_x, self.physics.ball_y,
            self.cpu_paddle_x, self.cpu_y)
        
        score_result = self.collision_detector.check_scoring(self.physics.ball_x)
        
        # Check player paddle collision
        if player_collision:
            self.ball_mechanics.on_paddle_hit()
            self.physics.ball_vx = -abs(self.physics.ball_vx)
            
            # Apply angle effect
            hit_offset = self.collision_detector.get_paddle_hit_offset(
                self.physics.ball_y, self.player_y)
            self.physics.ball_vy = self.ball_mechanics.apply_paddle_angle_effect(
                self.physics.ball_vy, hit_offset)
            
            # Record collision for analyzer
            if COLLISION_ANALYZER is not None:
                # Calculate hit position (0 to paddle_height-1)
                hit_position = self.physics.ball_y - self.player_y
                hit_position = np.clip(hit_position, 0, self.paddle_height - 1)
                
                # Calculate resulting angle (matches ball_mechanics)
                max_angle = 75
                bounce_angle = hit_offset * max_angle
                
                COLLISION_ANALYZER.record_collision(
                    hit_position=float(hit_position),
                    bounce_angle=float(abs(bounce_angle)),
                    episode=self._current_episode
                )
            
            # Normalize speed if configured
            if EnvConfig.BALL_SPEED_CONSTANT or EnvConfig.BALL_SPEED_NORMALIZE_AFTER_EDGE:
                self.physics.normalize_speed()
            
            reward = self.reward_ball_hit
        
        # Check CPU paddle collision
        elif cpu_collision:
            self.ball_mechanics.on_paddle_hit()
            self.physics.ball_vx = abs(self.physics.ball_vx)
            
            # Apply angle effect
            hit_offset = self.collision_detector.get_paddle_hit_offset(
                self.physics.ball_y, self.cpu_y)
            self.physics.ball_vy = self.ball_mechanics.apply_paddle_angle_effect(
                self.physics.ball_vy, hit_offset)
            
            # Normalize speed if configured
            if EnvConfig.BALL_SPEED_CONSTANT or EnvConfig.BALL_SPEED_NORMALIZE_AFTER_EDGE:
                self.physics.normalize_speed()
            
            reward = self.reward_ball_hit
        
        # Check scoring
        elif score_result == 'player_scored':
            self.player_score += 1
            self.last_scorer = 'player'
            reward = self.reward_score
            point_scored = True
        
        elif score_result == 'cpu_scored':
            self.cpu_score += 1
            self.last_scorer = 'cpu'
            reward = self.reward_opponent_score
            point_scored = True
        
        return reward, point_scored
    
    def _reset_ball_for_new_point(self):
        """Reset ball with loser serve rule if enabled"""
        if self.loser_serves and self.last_scorer is not None:
            if self.last_scorer == 'cpu':
                # CPU scored → player serves (toward CPU)
                self.physics.reset_ball(vx_direction='left', vy_direction='random')
            else:
                # Player scored → CPU serves (toward player)
                self.physics.reset_ball(vx_direction='right', vy_direction='random')
        else:
            self.physics.reset_ball(vx_direction='random', vy_direction='random')
        
        self.ball_mechanics.reset_rally()
    
    def _build_state(self):
        """Build state representation"""
        state = self.state_repr.get_state(
            self.player_y, self.cpu_y,
            self.physics.ball_x, self.physics.ball_y,
            self.player_paddle_x, self.cpu_paddle_x,
            self.paddle_width, self.paddle_height,
            EnvConfig.BALL_SIZE
        )
        
        if self.state_repr.use_frame_stacking:
            return self.state_repr.update_frame_stack(state)
        
        return state
    
    def _check_termination(self):
        """Check if episode should terminate"""
        if self.player_score >= self.max_score:
            return True, "player_won"
        if self.cpu_score >= self.max_score:
            return True, "cpu_won"
        if self.step_count >= self.max_steps:
            return True, "max_steps"
        return False, None
    
    def get_state(self):
        """Get current state (for compatibility)"""
        return self._build_state()
    
    def get_game_info(self):
        """Get detailed game information"""
        return {
            'ball_position': self.physics.get_position(),
            'ball_velocity': self.physics.get_velocity(),
            'ball_speed': self.physics.get_speed(),
            'player_paddle_y': self.player_y,
            'cpu_paddle_y': self.cpu_y,
            'player_score': self.player_score,
            'cpu_score': self.cpu_score,
            'step_count': self.step_count,
            'last_scorer': self.last_scorer,
            **self.ball_mechanics.get_game_info()
        }
    
    def render(self, mode=None):
        """Render the environment"""
        if mode is None:
            mode = VisualConfig.DEFAULT_RENDER_MODE
        
        if mode == 'rgb_array':
            return self._render_rgb()
        elif mode == 'ascii':
            return self._render_ascii()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_rgb(self):
        """Render to RGB/grayscale array"""
        render_height = VisualConfig.RENDER_HEIGHT
        render_width = VisualConfig.RENDER_WIDTH
        score_height = VisualConfig.SCORE_AREA_HEIGHT if VisualConfig.INCLUDE_SCORES_IN_RENDER else 0
        
        if VisualConfig.COLOR_MODE == 'grayscale':
            pil_image = Image.new('L', (render_width, render_height), color=0)
        else:
            pil_image = Image.new('RGB', (render_width, render_height), color=VisualConfig.RENDER_BACKGROUND)
        
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate game area
        if VisualConfig.SCORE_POSITION == 'bottom':
            game_area_height = render_height - score_height
            game_y_offset = 0
            score_y_offset = game_area_height
        else:
            game_area_height = render_height - score_height
            game_y_offset = score_height
            score_y_offset = 0
        
        scale_x = render_width / self.width
        scale_y = game_area_height / self.height
        
        # Render scores
        if VisualConfig.INCLUDE_SCORES_IN_RENDER and score_height > 0:
            self._draw_scores(draw, render_width, score_height, score_y_offset)
        
        # Draw field boundaries
        if VisualConfig.DRAW_FIELD_BOUNDARIES:
            boundary_color = 255 if VisualConfig.COLOR_MODE == 'grayscale' else VisualConfig.RENDER_FOREGROUND
            line_width = VisualConfig.BOUNDARY_LINE_WIDTH
            
            for i in range(line_width):
                y = game_y_offset + i
                if y < render_height:
                    draw.line([(0, y), (render_width, y)], fill=boundary_color, width=1)
            
            for i in range(line_width):
                y = game_y_offset + game_area_height - line_width + i
                if 0 <= y < render_height:
                    draw.line([(0, y), (render_width, y)], fill=boundary_color, width=1)
        
        # Draw center line
        center_x = int(render_width / 2)
        line_color = 128 if VisualConfig.COLOR_MODE == 'grayscale' else (128, 128, 128)
        
        if VisualConfig.CENTER_LINE_DOTTED:
            dot_height = VisualConfig.CENTER_LINE_DOT_HEIGHT
            dot_spacing = VisualConfig.CENTER_LINE_DOT_SPACING
            y = game_y_offset + (VisualConfig.BOUNDARY_LINE_WIDTH if VisualConfig.DRAW_FIELD_BOUNDARIES else 0)
            y_max = game_y_offset + game_area_height - (VisualConfig.BOUNDARY_LINE_WIDTH if VisualConfig.DRAW_FIELD_BOUNDARIES else 0)
            
            while y < y_max:
                y_end = min(y + dot_height, y_max)
                if y_end > y:
                    draw.line([(center_x, y), (center_x, y_end)], fill=line_color, width=1)
                y += dot_height + dot_spacing
        else:
            y_start = game_y_offset + (VisualConfig.BOUNDARY_LINE_WIDTH if VisualConfig.DRAW_FIELD_BOUNDARIES else 0)
            y_end = game_y_offset + game_area_height - (VisualConfig.BOUNDARY_LINE_WIDTH if VisualConfig.DRAW_FIELD_BOUNDARIES else 0)
            draw.line([(center_x, y_start), (center_x, y_end)], fill=line_color, width=1)
        
        # Draw objects
        fg_color = 255 if VisualConfig.COLOR_MODE == 'grayscale' else VisualConfig.RENDER_FOREGROUND
        
        # Player paddle
        paddle_x_scaled = int(self.player_paddle_x * scale_x)
        paddle_width_scaled = max(2, int(self.paddle_width * scale_x))
        
        for i in range(self.paddle_height):
            y_game = int(self.player_y) + i
            if 0 <= y_game < self.height:
                y_render = int(y_game * scale_y) + game_y_offset
                if game_y_offset <= y_render < game_y_offset + game_area_height:
                    draw.rectangle([paddle_x_scaled, y_render, paddle_x_scaled + paddle_width_scaled, y_render + 1], fill=fg_color)
        
        # CPU paddle
        cpu_paddle_x_scaled = int(self.cpu_paddle_x * scale_x)
        
        for i in range(self.paddle_height):
            y_game = int(self.cpu_y) + i
            if 0 <= y_game < self.height:
                y_render = int(y_game * scale_y) + game_y_offset
                if game_y_offset <= y_render < game_y_offset + game_area_height:
                    draw.rectangle([cpu_paddle_x_scaled, y_render, cpu_paddle_x_scaled + paddle_width_scaled, y_render + 1], fill=fg_color)
        
        # Ball
        ball_x_scaled = int(self.physics.ball_x * scale_x)
        ball_y_scaled = int(self.physics.ball_y * scale_y) + game_y_offset
        ball_size_scaled = max(2, int(EnvConfig.BALL_SIZE * scale_x))
        
        if game_y_offset <= ball_y_scaled < game_y_offset + game_area_height:
            draw.rectangle([ball_x_scaled, ball_y_scaled, ball_x_scaled + ball_size_scaled, ball_y_scaled + ball_size_scaled], fill=fg_color)
        
        image = np.array(pil_image)
        if VisualConfig.COLOR_MODE == 'grayscale' and VisualConfig.CHANNELS == 1:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def _draw_scores(self, draw, render_width, score_height, y_offset):
        """Draw score display"""
        try:
            font_size = max(12, score_height - 10)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        text_color = 200 if VisualConfig.COLOR_MODE == 'grayscale' else VisualConfig.RENDER_SCORE_COLOR
        
        cpu_score_text = str(self.cpu_score)
        player_score_text = str(self.player_score)
        text_y = y_offset + (score_height // 2) - 6
        
        if font:
            bbox_cpu = draw.textbbox((0, 0), cpu_score_text, font=font)
            text_width_cpu = bbox_cpu[2] - bbox_cpu[0]
            draw.text((render_width // 4 - text_width_cpu // 2, text_y), cpu_score_text, fill=text_color, font=font)
            
            bbox_player = draw.textbbox((0, 0), player_score_text, font=font)
            text_width_player = bbox_player[2] - bbox_player[0]
            draw.text(((render_width * 3) // 4 - text_width_player // 2, text_y), player_score_text, fill=text_color, font=font)
        else:
            draw.text((render_width // 4, text_y), cpu_score_text, fill=text_color)
            draw.text(((render_width * 3) // 4, text_y), player_score_text, fill=text_color)
    
    def _render_ascii(self):
        """Render ASCII representation"""
        court = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Center line
        center_x = self.width // 2
        if VisualConfig.CENTER_LINE_DOTTED:
            dot_height = VisualConfig.CENTER_LINE_DOT_HEIGHT
            dot_spacing = VisualConfig.CENTER_LINE_DOT_SPACING
            y = 0
            while y < self.height:
                for i in range(dot_height):
                    if y + i < self.height:
                        court[y + i][center_x] = '│'
                y += dot_height + dot_spacing
        else:
            for y in range(self.height):
                court[y][center_x] = '│'
        
        # Paddles
        for i in range(self.paddle_height):
            y = int(self.player_y) + i
            if 0 <= y < self.height:
                for x in range(self.player_paddle_x, min(self.player_paddle_x + self.paddle_width, self.width)):
                    court[y][x] = '█'
        
        for i in range(self.paddle_height):
            y = int(self.cpu_y) + i
            if 0 <= y < self.height:
                for x in range(self.cpu_paddle_x, min(self.cpu_paddle_x + self.paddle_width, self.width)):
                    court[y][x] = '█'
        
        # Ball
        for i in range(EnvConfig.BALL_SIZE):
            for j in range(EnvConfig.BALL_SIZE):
                y = int(self.physics.ball_y) + i
                x = int(self.physics.ball_x) + j
                if 0 <= y < self.height and 0 <= x < self.width:
                    court[y][x] = '●'
        
        # Build output
        output_lines = []
        
        if VisualConfig.ASCII_SHOW_SCORES:
            output_lines.append(f"CPU: {self.cpu_score}          Player: {self.player_score}")
            output_lines.append("")
        
        if VisualConfig.ASCII_SHOW_VELOCITY:
            output_lines.append(f"Ball velocity: ({self.physics.ball_vx:.1f}, {self.physics.ball_vy:.1f})")
            output_lines.append(f"Step: {self.step_count}")
            output_lines.append("")
        
        if VisualConfig.DRAW_FIELD_BOUNDARIES:
            output_lines.append("┏" + "━" * self.width + "┓")
        else:
            output_lines.append("┌" + "─" * self.width + "┐")
        
        for row in court:
            output_lines.append("│" + "".join(row) + "│")
        
        if VisualConfig.DRAW_FIELD_BOUNDARIES:
            output_lines.append("┗" + "━" * self.width + "┛")
        else:
            output_lines.append("└" + "─" * self.width + "┘")
        
        return "\n".join(output_lines)