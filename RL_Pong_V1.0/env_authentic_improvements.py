# env_authentic_improvements.py
"""
Authentic Pong Improvements - Drop-in Enhancements
Add these features to make your Pong even more authentic to the original.

Features:
1. 8-Segment Paddle Physics (discrete angles like original)
2. Sound Effects System (authentic beeps and boops)
3. 7-Segment Score Display
4. Serve Delay
5. CRT-style Visual Effects

Usage:
    Add these classes to your custom_pong_simulator.py or import them
"""

import numpy as np
import math

# ============================================================================
# 1. AUTHENTIC 8-SEGMENT PADDLE PHYSICS
# ============================================================================

class AuthenticPaddlePhysics:
    """
    Implements the original Pong 8-segment paddle physics.
    
    The paddle is divided into 8 equal segments:
    - Center segments (3,4,5,6): Shallow return angles
    - Edge segments (1,2,7,8): Steep return angles
    - Dead center: Returns perpendicular
    
    This creates strategic depth: hitting ball with edge of paddle
    makes it harder for opponent to return.
    """
    
    def __init__(self, paddle_height, ball_size, ball_speed):
        self.paddle_height = paddle_height
        self.ball_size = ball_size
        self.ball_speed = ball_speed
        
        # Define 8 segments with discrete angles (in degrees)
        # Negative = upward, Positive = downward
        self.segment_angles = [
            -60,  # Top edge (segment 0)
            -45,  # segment 1
            -30,  # segment 2
            -15,  # segment 3
            15,   # segment 4
            30,   # segment 5
            45,   # segment 6
            60    # Bottom edge (segment 7)
        ]
    
    def get_bounce_velocity(self, paddle_y, ball_y, ball_coming_from_right):
        """
        Calculate ball velocity after paddle hit using 8-segment physics.
        
        Args:
            paddle_y: Y-position of paddle top
            ball_y: Y-position of ball top
            ball_coming_from_right: If True, ball bounces left; else right
            
        Returns:
            tuple: (vx, vy) - New velocity components
        """
        # Calculate which segment was hit
        ball_center_y = ball_y + self.ball_size / 2
        paddle_center_y = paddle_y + self.paddle_height / 2
        
        # Relative hit position on paddle (0 to paddle_height)
        hit_position = ball_center_y - paddle_y
        
        # Clamp to paddle bounds
        hit_position = max(0, min(self.paddle_height, hit_position))
        
        # Determine segment (0-7)
        segment_height = self.paddle_height / 8
        segment = int(hit_position / segment_height)
        segment = min(7, segment)  # Ensure within bounds
        
        # Get angle for this segment
        angle_degrees = self.segment_angles[segment]
        angle_radians = math.radians(angle_degrees)
        
        # Calculate velocity components maintaining constant speed
        # Horizontal direction depends on which side was hit
        direction = -1 if ball_coming_from_right else 1
        
        vx = direction * self.ball_speed * math.cos(angle_radians)
        vy = self.ball_speed * math.sin(angle_radians)
        
        return vx, vy
    
    def apply_to_ball(self, ball, paddle_y, coming_from_right):
        """
        Convenience method to apply physics directly to ball object.
        
        Args:
            ball: Object with ball_x, ball_y, ball_vx, ball_vy attributes
            paddle_y: Y-position of paddle
            coming_from_right: Boolean for direction
        """
        ball.ball_vx, ball.ball_vy = self.get_bounce_velocity(
            paddle_y, ball.ball_y, coming_from_right
        )


# ============================================================================
# 2. AUTHENTIC SOUND EFFECTS SYSTEM
# ============================================================================

class PongSoundSystem:
    """
    Recreates the iconic Pong sound effects.
    
    Original Pong sounds:
    - Paddle hit: ~220 Hz (low beep)
    - Wall hit: ~440 Hz (high beep)  
    - Score: ~330 Hz (mid tone, longer)
    """
    
    def __init__(self, sample_rate=22050):
        """
        Initialize sound system.
        
        Args:
            sample_rate: Audio sample rate (22050 Hz is retro-appropriate)
        """
        self.sample_rate = sample_rate
        self.enabled = False
        
        try:
            import pygame
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=512)
            self.pygame = pygame
            self.enabled = True
            
            # Generate sound effects
            self.paddle_hit_sound = self._generate_beep(220, 0.08)
            self.wall_hit_sound = self._generate_beep(440, 0.06)
            self.score_sound = self._generate_beep(330, 0.15)
            self.game_over_sound = self._generate_victory_sound()
            
            print("✓ Pong sound system initialized")
            
        except ImportError:
            print("⚠ pygame not available - sound effects disabled")
        except Exception as e:
            print(f"⚠ Sound initialization failed: {e}")
    
    def _generate_beep(self, frequency, duration):
        """Generate a simple sine wave beep."""
        if not self.enabled:
            return None
        
        num_samples = int(self.sample_rate * duration)
        samples = np.zeros(num_samples, dtype=np.int16)
        
        # Generate sine wave
        for i in range(num_samples):
            t = float(i) / self.sample_rate
            # Add envelope (fade out) for less harsh sound
            envelope = 1.0 - (i / num_samples) ** 0.5
            amplitude = 4096 * envelope  # 16-bit audio range
            samples[i] = int(amplitude * math.sin(2.0 * math.pi * frequency * t))
        
        # Create pygame Sound object
        sound = self.pygame.sndarray.make_sound(samples)
        return sound
    
    def _generate_victory_sound(self):
        """Generate a simple victory jingle."""
        if not self.enabled:
            return None
        
        # Three ascending tones
        notes = [330, 415, 523]  # E4, G#4, C5
        duration = 0.12
        
        total_duration = len(notes) * duration
        num_samples = int(self.sample_rate * total_duration)
        samples = np.zeros(num_samples, dtype=np.int16)
        
        samples_per_note = int(self.sample_rate * duration)
        
        for note_idx, freq in enumerate(notes):
            start = note_idx * samples_per_note
            end = min(start + samples_per_note, num_samples)
            
            for i in range(start, end):
                t = float(i - start) / self.sample_rate
                envelope = 1.0 - ((i - start) / samples_per_note) ** 0.5
                amplitude = 3072 * envelope
                samples[i] = int(amplitude * math.sin(2.0 * math.pi * freq * t))
        
        sound = self.pygame.sndarray.make_sound(samples)
        return sound
    
    def play_paddle_hit(self):
        """Play paddle hit sound."""
        if self.enabled and self.paddle_hit_sound:
            self.paddle_hit_sound.play()
    
    def play_wall_hit(self):
        """Play wall hit sound."""
        if self.enabled and self.wall_hit_sound:
            self.wall_hit_sound.play()
    
    def play_score(self):
        """Play score sound."""
        if self.enabled and self.score_sound:
            self.score_sound.play()
    
    def play_game_over(self):
        """Play game over sound."""
        if self.enabled and self.game_over_sound:
            self.game_over_sound.play()


# ============================================================================
# 3. 7-SEGMENT SCORE DISPLAY
# ============================================================================

class SevenSegmentDisplay:
    """
    Renders numbers in 7-segment display style (like original Pong).
    
    Segment layout:
         _a_
        |   |
       f| g |b
        |___|
        |   |
       e|   |c
        |_d_|
    """
    
    # Define which segments are lit for each digit (0-9)
    SEGMENTS = {
        0: [1, 1, 1, 1, 1, 1, 0],  # abcdef
        1: [0, 1, 1, 0, 0, 0, 0],  # bc
        2: [1, 1, 0, 1, 1, 0, 1],  # abdeg
        3: [1, 1, 1, 1, 0, 0, 1],  # abcdg
        4: [0, 1, 1, 0, 0, 1, 1],  # bcfg
        5: [1, 0, 1, 1, 0, 1, 1],  # acdfg
        6: [1, 0, 1, 1, 1, 1, 1],  # acdefg
        7: [1, 1, 1, 0, 0, 0, 0],  # abc
        8: [1, 1, 1, 1, 1, 1, 1],  # all
        9: [1, 1, 1, 1, 0, 1, 1],  # abcdfg
    }
    
    @staticmethod
    def render_digit(digit, width=9, height=13):
        """
        Render a single digit in 7-segment style.
        
        Args:
            digit: 0-9 to render
            width: Width of digit in pixels
            height: Height of digit in pixels
            
        Returns:
            np.ndarray: Binary array with digit shape
        """
        display = np.zeros((height, width), dtype=np.uint8)
        
        if digit not in SevenSegmentDisplay.SEGMENTS:
            return display
        
        segments = SevenSegmentDisplay.SEGMENTS[digit]
        
        # Segment dimensions
        h_seg_len = width - 2
        v_seg_len = (height - 3) // 2
        thickness = 1
        
        # Segment positions (y, x, length, horizontal?)
        # a: top horizontal
        if segments[0]:
            display[0, 1:1+h_seg_len] = 255
        
        # b: top right vertical
        if segments[1]:
            display[1:1+v_seg_len, width-1] = 255
        
        # c: bottom right vertical
        if segments[2]:
            display[1+v_seg_len+1:height, width-1] = 255
        
        # d: bottom horizontal
        if segments[3]:
            display[height-1, 1:1+h_seg_len] = 255
        
        # e: bottom left vertical
        if segments[4]:
            display[1+v_seg_len+1:height, 0] = 255
        
        # f: top left vertical
        if segments[5]:
            display[1:1+v_seg_len, 0] = 255
        
        # g: middle horizontal
        if segments[6]:
            mid_y = height // 2
            display[mid_y, 1:1+h_seg_len] = 255
        
        return display
    
    @staticmethod
    def render_score(score, max_digits=2, digit_width=9, digit_height=13, spacing=2):
        """
        Render a score (multi-digit number) in 7-segment style.
        
        Args:
            score: Number to render
            max_digits: Maximum digits to display
            digit_width: Width of each digit
            digit_height: Height of each digit
            spacing: Pixels between digits
            
        Returns:
            np.ndarray: Binary array with score rendered
        """
        score_str = str(score).zfill(max_digits)[-max_digits:]
        
        total_width = (digit_width * len(score_str)) + (spacing * (len(score_str) - 1))
        display = np.zeros((digit_height, total_width), dtype=np.uint8)
        
        x_offset = 0
        for digit_char in score_str:
            digit = int(digit_char)
            digit_img = SevenSegmentDisplay.render_digit(digit, digit_width, digit_height)
            display[:, x_offset:x_offset+digit_width] = digit_img
            x_offset += digit_width + spacing
        
        return display


# ============================================================================
# 4. SERVE DELAY SYSTEM
# ============================================================================

class ServeDelayManager:
    """
    Manages the delay between points (like original Pong).
    After a point is scored, game pauses for ~2 seconds before serving.
    """
    
    def __init__(self, delay_steps=120):  # 2 seconds at 60 FPS
        """
        Initialize serve delay manager.
        
        Args:
            delay_steps: Number of steps to delay (120 = 2 sec at 60 FPS)
        """
        self.delay_steps = delay_steps
        self.current_delay = 0
        self.is_delaying = False
    
    def start_delay(self):
        """Start the serve delay."""
        self.is_delaying = True
        self.current_delay = self.delay_steps
    
    def update(self):
        """
        Update delay counter.
        
        Returns:
            bool: True if still delaying, False if delay finished
        """
        if not self.is_delaying:
            return False
        
        self.current_delay -= 1
        
        if self.current_delay <= 0:
            self.is_delaying = False
            return False
        
        return True
    
    def is_active(self):
        """Check if delay is currently active."""
        return self.is_delaying


# ============================================================================
# 5. USAGE EXAMPLE
# ============================================================================

class AuthenticPongEnhancements:
    """
    Example integration of all authentic improvements.
    Add these to your CustomPongSimulator class.
    """
    
    def __init__(self, env):
        """
        Initialize all authentic enhancements.
        
        Args:
            env: CustomPongSimulator instance
        """
        self.env = env
        
        # Initialize 8-segment paddle physics
        self.paddle_physics = AuthenticPaddlePhysics(
            paddle_height=env.paddle_height,
            ball_size=env.ball_size,
            ball_speed=env.target_ball_speed
        )
        
        # Initialize sound system
        self.sound = PongSoundSystem()
        
        # Initialize serve delay
        self.serve_delay = ServeDelayManager(delay_steps=120)
        
        print("✓ Authentic Pong enhancements initialized")
    
    def enhanced_paddle_collision(self, is_player_paddle):
        """
        Replace standard collision with 8-segment physics.
        
        Args:
            is_player_paddle: True if player paddle, False if AI paddle
        """
        if is_player_paddle:
            paddle_y = self.env.player_y
            coming_from_right = True
        else:
            paddle_y = self.env.ai_y
            coming_from_right = False
        
        # Apply authentic physics
        self.paddle_physics.apply_to_ball(self.env, paddle_y, coming_from_right)
        
        # Play sound
        self.sound.play_paddle_hit()
    
    def on_wall_hit(self):
        """Call when ball hits top or bottom wall."""
        self.sound.play_wall_hit()
    
    def on_score(self, player_scored):
        """
        Call when a point is scored.
        
        Args:
            player_scored: True if player scored, False if AI scored
        """
        self.sound.play_score()
        self.serve_delay.start_delay()
    
    def on_game_over(self):
        """Call when game ends."""
        self.sound.play_game_over()
    
    def should_pause_for_serve(self):
        """Check if game should be paused for serve delay."""
        return self.serve_delay.is_active()
    
    def update_serve_delay(self):
        """Update serve delay counter."""
        return self.serve_delay.update()


# ============================================================================
# INTEGRATION INSTRUCTIONS
# ============================================================================

"""
To integrate these enhancements into your custom_pong_simulator.py:

1. ADD TO __init__:
   self.enhancements = AuthenticPongEnhancements(self)

2. REPLACE _apply_paddle_angle_effect WITH:
   self.enhancements.enhanced_paddle_collision(is_player_paddle=True)

3. ADD TO _update_ball_physics (wall collision):
   if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
       self.enhancements.on_wall_hit()

4. ADD TO _check_collisions (scoring):
   if point_scored:
       self.enhancements.on_score(player_scored=(self.last_scorer == 'player'))

5. ADD TO step METHOD (serve delay):
   if self.enhancements.should_pause_for_serve():
       self.enhancements.update_serve_delay()
       return self.get_state(), 0.0, False, info  # Pause during serve

6. ADD TO _check_termination (game over):
   if done:
       self.enhancements.on_game_over()

7. OPTIONAL - Use 7-segment display in render():
   score_img = SevenSegmentDisplay.render_score(self.player_score)
"""