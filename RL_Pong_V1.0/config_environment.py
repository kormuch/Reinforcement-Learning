# config_environment.py
# =============================================================================
# ENVIRONMENT PARAMETERS - ATARI 2600 AUTHENTIC
# =============================================================================

class EnvConfig:
    """Pong environment configuration - Atari 2600 authentic specifications"""
    
    # Court dimensions (Atari 2600 standard resolution)
    WIDTH = 160
    HEIGHT = 192
    
    # Paddle settings (Atari authentic) - SLIMMER AND FASTER
    PADDLE_HEIGHT = 12  # Reduced from 20 for smaller paddles
    PADDLE_WIDTH = 2    # SLIMMER: Reduced from 3 to 2 pixels
    PADDLE_SPEED_PLAYER = 4 # FASTER: Increased from 4 to 6 pixels per action
    PADDLE_OFFSET_FROM_EDGE = 16  # Distance from side walls
    
    # Ball settings (Atari authentic - 1972 original)
    BALL_SIZE = 1  # Reduced from 2 for smaller ball
    BALL_SPEED_X = 2  # Horizontal velocity magnitude (CONSTANT - never increases)
    BALL_SPEED_Y = 2  # Vertical velocity magnitude (CONSTANT - never increases)
    BALL_SPEED_CONSTANT = True  # Original Pong: speed was always constant
    BALL_SPEED_NORMALIZE_AFTER_EDGE = True  # Maintain constant speed
    
    # CPU opponent settings (distinct from RL agent)
    OPPONENT_DIFFICULTY = 'medium'  # Options: 'easy', 'medium', 'hard'
    OPPONENT_SPEEDS = {
        'easy': 3,      # Slower, more beatable
        'medium': 4,    # Balanced, challenging but fair
        'hard': 5       # Fast but still beatable
    }
    OPPONENT_REACTION_DELAY = True  # CPU only moves when ball crosses midline
    
    
    # Episode structure (Atari standard)
    MAX_SCORE = 21  # Points to win (Atari standard)
    MAX_STEPS_PER_EPISODE = 20000  # Safety limit
    
    # Physics (Atari-style)
    ANGLE_VARIATION = 0.5   # Increased from 0.3 - edge hits add more angle
    MAX_BALL_ANGLE = 4.0    # Increased from 3.0 - allows steeper angles
    
    # Progressive angle difficulty (AUTHENTIC 1972 ATARI PONG)
    USE_PROGRESSIVE_ANGLES = True  # Authentic: speed constant, angles get steeper
    ANGLE_INCREASE_VOLLEYS = [4, 12, 20, 28]  # Volleys at which to increase angle multiplier
    ANGLE_MULTIPLIER_PER_THRESHOLD = 1.15     # Multiplier for angle progression
    MAX_ANGLE_MULTIPLIER = 2.0                # Maximum multiplier for angles
    
    # State representation
    USE_DISTINCT_VALUES = True  # Use different values for player/CPU/ball in state
    STATE_VALUE_PLAYER = 1.0    # Value for player paddle pixels
    STATE_VALUE_OPPONENT = 0.5  # Value for CPU paddle pixels (distinguishable!)
    STATE_VALUE_BALL = 0.75     # Value for ball pixels
    
    # Frame stacking (for velocity inference)
    USE_FRAME_STACKING = False  # Stack multiple frames for velocity info
    FRAME_STACK_SIZE = 2        # Number of frames to stack
    
    # Serve mechanism (Atari authentic)
    LOSER_SERVES = True  # Loser of previous point serves (Atari style)
    
    @classmethod
    def get_env_params(cls):
        """Returns dictionary of parameters for CustomPongSimulator initialization"""
        return {
            'width': cls.WIDTH,
            'height': cls.HEIGHT,
            'max_score': cls.MAX_SCORE,
            'max_steps': cls.MAX_STEPS_PER_EPISODE,
            'cpu_difficulty': cls.OPPONENT_DIFFICULTY
        }


# =============================================================================
# VISUAL RENDERING PARAMETERS - ATARI AUTHENTIC
# =============================================================================

class VisualConfig:
    """Configuration for visual rendering - Atari 2600 authentic"""
    
    # Rendering dimensions (match Atari 2600)
    RENDER_WIDTH = 160
    RENDER_HEIGHT = 192
    
    # Rendering modes
    DEFAULT_RENDER_MODE = 'rgb_array'
    
    # Visual properties
    COLOR_MODE = 'grayscale'  # Atari was monochrome
    CHANNELS = 1
    
    # Score display (Atari style - at TOP, above playing field)
    SCORE_AREA_HEIGHT = 30  # Increased from 24 to prevent cutoff
    INCLUDE_SCORES_IN_RENDER = True
    SCORE_POSITION = 'top'  # Scores at top, field below
    SCORE_DIGIT_HEIGHT = 13  # Height of 7-segment digits
    SCORE_DIGIT_WIDTH = 9    # Width of 7-segment digits
    
    # Colors for rendering (grayscale Atari style)
    RENDER_BACKGROUND = 0        # Black
    RENDER_FOREGROUND = 255      # White
    RENDER_SCORE_COLOR = 200     # Light gray for scores
    RENDER_LINE_COLOR = 128      # Gray for center line
    
    # Center line (Atari dotted line) - STRIPED/DASHED
    CENTER_LINE_DOTTED = True
    CENTER_LINE_DOT_HEIGHT = 3  # Height of each dash
    CENTER_LINE_DOT_SPACING = 3  # Spacing between dashes
    
    # Playing field boundaries (NEW)
    DRAW_FIELD_BOUNDARIES = True  # Draw top and bottom lines
    BOUNDARY_LINE_WIDTH = 2       # Width of boundary lines in pixels
    
    # ASCII rendering
    ASCII_SHOW_SCORES = True
    ASCII_SHOW_VELOCITY = False