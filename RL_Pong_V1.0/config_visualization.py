# config_visualization.py
# =============================================================================
# GUI PARAMETERS
# =============================================================================

from config_environment import EnvConfig


class GUIConfig:
    """Visualization and GUI configuration"""
    
    # GUI enable/disable
    ENABLE_GUI = True
    
    # Display settings (scale up Atari resolution for visibility)
    PIXEL_SIZE = 3  # Scale factor: 160×3 = 480 width, 192×3 = 576 height
    
    # Colors (RGB tuples)
    COLOR_BLACK = (0, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (50, 50, 50)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (255, 0, 0)
    
    # Frame rate
    GUI_FPS = 60  # Atari-authentic 60 FPS
    VISUALIZE_EVERY_N_STEPS = 5
    
    # Font sizes
    FONT_SIZE_LARGE = 48
    FONT_SIZE_SMALL = 24
    
    # Demonstration settings
    DEMO_EPISODES = 3
    DEMO_MAX_STEPS = 200
    DEMO_UPDATE_EVERY = 5
    
    @classmethod
    def get_screen_dimensions(cls):
        """Calculate screen dimensions based on environment and scaling"""
        width = EnvConfig.WIDTH * cls.PIXEL_SIZE
        height = EnvConfig.HEIGHT * cls.PIXEL_SIZE
        return width, height


# =============================================================================
# LOGGING & ANALYSIS PARAMETERS
# =============================================================================

class LoggingConfig:
    """Logging and performance analysis configuration"""
    
    # File paths
    LEARNING_CURVE_PATH = 'learning_curve.png'
    TRAINING_REPORT_PATH = 'pong_training_report.txt'
    
    # Plot settings
    PLOT_DPI = 300
    PLOT_FIGURE_SIZE = (12, 10)
    PLOT_ALPHA_RAW = 0.3
    PLOT_ALPHA_GRID = 0.3
    
    # Moving average windows
    WINDOW_SIZE_SMALL = 100
    WINDOW_SIZE_LARGE = 500
    
    # Performance metrics
    RECENT_EPISODES_WINDOW = 100