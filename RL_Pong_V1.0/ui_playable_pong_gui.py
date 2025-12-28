# ui_playable_gui.py
"""
Playable Pong GUI - Human vs CPU Interface
Allows human players to play against the CPU opponent using keyboard controls.

Controls:
    ↑ / UP ARROW    : Move paddle up
    ↓ / DOWN ARROW  : Move paddle down
    R               : Restart game
    ESC             : Quit

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""
import pygame
import sys
import numpy as np
from env_custom_pong_simulator import CustomPongSimulator
from config import EnvConfig, GUIConfig, VisualConfig

# Initialize Pygame
pygame.init()

class PlayablePongGUI:
    """GUI for human-playable Pong using CustomPongSimulator"""
    def __init__(self, player_mode='human', model_path=None):
        """Initialize the playable GUI
        Args:
            player_mode: 'human' or 'ai'
            model_path: Path to trained agent (required if player_mode='ai')
        """
        self.player_mode = player_mode
        # Create environment
        self.env = CustomPongSimulator()
        
        # Display settings from config
        self.pixel_size = GUIConfig.PIXEL_SIZE
        self.fps = GUIConfig.GUI_FPS  # Now using config value (60 FPS Atari-authentic)
        
        # Calculate screen dimensions
        # We render the game at VisualConfig resolution, then scale up for display
        self.render_width = VisualConfig.RENDER_WIDTH
        self.render_height = VisualConfig.RENDER_HEIGHT
        
        # Display window size (scaled up for visibility)
        self.display_width = self.render_width * self.pixel_size
        self.display_height = self.render_height * self.pixel_size
        
        # Create display
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("Atari Pong - Human vs CPU")
        
        # Clock for frame rate control
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_WHITE = GUIConfig.COLOR_WHITE
        self.COLOR_BLACK = GUIConfig.COLOR_BLACK
        self.COLOR_GREEN = GUIConfig.COLOR_GREEN
        self.COLOR_RED = GUIConfig.COLOR_RED
        
        # Game state
        self.running = True
        self.game_over = False
        self.winner = None
        
        # Debug info display
        self.show_debug = False
        
        # Check if AI-mode is set
        if self.player_mode == 'ai':
            self._load_ai_agent(model_path)
            
        # Reset game
        self.reset_game()
    
    def _load_ai_agent(self, model_path):
        """Load trained AI agent for demonstration"""
        from rl_feature_extractor import FeatureExtractor
        from rl_policy_gradient_agent import PolicyGradientAgent
        from config import PolicyGradientAgentConfig
        
        print(f"  Loading AI agent from: {model_path}")
        
        # Create feature extractor
        self.feature_extractor = FeatureExtractor(self.env)
        
        # Create agent
        self.agent = PolicyGradientAgent(
            input_size=PolicyGradientAgentConfig.INPUT_SIZE,
            hidden_size=PolicyGradientAgentConfig.HIDDEN_SIZE,
            learning_rate=PolicyGradientAgentConfig.LEARNING_RATE,
            discount=PolicyGradientAgentConfig.DISCOUNT_FACTOR
        )
        # Load trained weights
        self.agent.load(model_path)
        print(f"  ✓ AI agent loaded successfully")
        
    def reset_game(self):
        """Reset the game to initial state"""
        self.state = self.env.reset()
        self.game_over = False
        self.winner = None
    
    def handle_events(self):
        """Handle keyboard input and window events"""
        action = 0  # Default: stay
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return action
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return action
                
                if event.key == pygame.K_r:
                    self.reset_game()
                    return action
                
                if event.key == pygame.K_d:
                    # Toggle debug info
                    self.show_debug = not self.show_debug
        
        # Continuous key checking for paddle movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = 1  # Move up
        elif keys[pygame.K_DOWN]:
            action = 2  # Move down
        
        return action
    
    def update_game(self, action):
        """Update game state based on action"""
        if not self.game_over:
            self.state, reward, done, info = self.env.step(action)
            
            if done:
                self.game_over = True
                if info['done_reason'] == 'player_won':
                    self.winner = 'PLAYER'
                elif info['done_reason'] == 'cpu_won':
                    self.winner = 'CPU'
                else:
                    self.winner = 'DRAW'
    
    def render_game(self):
        """Render the game to the screen"""
        # Get rendered image from environment
        rendered_image = self.env.render('rgb_array')
        
        # Handle grayscale vs RGB
        if VisualConfig.COLOR_MODE == 'grayscale':
            # Convert grayscale to RGB for pygame (expects 3 channels)
            if rendered_image.shape[-1] == 1:
                rendered_image = np.squeeze(rendered_image, axis=-1)
            # Create RGB surface from grayscale
            rendered_rgb = np.stack([rendered_image, rendered_image, rendered_image], axis=-1)
        else:
            rendered_rgb = rendered_image
        
        # Scale up to display size
        small_surface = pygame.surfarray.make_surface(
            np.transpose(rendered_rgb, (1, 0, 2))  # Pygame expects (width, height, channels)
        )
        scaled_surface = pygame.transform.scale(
            small_surface,
            (self.display_width, self.display_height)
        )
        
        # Draw to screen
        self.screen.blit(scaled_surface, (0, 0))
        
        # Draw debug info if enabled
        if self.show_debug:
            self._draw_debug_info()
        
        # Draw game over overlay if needed
        if self.game_over:
            self._draw_game_over_overlay()
        
        # Update display
        pygame.display.flip()
    
    def _draw_debug_info(self):
        """Draw debug information overlay"""
        info = self.env.get_game_info()
        
        debug_lines = [
            f"Ball Speed: {info['ball_speed']:.2f} (constant)",
            f"Volley Count: {info['volley_count']}",
            f"Angle Multiplier: {info['angle_multiplier']:.2f}x",
            f"Ball Vel: ({info['ball_velocity'][0]:.2f}, {info['ball_velocity'][1]:.2f})",
            f"Steps: {info['step_count']}",
        ]
        
        y_offset = 10
        for line in debug_lines:
            text_surface = self.font_small.render(line, True, self.COLOR_GREEN)
            # Draw background
            bg_rect = text_surface.get_rect(topleft=(10, y_offset))
            bg_rect.inflate_ip(10, 5)
            pygame.draw.rect(self.screen, self.COLOR_BLACK, bg_rect)
            # Draw text
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
    
    def _draw_game_over_overlay(self):
        """Draw game over message"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.display_width, self.display_height))
        overlay.set_alpha(180)
        overlay.fill(self.COLOR_BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Winner text
        if self.winner == 'PLAYER':
            text = "YOU WIN!"
            color = self.COLOR_GREEN
        elif self.winner == 'CPU':
            text = "CPU WINS!"
            color = self.COLOR_RED
        else:
            text = "DRAW!"
            color = self.COLOR_WHITE
        
        winner_surface = self.font_large.render(text, True, color)
        winner_rect = winner_surface.get_rect(
            center=(self.display_width // 2, self.display_height // 2 - 30)
        )
        self.screen.blit(winner_surface, winner_rect)
        
        # Final score
        score_text = f"{self.env.cpu_score} - {self.env.player_score}"
        score_surface = self.font_medium.render(score_text, True, self.COLOR_WHITE)
        score_rect = score_surface.get_rect(
            center=(self.display_width // 2, self.display_height // 2 + 20)
        )
        self.screen.blit(score_surface, score_rect)
        
        # Restart hint
        restart_surface = self.font_small.render("Press R to restart", True, self.COLOR_WHITE)
        restart_rect = restart_surface.get_rect(
            center=(self.display_width // 2, self.display_height // 2 + 60)
        )
        self.screen.blit(restart_surface, restart_rect)
    
    def run(self):
        """Main game loop"""
        print("=" * 60)
        print("ATARI PONG - Human vs CPU")
        print("=" * 60)
        print(f"Playing to {EnvConfig.MAX_SCORE} points")
        print(f"CPU Difficulty: {EnvConfig.OPPONENT_DIFFICULTY}")
        print(f"CPU Speed: {self.env.cpu_speed} pixels/step")
        print(f"CPU Reaction: {'Waits for midline' if self.env.cpu_reaction_enabled else 'Always tracking'}")
        print(f"Serve Rule: {'Loser serves' if EnvConfig.LOSER_SERVES else 'Random'}")
        print("\nControls:")
        print("  ↑ / ↓ ARROW : Move paddle")
        print("  R           : Restart game")
        print("  D           : Toggle debug info")
        print("  ESC         : Quit")
        print("=" * 60)
        
        while self.running:
            # Handle input
            action = self.handle_events()
            
            # Update game state
            self.update_game(action)
            
            # Render
            self.render_game()
            
            # Control frame rate
            self.clock.tick(self.fps)
        
        # Cleanup
        pygame.quit()
        print("\nThanks for playing!")
        print(f"Final Score: CPU {self.env.cpu_score} - {self.env.player_score} Player")