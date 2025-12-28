# =============================================================================
# PONG CONFIG PACKAGE
# Central imports and validation for all configuration modules
# =============================================================================

# Import all config classes for convenient access
from config_environment import EnvConfig, VisualConfig
from config_agent_base import AgentConfig
from config_agent_policygradient import PolicyGradientAgentConfig
from config_agent_cnn import CNNAgentConfig
from config_training import TrainingConfig
from config_visualization import GUIConfig, LoggingConfig
from config_seed import SeedConfig

# Export all public classes
__all__ = [
    'EnvConfig',
    'VisualConfig',
    'AgentConfig',
    'PolicyGradientAgentConfig',
    'CNNAgentConfig',
    'TrainingConfig',
    'GUIConfig',
    'LoggingConfig',
    'SeedConfig',
    'validate_config',
    'print_config_summary',
    'get_all_params',
]


# =============================================================================
# VALIDATION & HELPER FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration parameters across all modules"""
    if EnvConfig.WIDTH <= 0 or EnvConfig.HEIGHT <= 0:
        raise ValueError("Environment dimensions must be positive")
    
    if EnvConfig.PADDLE_HEIGHT >= EnvConfig.HEIGHT:
        raise ValueError("Paddle height must be smaller than court height")
    
    if EnvConfig.OPPONENT_DIFFICULTY not in EnvConfig.OPPONENT_SPEEDS:
        raise ValueError(f"CPU difficulty must be one of {list(EnvConfig.OPPONENT_SPEEDS.keys())}")
    
    if PolicyGradientAgentConfig.INPUT_SIZE <= 0:
        raise ValueError("PolicyGradient agent input size must be positive")
    
    if CNNAgentConfig.INPUT_SIZE != EnvConfig.WIDTH * EnvConfig.HEIGHT:
        print(f"⚠ Warning: CNN input size ({CNNAgentConfig.INPUT_SIZE}) doesn't match environment ({EnvConfig.WIDTH * EnvConfig.HEIGHT})")
    
    if PolicyGradientAgentConfig.HIDDEN_SIZE <= 0:
        raise ValueError("Hidden layer size must be positive")
    
    if TrainingConfig.BATCH_SIZE <= 0:
        raise ValueError("Batch size must be positive")
    
    if not 0 < TrainingConfig.RUNNING_REWARD_DECAY < 1:
        raise ValueError("Running reward decay must be between 0 and 1")
    
    if not 0 < PolicyGradientAgentConfig.LEARNING_RATE < 1:
        raise ValueError("Learning rate must be between 0 and 1")
    
    print("✓ Configuration validation passed")


def print_config_summary():
    """Print summary of current configuration"""
    print("=" * 60)
    print("ATARI-AUTHENTIC PONG RL CONFIGURATION")
    print("=" * 60)
    
    print("\n[ENVIRONMENT - Atari 2600 Specifications]")
    print(f"  Court Size: {EnvConfig.WIDTH}×{EnvConfig.HEIGHT} (Atari standard)")
    print(f"  Paddle: {EnvConfig.PADDLE_WIDTH}×{EnvConfig.PADDLE_HEIGHT} pixels (slim)")
    print(f"  Paddle Speed: {EnvConfig.PADDLE_SPEED_PLAYER} pixels/action")
    print(f"  Paddle Offset: {EnvConfig.PADDLE_OFFSET_FROM_EDGE} pixels from edge")
    print(f"  Ball: {EnvConfig.BALL_SIZE}×{EnvConfig.BALL_SIZE} pixels (small)")
    print(f"  Max Score: {EnvConfig.MAX_SCORE} points")
    print(f"  Loser Serves: {EnvConfig.LOSER_SERVES}")
    print(f"  Progressive Angles: {EnvConfig.USE_PROGRESSIVE_ANGLES}")
    
    print("\n[CPU OPPONENT - Beatable & Realistic]")
    print(f"  Difficulty: {EnvConfig.OPPONENT_DIFFICULTY}")
    print(f"  Speed: {EnvConfig.OPPONENT_SPEEDS[EnvConfig.OPPONENT_DIFFICULTY]} pixels/step")
    print(f"  Reaction Rule: {'Midline-based' if EnvConfig.OPPONENT_REACTION_DELAY else 'Always active'}")
    
    print("\n[POLICY GRADIENT AGENT - Non-Visual (Current)]")
    print(f"  Input Size: {PolicyGradientAgentConfig.INPUT_SIZE} (features)")
    print(f"  Hidden Size: {PolicyGradientAgentConfig.HIDDEN_SIZE}")
    print(f"  Output Size: {PolicyGradientAgentConfig.OUTPUT_SIZE} (actions)")
    print(f"  Total Parameters: {PolicyGradientAgentConfig.get_total_parameters()}")
    print(f"  Learning Rate: {PolicyGradientAgentConfig.LEARNING_RATE}")
    print(f"  Discount Factor: {PolicyGradientAgentConfig.DISCOUNT_FACTOR}")
    
    print("\n[CNN AGENT - Visual]")
    print(f"  Input Size: {CNNAgentConfig.INPUT_SIZE} (pixels)")
    print(f"  Hidden Size: {CNNAgentConfig.HIDDEN_SIZE}")
    print(f"  Output Size: {CNNAgentConfig.OUTPUT_SIZE} (actions)")
    print(f"  Total Parameters: {CNNAgentConfig.get_total_parameters()}")
    print(f"  Learning Rate: {CNNAgentConfig.LEARNING_RATE}")
    print(f"  Discount Factor: {CNNAgentConfig.DISCOUNT_FACTOR}")
    
    print("\n[TRAINING]")
    print(f"  Max Episodes: {TrainingConfig.MAX_EPISODES}")
    print(f"  Batch Size: {TrainingConfig.BATCH_SIZE}")
    print(f"  Print Every N Episodes: {TrainingConfig.PRINT_EVERY_N_EPISODES}")
    print(f"  Running Reward Decay: {TrainingConfig.RUNNING_REWARD_DECAY}")
    
    print("\n[REWARDS]")
    print(f"  Score Point: +{TrainingConfig.REWARD_SCORE}")
    print(f"  Opponent Scores: {TrainingConfig.REWARD_OPPONENT_SCORE}")
    print(f"  Ball Hit (Shaping): +{TrainingConfig.REWARD_BALL_HIT}")
    print(f"  Game Win: +{TrainingConfig.REWARD_WIN}")
    print(f"  Game Loss: {TrainingConfig.REWARD_LOSS}")
    
    print("\n[EVALUATION & SAVING]")
    print(f"  Eval Episodes: {TrainingConfig.EVAL_EPISODES}")
    print(f"  Eval After Training: {TrainingConfig.EVAL_AFTER_TRAINING}")
    print(f"  Model Save Path: {TrainingConfig.MODEL_SAVE_PATH}")
    print(f"  Save After Training: {TrainingConfig.SAVE_AFTER_TRAINING}")
    print(f"  Generate Plots: {TrainingConfig.GENERATE_PLOTS}")
    print(f"  Plot File Prefix: {TrainingConfig.PLOT_PREFIX}")
    print("=" * 60)


def get_all_params():
    """Returns all parameters as nested dictionary"""
    return {
        'environment': {
            'width': EnvConfig.WIDTH,
            'height': EnvConfig.HEIGHT,
            'paddle_size': f"{EnvConfig.PADDLE_WIDTH}×{EnvConfig.PADDLE_HEIGHT}",
            'ball_size': EnvConfig.BALL_SIZE,
            'max_score': EnvConfig.MAX_SCORE,
            'cpu_difficulty': EnvConfig.OPPONENT_DIFFICULTY,
            'constant_ball_speed': EnvConfig.BALL_SPEED_CONSTANT,
            'loser_serves': EnvConfig.LOSER_SERVES,
        },
        'rewards': TrainingConfig.get_reward_structure(),
        'policy_gradient_agent': {
            'input_size': PolicyGradientAgentConfig.INPUT_SIZE,
            'hidden_size': PolicyGradientAgentConfig.HIDDEN_SIZE,
            'total_parameters': PolicyGradientAgentConfig.get_total_parameters(),
            'learning_rate': PolicyGradientAgentConfig.LEARNING_RATE,
        },
        'cnn_agent': {
            'input_size': CNNAgentConfig.INPUT_SIZE,
            'hidden_size': CNNAgentConfig.HIDDEN_SIZE,
            'total_parameters': CNNAgentConfig.get_total_parameters(),
            'learning_rate': CNNAgentConfig.LEARNING_RATE,
        },
        'training': TrainingConfig.get_training_params(),
        'visual': {
            'resolution': f"{VisualConfig.RENDER_WIDTH}×{VisualConfig.RENDER_HEIGHT}",
            'atari_authentic': True,
        }
    }


# =============================================================================
# AUTO-VALIDATION
# =============================================================================

if __name__ == "__main__":
    validate_config()
    print_config_summary()
else:
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠ Configuration Error: {e}")