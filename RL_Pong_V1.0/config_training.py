# config_training.py
# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
import json
import os
# The datetime import is removed as timestamp generation is moved to the Orchestrator.

json_path = "config/config_training.json"
model_path = "MODEL_SAVE_PATH" # Placeholder, Orchestrator creates the final path

# CRITICAL FIX: The global timestamp variable that caused NameError is removed.
# All timestamp logic is centralized in TrainingOrchestrator.

class TrainingConfig:
    """Training loop configuration - All hyperparameters for agent learning"""
    
    # =========================================================================
    # DEFAULT PARAMETERS
    # =========================================================================
    
    EXPERIMENT_NAME = "default_experiment"
    DESCRIPTION = "Default description"
    
    # REWARD STRUCTURE
    REWARD_SCORE = 1.0
    REWARD_OPPONENT_SCORE = -1.0
    REWARD_BALL_HIT = 0.1
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    
    # TRAINING LOOP
    MAX_EPISODES = 1000
    BATCH_SIZE = 1
    PRINT_EVERY_N_EPISODES = 50
    RUNNING_REWARD_DECAY = 0.99
    
    # EVALUATION
    EVAL_EPISODES = 10
    EVAL_AFTER_TRAINING = True
    
    # MODEL PERSISTENCE
    # NOTE: The Orchestrator will insert the experiment name and timestamp here.
    MODEL_SAVE_PATH = "models/defaultname" 
    SAVE_AFTER_TRAINING = True
    LOAD_EXISTING_MODEL = False
    
    # PLOTTING
    GENERATE_PLOTS = True
    PLOT_WINDOW_SIZE = 50
    
    # CRITICAL FIX: Replaced dynamic path with a static prefix to fix NameError.
    # TrainingAnalytics will handle the final path construction: 
    # {output_dir}/{timestamp}_pong_training_results_{plot_number}.png
    PLOT_PREFIX = "pong_training_results"
    
    # The original PLOT_SAVE_PATH attribute is now obsolete in this file.
    
    # EARLY STOPPING
    USE_EARLY_STOPPING = False
    EARLY_STOP_THRESHOLD = 0.5
    EARLY_STOP_PATIENCE = 100
    
    # =========================================================================
    # LOAD CONFIG FROM JSON
    # =========================================================================
    
    @classmethod
    def load_from_json(cls, json_path=json_path):
        """
        Override default parameters from a JSON file if it exists.
        """
        if not os.path.exists(json_path):
            print(f"⚠ No config file found at {json_path}. Using defaults.")
            return
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        for key, value in data.items():
            key_upper = key.upper()
            if hasattr(cls, key_upper):
                setattr(cls, key_upper, value)
                print(f"✓ Loaded {key_upper} = {value}")
            else:
                print(f"⚠ Unknown config key: {key}")

    
    @classmethod
    def save_active_config(cls, save_dir="logs"):
        """
        [DEPRECATED] This method's logic is now handled by TrainingOrchestrator
        to ensure the correct timestamp and save location are used.
        """
        # The Orchestrator handles saving the config to the correct timestamped folder.
        pass
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @classmethod
    def get_reward_structure(cls):
        return {
            "score": cls.REWARD_SCORE,
            "opponent_score": cls.REWARD_OPPONENT_SCORE,
            "ball_hit": cls.REWARD_BALL_HIT,
            "win": cls.REWARD_WIN,
            "loss": cls.REWARD_LOSS,
        }
    
    @classmethod
    def get_training_params(cls):
        return {
            "max_episodes": cls.MAX_EPISODES,
            "batch_size": cls.BATCH_SIZE,
            "print_every": cls.PRINT_EVERY_N_EPISODES,
            "running_reward_decay": cls.RUNNING_REWARD_DECAY,
            "eval_episodes": cls.EVAL_EPISODES,
        }

# Auto-load from JSON when module is imported
TrainingConfig.load_from_json()