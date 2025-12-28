# main_train_pong_gradient_agent.py
"""
Unified Main Training Script with Integrated Heatmap Recording
Completely modular - swappable trainers, agents, and environments.

The TrainingOrchestrator now handles heatmap recording automatically!
Experiment name and parameters loaded from config files!

To use a different trainer/agent:
    1. Create your new trainer class (implements Trainer interface)
    2. Create your new agent class (implements Agent interface)
    3. Change the imports below
    4. Everything else stays the same!

Usage:
    python main_train_pong_gradient_agent.py

To run different experiment:
    
    1. Edit config/config_training.json
       - Change "EXPERIMENT_NAME": "pong_baseline" → "pong_my_experiment"
       - Change other parameters as needed
    2. Run: python main_train_pong_gradient_agent.py
    3. Output automatically named from config!

Output:
    outputs/
    └── {EXPERIMENT_NAME}_{TIMESTAMP}/
        ├── model.npz
        ├── config_used.json
        ├── training.csv
        ├── evaluation.csv
        ├── training.log
        ├── heatmap.json
        └── [plots]
"""
import os
import sys
import datetime
from pathlib import Path

from env_custom_pong_simulator import CustomPongSimulator
from rl_feature_extractor import FeatureExtractor
from rl_policy_gradient_agent import PolicyGradientAgent
from rl_pong_training_complete import PongTrainer
from training_orchestrator import TrainingOrchestrator
from heatmap_recorder import HeatmapRecorder
from config import EnvConfig, TrainingConfig


class DualLogger:
    """Logs to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.console = sys.stdout
        
    def write(self, message):
        self.console.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        self.console.flush()


def main():
    """
    Main training entry point with config-driven naming.
    
    Flow:
    1. Load configs from JSON files
    2. Create environment
    3. Create agent with config parameters
    4. Create trainer
    5. Create heatmap recorder
    6. Pass all to TrainingOrchestrator (with experiment_name from config)
    7. Orchestrator handles: training, analytics, heatmap recording, evaluation, saving
    
    All parameters controlled by config files - no code changes needed!
    """
    
    # =========================================================================
    # STEP 0: LOAD CONFIGURATION
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("LOADING CONFIGURATION")
    print("=" * 70)
    
    # Load training config
    print("\n✓ Loading training config from config/config_training.json...")
    TrainingConfig.load_from_json("config/config_training.json")
    
    # Get experiment name from config (NEW!)
    experiment_name = TrainingConfig.EXPERIMENT_NAME
    experiment_description = getattr(TrainingConfig, 'DESCRIPTION', '')
    
    print(f"  Experiment Name: {experiment_name}")
    print(f"  Description: {experiment_description}")
    print(f"  Max Episodes: {TrainingConfig.MAX_EPISODES}")
    print(f"  Batch Size: {TrainingConfig.BATCH_SIZE}")
    
    # Save active config
    TrainingConfig.save_active_config()
    
    # =========================================================================
    # STEP 1: CREATE OUTPUT DIRECTORY FOR LOGS
    # =========================================================================
    
    # Create outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Setup logging (will be replaced by orchestrator logging later)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_log_path = outputs_dir / "training_init.log"
    
    original_stdout = sys.stdout
    sys.stdout = DualLogger(str(temp_log_path))
    
    try:
        # =====================================================================
        # STEP 2: CREATE ENVIRONMENT
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("INITIALIZING TRAINING COMPONENTS")
        print("=" * 70)
        
        print("\n✓ Creating environment...")
        env = CustomPongSimulator(**EnvConfig.get_env_params())
        feature_extractor = FeatureExtractor(env)
        print(f"  Court: {env.width}×{env.height}")
        print(f"  Paddle: {EnvConfig.PADDLE_WIDTH}×{EnvConfig.PADDLE_HEIGHT}")
        print(f"  CPU Difficulty: {EnvConfig.OPPONENT_DIFFICULTY}")
        
        # =====================================================================
        # STEP 3: CREATE AGENT
        # =====================================================================
        
        print("\n✓ Creating agent (PolicyGradient)...")
        agent = PolicyGradientAgent()
        print(f"  Input Size: {agent.INPUT_SIZE}")
        print(f"  Hidden Size: {agent.HIDDEN_SIZE}")
        print(f"  Output Size: {agent.OUTPUT_SIZE}")
        print(f"  Learning Rate: {agent.LEARNING_RATE}")
        print(f"  Discount Factor: {agent.DISCOUNT_FACTOR}")
        
        # =====================================================================
        # STEP 4: CREATE TRAINER
        # =====================================================================
        
        print("\n✓ Creating trainer (PolicyGradient)...")
        trainer = PongTrainer(
            agent=agent,
            env=env,
            feature_extractor=feature_extractor,
            max_episodes=TrainingConfig.MAX_EPISODES
        )
        print(f"  Max Episodes: {TrainingConfig.MAX_EPISODES}")
        print(f"  Batch Size: {TrainingConfig.BATCH_SIZE}")
        print(f"  Running Reward Decay: {TrainingConfig.RUNNING_REWARD_DECAY}")
        
        # =====================================================================
        # STEP 5: CREATE HEATMAP RECORDER
        # =====================================================================
        
        print("\n✓ Creating heatmap recorder...")
        heatmap_recorder = HeatmapRecorder(
            court_width=EnvConfig.WIDTH,
            court_height=EnvConfig.HEIGHT,
            grid_resolution=20
        )
        print(f"  Grid Resolution: {heatmap_recorder.grid_res}×{heatmap_recorder.grid_height}")
        print(f"  Court Size: {EnvConfig.WIDTH}×{EnvConfig.HEIGHT}")
        
        # =====================================================================
        # STEP 6: CREATE ORCHESTRATOR (with experiment name from config!)
        # =====================================================================
        
        print("\n✓ Creating training orchestrator...")
        print(f"  Experiment: {experiment_name}")
        print("  Orchestrator will handle:")
        print("    - Training pipeline")
        print("    - Analytics logging")
        print("    - Heatmap recording")
        print("    - Model evaluation")
        print("    - Automatic saving")
                
        orchestrator = TrainingOrchestrator(
            agent=agent,
            trainer=trainer,
            heatmap_recorder= None
        )
        
        # =====================================================================
        # STEP 7: RUN TRAINING
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        results = orchestrator.run()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\n✓ Results saved to: {orchestrator.output_dir}/")
        print(f"  - Model: model.npz")
        print(f"  - Config: config_used.json")
        print(f"  - Metrics: training.csv, evaluation.csv")
        print(f"  - Heatmap: heatmap.json")
        print(f"  - Log: training.log")
        
        return results
    
    finally:
        # Restore stdout
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()